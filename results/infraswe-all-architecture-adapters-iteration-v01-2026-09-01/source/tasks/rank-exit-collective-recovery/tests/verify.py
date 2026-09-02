from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

repo = Path(os.environ["INFRASWE_REPO"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
workload_dir = Path(os.environ["INFRASWE_WORKLOAD_DIR"])
faults_dir = Path(os.environ["INFRASWE_FAULTS_DIR"])
evidence.mkdir(parents=True, exist_ok=True)


def write_json(name: str, value: object) -> None:
    (evidence / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def load_policy():
    spec = importlib.util.spec_from_file_location(
        "candidate_failure_policy", repo / "failure_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load failure_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, event, state, config) -> tuple[Any, str | None, bool]:
    event_copy = copy.deepcopy(event)
    state_copy = copy.deepcopy(state)
    config_copy = copy.deepcopy(config)
    try:
        plan = module.plan_rank_failure(event_copy, state_copy, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    immutable = event_copy == event and state_copy == state and config_copy == config
    return plan, error, immutable


def valid_restart(plan: Any) -> bool:
    return bool(
        isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == "restart_group"
        and plan.get("failed_rank") == 1
        and plan.get("world_size") == 2
        and plan.get("abort_ranks") == [0, 1]
        and plan.get("restart_ranks") == [0, 1]
        and plan.get("resume_step") == 40
        and plan.get("replay_request_ids") == ["request-2", "request-7"]
        and plan.get("reinitialize_process_group") is True
        and plan.get("reason") == "abort_and_reinitialize_full_group"
        and plan.get("fallback_reported") is True
    )


def valid_abort(plan: Any) -> bool:
    return bool(
        isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == "abort"
        and plan.get("abort_ranks") == [0, 1]
        and plan.get("restart_ranks") == []
        and plan.get("resume_step") == 40
        and plan.get("replay_request_ids") == ["request-2", "request-7"]
        and plan.get("reinitialize_process_group") is False
        and plan.get("reason") == "restart_budget_exhausted"
        and plan.get("fallback_reported") is True
    )


def torchrun(script: str, port: int, environment: dict[str, str], timeout: int):
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            "--max-restarts=0",
            "--monitor-interval=0.1",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            str(Path(__file__).with_name(script)),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return completed, time.monotonic() - started


try:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()}")
    workload = load_json(workload_dir / "recovery.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "recovery_config.json")
    module = load_policy()
    event = {
        "failed_operation": "all_reduce",
        "failed_rank": 1,
        "kind": "rank_exit",
        "observed_step": 41,
    }
    state = {
        "last_committed_step": 40,
        "pending_request_ids": ["request-7", "request-2"],
        "ranks": [
            {"rank": 0, "status": "running"},
            {"rank": 1, "status": "exited"},
        ],
        "restart_count": 0,
        "world_size": 2,
    }
    plan, plan_error, immutable = call(module, event, state, config)
    restart_valid = valid_restart(plan)
    reordered, reordered_error, reordered_immutable = call(
        module,
        event,
        {
            **state,
            "pending_request_ids": list(reversed(state["pending_request_ids"])),
            "ranks": list(reversed(state["ranks"])),
        },
        config,
    )
    deterministic = plan_error is None and reordered_error is None and plan == reordered
    exhausted_state = {**state, "restart_count": 1}
    exhausted, exhausted_error, exhausted_immutable = call(module, event, exhausted_state, config)
    exhausted_valid = exhausted_error is None and valid_abort(exhausted)
    unsafe_config = {**config, "forbid_world_size_shrink": False}
    _, unsafe_error, unsafe_immutable = call(module, event, state, unsafe_config)
    unsafe_rejected = unsafe_error is not None
    malformed_rejected = False
    try:
        module.plan_rank_failure({}, state, config)
    except Exception:
        malformed_rejected = True
    regression = {
        "checkpoint_resume": restart_valid,
        "deterministic": deterministic,
        "full_group_abort": restart_valid,
        "input_immutable": immutable
        and reordered_immutable
        and exhausted_immutable
        and unsafe_immutable,
        "malformed_event_rejected": malformed_rejected,
        "restart_budget_enforced": exhausted_valid,
        "world_size_shrink_rejected": unsafe_rejected,
    }
    plan_path = evidence / "recovery-plan.json"
    write_json(
        "recovery-plan.json",
        plan if isinstance(plan, dict) else {"error": plan_error, "plan": plan},
    )
    common_env = os.environ.copy()
    common_env.update(
        {
            "INFRASWE_EVIDENCE_DIR": str(evidence),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "WARN",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    fault_run, detection_seconds = torchrun("rank_exit_probe.py", 29741, common_env, 60)
    (evidence / "fault.stdout.log").write_text(fault_run.stdout, encoding="utf-8")
    (evidence / "fault.stderr.log").write_text(fault_run.stderr, encoding="utf-8")
    injected = (
        load_json(evidence / "injected-rank-exit.json")
        if (evidence / "injected-rank-exit.json").is_file()
        else {}
    )
    actual_exit_observed = (
        fault_run.returncode != 0
        and injected.get("rank") == 1
        and injected.get("exit_code") == 17
        and detection_seconds <= float(workload["maximum_failure_detection_seconds"])
    )

    recovery_seconds = 999999.0
    recovery_returncode = 125
    if restart_valid:
        recovery_env = {**common_env, "INFRASWE_RECOVERY_PLAN": str(plan_path)}
        recovery_run, recovery_seconds = torchrun("recovery_workload.py", 29742, recovery_env, 120)
        recovery_returncode = recovery_run.returncode
        (evidence / "recovery.stdout.log").write_text(recovery_run.stdout, encoding="utf-8")
        (evidence / "recovery.stderr.log").write_text(recovery_run.stderr, encoding="utf-8")
    rank_results = [
        load_json(evidence / f"recovered-rank-{rank}.json")
        for rank in range(2)
        if (evidence / f"recovered-rank-{rank}.json").is_file()
    ]
    recovery_correct = len(rank_results) == 2 and all(
        result.get("correct") is True
        and result.get("resume_step") == 40
        and result.get("replayed_request_ids") == ["request-2", "request-7"]
        for result in rank_results
    )
    resources_cleaned = (
        detection_seconds < 60
        and recovery_returncode != 124
        and all(
            result["memory_after_bytes"] <= result["memory_before_bytes"] + 1_048_576
            for result in rank_results
        )
    )
    goodput = 1.0 if recovery_correct else 0.0
    minimum_goodput = float(workload["minimum_recovery_goodput_ratio"])
    assertions = {
        "functional": {
            "actual_rank_exit_observed": actual_exit_observed,
            "recovered_collective_correct": recovery_correct,
            "two_ranks_rejoined": len(rank_results) == 2 and recovery_returncode == 0,
        },
        "mechanism": {
            "full_group_reinitialized": restart_valid,
            "resumed_last_checkpoint": recovery_correct,
        },
        "regression": regression,
        "safety": {
            "no_world_size_shrink": restart_valid,
            "resources_cleaned": resources_cleaned,
            "restart_budget_explicit": exhausted_valid,
        },
        "slo": {
            "failure_detection": detection_seconds
            <= float(workload["maximum_failure_detection_seconds"]),
            "recovery_time": recovery_seconds <= float(workload["maximum_recovery_seconds"]),
            "replay_goodput": goodput >= minimum_goodput,
        },
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "abrupt-rank-exit": actual_exit_observed and restart_valid and recovery_correct,
        "restart-budget-exhausted": exhausted_valid,
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "detection_seconds": detection_seconds,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    policy_result = {
        "data_corruption": not recovery_correct,
        "hard_failures": [],
        "passed": recovery_correct and resources_cleaned and restart_valid,
        "resource_leak": not resources_cleaned,
        "silent_fallback": not restart_valid,
    }
    metrics = {
        "failure_detection_seconds": detection_seconds,
        "observability_ratio": float(actual_exit_observed and plan_path.is_file()),
        "recovery_goodput_ratio": min(1.0, goodput / minimum_goodput),
        "recovery_seconds": recovery_seconds,
        "resource_efficiency_ratio": float(resources_cleaned),
        "topology_robustness_ratio": sum(regression.values()) / len(regression),
    }
except Exception as error:
    assertions = {"functional": {"verifier_completed": False}}
    faults_result = {"error": str(error), "passed": False, "score": 0.0}
    policy_result = {
        "data_corruption": False,
        "hard_failures": [],
        "passed": False,
        "resource_leak": False,
        "silent_fallback": False,
        "verifier_error": str(error),
    }
    metrics = {
        "observability_ratio": 0.25,
        "recovery_goodput_ratio": 0.0,
        "resource_efficiency_ratio": 0.0,
        "topology_robustness_ratio": 0.0,
    }

write_json("assertions.json", assertions)
write_json("faults.json", faults_result)
write_json("policy.json", policy_result)
write_json("metrics.json", metrics)
passed = all(
    value
    for group in assertions.values()
    for value in (group.values() if isinstance(group, dict) else [group])
)
raise SystemExit(0 if passed and faults_result["passed"] and policy_result["passed"] else 1)
