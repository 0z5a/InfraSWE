from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
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
        "candidate_overlap_policy", repo / "overlap_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load overlap_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, stages, topology, config) -> tuple[Any, str | None, bool]:
    stage_copy = copy.deepcopy(stages)
    topology_copy = copy.deepcopy(topology)
    config_copy = copy.deepcopy(config)
    try:
        plan = module.build_overlap_plan(stage_copy, topology_copy, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    immutable = stage_copy == stages and topology_copy == topology and config_copy == config
    return plan, error, immutable


def ready_checks(plan: Any, stages: list[dict[str, Any]]) -> dict[str, bool]:
    ordered = sorted(stages, key=lambda stage: (stage["sequence"], stage["id"]))
    expected_stages = [
        {
            "collective_stream": "communication",
            "compute_stream": "default",
            "overlap_with_next": index < len(ordered) - 1,
            "sequence": stage["sequence"],
            "stage_id": stage["id"],
            "wait_with_event": True,
        }
        for index, stage in enumerate(ordered)
    ]
    return {
        "async_collectives": isinstance(plan, dict) and plan.get("async_collectives") is True,
        "dedicated_comm_stream": isinstance(plan, dict) and plan.get("comm_stream") is True,
        "event_fencing": isinstance(plan, dict) and plan.get("event_fencing") is True,
        "overlap_next_compute": isinstance(plan, dict) and plan.get("overlap_next_compute") is True,
        "ready_schema": isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == "ready"
        and plan.get("fallback_reported") is False,
        "stage_pipeline": isinstance(plan, dict) and plan.get("stages") == expected_stages,
    }


def explicitly_blocked(plan: Any) -> bool:
    return bool(
        isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == "blocked"
        and plan.get("stages") == []
        and plan.get("fallback_reported") is True
        and all(
            plan.get(key) is False
            for key in (
                "comm_stream",
                "async_collectives",
                "event_fencing",
                "overlap_next_compute",
            )
        )
    )


try:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()}")
    workload = load_json(workload_dir / "stages.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "overlap_config.json")
    module = load_policy()
    stages = workload["stages"]
    topology = {"concurrent_kernels": True, "device_count": 2}
    plan, plan_error, immutable = call(module, stages, topology, config)
    checks = ready_checks(plan, stages)
    reversed_plan, reversed_error, reversed_immutable = call(
        module, list(reversed(stages)), topology, config
    )
    order_independent = plan_error is None and reversed_error is None and plan == reversed_plan
    unsupported, unsupported_error, unsupported_immutable = call(
        module, stages, {"concurrent_kernels": False, "device_count": 2}, config
    )
    topology_blocked = unsupported_error is None and explicitly_blocked(unsupported)
    unsafe_config = {**config, "event_fencing": False}
    _, unsafe_error, unsafe_immutable = call(module, stages, topology, unsafe_config)
    unsafe_rejected = unsafe_error is not None
    malformed_rejected = False
    try:
        module.build_overlap_plan([], topology, config)
    except Exception:
        malformed_rejected = True
    regression = {
        **checks,
        "input_immutable": immutable
        and reversed_immutable
        and unsupported_immutable
        and unsafe_immutable,
        "malformed_input_rejected": malformed_rejected,
        "order_independent": order_independent,
        "unsupported_topology_blocked": topology_blocked,
        "unsafe_event_config_rejected": unsafe_rejected,
    }
    plan_path = evidence / "overlap-plan.json"
    write_json(
        "overlap-plan.json",
        plan if isinstance(plan, dict) else {"error": plan_error, "plan": plan},
    )
    executable_plan = isinstance(plan, dict) and plan.get("status") == "ready"
    workload_env = os.environ.copy()
    workload_env.update(
        {
            "INFRASWE_EVIDENCE_DIR": str(evidence),
            "INFRASWE_OVERLAP_CASES": str(workload_dir / "stages.json"),
            "INFRASWE_OVERLAP_PLAN": str(plan_path),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "WARN",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    if executable_plan:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--master-addr=127.0.0.1",
                "--master-port=29731",
                str(Path(__file__).with_name("overlap_workload.py")),
            ],
            env=workload_env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        (evidence / "torchrun.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (evidence / "torchrun.stderr.log").write_text(completed.stderr, encoding="utf-8")
        returncode = completed.returncode
    else:
        returncode = 125
    rank_results = [
        load_json(evidence / f"rank-{rank}.json")
        for rank in range(2)
        if (evidence / f"rank-{rank}.json").is_file()
    ]
    correct = len(rank_results) == 2 and all(
        result.get("correct") is True for result in rank_results
    )
    resources_cleaned = returncode != 124 and all(
        result["memory_after_bytes"] <= result["memory_before_bytes"] + 1_048_576
        for result in rank_results
    )
    speedup = min((float(result["speedup_ratio"]) for result in rank_results), default=0.0)
    candidate_ms = max((float(result["candidate_ms"]) for result in rank_results), default=999999.0)
    minimum_speedup = float(workload["minimum_speedup_ratio"])
    maximum_candidate = float(workload["maximum_candidate_ms"])
    observed_overlap = speedup >= minimum_speedup
    assertions = {
        "functional": {
            "collectives_correct": correct,
            "two_ranks_completed": len(rank_results) == 2 and returncode == 0,
        },
        "mechanism": {
            "async_comm_stream": checks.get("async_collectives", False)
            and checks.get("dedicated_comm_stream", False),
            "event_fenced_pipeline": checks.get("event_fencing", False)
            and checks.get("stage_pipeline", False),
        },
        "regression": regression,
        "safety": {
            "no_data_corruption": correct,
            "resources_cleaned": resources_cleaned,
            "unsupported_topology_explicit": topology_blocked,
        },
        "slo": {
            "candidate_latency": candidate_ms <= maximum_candidate,
            "overlap_speedup_observed": observed_overlap,
        },
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "missing-event-fence": unsafe_rejected,
        "serialized-default-stream": observed_overlap
        and checks.get("dedicated_comm_stream", False),
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    policy_result = {
        "data_corruption": not correct,
        "hard_failures": [],
        "passed": correct and resources_cleaned and observed_overlap,
        "resource_leak": not resources_cleaned,
        "silent_fallback": not checks.get("dedicated_comm_stream", False),
    }
    metrics = {
        "candidate_ms": candidate_ms,
        "observability_ratio": float(plan_path.is_file() and len(rank_results) == 2),
        "overlap_speedup_raw": speedup,
        "overlap_speedup_ratio": min(1.0, speedup / minimum_speedup),
        "resource_efficiency_ratio": min(
            1.0, maximum_candidate / max(candidate_ms, maximum_candidate)
        ),
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
        "overlap_speedup_ratio": 0.0,
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
