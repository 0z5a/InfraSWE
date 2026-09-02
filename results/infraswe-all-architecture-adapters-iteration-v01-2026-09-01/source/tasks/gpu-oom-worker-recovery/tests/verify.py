from __future__ import annotations

import copy
import gc
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
    spec = importlib.util.spec_from_file_location("candidate_recovery", repo / "recovery.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, event, state, config) -> tuple[Any, str | None, bool]:
    event_copy = copy.deepcopy(event)
    state_copy = copy.deepcopy(state)
    config_copy = copy.deepcopy(config)
    try:
        plan = module.plan_recovery(event_copy, state_copy, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    immutable = event_copy == event and state_copy == state and config_copy == config
    return plan, error, immutable


def run_oom_probe() -> tuple[dict[str, Any], dict[str, Any], float]:
    output = evidence / "oom-probe.json"
    environment = os.environ.copy()
    environment["INFRASWE_OOM_PROBE_OUTPUT"] = str(output)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("oom_probe.py"))],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )
    elapsed = time.monotonic() - started
    process = {
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-4000:],
        "stdout": completed.stdout[-4000:],
    }
    probe = load_json(output) if output.is_file() else {}
    return probe, process, elapsed


def valid_retry_plan(plan: Any, expected_status: str, expected_worker: str) -> bool:
    return (
        isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == expected_status
        and plan.get("failed_worker_id") == "worker-0"
        and plan.get("retry_worker_id") == expected_worker
        and plan.get("retry_batch_size") == 4
        and plan.get("retry_request_ids") == ["request-2", "request-7"]
        and plan.get("clear_cache") is True
        and plan.get("preserve_worker_ids") == ["worker-1", "worker-2"]
        and plan.get("fallback_reported") is True
    )


def recovery_kernel() -> tuple[dict[str, Any], bool, bool]:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    before = int(torch.cuda.memory_allocated(0))
    source = torch.arange(65536, device="cuda:0", dtype=torch.float32)
    output = source.square()
    torch.cuda.synchronize(0)
    expected = torch.tensor([0, 1, 4, 9], device="cuda:0", dtype=torch.float32)
    correct = bool(torch.equal(output[:4], expected))
    steady = int(torch.cuda.memory_allocated(0))
    del expected, output, source
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    after = int(torch.cuda.memory_allocated(0))
    return (
        {
            "memory_after_bytes": after,
            "memory_before_bytes": before,
            "memory_steady_bytes": steady,
        },
        correct,
        after <= before + 1_048_576,
    )


try:
    workload = load_json(workload_dir / "recovery.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "recovery_config.json")
    module = load_policy()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}")
    probe, process, recovery_seconds = run_oom_probe()
    actual_oom = (
        process["returncode"] == 0
        and probe.get("caught_cuda_oom") is True
        and probe.get("message_contains_oom") is True
    )
    child_cleaned = (
        isinstance(probe.get("memory_before_bytes"), int)
        and isinstance(probe.get("memory_after_bytes"), int)
        and probe["memory_after_bytes"] <= probe["memory_before_bytes"] + 1_048_576
    )

    event = {
        "batch_size": 8,
        "failed_request_ids": ["request-7", "request-2"],
        "kind": "cuda_oom",
        "worker_id": "worker-0",
    }
    state = {
        "oom_count_by_worker": {"worker-0": 1},
        "workers": [
            {"id": "worker-0", "status": "failed"},
            {"id": "worker-2", "status": "healthy"},
            {"id": "worker-1", "status": "healthy"},
        ],
    }
    first, first_error, immutable = call(module, event, state, config)
    first_valid = valid_retry_plan(first, "retry", "worker-0") and (
        first.get("restart_worker") is True
        and first.get("restarted_worker_ids") == ["worker-0"]
        and first.get("quarantined_worker_ids") == []
        and first.get("reason") == "cuda_oom_retry_smaller_batch"
    )
    repeated_state = copy.deepcopy(state)
    repeated_state["oom_count_by_worker"]["worker-0"] = 2
    repeated, repeated_error, repeated_immutable = call(module, event, repeated_state, config)
    repeated_valid = valid_retry_plan(repeated, "quarantine", "worker-1") and (
        repeated.get("restart_worker") is False
        and repeated.get("restarted_worker_ids") == []
        and repeated.get("quarantined_worker_ids") == ["worker-0"]
        and repeated.get("reason") == "repeated_cuda_oom_quarantine"
    )
    reordered, reordered_error, reordered_immutable = call(
        module,
        {**event, "failed_request_ids": list(reversed(event["failed_request_ids"]))},
        {**state, "workers": list(reversed(state["workers"]))},
        config,
    )
    deterministic = reordered_error is None and first_error is None and reordered == first
    minimum_event = {**event, "batch_size": 1}
    minimum, minimum_error, minimum_immutable = call(module, minimum_event, state, config)
    minimum_bounded = minimum_error is None and minimum.get("retry_batch_size") == 1
    malformed_rejected = False
    try:
        module.plan_recovery({}, state, config)
    except Exception:
        malformed_rejected = True
    no_failover_rejected = False
    try:
        module.plan_recovery(
            event,
            {
                "oom_count_by_worker": {"worker-0": 2},
                "workers": [{"id": "worker-0", "status": "failed"}],
            },
            config,
        )
    except Exception:
        no_failover_rejected = True
    kernel_evidence, gpu_correct, parent_cleaned = recovery_kernel()
    all_requests_retried = (
        first_valid
        and repeated_valid
        and set(first["retry_request_ids"]) == set(event["failed_request_ids"])
        and set(repeated["retry_request_ids"]) == set(event["failed_request_ids"])
    )
    healthy_preserved = first_valid and repeated_valid
    explicit_recovery = first_valid and repeated_valid
    resources_cleaned = child_cleaned and parent_cleaned
    regression = {
        "actual_cuda_oom_observed": actual_oom,
        "deterministic": deterministic,
        "healthy_workers_preserved": healthy_preserved,
        "input_immutable": immutable
        and repeated_immutable
        and reordered_immutable
        and minimum_immutable,
        "malformed_event_rejected": malformed_rejected,
        "minimum_batch_bounded": minimum_bounded,
        "no_failover_fails_closed": no_failover_rejected,
        "repeated_oom_quarantined": repeated_valid,
    }
    write_json(
        "recovery-evidence.json",
        {
            "first_error": first_error,
            "first_plan": first,
            "gpu_recovery": kernel_evidence,
            "oom_probe": probe,
            "oom_probe_process": process,
            "repeated_error": repeated_error,
            "repeated_plan": repeated,
        },
    )
    maximum_recovery = float(workload["maximum_recovery_seconds"])
    minimum_goodput = float(workload["minimum_goodput_ratio"])
    goodput_ratio = 1.0 if all_requests_retried else 0.0
    assertions = {
        "functional": {
            "all_failed_requests_retried": all_requests_retried,
            "gpu_usable_after_oom": gpu_correct,
        },
        "mechanism": {
            "actual_cuda_oom_observed": actual_oom,
            "failed_worker_cache_cleared": first_valid,
            "repeated_oom_quarantined": repeated_valid,
        },
        "regression": regression,
        "safety": {
            "healthy_workers_not_restarted": healthy_preserved,
            "recovery_explicitly_reported": explicit_recovery,
            "resources_cleaned": resources_cleaned,
        },
        "slo": {
            "recovery_time": recovery_seconds <= maximum_recovery,
            "retry_goodput": goodput_ratio >= minimum_goodput,
        },
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "actual-cuda-oom": actual_oom and first_valid,
        "repeated-oom-quarantine": repeated_valid,
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    policy_result = {
        "data_corruption": not gpu_correct,
        "hard_failures": [],
        "passed": gpu_correct and resources_cleaned and explicit_recovery,
        "resource_leak": not resources_cleaned,
        "silent_fallback": not explicit_recovery,
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "observability_ratio": float((evidence / "recovery-evidence.json").is_file()),
        "recovery_goodput_ratio": min(1.0, goodput_ratio / minimum_goodput),
        "recovery_seconds": recovery_seconds,
        "resource_efficiency_ratio": float(healthy_preserved and resources_cleaned),
        "topology_robustness_ratio": regression_ratio,
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
