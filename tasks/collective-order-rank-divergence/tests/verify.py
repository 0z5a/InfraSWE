from __future__ import annotations

import copy
import hashlib
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
        "candidate_collective_policy", repo / "collective_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load collective_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, steps, world_size, config) -> tuple[Any, str | None, bool]:
    step_copy = copy.deepcopy(steps)
    config_copy = copy.deepcopy(config)
    try:
        plan = module.build_collective_schedule(step_copy, world_size, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    return plan, error, step_copy == steps and config_copy == config


def valid_ready(
    plan: Any, steps: list[list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, bool]:
    canonical = config.get("canonical_order")
    metadata = {step["id"]: step for step in steps[0]}
    encoded = (
        json.dumps(
            [metadata[identifier] for identifier in canonical],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if isinstance(canonical, list) and all(identifier in metadata for identifier in canonical)
        else b""
    )
    expected_fingerprint = hashlib.sha256(encoded).hexdigest()
    return {
        "canonical_schedule": isinstance(plan, dict) and plan.get("schedule") == canonical,
        "divergence_reported": isinstance(plan, dict) and plan.get("divergence_detected") is True,
        "fingerprint_valid": isinstance(plan, dict)
        and plan.get("fingerprint") == expected_fingerprint,
        "rank_schedules_identical": isinstance(plan, dict)
        and plan.get("rank_schedules") == {str(rank): canonical for rank in range(len(steps))},
        "ready_schema": isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == "ready"
        and plan.get("world_size") == len(steps)
        and plan.get("fallback_reported") is False,
    }


def explicitly_blocked(plan: Any, reason: str) -> bool:
    return bool(
        isinstance(plan, dict)
        and plan.get("schema_version") == "1"
        and plan.get("status") == "blocked"
        and plan.get("schedule") == []
        and plan.get("rank_schedules") == {}
        and plan.get("fingerprint") is None
        and plan.get("divergence_detected") is True
        and plan.get("reason") == reason
        and plan.get("fallback_reported") is True
    )


try:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()}")
    workload = load_json(workload_dir / "orders.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "order_config.json")
    module = load_policy()
    steps = workload["rank_steps"]
    world_size = int(workload["world_size"])
    plan, plan_error, immutable = call(module, steps, world_size, config)
    checks = valid_ready(plan, steps, config)
    reordered_steps = [list(reversed(rank_steps)) for rank_steps in steps]
    reordered, reordered_error, reordered_immutable = call(
        module, reordered_steps, world_size, config
    )
    order_independent = plan_error is None and reordered_error is None and plan == reordered
    metadata_mismatch = copy.deepcopy(steps)
    metadata_mismatch[1][2]["elements"] += 1
    mismatch_plan, mismatch_error, mismatch_immutable = call(
        module, metadata_mismatch, world_size, config
    )
    metadata_blocked = mismatch_error is None and explicitly_blocked(
        mismatch_plan, "rank_collective_metadata_mismatch"
    )
    missing = copy.deepcopy(steps)
    missing[1].pop()
    missing_plan, missing_error, missing_immutable = call(module, missing, world_size, config)
    missing_blocked = missing_error is None and explicitly_blocked(
        missing_plan, "rank_collective_set_mismatch"
    )
    malformed_rejected = False
    try:
        module.build_collective_schedule([], world_size, config)
    except Exception:
        malformed_rejected = True
    regression = {
        **checks,
        "input_immutable": immutable
        and reordered_immutable
        and mismatch_immutable
        and missing_immutable,
        "malformed_input_rejected": malformed_rejected,
        "metadata_mismatch_blocked": metadata_blocked,
        "missing_collective_blocked": missing_blocked,
        "order_independent": order_independent,
    }
    plan_path = evidence / "collective-plan.json"
    write_json(
        "collective-plan.json",
        plan if isinstance(plan, dict) else {"error": plan_error, "plan": plan},
    )
    safe_to_execute = all(checks.values()) and order_independent
    workload_env = os.environ.copy()
    workload_env.update(
        {
            "INFRASWE_EVIDENCE_DIR": str(evidence),
            "INFRASWE_ORDER_CASES": str(workload_dir / "orders.json"),
            "INFRASWE_ORDER_PLAN": str(plan_path),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "WARN",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    if safe_to_execute:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--master-addr=127.0.0.1",
                "--master-port=29721",
                str(Path(__file__).with_name("order_workload.py")),
            ],
            env=workload_env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        (evidence / "torchrun.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (evidence / "torchrun.stderr.log").write_text(completed.stderr, encoding="utf-8")
        returncode = completed.returncode
    else:
        returncode = 125
    rank_results = [
        load_json(evidence / f"rank-{rank}.json")
        for rank in range(world_size)
        if (evidence / f"rank-{rank}.json").is_file()
    ]
    correct = len(rank_results) == world_size and all(
        result.get("correct") is True for result in rank_results
    )
    resources_cleaned = returncode != 124 and all(
        result["memory_after_bytes"] <= result["memory_before_bytes"] + 1_048_576
        for result in rank_results
    )
    collective_ms = max(
        (float(result["collective_ms"]) for result in rank_results), default=999999.0
    )
    maximum_collective = float(workload["maximum_collective_ms"])
    assertions = {
        "functional": {
            "collectives_correct": correct,
            "two_ranks_completed": len(rank_results) == world_size and returncode == 0,
        },
        "mechanism": {
            "canonical_order_before_nccl": checks.get("rank_schedules_identical", False),
            "rank_divergence_detected": checks.get("divergence_reported", False),
        },
        "regression": regression,
        "safety": {
            "mismatched_metadata_blocked": metadata_blocked,
            "no_data_corruption": correct,
            "resources_cleaned": resources_cleaned,
        },
        "slo": {"collective_latency": collective_ms <= maximum_collective},
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "collective-metadata-mismatch": metadata_blocked,
        "rank-order-divergence": checks.get("divergence_reported", False)
        and checks.get("rank_schedules_identical", False),
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
        "passed": correct and resources_cleaned,
        "resource_leak": not resources_cleaned,
        "silent_fallback": not checks.get("divergence_reported", False),
    }
    metrics = {
        "collective_goodput_ratio": float(correct),
        "collective_ms": collective_ms,
        "observability_ratio": float(plan_path.is_file() and len(rank_results) == world_size),
        "resource_efficiency_ratio": min(
            1.0, maximum_collective / max(collective_ms, maximum_collective)
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
        "collective_goodput_ratio": 0.0,
        "observability_ratio": 0.25,
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
