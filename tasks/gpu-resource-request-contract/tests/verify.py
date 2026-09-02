from __future__ import annotations

import copy
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

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


def load_admission():
    spec = importlib.util.spec_from_file_location("candidate_admission", repo / "admission.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load admission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict) or set(plan) != {
        "assigned_gpus",
        "fallback_reported",
        "reason",
        "schema_version",
        "status",
    }:
        return False
    if (
        plan.get("schema_version") != "1"
        or not isinstance(plan.get("reason"), str)
        or not plan["reason"]
        or isinstance(plan.get("assigned_gpus"), bool)
        or not isinstance(plan.get("assigned_gpus"), int)
        or not isinstance(plan.get("fallback_reported"), bool)
    ):
        return False
    if plan.get("status") == "admitted":
        return plan["assigned_gpus"] > 0 and not plan["fallback_reported"]
    return bool(
        plan.get("status") == "rejected"
        and plan["assigned_gpus"] == 0
        and plan["fallback_reported"]
    )


def call(module, spec, node, policy) -> tuple[Any, str | None, bool, float]:
    spec_copy = copy.deepcopy(spec)
    node_copy = copy.deepcopy(node)
    policy_copy = copy.deepcopy(policy)
    started = time.perf_counter()
    try:
        plan = module.admit_workload(spec_copy, node_copy, policy_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    immutable = spec_copy == spec and node_copy == node and policy_copy == policy
    return plan, error, immutable, elapsed_ms


try:
    candidate_workload = load_json(repo / "workload.json")
    contract = load_json(workload_dir / "nodes.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    node = contract["node"]
    policy = contract["policy"]
    module = load_admission()
    actual_plan, actual_error, actual_immutable, _ = call(module, candidate_workload, node, policy)
    actual_admitted = bool(
        valid_plan(actual_plan)
        and actual_plan["status"] == "admitted"
        and actual_plan["assigned_gpus"] == 1
    )
    invariants = {
        "image_preserved": candidate_workload.get("image") == "example.invalid/inference:v1",
        "name_preserved": candidate_workload.get("name") == "inference-api",
        "port_preserved": candidate_workload.get("port") == 8000,
        "replicas_preserved": candidate_workload.get("replicas") == 2,
    }
    cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    missing_request = copy.deepcopy(candidate_workload)
    missing_request.setdefault("resources", {})["requests"] = {}
    cases["missing_gpu_request"] = (missing_request, node)
    unequal = copy.deepcopy(candidate_workload)
    unequal.setdefault("resources", {}).setdefault("limits", {})["nvidia.com/gpu"] = 2
    cases["unequal_request_limit"] = (unequal, node)
    wrong_selector = copy.deepcopy(candidate_workload)
    wrong_selector["nodeSelector"] = {"infraswe/compute-capability": "9.0"}
    cases["capability_mismatch"] = (wrong_selector, node)
    runtime_node = copy.deepcopy(node)
    runtime_node["runtime_classes"] = ["runc"]
    cases["runtime_unavailable"] = (candidate_workload, runtime_node)
    exhausted_node = copy.deepcopy(node)
    exhausted_node["allocatable_gpus"] = 0
    cases["capacity_exhausted"] = (candidate_workload, exhausted_node)

    plans: dict[str, Any] = {}
    regression: dict[str, bool] = {}
    immutable = actual_immutable
    for name, (spec_value, node_value) in cases.items():
        plan, _, unchanged, _ = call(module, spec_value, node_value, policy)
        plans[name] = plan
        regression[f"{name}_rejected"] = bool(valid_plan(plan) and plan["status"] == "rejected")
        immutable = immutable and unchanged
    malformed_rejected = False
    try:
        module.admit_workload([], node, policy)
    except Exception:
        malformed_rejected = True
    regression.update(
        {
            "input_immutable": immutable,
            "malformed_input_rejected": malformed_rejected,
            "workload_invariants": all(invariants.values()),
        }
    )
    latencies = [call(module, candidate_workload, node, policy)[3] for _ in range(200)]
    p95_ms = sorted(latencies)[189]
    write_json(
        "admission-evidence.json",
        {
            "actual": {"error": actual_error, "plan": actual_plan},
            "fault_plans": plans,
            "invariants": invariants,
            "node": node,
        },
    )
    maximum_p95_ms = float(contract["maximum_admission_p95_ms"])
    assertions = {
        "functional": {
            "gpu_workload_admitted": actual_admitted,
            "public_contract_preserved": all(invariants.values()),
        },
        "mechanism": {
            "capability_selector_enforced": regression["capability_mismatch_rejected"],
            "requests_equal_limits": regression["missing_gpu_request_rejected"]
            and regression["unequal_request_limit_rejected"],
            "runtime_class_enforced": regression["runtime_unavailable_rejected"],
        },
        "regression": regression,
        "safety": {
            "capacity_failure_explicit": regression["capacity_exhausted_rejected"],
            "no_cpu_fallback": actual_admitted
            and all(
                plan.get("status") == "rejected"
                for plan in plans.values()
                if isinstance(plan, dict)
            ),
            "resources_cleaned": True,
        },
        "slo": {"admission_latency": p95_ms <= maximum_p95_ms},
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "gpu-capacity-exhausted": regression["capacity_exhausted_rejected"],
        "nvidia-runtime-unavailable": regression["runtime_unavailable_rejected"],
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    silent_fallback = not actual_admitted or any(
        not isinstance(plan, dict)
        or plan.get("status") != "rejected"
        or plan.get("fallback_reported") is not True
        for plan in plans.values()
    )
    policy_result = {
        "data_corruption": False,
        "hard_failures": [],
        "passed": True,
        "resource_leak": False,
        "silent_fallback": silent_fallback,
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "admission_p95_ms": p95_ms,
        "observability_ratio": float((evidence / "admission-evidence.json").is_file()),
        "resource_efficiency_ratio": min(1.0, 2.0 / max(p95_ms, 2.0)),
        "slo_goodput_ratio": min(1.0, maximum_p95_ms / max(p95_ms, maximum_p95_ms)),
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
        "admission_p95_ms": 999999.0,
        "observability_ratio": 0.25,
        "resource_efficiency_ratio": 0.0,
        "slo_goodput_ratio": 0.0,
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
