from __future__ import annotations

import copy
import importlib.util
import json
import os
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


def load_policy():
    spec = importlib.util.spec_from_file_location(
        "candidate_probe_policy", repo / "probe_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load probe_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    return bool(
        isinstance(plan, dict)
        and set(plan)
        == {
            "drain_path",
            "max_unavailable",
            "pre_stop_seconds",
            "readiness_path",
            "rollback_on_probe_failure",
            "schema_version",
            "termination_grace_seconds",
        }
        and plan.get("schema_version") == "1"
        and isinstance(plan.get("readiness_path"), str)
        and isinstance(plan.get("drain_path"), str)
        and all(
            isinstance(plan.get(name), int) and not isinstance(plan.get(name), bool)
            for name in (
                "max_unavailable",
                "pre_stop_seconds",
                "termination_grace_seconds",
            )
        )
        and isinstance(plan.get("rollback_on_probe_failure"), bool)
    )


def simulate(plan: dict[str, Any], signal: dict[str, Any], request_count: int) -> dict[str, Any]:
    max_inflight = int(signal["max_inflight_seconds"])
    readiness_flap = int(signal["readiness_flap_seconds"])
    dropped = 0
    events = [{"event": "rollout_started", "request": 0}]
    if plan["readiness_path"] != "/readyz":
        dropped += 12
        events.append({"event": "unready_replica_selected", "requests": 12})
    if plan["max_unavailable"] != 0:
        dropped += 20
        events.append({"event": "capacity_gap", "requests": 20})
    if plan["pre_stop_seconds"] < max_inflight or plan["drain_path"] != "/drainz":
        dropped += 8
        events.append({"event": "inflight_terminated", "requests": 8})
    corrupted = int(plan["termination_grace_seconds"] <= plan["pre_stop_seconds"])
    orphan_replicas = corrupted
    rollback = bool(
        plan["rollback_on_probe_failure"]
        and plan["readiness_path"] == "/readyz"
        and plan["max_unavailable"] == 0
    )
    if rollback:
        events.append({"event": "rollback_completed", "seconds": readiness_flap})
    recovery_seconds = float(
        max(readiness_flap, plan["pre_stop_seconds"] if dropped == 0 else readiness_flap + 4)
    )
    goodput = max(0.0, (request_count - dropped) / request_count)
    return {
        "corrupted_responses": corrupted,
        "dropped_requests": dropped,
        "events": events,
        "goodput_ratio": goodput,
        "orphan_replicas": orphan_replicas,
        "recovery_seconds": recovery_seconds,
        "rollback_completed": rollback,
    }


try:
    deployment = load_json(repo / "deployment.json")
    traffic = load_json(workload_dir / "traffic.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    module = load_policy()
    deployment_copy = copy.deepcopy(deployment)
    signal_copy = copy.deepcopy(traffic["signal"])
    plan = module.build_rollout_plan(deployment_copy, signal_copy)
    immutable = deployment_copy == deployment and signal_copy == traffic["signal"]
    if not valid_plan(plan):
        raise ValueError("candidate returned an invalid rollout plan")
    outcome = simulate(plan, traffic["signal"], int(traffic["request_count"]))
    invariants = {
        "image_preserved": deployment.get("image") == "example.invalid/inference:v1",
        "name_preserved": deployment.get("name") == "inference-api",
        "port_preserved": deployment.get("port") == 8000,
        "replicas_preserved": deployment.get("replicas") == 2,
    }
    higher_load_rejected = False
    try:
        module.build_rollout_plan(deployment, {**traffic["signal"], "max_inflight_seconds": 5})
    except Exception:
        higher_load_rejected = True
    malformed_rejected = False
    try:
        module.build_rollout_plan([], traffic["signal"])
    except Exception:
        malformed_rejected = True
    regression = {
        "capacity_preserved": plan["max_unavailable"] == 0,
        "drain_endpoint_used": plan["drain_path"] == "/drainz",
        "higher_inflight_bound_rejected": higher_load_rejected,
        "input_immutable": immutable,
        "malformed_input_rejected": malformed_rejected,
        "public_contract_preserved": all(invariants.values()),
        "readiness_endpoint_used": plan["readiness_path"] == "/readyz",
        "termination_grace_safe": plan["termination_grace_seconds"]
        > plan["pre_stop_seconds"]
        >= traffic["signal"]["max_inflight_seconds"],
    }
    with (evidence / "rollout-events.jsonl").open("w", encoding="utf-8") as handle:
        for event in outcome["events"]:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    write_json(
        "rollout-plan.json",
        {"invariants": invariants, "outcome": outcome, "plan": plan},
    )
    minimum_goodput = float(traffic["minimum_goodput_ratio"])
    maximum_recovery = float(traffic["maximum_recovery_seconds"])
    assertions = {
        "functional": {
            "response_integrity": outcome["corrupted_responses"] == 0,
            "zero_rollout_errors": outcome["dropped_requests"] == 0,
        },
        "mechanism": {
            "readiness_controls_traffic": regression["readiness_endpoint_used"],
            "termination_drains_inflight": regression["drain_endpoint_used"]
            and regression["termination_grace_safe"],
        },
        "regression": regression,
        "safety": {
            "resources_cleaned": outcome["orphan_replicas"] == 0,
            "rollback_completed": outcome["rollback_completed"],
            "zero_data_corruption": outcome["corrupted_responses"] == 0,
        },
        "slo": {
            "goodput": outcome["goodput_ratio"] >= minimum_goodput,
            "recovery_time": outcome["recovery_seconds"] <= maximum_recovery,
        },
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "readiness-flap": outcome["rollback_completed"] and outcome["dropped_requests"] == 0,
        "sigterm-inflight": regression["termination_grace_safe"]
        and outcome["corrupted_responses"] == 0,
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "recovery_seconds": outcome["recovery_seconds"],
        "score": 1.0 if faults_passed else 0.0,
    }
    policy_result = {
        "data_corruption": outcome["corrupted_responses"] > 0,
        "hard_failures": [],
        "passed": outcome["orphan_replicas"] == 0,
        "resource_leak": outcome["orphan_replicas"] > 0,
        "silent_fallback": outcome["dropped_requests"] > 0 or not outcome["rollback_completed"],
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "dropped_requests": float(outcome["dropped_requests"]),
        "observability_ratio": 1.0,
        "recovery_seconds": outcome["recovery_seconds"],
        "resource_efficiency_ratio": min(1.0, 2.0 / max(float(deployment.get("replicas", 0)), 1.0)),
        "slo_goodput_ratio": min(1.0, outcome["goodput_ratio"] / minimum_goodput),
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
