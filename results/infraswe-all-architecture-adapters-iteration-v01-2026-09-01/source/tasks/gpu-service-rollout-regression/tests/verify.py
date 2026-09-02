from __future__ import annotations

import json
import os
import sys
from pathlib import Path

repo = Path(os.environ["INFRASWE_REPO"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
evidence.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))

from service_model import load_deployment, simulate_rollout  # noqa: E402


def write_json(name: str, value: object) -> None:
    (evidence / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


try:
    deployment = load_deployment(repo / "deployment.json")
    outcome = simulate_rollout(deployment)
    spec = deployment["spec"]
    regression = {
        "replica_count_preserved": spec.get("replicas") == 2,
        "image_preserved": spec.get("image") == "example.invalid/inference:v1",
        "port_preserved": spec.get("containerPort") == 8000,
    }
    functional = {
        "ready_endpoint_selected": spec.get("readinessProbe", {}).get("path") == "/readyz",
        "zero_rollout_errors": outcome.dropped_requests == 0,
        "response_integrity": outcome.corrupted_responses == 0,
    }
    safety = {
        "rollback_completed": outcome.rollback_completed,
        "zero_data_corruption": outcome.corrupted_responses == 0,
        "resources_cleaned": outcome.orphan_replicas == 0,
    }
    slo = {
        "goodput_budget": outcome.goodput_ratio >= 0.995,
        "recovery_time": outcome.recovery_time_seconds <= 3.0,
    }
    assertions = {
        "functional": functional,
        "regression": regression,
        "safety": safety,
        "slo": slo,
    }
    faults = {
        "passed": outcome.recovery_time_seconds <= 3.0 and outcome.rollback_completed,
        "score": 1.0 if outcome.recovery_time_seconds <= 3.0 else 0.0,
        "termination_recovery_seconds": outcome.recovery_time_seconds,
    }
    policy = {
        "passed": outcome.corrupted_responses == 0 and outcome.orphan_replicas == 0,
        "hard_failures": [],
        "silent_fallback": False,
        "data_corruption": outcome.corrupted_responses > 0,
        "resource_leak": outcome.orphan_replicas > 0,
    }
    metrics = {
        "slo_goodput_ratio": outcome.goodput_ratio,
        "resource_efficiency_ratio": min(1.0, 2 / max(float(spec.get("replicas", 0)), 1.0)),
        "topology_robustness_ratio": 1.0,
        "observability_ratio": 1.0,
        "recovery_time_seconds": outcome.recovery_time_seconds,
        "dropped_requests": float(outcome.dropped_requests),
    }
    with (evidence / "rollout-events.jsonl").open("w", encoding="utf-8") as handle:
        for event in outcome.events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
except Exception as error:  # task verifier converts malformed candidate state to evidence
    assertions = {"functional": {"valid_deployment": False}}
    faults = {"passed": False, "score": 0.0}
    policy = {
        "passed": False,
        "hard_failures": [],
        "silent_fallback": False,
        "data_corruption": False,
        "resource_leak": False,
        "verifier_error": str(error),
    }
    metrics = {
        "slo_goodput_ratio": 0.0,
        "resource_efficiency_ratio": 0.0,
        "topology_robustness_ratio": 0.0,
        "observability_ratio": 0.5,
    }

write_json("assertions.json", assertions)
write_json("faults.json", faults)
write_json("policy.json", policy)
write_json("metrics.json", metrics)
passed = all(
    value
    for group in assertions.values()
    for value in (group.values() if isinstance(group, dict) else [group])
)
raise SystemExit(0 if passed and faults["passed"] and policy["passed"] else 1)
