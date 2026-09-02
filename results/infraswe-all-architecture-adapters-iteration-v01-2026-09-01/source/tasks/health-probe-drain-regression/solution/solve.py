from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def build_rollout_plan(
    deployment: dict[str, Any], signal: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(deployment, dict) or not isinstance(signal, dict):
        raise ValueError("deployment and signal must be objects")
    readiness = deployment.get("readinessProbe")
    strategy = deployment.get("strategy")
    lifecycle = deployment.get("lifecycle")
    if not all(isinstance(value, dict) for value in (readiness, strategy, lifecycle)):
        raise ValueError("readinessProbe, strategy, and lifecycle must be objects")
    max_inflight = _integer(signal.get("max_inflight_seconds"), "max_inflight_seconds")
    pre_stop = _integer(lifecycle.get("preStopSeconds"), "preStopSeconds")
    grace = _integer(
        deployment.get("terminationGracePeriodSeconds"),
        "terminationGracePeriodSeconds",
        minimum=1,
    )
    max_unavailable = _integer(strategy.get("maxUnavailable"), "maxUnavailable")
    if pre_stop < max_inflight:
        raise ValueError("preStopSeconds is shorter than the in-flight request bound")
    if grace <= pre_stop:
        raise ValueError("termination grace must exceed pre-stop drain time")
    return {
        "schema_version": "1",
        "readiness_path": readiness.get("path"),
        "drain_path": deployment.get("drainEndpoint"),
        "max_unavailable": max_unavailable,
        "pre_stop_seconds": pre_stop,
        "termination_grace_seconds": grace,
        "rollback_on_probe_failure": deployment.get("rollbackOnProbeFailure") is True,
    }
"""

deployment = {
    "drainEndpoint": "/drainz",
    "image": "example.invalid/inference:v1",
    "lifecycle": {"preStopSeconds": 4},
    "name": "inference-api",
    "port": 8000,
    "readinessProbe": {"failureThreshold": 2, "path": "/readyz", "periodSeconds": 1},
    "replicas": 2,
    "rollbackOnProbeFailure": True,
    "strategy": {"maxSurge": 1, "maxUnavailable": 0},
    "terminationGracePeriodSeconds": 5,
}

Path("probe_policy.py").write_text(source, encoding="utf-8")
Path("deployment.json").write_text(
    json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
