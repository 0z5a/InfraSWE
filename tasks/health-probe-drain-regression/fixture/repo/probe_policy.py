from __future__ import annotations

from typing import Any


def build_rollout_plan(deployment: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    del signal
    return {
        "schema_version": "1",
        "readiness_path": deployment.get("readinessProbe", {}).get("path", "/healthz"),
        "drain_path": deployment.get("drainEndpoint", ""),
        "max_unavailable": deployment.get("strategy", {}).get("maxUnavailable", 1),
        "pre_stop_seconds": deployment.get("lifecycle", {}).get("preStopSeconds", 0),
        "termination_grace_seconds": deployment.get("terminationGracePeriodSeconds", 1),
        "rollback_on_probe_failure": deployment.get("rollbackOnProbeFailure", False),
    }
