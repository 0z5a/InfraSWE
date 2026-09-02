from __future__ import annotations

from typing import Any


def admit_workload(
    spec: dict[str, Any], node: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    del node, policy
    limits = spec.get("resources", {}).get("limits", {})
    assigned = int(limits.get("nvidia.com/gpu", 0))
    return {
        "schema_version": "1",
        "status": "admitted",
        "assigned_gpus": assigned,
        "reason": "limits_only",
        "fallback_reported": False,
    }
