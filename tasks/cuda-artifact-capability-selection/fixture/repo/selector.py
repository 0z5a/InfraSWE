from __future__ import annotations

from typing import Any


def select_artifact(
    request: dict[str, Any],
    artifacts: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Select the first configured package artifact."""
    del request, policy
    if artifacts:
        return {
            "schema_version": "1",
            "status": "ready",
            "artifact_id": artifacts[0].get("id"),
            "mechanism": "cuda",
            "reason": "configured_order",
            "fallback_reported": False,
        }
    return {
        "schema_version": "1",
        "status": "ready",
        "artifact_id": "builtin-cpu",
        "mechanism": "cpu",
        "reason": "no_cuda_artifact",
        "fallback_reported": False,
    }
