from __future__ import annotations

from typing import Any


def resolve_image(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    del request, policy
    if not candidates:
        return {
            "schema_version": "1",
            "status": "blocked",
            "candidate_id": None,
            "reason": "registry_empty",
            "fallback_reported": False,
            "lock": [],
        }
    candidate = candidates[0]
    return {
        "schema_version": "1",
        "status": "ready",
        "candidate_id": candidate.get("id"),
        "reason": "registry_order",
        "fallback_reported": False,
        "lock": candidate.get("dependencies", []),
    }
