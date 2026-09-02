from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

import hashlib
import json
from typing import Any


def _blocked(world_size: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "blocked",
        "world_size": world_size,
        "schedule": [],
        "rank_schedules": {},
        "fingerprint": None,
        "divergence_detected": True,
        "reason": reason,
        "fallback_reported": True,
    }


def build_collective_schedule(
    rank_steps: list[list[dict[str, Any]]], world_size: int, config: dict[str, Any]
) -> dict[str, Any]:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 2
        or not isinstance(rank_steps, list)
        or len(rank_steps) != world_size
        or any(
            not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, dict) for step in steps)
            for steps in rank_steps
        )
        or not isinstance(config, dict)
    ):
        raise ValueError("rank_steps, world_size, or config are invalid")
    canonical = config.get("canonical_order")
    if (
        not isinstance(canonical, list)
        or not canonical
        or any(not isinstance(item, str) or not item for item in canonical)
        or len(canonical) != len(set(canonical))
        or config.get("require_identical_metadata") is not True
        or config.get("require_identical_operation_set") is not True
    ):
        raise ValueError("canonical fail-closed ordering must be configured")

    normalized: list[dict[str, dict[str, Any]]] = []
    discovered_orders: list[list[str]] = []
    for steps in rank_steps:
        by_id: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for step in steps:
            identifier = step.get("id")
            kind = step.get("kind")
            elements = step.get("elements")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in by_id
                or kind != "all_reduce"
                or isinstance(elements, bool)
                or not isinstance(elements, int)
                or elements <= 0
            ):
                raise ValueError(
                    "collectives require unique IDs, all_reduce, and positive elements"
                )
            by_id[identifier] = {"elements": elements, "id": identifier, "kind": kind}
            order.append(identifier)
        normalized.append(by_id)
        discovered_orders.append(order)
    expected_ids = set(canonical)
    if any(set(by_id) != expected_ids for by_id in normalized):
        return _blocked(world_size, "rank_collective_set_mismatch")
    reference = normalized[0]
    if any(by_id != reference for by_id in normalized[1:]):
        return _blocked(world_size, "rank_collective_metadata_mismatch")

    canonical_metadata = [reference[identifier] for identifier in canonical]
    encoded = json.dumps(canonical_metadata, separators=(",", ":"), sort_keys=True).encode()
    return {
        "schema_version": "1",
        "status": "ready",
        "world_size": world_size,
        "schedule": list(canonical),
        "rank_schedules": {
            str(rank): list(canonical) for rank in range(world_size)
        },
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "divergence_detected": any(order != canonical for order in discovered_orders),
        "reason": "canonical_collective_order",
        "fallback_reported": False,
    }
"""

config = {
    "canonical_order": ["reduce-a", "reduce-b", "reduce-c"],
    "require_identical_metadata": True,
    "require_identical_operation_set": True,
}

Path("collective_policy.py").write_text(source, encoding="utf-8")
Path("order_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
