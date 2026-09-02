from __future__ import annotations

import hashlib
import json
from typing import Any


def build_collective_schedule(
    rank_steps: list[list[dict[str, Any]]], world_size: int, config: dict[str, Any]
) -> dict[str, Any]:
    rank_schedules = {
        str(rank): [step["id"] for step in steps] for rank, steps in enumerate(rank_steps)
    }
    schedule = rank_schedules.get("0", [])
    return {
        "schema_version": "1",
        "status": "ready",
        "world_size": world_size,
        "schedule": schedule,
        "rank_schedules": rank_schedules,
        "fingerprint": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "divergence_detected": False,
        "reason": "rank_local_discovery_order",
        "fallback_reported": False,
    }
