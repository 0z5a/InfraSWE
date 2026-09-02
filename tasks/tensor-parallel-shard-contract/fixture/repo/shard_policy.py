from __future__ import annotations

from typing import Any


def build_shard_plan(
    parameters: list[dict[str, Any]], world_size: int, config: dict[str, Any]
) -> dict[str, Any]:
    entries = []
    for parameter in parameters:
        size = parameter["shape"][0] // world_size
        entries.append(
            {
                "axis": 0,
                "name": parameter["name"],
                "replicated": False,
                "shape": parameter["shape"],
                "shards": [
                    {"end": (rank + 1) * size, "rank": rank, "start": rank * size}
                    for rank in range(world_size)
                ],
            }
        )
    return {
        "schema_version": "1",
        "status": "ready",
        "world_size": world_size,
        "parameters": entries,
        "reason": "axis_zero_for_everything",
        "fallback_reported": False,
    }
