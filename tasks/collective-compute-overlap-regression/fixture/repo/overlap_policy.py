from __future__ import annotations

from typing import Any


def build_overlap_plan(
    stages: list[dict[str, Any]], topology: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "ready",
        "comm_stream": False,
        "async_collectives": False,
        "event_fencing": False,
        "overlap_next_compute": False,
        "stages": [
            {
                "compute_stream": "default",
                "collective_stream": "default",
                "overlap_with_next": False,
                "sequence": stage.get("sequence"),
                "stage_id": stage.get("id"),
                "wait_with_event": False,
            }
            for stage in stages
        ],
        "reason": "single_default_stream",
        "fallback_reported": False,
    }
