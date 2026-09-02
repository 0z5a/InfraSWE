from __future__ import annotations

from typing import Any


def plan_rank_failure(
    event: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    failed = event.get("failed_rank", 0)
    return {
        "schema_version": "1",
        "status": "restart_group",
        "failed_rank": failed,
        "world_size": state.get("world_size", 0),
        "abort_ranks": [failed],
        "restart_ranks": [failed],
        "resume_step": event.get("observed_step", 0),
        "replay_request_ids": state.get("pending_request_ids", []),
        "reinitialize_process_group": False,
        "reason": "restart_failed_rank_only",
        "fallback_reported": False,
    }
