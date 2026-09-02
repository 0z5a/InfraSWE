from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def plan_rank_failure(
    event: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if not all(isinstance(value, dict) for value in (event, state, config)):
        raise ValueError("event, state, and config must be objects")
    if event.get("kind") != "rank_exit":
        raise ValueError("only rank_exit events are supported")
    failed_rank = _integer(event.get("failed_rank"), "failed_rank")
    observed_step = _integer(event.get("observed_step"), "observed_step")
    failed_operation = event.get("failed_operation")
    if not isinstance(failed_operation, str) or not failed_operation:
        raise ValueError("failed_operation must be a non-empty string")

    world_size = _integer(state.get("world_size"), "world_size", minimum=2)
    restart_count = _integer(state.get("restart_count"), "restart_count")
    committed = _integer(state.get("last_committed_step"), "last_committed_step")
    ranks = state.get("ranks")
    requests = state.get("pending_request_ids")
    if (
        not isinstance(ranks, list)
        or len(ranks) != world_size
        or any(not isinstance(rank, dict) for rank in ranks)
        or not isinstance(requests, list)
        or not requests
        or any(not isinstance(item, str) or not item for item in requests)
        or len(requests) != len(set(requests))
        or committed > observed_step
        or failed_rank >= world_size
    ):
        raise ValueError("rank state, requests, or checkpoint are invalid")
    rank_states: dict[int, str] = {}
    for rank in ranks:
        identifier = rank.get("rank")
        status = rank.get("status")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier in rank_states
            or status not in {"exited", "running"}
        ):
            raise ValueError("rank entries are invalid")
        rank_states[identifier] = status
    if set(rank_states) != set(range(world_size)) or rank_states[failed_rank] != "exited":
        raise ValueError("failed rank must be present with exited status")

    required = {
        "abort_process_group": True,
        "forbid_world_size_shrink": True,
        "reinitialize_process_group": True,
        "replay_from_checkpoint": True,
    }
    if any(config.get(key) is not value for key, value in required.items()):
        raise ValueError("rank recovery safety controls must be enabled")
    max_restarts = _integer(config.get("max_restarts"), "max_restarts", minimum=1)
    all_ranks = list(range(world_size))
    exhausted = restart_count >= max_restarts
    return {
        "schema_version": "1",
        "status": "abort" if exhausted else "restart_group",
        "failed_rank": failed_rank,
        "world_size": world_size,
        "abort_ranks": all_ranks,
        "restart_ranks": [] if exhausted else all_ranks,
        "resume_step": committed,
        "replay_request_ids": sorted(requests),
        "reinitialize_process_group": not exhausted,
        "reason": "restart_budget_exhausted"
        if exhausted
        else "abort_and_reinitialize_full_group",
        "fallback_reported": True,
    }
"""

config = {
    "abort_process_group": True,
    "forbid_world_size_shrink": True,
    "max_restarts": 1,
    "reinitialize_process_group": True,
    "replay_from_checkpoint": True,
}

Path("failure_policy.py").write_text(source, encoding="utf-8")
Path("recovery_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
