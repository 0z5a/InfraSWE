from __future__ import annotations

from typing import Any


def plan_recovery(
    event: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Broken policy: retries the same batch after restarting the entire pool."""
    worker_ids = [worker["id"] for worker in state.get("workers", [])]
    return {
        "schema_version": "1",
        "status": "retry",
        "failed_worker_id": event.get("worker_id"),
        "retry_worker_id": event.get("worker_id"),
        "retry_batch_size": event.get("batch_size", 0),
        "retry_request_ids": event.get("failed_request_ids", []),
        "clear_cache": False,
        "restart_worker": True,
        "preserve_worker_ids": [],
        "quarantined_worker_ids": [],
        "restarted_worker_ids": worker_ids,
        "reason": "restart_pool",
        "fallback_reported": False,
    }
