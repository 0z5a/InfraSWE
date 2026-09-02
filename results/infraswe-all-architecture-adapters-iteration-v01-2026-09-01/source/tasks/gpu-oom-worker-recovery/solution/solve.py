from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def plan_recovery(
    event: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if not all(isinstance(value, dict) for value in (event, state, config)):
        raise ValueError("event, state, and config must be objects")
    if event.get("kind") != "cuda_oom":
        raise ValueError("only cuda_oom events are recoverable")
    failed_worker = event.get("worker_id")
    if not isinstance(failed_worker, str) or not failed_worker:
        raise ValueError("event.worker_id must be a non-empty string")
    batch_size = _integer(event.get("batch_size"), "batch_size", minimum=1)
    request_ids = event.get("failed_request_ids")
    if (
        not isinstance(request_ids, list)
        or not request_ids
        or any(not isinstance(item, str) or not item for item in request_ids)
        or len(request_ids) != len(set(request_ids))
    ):
        raise ValueError("failed_request_ids must contain unique non-empty strings")

    workers = state.get("workers")
    counts = state.get("oom_count_by_worker")
    if (
        not isinstance(workers, list)
        or not workers
        or any(not isinstance(worker, dict) for worker in workers)
        or not isinstance(counts, dict)
    ):
        raise ValueError("state must contain workers and oom_count_by_worker")
    worker_ids: list[str] = []
    statuses: dict[str, str] = {}
    for worker in workers:
        identifier = worker.get("id")
        status = worker.get("status")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in statuses
            or status not in {"failed", "healthy", "quarantined"}
        ):
            raise ValueError("workers must have unique IDs and valid statuses")
        worker_ids.append(identifier)
        statuses[identifier] = status
    if failed_worker not in statuses or statuses[failed_worker] != "failed":
        raise ValueError("the failed worker must exist with failed status")
    oom_count = _integer(counts.get(failed_worker), "oom_count", minimum=1)

    clear_cache = config.get("clear_failed_worker_cache") is True
    preserve_healthy = config.get("preserve_healthy_workers") is True
    minimum_batch = _integer(config.get("min_batch_size"), "min_batch_size", minimum=1)
    max_retries = _integer(config.get("max_retries"), "max_retries", minimum=1)
    quarantine_after = _integer(
        config.get("quarantine_after_ooms"), "quarantine_after_ooms", minimum=1
    )
    if not clear_cache or not preserve_healthy or quarantine_after > max_retries:
        raise ValueError("recovery safety controls are invalid")

    healthy_workers = sorted(
        identifier for identifier in worker_ids if statuses[identifier] == "healthy"
    )
    retry_batch_size = max(minimum_batch, batch_size // 2)
    ordered_requests = sorted(request_ids)
    if oom_count >= quarantine_after:
        if not healthy_workers:
            raise ValueError("no healthy worker is available for quarantined failover")
        status = "quarantine"
        retry_worker = healthy_workers[0]
        restart_worker = False
        quarantined = [failed_worker]
        restarted: list[str] = []
        reason = "repeated_cuda_oom_quarantine"
    else:
        status = "retry"
        retry_worker = failed_worker
        restart_worker = True
        quarantined = []
        restarted = [failed_worker]
        reason = "cuda_oom_retry_smaller_batch"
    return {
        "schema_version": "1",
        "status": status,
        "failed_worker_id": failed_worker,
        "retry_worker_id": retry_worker,
        "retry_batch_size": retry_batch_size,
        "retry_request_ids": ordered_requests,
        "clear_cache": True,
        "restart_worker": restart_worker,
        "preserve_worker_ids": healthy_workers,
        "quarantined_worker_ids": quarantined,
        "restarted_worker_ids": restarted,
        "reason": reason,
        "fallback_reported": True,
    }
"""

config = {
    "clear_failed_worker_cache": True,
    "max_retries": 2,
    "min_batch_size": 1,
    "preserve_healthy_workers": True,
    "quarantine_after_ooms": 2,
}

Path("recovery.py").write_text(source, encoding="utf-8")
Path("recovery_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
