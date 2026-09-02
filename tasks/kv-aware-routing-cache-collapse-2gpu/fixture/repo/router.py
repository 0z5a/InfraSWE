from __future__ import annotations

from typing import Any


def choose_worker(
    request: dict[str, Any],
    workers: list[dict[str, Any]],
    available: list[bool],
    config: dict[str, Any],
) -> int:
    """Return a worker index for a request."""
    worker_count = len(workers)
    del available, config
    if worker_count == 0:
        raise ValueError("at least one worker is required")
    return int(request.get("request_id", 0)) % worker_count
