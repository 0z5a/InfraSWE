from __future__ import annotations

import json
from pathlib import Path

router_source = '''from __future__ import annotations

import hashlib
import math
from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _score(prefix_id: str, salt: str, worker_index: int) -> bytes:
    key = prefix_id if not salt else f"{salt}\\0{prefix_id}"
    return hashlib.sha256(f"{key}\\0{worker_index}".encode()).digest()


def _owner(prefix_id: str, salt: str, worker_count: int) -> int:
    key = prefix_id if not salt else f"{salt}\\0{prefix_id}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % worker_count


def _has_fresh_cache(
    worker: dict[str, Any], prefix_id: str, step: int, ttl_steps: int
) -> bool:
    cache = worker.get("cache", {})
    if not isinstance(cache, dict):
        return False
    cached_at = cache.get(prefix_id)
    return (
        isinstance(cached_at, int)
        and not isinstance(cached_at, bool)
        and 0 <= cached_at <= step
        and step - cached_at <= ttl_steps
    )


def _load(worker: dict[str, Any]) -> float:
    value = worker.get("load", 0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return math.inf
    return max(0.0, float(value))


def choose_worker(
    request: dict[str, Any],
    workers: list[dict[str, Any]],
    available: list[bool],
    config: dict[str, Any],
) -> int:
    """Choose a stable KV owner, with a cache-aware fallback during an outage."""
    if not isinstance(request, dict) or not isinstance(config, dict):
        raise ValueError("request and config must be objects")
    if not isinstance(workers, list) or not workers or any(
        not isinstance(worker, dict) for worker in workers
    ):
        raise ValueError("workers must be a non-empty list of objects")
    if (
        not isinstance(available, list)
        or len(available) != len(workers)
        or any(not isinstance(item, bool) for item in available)
    ):
        raise ValueError("available must contain one boolean per worker")
    candidates = [index for index, is_available in enumerate(available) if is_available]
    if not candidates:
        raise ValueError("no workers are available")

    prefix_id = request.get("prefix_id")
    if not isinstance(prefix_id, str) or not prefix_id:
        raise ValueError("request.prefix_id must be a non-empty string")
    step = _integer(request.get("step"), "request.step")
    ttl_steps = _integer(config.get("cache_ttl_steps"), "cache_ttl_steps", minimum=1)
    salt = config.get("hash_salt", "")
    if not isinstance(salt, str):
        raise ValueError("hash_salt must be a string")

    owner = _owner(prefix_id, salt, len(workers))
    if available[owner]:
        return owner

    cached = [
        index
        for index in candidates
        if _has_fresh_cache(workers[index], prefix_id, step, ttl_steps)
    ]
    if cached:
        return min(cached, key=lambda index: (_load(workers[index]), index))
    return max(candidates, key=lambda index: (_score(prefix_id, salt, index), -index))
'''

config = {
    "cache_capacity_entries": 4,
    "cache_ttl_steps": 64,
    "hash_salt": "",
    "strategy": "kv_affinity",
}

Path("router.py").write_text(router_source, encoding="utf-8")
Path("routing_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
