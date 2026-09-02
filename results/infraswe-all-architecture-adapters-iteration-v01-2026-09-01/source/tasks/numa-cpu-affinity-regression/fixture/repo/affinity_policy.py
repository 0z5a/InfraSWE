from __future__ import annotations

from typing import Any


def select_affinity(
    gpu: dict[str, Any], topology: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    del gpu
    allowed = topology.get("allowed_cpus", [])
    count = int(config.get("worker_threads", 8))
    return {
        "schema_version": "1",
        "status": "ready",
        "numa_node": 0,
        "cpu_ids": sorted(allowed)[:count],
        "reason": "lowest_cpu_ids",
        "fallback_reported": False,
    }
