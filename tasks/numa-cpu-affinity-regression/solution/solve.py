from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "blocked",
        "numa_node": None,
        "cpu_ids": [],
        "reason": reason,
        "fallback_reported": True,
    }


def _cpu_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in value
    ):
        raise ValueError(f"{name} must be a list of non-negative CPU IDs")
    return value


def select_affinity(
    gpu: dict[str, Any], topology: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(gpu, dict) or not isinstance(topology, dict) or not isinstance(config, dict):
        raise ValueError("gpu, topology, and config must be objects")
    numa_node = gpu.get("numa_node")
    threads = config.get("worker_threads")
    if (
        isinstance(numa_node, bool)
        or not isinstance(numa_node, int)
        or numa_node < 0
        or isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads <= 0
    ):
        raise ValueError("GPU NUMA node and worker_threads must be valid integers")
    if config.get("require_gpu_local") is not True or config.get("forbid_cross_numa") is not True:
        raise ValueError("GPU-local fail-closed affinity must be enabled")
    nodes = topology.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("topology.nodes must be an object")
    local = _cpu_list(nodes.get(str(numa_node)), "local NUMA CPUs")
    allowed = set(_cpu_list(topology.get("allowed_cpus"), "allowed_cpus"))
    reserved = set(_cpu_list(topology.get("reserved_cpus"), "reserved_cpus"))
    candidates = sorted(set(local) & allowed - reserved)
    if len(candidates) < threads:
        return _blocked("insufficient_gpu_local_cpus")
    return {
        "schema_version": "1",
        "status": "ready",
        "numa_node": numa_node,
        "cpu_ids": candidates[:threads],
        "reason": "gpu_local_allowed_non_reserved_cpus",
        "fallback_reported": False,
    }
"""

config = {
    "forbid_cross_numa": True,
    "require_gpu_local": True,
    "worker_threads": 4,
}

Path("affinity_policy.py").write_text(source, encoding="utf-8")
Path("affinity_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
