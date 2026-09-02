from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "blocked",
        "comm_stream": False,
        "async_collectives": False,
        "event_fencing": False,
        "overlap_next_compute": False,
        "stages": [],
        "reason": reason,
        "fallback_reported": True,
    }


def build_overlap_plan(
    stages: list[dict[str, Any]], topology: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(stages, list)
        or not stages
        or any(not isinstance(stage, dict) for stage in stages)
        or not isinstance(topology, dict)
        or not isinstance(config, dict)
    ):
        raise ValueError("stages, topology, and config must be valid objects")
    required_config = {
        "async_collectives": True,
        "dedicated_comm_stream": True,
        "event_fencing": True,
        "overlap_next_compute": True,
    }
    if any(config.get(key) is not value for key, value in required_config.items()):
        raise ValueError("all overlap safety controls must be enabled")
    device_count = topology.get("device_count")
    if device_count != 2 or topology.get("concurrent_kernels") is not True:
        return _blocked("topology_does_not_support_two_gpu_overlap")

    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    sequences: set[int] = set()
    for stage in stages:
        identifier = stage.get("id")
        sequence = stage.get("sequence")
        elements = stage.get("collective_elements")
        cycles = stage.get("compute_cycles")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or sequence in sequences
            or isinstance(elements, bool)
            or not isinstance(elements, int)
            or elements <= 0
            or isinstance(cycles, bool)
            or not isinstance(cycles, int)
            or cycles <= 0
        ):
            raise ValueError("stages require unique IDs/sequences and positive work sizes")
        identifiers.add(identifier)
        sequences.add(sequence)
        normalized.append({"id": identifier, "sequence": sequence})
    if sequences != set(range(len(stages))):
        raise ValueError("stage sequences must be contiguous from zero")
    ordered = sorted(normalized, key=lambda stage: (stage["sequence"], stage["id"]))
    return {
        "schema_version": "1",
        "status": "ready",
        "comm_stream": True,
        "async_collectives": True,
        "event_fencing": True,
        "overlap_next_compute": True,
        "stages": [
            {
                "compute_stream": "default",
                "collective_stream": "communication",
                "overlap_with_next": index < len(ordered) - 1,
                "sequence": stage["sequence"],
                "stage_id": stage["id"],
                "wait_with_event": True,
            }
            for index, stage in enumerate(ordered)
        ],
        "reason": "event_fenced_async_communication_stream",
        "fallback_reported": False,
    }
"""

config = {
    "async_collectives": True,
    "dedicated_comm_stream": True,
    "event_fencing": True,
    "overlap_next_compute": True,
}

Path("overlap_policy.py").write_text(source, encoding="utf-8")
Path("overlap_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
