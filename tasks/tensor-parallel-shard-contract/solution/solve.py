from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _names(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain unique non-empty parameter names")
    return value


def build_shard_plan(
    parameters: list[dict[str, Any]], world_size: int, config: dict[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(parameters, list)
        or not parameters
        or any(not isinstance(parameter, dict) for parameter in parameters)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 2
        or not isinstance(config, dict)
    ):
        raise ValueError("parameters, world_size, or config are invalid")
    if config.get("require_even_partition") is not True:
        raise ValueError("even partitioning must be required")
    column = _names(config.get("column_parallel"), "column_parallel")
    row = _names(config.get("row_parallel"), "row_parallel")
    replicated = _names(config.get("replicated"), "replicated")
    classifications = column + row + replicated
    if len(classifications) != len(set(classifications)):
        raise ValueError("parameter classifications must be disjoint")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parameter in parameters:
        name = parameter.get("name")
        shape = parameter.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or name in seen
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise ValueError("parameters must have unique names and positive shapes")
        if name not in classifications:
            raise ValueError(f"unclassified parameter: {name}")
        seen.add(name)
        normalized.append({"name": name, "shape": list(shape)})
    if seen != set(classifications):
        raise ValueError("configuration and parameter set must match exactly")

    entries: list[dict[str, Any]] = []
    for parameter in sorted(normalized, key=lambda item: item["name"]):
        name = parameter["name"]
        shape = parameter["shape"]
        is_replicated = name in replicated
        axis = 0 if name in column or is_replicated else 1
        if axis >= len(shape):
            raise ValueError(f"partition axis is absent for {name}")
        dimension = shape[axis]
        if is_replicated:
            shards = [
                {"end": dimension, "rank": rank, "start": 0}
                for rank in range(world_size)
            ]
        else:
            if dimension % world_size:
                raise ValueError(f"{name} is not evenly divisible by world_size")
            size = dimension // world_size
            shards = [
                {"end": (rank + 1) * size, "rank": rank, "start": rank * size}
                for rank in range(world_size)
            ]
        entries.append(
            {
                "axis": axis,
                "name": name,
                "replicated": is_replicated,
                "shape": shape,
                "shards": shards,
            }
        )
    return {
        "schema_version": "1",
        "status": "ready",
        "world_size": world_size,
        "parameters": entries,
        "reason": "complete_even_tensor_parallel_partition",
        "fallback_reported": False,
    }
"""

config = {
    "column_parallel": ["attention.qkv.weight", "mlp.gate_up.weight"],
    "replicated": ["norm.weight"],
    "require_even_partition": True,
    "row_parallel": ["attention.out.weight"],
}

Path("shard_policy.py").write_text(source, encoding="utf-8")
Path("tp_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
