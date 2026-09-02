from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _rejected(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "rejected",
        "assigned_gpus": 0,
        "reason": reason,
        "fallback_reported": True,
    }


def _gpu_count(resources: dict[str, Any], section: str) -> int | None:
    values = resources.get(section)
    if not isinstance(values, dict):
        return None
    value = values.get("nvidia.com/gpu")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def admit_workload(
    spec: dict[str, Any], node: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(spec, dict) or not isinstance(node, dict) or not isinstance(policy, dict):
        raise ValueError("spec, node, and policy must be objects")
    required_policy = {
        "forbid_cpu_fallback": True,
        "require_capability_selector": True,
        "require_equal_requests_limits": True,
        "required_runtime_class": "nvidia",
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ValueError("admission policy is not fail-closed")
    resources = spec.get("resources")
    if not isinstance(resources, dict):
        return _rejected("gpu_resources_missing")
    requested = _gpu_count(resources, "requests")
    limited = _gpu_count(resources, "limits")
    if requested is None or limited is None or requested <= 0 or requested != limited:
        return _rejected("gpu_requests_limits_invalid")
    allocatable = node.get("allocatable_gpus")
    if isinstance(allocatable, bool) or not isinstance(allocatable, int) or allocatable < requested:
        return _rejected("insufficient_gpu_capacity")
    runtime = spec.get("runtimeClassName")
    runtime_classes = node.get("runtime_classes")
    if (
        runtime != policy["required_runtime_class"]
        or not isinstance(runtime_classes, list)
        or runtime not in runtime_classes
    ):
        return _rejected("gpu_runtime_unavailable")
    selector = spec.get("nodeSelector")
    capability = node.get("compute_capability")
    if (
        not isinstance(selector, dict)
        or not isinstance(capability, str)
        or selector.get("infraswe/compute-capability") != capability
    ):
        return _rejected("compute_capability_mismatch")
    return {
        "schema_version": "1",
        "status": "admitted",
        "assigned_gpus": requested,
        "reason": "gpu_contract_satisfied",
        "fallback_reported": False,
    }
"""

workload = {
    "image": "example.invalid/inference:v1",
    "name": "inference-api",
    "nodeSelector": {"infraswe/compute-capability": "8.0"},
    "port": 8000,
    "replicas": 2,
    "resources": {
        "limits": {"nvidia.com/gpu": 1},
        "requests": {"nvidia.com/gpu": 1},
    },
    "runtimeClassName": "nvidia",
}

Path("admission.py").write_text(source, encoding="utf-8")
Path("workload.json").write_text(
    json.dumps(workload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
