from __future__ import annotations

import importlib.util
import platform
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from infraswe.models.training import (
    TrainingAdapterCapability,
    TrainingCapabilityManifest,
)

PROBE_VERSION = "training-capability-probe-v0.1"

_ADAPTERS = {
    "native-pytorch": {
        "modules": ("torch",),
        "distribution": "torch",
        "implemented": True,
        "algorithms": ("sft", "grpo-contract", "dapo-loss-contract", "muon"),
    },
    "hf-transformers": {
        "modules": ("torch", "transformers"),
        "distribution": "transformers",
        "implemented": False,
        "algorithms": ("sft",),
    },
    "trl": {
        "modules": ("torch", "transformers", "trl"),
        "distribution": "trl",
        "implemented": False,
        "algorithms": ("sft", "grpo", "dapo-loss-contract"),
    },
    "verl": {
        "modules": ("torch", "verl"),
        "distribution": "verl",
        "implemented": False,
        "algorithms": ("grpo", "dapo-recipe-contract", "dapo-online"),
    },
    "torchtune": {
        "modules": ("torch", "torchtune"),
        "distribution": "torchtune",
        "implemented": False,
        "algorithms": ("sft", "muon"),
    },
    "axolotl": {
        "modules": ("torch", "axolotl"),
        "distribution": "axolotl",
        "implemented": False,
        "algorithms": ("sft", "grpo"),
    },
    "megatron-core": {
        "modules": ("torch", "megatron.core"),
        "distribution": "megatron-core",
        "implemented": False,
        "algorithms": ("sft", "muon"),
    },
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _platform_probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cuda_available": False,
        "devices": [],
    }
    if not _module_available("torch"):
        return payload
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            }
        )
        if torch.cuda.is_available():
            payload["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": ".".join(
                        str(part) for part in torch.cuda.get_device_capability(index)
                    ),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as error:  # pragma: no cover - depends on external framework installation
        payload["torch_probe_error"] = f"{type(error).__name__}: {error}"
    return payload


def probe_capabilities() -> TrainingCapabilityManifest:
    adapters: dict[str, TrainingAdapterCapability] = {}
    for adapter_id, descriptor in _ADAPTERS.items():
        modules = tuple(str(name) for name in descriptor["modules"])
        missing = [name for name in modules if not _module_available(name)]
        runtime_available = not missing
        implemented = bool(descriptor["implemented"])
        level = "adapter-implemented" if implemented else "protocol-supported"
        algorithm_level = level
        adapters[adapter_id] = TrainingAdapterCapability(
            adapter_id=adapter_id,
            capability_level=level,
            runtime_available=runtime_available,
            framework_version=_version(str(descriptor["distribution"])),
            algorithms={str(name): algorithm_level for name in descriptor["algorithms"]},
            missing_dependencies=missing,
        )
    platform_payload = _platform_probe()
    torch_available = adapters["native-pytorch"].runtime_available
    failures = []
    if not torch_available:
        failures.append("TRAIN_TORCH_UNAVAILABLE")
    if not platform_payload["cuda_available"]:
        failures.append("TRAIN_CUDA_UNAVAILABLE")
    status = (
        "ready"
        if torch_available and platform_payload["cuda_available"]
        else ("partial" if torch_available else "protocol_only")
    )
    return TrainingCapabilityManifest(
        generated_at=datetime.now(UTC).isoformat(),
        probe_version=PROBE_VERSION,
        status=status,
        adapters=adapters,
        platform=platform_payload,
        failure_codes=failures,
    )
