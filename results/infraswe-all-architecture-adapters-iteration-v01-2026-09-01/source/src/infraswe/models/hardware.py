from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "0.2"] = "0.1"
    id: str
    cpu_min: int = Field(ge=1)
    ram_gib_min: int = Field(ge=1)
    gpu_count: int = Field(ge=0, le=16)
    accelerator_vendor: Literal["nvidia", "amd"] | None = None
    architecture: str | None = None
    gpu_model: str | None = None
    runtime: Literal["cuda", "rocm"] | None = None
    runtime_version: str | None = None
    compute_capability: str | None = None
    interconnect: str | None = None
    experimental: bool = False

    @classmethod
    def load(cls, path: Path) -> HardwareProfile:
        with path.open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))


def _compute_capabilities(manifest: dict[str, Any]) -> list[str]:
    generic = [
        str(accelerator["compute_capability"])
        for accelerator in manifest.get("accelerators", [])
        if accelerator.get("compute_capability")
    ]
    if generic:
        return generic
    output = manifest.get("commands", {}).get("nvidia_query", {}).get("raw", "")
    capabilities: list[str] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 7:
            capabilities.append(fields[-1])
    return capabilities


def _accelerator_values(manifest: dict[str, Any], field: str) -> list[str]:
    return [
        str(accelerator[field])
        for accelerator in manifest.get("accelerators", [])
        if accelerator.get(field) is not None
    ]


def validate_hardware_manifest(
    profile: HardwareProfile, manifest: dict[str, Any]
) -> dict[str, list[str] | bool]:
    errors: list[str] = []
    warnings: list[str] = []
    host = manifest.get("host", {})
    if int(host.get("cpu_count", 0)) < profile.cpu_min:
        errors.append(f"CPU count {host.get('cpu_count', 0)} < required {profile.cpu_min}")
    if float(host.get("memory_gib", 0)) < profile.ram_gib_min:
        errors.append(f"RAM {host.get('memory_gib', 0)} GiB < required {profile.ram_gib_min}")
    available_gpus = int(manifest.get("gpu_count", 0))
    if available_gpus < profile.gpu_count:
        errors.append(f"GPU count {available_gpus} < required {profile.gpu_count}")
    elif available_gpus > profile.gpu_count:
        warnings.append(
            f"host exposes {available_gpus} GPUs; lease must constrain visibility to "
            f"{profile.gpu_count}"
        )
    vendor = manifest.get("accelerator_vendor")
    if profile.accelerator_vendor and vendor != profile.accelerator_vendor:
        errors.append(
            f"accelerator vendor {vendor or 'unknown'} != required {profile.accelerator_vendor}"
        )
    if profile.architecture and profile.gpu_count:
        architectures = _accelerator_values(manifest, "architecture")
        compatible_count = sum(value == profile.architecture for value in architectures)
        if compatible_count < profile.gpu_count:
            errors.append(
                f"only {compatible_count} GPU(s) match architecture {profile.architecture}; "
                f"required {profile.gpu_count}"
            )
    if profile.gpu_model and profile.gpu_count:
        models = _accelerator_values(manifest, "name")
        compatible_count = sum(profile.gpu_model.lower() in value.lower() for value in models)
        if compatible_count < profile.gpu_count:
            errors.append(
                f"only {compatible_count} GPU(s) match model {profile.gpu_model}; "
                f"required {profile.gpu_count}"
            )
    runtime = manifest.get("runtime")
    if profile.runtime and runtime != profile.runtime:
        errors.append(f"accelerator runtime {runtime or 'unknown'} != required {profile.runtime}")
    runtime_version = str(manifest.get("runtime_version") or "")
    if profile.runtime_version and not runtime_version.startswith(profile.runtime_version):
        errors.append(
            f"runtime version {runtime_version or 'unknown'} does not match required "
            f"prefix {profile.runtime_version}"
        )
    if profile.compute_capability and profile.gpu_count:
        capabilities = _compute_capabilities(manifest)
        compatible_count = sum(value == profile.compute_capability for value in capabilities)
        if compatible_count < profile.gpu_count:
            errors.append(
                f"only {compatible_count} GPU(s) match compute capability "
                f"{profile.compute_capability}; required {profile.gpu_count}"
            )
    topology_command = "amd_topology" if vendor == "amd" else "nvidia_topology"
    topology = manifest.get("commands", {}).get(topology_command, {}).get("raw", "")
    if profile.interconnect == "nvlink" and "NV" not in topology:
        errors.append("NVLink was required but not present in accelerator topology")
    if profile.interconnect == "pcie-phb" and profile.gpu_count > 1 and "PHB" not in topology:
        errors.append("PHB PCIe topology was required but not observed")
    if profile.experimental:
        warnings.append("profile is experimental and must not enter the default leaderboard")
    return {"passed": not errors, "errors": errors, "warnings": warnings}
