from __future__ import annotations

from infraswe.environments.hardware_manifest import _amd_accelerators, _nvidia_accelerators
from infraswe.models.hardware import HardwareProfile, validate_hardware_manifest


def test_hardware_profile_checks_topology_and_compute_capability() -> None:
    profile = HardwareProfile(
        id="gpu-2x-sm120-pcie",
        cpu_min=16,
        ram_gib_min=64,
        gpu_count=2,
        compute_capability="12.0",
        interconnect="pcie-phb",
        experimental=True,
    )
    manifest = {
        "host": {"cpu_count": 30, "memory_gib": 98},
        "gpu_count": 2,
        "commands": {
            "nvidia_query": {
                "raw": (
                    "0, GPU-a, RTX PRO 5000, 48935, 580.95.05, 0000:01:00.0, 12.0\n"
                    "1, GPU-b, RTX PRO 5000, 48935, 580.95.05, 0000:02:00.0, 12.0"
                )
            },
            "nvidia_topology": {"raw": "GPU0 X PHB\nGPU1 PHB X"},
        },
    }
    result = validate_hardware_manifest(profile, manifest)
    assert result["passed"]
    assert result["errors"] == []
    assert result["warnings"]


def test_hardware_profile_can_lease_a_gpu_subset() -> None:
    profile = HardwareProfile(
        id="gpu-1x-sm80",
        cpu_min=8,
        ram_gib_min=32,
        gpu_count=1,
        compute_capability="8.0",
    )
    manifest = {
        "host": {"cpu_count": 30, "memory_gib": 98},
        "gpu_count": 2,
        "commands": {
            "nvidia_query": {
                "raw": (
                    "0, GPU-a, A100, 40960, 580.105.08, 0000:01:00.0, 8.0\n"
                    "1, GPU-b, A100, 40960, 580.105.08, 0000:02:00.0, 8.0"
                )
            },
            "nvidia_topology": {"raw": "GPU0 X PHB\nGPU1 PHB X"},
        },
    }

    result = validate_hardware_manifest(profile, manifest)

    assert result["passed"]
    assert result["errors"] == []
    assert result["warnings"] == ["host exposes 2 GPUs; lease must constrain visibility to 1"]


def test_mi300x_profile_checks_vendor_architecture_model_and_runtime() -> None:
    profile = HardwareProfile(
        id="gpu-1x-gfx942-mi300x-rocm61",
        cpu_min=16,
        ram_gib_min=128,
        gpu_count=1,
        accelerator_vendor="amd",
        architecture="gfx942",
        gpu_model="MI300X",
        runtime="rocm",
        runtime_version="6.1",
        experimental=True,
    )
    manifest = {
        "host": {"cpu_count": 64, "memory_gib": 1024},
        "gpu_count": 1,
        "accelerator_vendor": "amd",
        "runtime": "rocm",
        "runtime_version": "6.1.2",
        "accelerators": [
            {
                "index": 0,
                "vendor": "amd",
                "name": "AMD Instinct MI300X",
                "architecture": "gfx942",
                "runtime": "rocm",
                "runtime_version": "6.1.2",
            }
        ],
        "commands": {"amd_topology": {"raw": "GPU0"}},
    }

    result = validate_hardware_manifest(profile, manifest)

    assert result["passed"]
    assert result["errors"] == []
    assert result["warnings"] == [
        "profile is experimental and must not enter the default leaderboard"
    ]


def test_mi300x_profile_rejects_wrong_rocm_minor() -> None:
    profile = HardwareProfile(
        id="gpu-1x-gfx942-mi300x-rocm61",
        cpu_min=1,
        ram_gib_min=1,
        gpu_count=1,
        accelerator_vendor="amd",
        architecture="gfx942",
        gpu_model="MI300X",
        runtime="rocm",
        runtime_version="6.1",
    )
    manifest = {
        "host": {"cpu_count": 64, "memory_gib": 1024},
        "gpu_count": 1,
        "accelerator_vendor": "amd",
        "runtime": "rocm",
        "runtime_version": "6.2.0",
        "accelerators": [
            {
                "name": "AMD Instinct MI300X",
                "architecture": "gfx942",
            }
        ],
        "commands": {},
    }

    result = validate_hardware_manifest(profile, manifest)

    assert not result["passed"]
    assert any("runtime version 6.2.0" in error for error in result["errors"])


def test_rocm_smi_and_rocminfo_are_normalized_to_generic_accelerator() -> None:
    accelerators = _amd_accelerators(
        {
            "card0": {
                "Card Series": "AMD Instinct MI300X OAM",
                "Unique ID": "0x1234",
                "VRAM Total Memory (B)": "206158430208",
                "Driver version": "6.8.5",
            }
        },
        rocminfo="  Name: gfx942\n  Name: gfx942\n",
        hipcc_version="HIP version: 6.1.40093-abc",
    )

    assert accelerators == [
        {
            "index": 0,
            "vendor": "amd",
            "name": "AMD Instinct MI300X OAM",
            "uuid": "0x1234",
            "memory_bytes": 206158430208,
            "driver_version": "6.8.5",
            "pci_bus_id": None,
            "architecture": "gfx942",
            "compute_capability": None,
            "runtime": "rocm",
            "runtime_version": "6.1.40093",
        }
    ]


def test_nvidia_toolkit_version_is_normalized_for_compiler_profiles() -> None:
    accelerators = _nvidia_accelerators(
        "0, GPU-b200, NVIDIA B200, 183359, 590.1, 0000:01:00.0, 10.0",
        nvcc_version="Cuda compilation tools, release 13.3, V13.3.42",
    )

    assert accelerators[0]["architecture"] == "sm100"
    assert accelerators[0]["runtime"] == "cuda"
    assert accelerators[0]["runtime_version"] == "13.3"


def test_nvidia_unified_memory_capacity_is_unknown_not_zero() -> None:
    accelerators = _nvidia_accelerators(
        "0, GPU-gb10, NVIDIA GB10, [N/A], 580.173.02, 0000:01:00.0, 12.1",
        nvcc_version="Cuda compilation tools, release 13.0, V13.0.88",
    )

    assert accelerators[0]["memory_bytes"] is None
    assert accelerators[0]["architecture"] == "sm121"


def test_b200_cuda133_profile_checks_full_compiler_cell() -> None:
    profile = HardwareProfile(
        id="gpu-1x-sm100-b200-cuda133",
        cpu_min=16,
        ram_gib_min=128,
        gpu_count=1,
        accelerator_vendor="nvidia",
        architecture="sm100",
        gpu_model="B200",
        runtime="cuda",
        runtime_version="13.3",
        compute_capability="10.0",
        experimental=True,
    )
    manifest = {
        "host": {"cpu_count": 64, "memory_gib": 1024},
        "gpu_count": 1,
        "accelerator_vendor": "nvidia",
        "runtime": "cuda",
        "runtime_version": "13.3.1",
        "accelerators": [
            {
                "name": "NVIDIA B200",
                "architecture": "sm100",
                "compute_capability": "10.0",
            }
        ],
        "commands": {"nvidia_topology": {"raw": "GPU0 X"}},
    }

    result = validate_hardware_manifest(profile, manifest)

    assert result["passed"]
    assert result["errors"] == []
