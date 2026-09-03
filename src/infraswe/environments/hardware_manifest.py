from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(command: list[str], *, timeout_seconds: int = 20) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return 127, str(error)
    output = completed.stdout.strip() or completed.stderr.strip()
    return completed.returncode, output


def _json_command(command: list[str]) -> Any:
    code, output = _run(command)
    if code:
        return {"available": False, "error": output}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"available": True, "raw": output}


def _cuda_version(raw: str) -> str | None:
    match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", raw, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _nvidia_accelerators(raw: str, *, nvcc_version: str = "") -> list[dict[str, Any]]:
    compiler_version = _cuda_version(nvcc_version)
    accelerators = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            continue
        index, uuid, name, memory_mib, driver, pci_bus_id, capability = fields[:7]
        try:
            memory_bytes = int(float(memory_mib) * 1024**2)
        except ValueError:
            # Unified-memory devices such as GB10 expose memory.total as [N/A].
            # Preserve that as unknown rather than fabricating a zero capacity.
            memory_bytes = None
        accelerators.append(
            {
                "index": int(index),
                "vendor": "nvidia",
                "name": name,
                "uuid": uuid,
                "memory_bytes": memory_bytes,
                "driver_version": driver,
                "pci_bus_id": pci_bus_id,
                "architecture": f"sm{capability.replace('.', '')}",
                "compute_capability": capability,
                "runtime": "cuda",
                # Kept for schema v0.2 compatibility. New profiles should use
                # compiler_version and framework_runtime_version explicitly.
                "runtime_version": compiler_version,
                "compiler_version": compiler_version,
            }
        )
    return accelerators


def _rocm_architectures(raw: str) -> list[str]:
    # rocminfo repeats an agent name in several sections. Preserve first-seen
    # order but collapse duplicates so a homogeneous node has one architecture.
    architectures: list[str] = []
    for match in re.finditer(r"(?m)^\s*Name:\s*(gfx[0-9a-z]+)\s*$", raw):
        architecture = match.group(1).lower()
        if architecture not in architectures:
            architectures.append(architecture)
    return architectures


def _rocm_version(raw: str) -> str | None:
    for pattern in (
        r"HIP version:\s*([0-9]+(?:\.[0-9]+)+)",
        r"HIP version\s*:\s*([0-9]+(?:\.[0-9]+)+)",
        r"rocm-([0-9]+(?:\.[0-9]+)+)",
    ):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _amd_accelerators(snapshot: Any, *, rocminfo: str, hipcc_version: str) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    cards = [
        (key, value)
        for key, value in snapshot.items()
        if re.fullmatch(r"card[0-9]+", str(key), flags=re.IGNORECASE) and isinstance(value, dict)
    ]
    architectures = _rocm_architectures(rocminfo)
    compiler_version = _rocm_version(hipcc_version)
    system = snapshot.get("system", {})
    normalized_system = (
        {str(key).lower(): value for key, value in system.items()}
        if isinstance(system, dict)
        else {}
    )
    accelerators = []
    for ordinal, (card, details) in enumerate(sorted(cards)):
        normalized = {str(key).lower(): value for key, value in details.items()}

        def first_value(*needles: str, values: dict[str, Any] = normalized) -> Any:
            for key, value in values.items():
                if any(needle in key for needle in needles):
                    return value
            return None

        index_match = re.search(r"[0-9]+", str(card))
        architecture = (
            architectures[ordinal]
            if len(architectures) == len(cards)
            else architectures[0]
            if len(architectures) == 1
            else None
        )
        memory = first_value("vram total memory", "total vram")
        try:
            memory_bytes = int(memory) if memory is not None else None
        except (TypeError, ValueError):
            memory_bytes = None
        accelerators.append(
            {
                "index": int(index_match.group()) if index_match else ordinal,
                "vendor": "amd",
                "name": first_value("card series", "product name", "device name", "card model"),
                "uuid": first_value("unique id", "serial"),
                "memory_bytes": memory_bytes,
                "driver_version": first_value("driver version")
                or first_value("driver version", values=normalized_system),
                "pci_bus_id": first_value("pci bus"),
                "architecture": architecture,
                "compute_capability": None,
                "runtime": "rocm",
                # Kept for schema v0.2 compatibility. New profiles should use
                # compiler_version and framework_runtime_version explicitly.
                "runtime_version": compiler_version,
                "compiler_version": compiler_version,
            }
        )
    return accelerators


def collect_hardware_manifest(profile: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "0.2",
        "collected_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os_cpu_count(),
            "memory_gib": round(total_memory_bytes() / 1024**3, 3),
        },
        "commands": {},
    }
    commands = {
        "lscpu": ["lscpu", "--json"],
        "numa": ["numactl", "-H"],
        "nvidia_query": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_topology": ["nvidia-smi", "topo", "-m"],
        "nvcc_version": ["nvcc", "--version"],
        "amd_query": [
            "rocm-smi",
            "--showproductname",
            "--showuniqueid",
            "--showdriverversion",
            "--showmeminfo",
            "vram",
            "--json",
        ],
        "amd_topology": ["rocm-smi", "--showtopo"],
        "rocminfo": ["rocminfo"],
        "hipcc_version": ["hipcc", "--version"],
        "framework_runtime": [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'framework': 'torch', 'framework_version': "
                "torch.__version__, 'cuda': torch.version.cuda, "
                "'hip': getattr(torch.version, 'hip', None)}))"
            ),
        ],
        "infiniband": ["ibv_devices"],
        "cgroup": ["sh", "-c", "cat /sys/fs/cgroup/cgroup.controllers"],
        "network": ["ip", "-json", "-brief", "link"],
    }
    for name, command in commands.items():
        if shutil.which(command[0]) is None:
            manifest["commands"][name] = {"available": False, "error": "command not found"}
            continue
        code, output = _run(
            command,
            timeout_seconds=120 if name == "framework_runtime" else 20,
        )
        manifest["commands"][name] = {
            "available": code == 0,
            "exit_code": code,
            "raw": output,
        }
        if name in {"amd_query", "framework_runtime"} and code == 0:
            with suppress(json.JSONDecodeError):
                manifest["commands"][name]["json"] = json.loads(output)

    nvidia = _nvidia_accelerators(
        manifest["commands"].get("nvidia_query", {}).get("raw", ""),
        nvcc_version=manifest["commands"].get("nvcc_version", {}).get("raw", ""),
    )
    amd_query = manifest["commands"].get("amd_query", {})
    amd = _amd_accelerators(
        amd_query.get("json", {}),
        rocminfo=manifest["commands"].get("rocminfo", {}).get("raw", ""),
        hipcc_version=manifest["commands"].get("hipcc_version", {}).get("raw", ""),
    )
    accelerators = nvidia or amd
    manifest["accelerators"] = accelerators
    manifest["accelerator_vendor"] = accelerators[0]["vendor"] if accelerators else None
    manifest["runtime"] = accelerators[0].get("runtime") if accelerators else None
    manifest["runtime_version"] = accelerators[0].get("runtime_version") if accelerators else None
    manifest["compiler_version"] = (
        accelerators[0].get("compiler_version") if accelerators else None
    )
    framework = manifest["commands"].get("framework_runtime", {}).get("json", {})
    framework_runtime = manifest["runtime"]
    manifest["framework"] = framework.get("framework") if isinstance(framework, dict) else None
    manifest["framework_version"] = (
        framework.get("framework_version") if isinstance(framework, dict) else None
    )
    manifest["framework_runtime_version"] = (
        framework.get("hip" if framework_runtime == "rocm" else "cuda")
        if isinstance(framework, dict) and framework_runtime in {"rocm", "cuda"}
        else None
    )
    manifest["driver_versions"] = sorted(
        {
            str(accelerator["driver_version"])
            for accelerator in accelerators
            if accelerator.get("driver_version")
        }
    )
    manifest["gpu_count"] = len(accelerators)
    return manifest


def os_cpu_count() -> int:
    return os.cpu_count() or 0


def total_memory_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return 0


def write_hardware_manifest(path: Path, profile: str) -> dict[str, Any]:
    from infraswe.io import atomic_write_json

    manifest = collect_hardware_manifest(profile)
    atomic_write_json(path, manifest)
    return manifest
