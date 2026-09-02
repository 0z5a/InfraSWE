from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
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


def collect_hardware_manifest(profile: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "0.1",
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
        "infiniband": ["ibv_devices"],
        "cgroup": ["sh", "-c", "cat /sys/fs/cgroup/cgroup.controllers"],
        "network": ["ip", "-json", "-brief", "link"],
    }
    for name, command in commands.items():
        if shutil.which(command[0]) is None:
            manifest["commands"][name] = {"available": False, "error": "command not found"}
            continue
        code, output = _run(command)
        manifest["commands"][name] = {
            "available": code == 0,
            "exit_code": code,
            "raw": output,
        }
    manifest["gpu_count"] = sum(
        1
        for line in manifest["commands"].get("nvidia_query", {}).get("raw", "").splitlines()
        if line.strip()
    )
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
