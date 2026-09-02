from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sha256_tree(root: Path, suffixes: tuple[str, ...] = (".py", ".so")) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path).removeprefix("sha256:")))
    return "sha256:" + digest.hexdigest()


def module_evidence(module: Any, *, tree: bool = False) -> dict[str, Any]:
    source = Path(module.__file__).resolve()
    if tree and source.name == "__init__.py":
        digest = sha256_tree(source.parent)
        root = source.parent
    else:
        digest = sha256_file(source)
        root = source
    return {"module": module.__name__, "path": str(root), "sha256": digest}


def nvidia_smi_snapshot() -> dict[str, str]:
    fields = [
        "uuid",
        "name",
        "driver_version",
        "pci.bus_id",
        "temperature.gpu",
        "pstate",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "memory.total",
        "memory.used",
        "utilization.gpu",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    try:
        values = subprocess.check_output(command, text=True, timeout=10).strip().split(", ")
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(zip(fields, values, strict=False))


def _nested_value(payload: Any, *needles: str) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(needle in normalized for needle in needles):
                return value
        for value in payload.values():
            nested = _nested_value(value, *needles)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for value in payload:
            nested = _nested_value(value, *needles)
            if nested is not None:
                return nested
    return None


def rocm_smi_snapshot() -> dict[str, Any]:
    command = [
        "rocm-smi",
        "--showproductname",
        "--showuniqueid",
        "--showdriverversion",
        "--showmeminfo",
        "vram",
        "--showuse",
        "--showtemp",
        "--showpower",
        "--json",
    ]
    try:
        raw = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return {
        "driver_version": _nested_value(parsed, "driver version"),
        "unique_id": _nested_value(parsed, "unique id", "serial"),
        "product_name": _nested_value(
            parsed, "card series", "product name", "device name", "card model"
        ),
        "raw": parsed,
    }


def accelerator_identity(properties: Any | None = None) -> dict[str, Any]:
    properties = properties or torch.cuda.get_device_properties(0)
    torch_hip = getattr(torch.version, "hip", None)
    if torch_hip:
        raw_architecture = str(getattr(properties, "gcnArchName", ""))
        architecture_match = re.search(r"gfx[0-9a-z]+", raw_architecture.lower())
        architecture = architecture_match.group(0) if architecture_match else None
        return {
            "accelerator_vendor": "amd",
            "runtime": "rocm",
            "runtime_version": str(torch_hip),
            "architecture": architecture,
            "compute_capability": None,
        }
    capability = f"{properties.major}.{properties.minor}"
    return {
        "accelerator_vendor": "nvidia",
        "runtime": "cuda",
        "runtime_version": torch.version.cuda,
        "architecture": f"sm{properties.major}{properties.minor}",
        "compute_capability": capability,
    }


def device_timer_name() -> str:
    return "hip-events" if getattr(torch.version, "hip", None) else "cuda-events"


def hardware_manifest() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    identity = accelerator_identity(properties)
    nvidia_smi = nvidia_smi_snapshot() if identity["accelerator_vendor"] == "nvidia" else {}
    rocm_smi = rocm_smi_snapshot() if identity["accelerator_vendor"] == "amd" else {}
    return {
        **identity,
        "gpu_name": properties.name,
        "compute_unit_count": properties.multi_processor_count,
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": getattr(torch.version, "hip", None),
        "cudnn_version": (
            torch.backends.cudnn.version() if identity["accelerator_vendor"] == "nvidia" else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "driver_version": nvidia_smi.get("driver_version") or rocm_smi.get("driver_version"),
        "nvidia_smi": nvidia_smi,
        "rocm_smi": rocm_smi,
    }


def event_latency_us(function: Callable[[], Any], repetitions: int) -> float:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(repetitions):
        output = function()
    end.record()
    end.synchronize()
    if isinstance(output, torch.Tensor):
        _ = output.data_ptr()
    return start.elapsed_time(end) * 1000 / repetitions


def choose_repetitions(
    function: Callable[[], Any],
    *,
    min_timed_span_ms: float,
    maximum: int = 8192,
) -> tuple[int, float]:
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    pilot_us = event_latency_us(function, 1)
    repetitions = max(1, min(maximum, math.ceil(min_timed_span_ms * 1000 / pilot_us)))
    return repetitions, pilot_us


def paired_blocks(
    *,
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    reference_repetitions: int,
    candidate_repetitions: int,
    blocks: int,
    seed: int,
) -> list[dict[str, Any]]:
    generator = random.Random(seed)
    output = []
    for index in range(blocks):
        order = "ABBA" if generator.randrange(2) == 0 else "BAAB"
        measurements: dict[str, list[float]] = {"A": [], "B": []}
        for label in order:
            if label == "A":
                latency = event_latency_us(reference, reference_repetitions)
            else:
                latency = event_latency_us(candidate, candidate_repetitions)
            measurements[label].append(latency)
        output.append(
            {
                "block_index": index + 1,
                "order": order,
                "reference_latency_us": statistics.median(measurements["A"]),
                "candidate_latency_us": statistics.median(measurements["B"]),
                "reference_positions_us": measurements["A"],
                "candidate_positions_us": measurements["B"],
            }
        )
    return output


def tensor_correctness(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = (reference_float - candidate_float).abs()
    reference_norm = torch.linalg.vector_norm(reference_float)
    relative_l2 = torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        reference_float.flatten(), candidate_float.flatten(), dim=0
    )
    return {
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "relative_l2_error": float(relative_l2.item()),
        "cosine_similarity": float(cosine.item()),
    }


def profiler_evidence(function: Callable[[], Any]) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    try:
        with torch.profiler.profile(activities=activities) as profile:
            function()
            torch.cuda.synchronize()
        events = []
        for event in profile.key_averages():
            self_device_time = getattr(
                event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)
            )
            device_time = getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0))
            if self_device_time <= 0 and device_time <= 0:
                continue
            events.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "self_device_time_us": self_device_time,
                    "device_time_us": device_time,
                }
            )
        events.sort(key=lambda item: item["device_time_us"], reverse=True)
        device_events = events[:64]
        return {
            "captured": True,
            "device_event_runtime": "hip" if getattr(torch.version, "hip", None) else "cuda",
            "device_events": device_events,
            # Kept for v0.3 evidence readers and already-produced CUDA packs.
            "cuda_events": device_events,
        }
    except Exception as error:  # profiler support differs across framework builds
        return {"captured": False, "error": f"{type(error).__name__}: {error}"}
