from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.ada_sm89 import (
    CANONICAL_CUDA_SERIES,
    CANONICAL_PTX_ISA,
    CAPABILITY_SCHEMA_VERSION,
    FEATURE_CONTRACTS,
    MINIMUM_CUDA_SERIES,
    NATIVE_TARGET,
    PLATFORM_CELLS,
    PTX_FALLBACK_TARGET,
    capability_contract_manifest,
    version_tuple,
)
from infraswe.models.ada_sm89 import AdaSM89CapabilityManifest, AdaSM89Gate

PROBE_VERSION = "ada-sm89-capability-v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def command(
    argv: list[str], *, timeout_seconds: int = 60, env: dict[str, str] | None = None
) -> dict[str, Any]:
    executable = shutil.which(argv[0]) if not Path(argv[0]).is_file() else argv[0]
    if executable is None:
        return {
            "argv": argv,
            "available": False,
            "returncode": 127,
            "stdout": "",
            "stderr": "command not found",
        }
    resolved = [str(executable), *argv[1:]]
    try:
        completed = subprocess.run(
            resolved,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        return {
            "argv": resolved,
            "available": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": resolved,
            "available": True,
            "returncode": 124 if isinstance(error, subprocess.TimeoutExpired) else 127,
            "stdout": getattr(error, "stdout", "") or "",
            "stderr": getattr(error, "stderr", "") or f"{type(error).__name__}: {error}",
        }


def parse_cuda_release(text: str) -> str | None:
    match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def cuda_series(release: str | None) -> str | None:
    if release is None:
        return None
    parts = release.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else None


def parse_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        index, uuid, name, memory_mib, driver, pci_bus_id, pci_device_id, compute_cap = fields
        try:
            parsed_index = int(index)
        except ValueError:
            continue
        try:
            memory_bytes = int(float(memory_mib) * 1024**2)
        except ValueError:
            memory_bytes = None
        rows.append(
            {
                "index": parsed_index,
                "uuid": uuid,
                "name": name,
                "memory_bytes": memory_bytes,
                "driver_version": driver,
                "pci_bus_id": pci_bus_id,
                "pci_device_id": pci_device_id,
                "compute_capability": compute_cap,
                "architecture": "sm" + compute_cap.replace(".", ""),
            }
        )
    return rows


def identify_platform_cell(gpu: dict[str, Any] | None, *, allow_generic_sm89: bool) -> str | None:
    if not gpu or gpu.get("compute_capability") != "8.9":
        return None
    name = str(gpu.get("name", ""))
    for cell_name in ("l40s-48gb-pcie", "l20-48gb-pcie"):
        pattern = str(PLATFORM_CELLS[cell_name]["product_pattern"])
        if re.search(pattern, name, flags=re.IGNORECASE):
            return cell_name
    return "generic-sm89" if allow_generic_sm89 else None


def target_probe_source() -> str:
    return """extern \"C\" __global__ void infraswe_sm89_target_probe(int* value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *value += 1;
}
"""


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def compile_targets(artifact_root: Path) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    source = artifact_root / "target-sm89.cu"
    ptx = artifact_root / "target-sm89.ptx"
    cubin = artifact_root / "target-sm89.cubin"
    fatbin = artifact_root / "target-sm89.fatbin"
    source.write_text(target_probe_source(), encoding="utf-8")
    commands = {
        "ptx": command(["nvcc", "-O2", "-arch=compute_89", "-ptx", str(source), "-o", str(ptx)]),
        "cubin": command(["nvcc", "-O2", "-arch=sm_89", "-cubin", str(source), "-o", str(cubin)]),
        "fatbin": command(
            [
                "nvcc",
                "-O2",
                "-fatbin",
                "-gencode",
                "arch=compute_89,code=sm_89",
                "-gencode",
                "arch=compute_89,code=compute_89",
                str(source),
                "-o",
                str(fatbin),
            ]
        ),
    }
    for name, record in commands.items():
        (artifact_root / f"{name}.build.log").write_text(
            record["stdout"] + record["stderr"], encoding="utf-8"
        )
    inspection = {
        "list_elf": command(["cuobjdump", "--list-elf", str(fatbin)]),
        "list_ptx": command(["cuobjdump", "--list-ptx", str(fatbin)]),
        "dump_sass": command(["cuobjdump", "--dump-sass", str(cubin)]),
    }
    for name, record in inspection.items():
        (artifact_root / f"{name}.txt").write_text(
            record["stdout"] + record["stderr"], encoding="utf-8"
        )
    ptx_text = ptx.read_text(encoding="utf-8", errors="replace") if ptx.exists() else ""
    ptx_version_match = re.search(r"(?m)^\s*\.version\s+([0-9.]+)\b", ptx_text)
    ptx_target_match = re.search(r"(?m)^\s*\.target\s+(sm_[0-9]+)\b", ptx_text)
    passed = bool(
        all(record["returncode"] == 0 for record in commands.values())
        and all(path.is_file() and path.stat().st_size > 0 for path in (ptx, cubin, fatbin))
        and ptx_target_match
        and ptx_target_match.group(1) == NATIVE_TARGET
    )
    return {
        "passed": passed,
        "native_target": NATIVE_TARGET,
        "ptx_fallback_target": PTX_FALLBACK_TARGET,
        "detected_ptx_isa": ptx_version_match.group(1) if ptx_version_match else None,
        "detected_ptx_target": ptx_target_match.group(1) if ptx_target_match else None,
        "source": _artifact_record(source),
        "ptx": _artifact_record(ptx),
        "cubin": _artifact_record(cubin),
        "fatbin": _artifact_record(fatbin),
        "commands": commands,
        "inspection": inspection,
    }


def compile_and_run_runtime_probe(
    source: Path, artifact_root: Path, *, device_index: int
) -> dict[str, Any]:
    binary = artifact_root / "cuda-runtime-attributes"
    build = command(["nvcc", "-std=c++17", str(source), "-lcuda", "-o", str(binary)])
    result: dict[str, Any] = {"build": build, "run": None, "parsed": {}}
    if build["returncode"] != 0:
        return result
    run = command([str(binary), str(device_index)])
    result["run"] = run
    if run["returncode"] == 0:
        try:
            result["parsed"] = json.loads(run["stdout"])
        except json.JSONDecodeError:
            result["parsed"] = {"available": False, "error": "invalid probe JSON"}
    return result


def _runtime_value(runtime_probe: dict[str, Any], name: str) -> Any:
    parsed = runtime_probe.get("parsed", {})
    if not isinstance(parsed, dict):
        return None
    if name in parsed:
        return parsed[name]
    attribute = parsed.get("attributes", {}).get(name, {})
    return attribute.get("value") if isinstance(attribute, dict) else None


def _feature_states(
    *, platform_passed: bool, compile_passed: bool, detected_ptx_isa: str | None
) -> dict[str, dict[str, Any]]:
    states = {}
    for feature_id, contract in FEATURE_CONTRACTS.items():
        reasons = []
        if not platform_passed:
            state = "unsupported_on_current_device"
            reasons.append("a canonical L40S/L20 CC 8.9 platform was not established")
        elif contract.required_target and not compile_passed:
            state = "misconfigured"
            reasons.append("sm_89 cubin plus compute_89 PTX compiler path is unavailable")
        elif contract.minimum_ptx_isa and (
            detected_ptx_isa is None
            or version_tuple(detected_ptx_isa) < version_tuple(contract.minimum_ptx_isa)
        ):
            state = "toolchain_too_old"
            reasons.append(f"feature requires PTX {contract.minimum_ptx_isa}+")
        else:
            state = "pending_evidence"
            reasons.append("capability is eligible; task-bound evidence is still required")
        states[feature_id] = {
            "namespace": contract.namespace,
            "phase": contract.phase,
            "required_target": contract.required_target,
            "state": state,
            "external_evidence": contract.external_evidence,
            "leaderboard_eligible": contract.leaderboard_eligible,
            "reason": "; ".join(reasons),
        }
    return states


def probe(
    *,
    device_index: int,
    artifact_root: Path,
    profile_id: str,
    runtime_probe_source: Path,
    allow_generic_sm89: bool = False,
    execution_mode: str = "bare_metal",
    cooling_variant: str = "unknown",
) -> dict[str, Any]:
    base_query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id,pci.device_id,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows = parse_gpu_rows(base_query["stdout"]) if base_query["returncode"] == 0 else []
    selected = next((row for row in gpu_rows if row["index"] == device_index), None)
    platform_cell = identify_platform_cell(selected, allow_generic_sm89=allow_generic_sm89)
    framebuffer_bytes = selected.get("memory_bytes") if selected else None
    memory_sane = bool(
        isinstance(framebuffer_bytes, int) and 44 * 1024**3 <= framebuffer_bytes <= 50 * 1024**3
    )
    canonical_cell = platform_cell in {"l40s-48gb-pcie", "l20-48gb-pcie"}
    platform_passed = bool(canonical_cell and memory_sane)
    platform_failures = []
    if not selected or selected.get("compute_capability") != "8.9":
        platform_failures.append("ADA_SM89_COMPUTE_CAPABILITY_NOT_ESTABLISHED")
    if selected and selected.get("compute_capability") == "8.9" and platform_cell is None:
        platform_failures.append("ADA_SM89_PRODUCT_NOT_CANONICAL")
    if platform_cell == "generic-sm89":
        platform_failures.append("ADA_GENERIC_SM89_NOT_A_BOARD_SCORE_CELL")
    if canonical_cell and not memory_sane:
        platform_failures.append("ADA_48GB_FRAMEBUFFER_NOT_ESTABLISHED")

    tool_commands = {
        "nvcc_version": command(["nvcc", "--version"]),
        "ptxas_version": command(["ptxas", "--version"]),
        "cuobjdump_version": command(["cuobjdump", "--version"]),
        "nvdisasm_version": command(["nvdisasm", "--version"]),
        "ncu_version": command(["ncu", "--version"]),
        "nsys_version": command(["nsys", "--version"]),
        "nvcc_gpu_arch": command(["nvcc", "--list-gpu-arch"]),
        "nvcc_gpu_code": command(["nvcc", "--list-gpu-code"]),
    }
    cuda_release = parse_cuda_release(
        tool_commands["nvcc_version"]["stdout"] + tool_commands["nvcc_version"]["stderr"]
    )
    detected_series = cuda_series(cuda_release)
    target_results = compile_targets(artifact_root / "compile-targets")
    detected_ptx_isa = target_results["detected_ptx_isa"]
    binary_tools_available = all(
        tool_commands[name]["returncode"] == 0
        for name in ("nvcc_version", "ptxas_version", "cuobjdump_version", "nvdisasm_version")
    )
    minimum_version = bool(
        detected_series and version_tuple(detected_series) >= version_tuple(MINIMUM_CUDA_SERIES)
    )
    compile_passed = bool(binary_tools_available and minimum_version and target_results["passed"])
    compile_failures = []
    if not binary_tools_available:
        compile_failures.append("CUDA_BINARY_UTILITIES_MISSING")
    if not minimum_version:
        compile_failures.append("CUDA_11_8_MINIMUM_NOT_ESTABLISHED")
    if not target_results["passed"]:
        compile_failures.append("SM89_NATIVE_AND_PTX_TARGETS_NOT_ESTABLISHED")

    runtime_probe = compile_and_run_runtime_probe(
        runtime_probe_source, artifact_root, device_index=device_index
    )
    detail_commands = {
        "nvidia_smi_query": command(["nvidia-smi", "-q", "-i", str(device_index)]),
        "nvidia_topology": command(["nvidia-smi", "topo", "-m"]),
        "lspci": command(["lspci", "-vv"]),
        "numactl": command(["numactl", "--hardware"]),
    }
    canonical_toolchain = bool(detected_series == CANONICAL_CUDA_SERIES)
    warnings = []
    if compile_passed and not canonical_toolchain:
        warnings.append(
            f"CUDA {cuda_release} passed the minimum compiler gate; canonical publishing uses "
            f"CUDA {CANONICAL_CUDA_SERIES}/PTX {CANONICAL_PTX_ISA}"
        )
    if platform_cell == "generic-sm89":
        warnings.append(
            "generic-sm89 is compile/compatibility-only and cannot publish board scores"
        )
    if tool_commands["ncu_version"]["returncode"] != 0:
        warnings.append("Nsight Compute is unavailable; E3 counters remain unresolved, not zero")
    if tool_commands["nsys_version"]["returncode"] != 0:
        warnings.append("Nsight Systems is unavailable; E2 evidence remains unresolved, not zero")

    gates = {
        "platform": AdaSM89Gate(
            status="pass" if platform_passed else "fail",
            evidence={
                "selected_gpu": selected,
                "platform_cell": platform_cell,
                "framebuffer_48gb_sanity": memory_sane,
            },
            failure_codes=platform_failures,
        ),
        "compile": AdaSM89Gate(
            status="pass" if compile_passed else "fail",
            evidence={
                "cuda_release": cuda_release,
                "ptx_isa": detected_ptx_isa,
                "canonical": canonical_toolchain,
                "native_target": NATIVE_TARGET,
                "ptx_fallback_target": PTX_FALLBACK_TARGET,
                "target_results": target_results,
            },
            failure_codes=compile_failures,
        ),
        "native_feature": AdaSM89Gate(
            status="unresolved",
            evidence={"reason": "per-task PTX/SASS and dispatch evidence is pending"},
            failure_codes=["ADA_NATIVE_TASK_EVIDENCE_PENDING"],
        ),
        "correctness_liveness": AdaSM89Gate(
            status="unresolved",
            evidence={"reason": "sealed task workloads have not completed"},
            failure_codes=["ADA_TASK_EXECUTION_PENDING"],
        ),
        "evidence_performance": AdaSM89Gate(
            status="unresolved",
            evidence={"reason": "E0/E1/E2/E3 and local anchors are incomplete"},
            failure_codes=["ADA_PROFILE_EVIDENCE_PENDING"],
        ),
    }
    all_failures = sorted({code for gate in gates.values() for code in gate.failure_codes})
    if platform_passed and compile_passed:
        status = "ready"
    elif compile_passed:
        status = "compile_only"
    elif canonical_cell:
        status = "partial"
    else:
        status = "not_ready"

    contract = capability_contract_manifest()
    runtime_parsed = runtime_probe.get("parsed", {})
    fingerprint_payload = {
        "profile_id": profile_id,
        "platform_cell": platform_cell,
        "selected_gpu": selected,
        "runtime": runtime_parsed,
        "cuda_release": cuda_release,
        "ptx_isa": detected_ptx_isa,
        "target_artifacts": {
            key: target_results[key].get("sha256") for key in ("ptx", "cubin", "fatbin")
        },
        "execution_mode": execution_mode,
        "cooling_variant": cooling_variant,
        "contract_sha256": canonical_sha256(contract),
    }
    manifest = AdaSM89CapabilityManifest(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        generated_at=utc_now(),
        probe_version=PROBE_VERSION,
        profile_id=profile_id,
        status=status,
        platform_cell=platform_cell,
        capability_fingerprint=canonical_sha256(fingerprint_payload),
        platform={
            "vendor": "NVIDIA",
            "product_name": selected.get("name") if selected else None,
            "pci_bus_id": selected.get("pci_bus_id") if selected else None,
            "pci_device_id": selected.get("pci_device_id") if selected else None,
            "pci_vendor_device": selected.get("pci_device_id") if selected else None,
            "uuid": selected.get("uuid") if selected else None,
            "architecture": "ada" if canonical_cell else None,
            "gpu_family": "AD102" if canonical_cell else None,
            "compute_capability": selected.get("compute_capability") if selected else None,
            "target": NATIVE_TARGET,
            "host_arch": platform.machine().lower(),
            "cpu_count": os.cpu_count(),
            "runtime_probe": runtime_probe,
            "all_gpus": gpu_rows,
            "base_query": base_query,
        },
        memory={
            "framebuffer_bytes": framebuffer_bytes,
            "ecc_enabled": _runtime_value(runtime_probe, "ecc_enabled"),
            "nominal_bandwidth_gbps": 864 if canonical_cell else None,
            "l2_bytes": _runtime_value(runtime_probe, "l2_bytes"),
            "memory_clock_mhz": (
                _runtime_value(runtime_probe, "memory_clock_khz") / 1000
                if isinstance(_runtime_value(runtime_probe, "memory_clock_khz"), int)
                else None
            ),
            "calibrated_read_gbps": None,
            "calibrated_write_gbps": None,
            "calibrated_bidir_gbps": None,
        },
        power_thermal={
            "cooling_variant": cooling_variant,
            "default_power_limit_watts": None,
            "enforced_power_limit_watts": None,
            "idle_temperature_c": None,
            "steady_temperature_c": None,
            "throttle_reasons": [],
            "clocks_locked": False,
            "raw_nvidia_smi_query": detail_commands["nvidia_smi_query"],
        },
        interconnect={
            "pcie_generation_max": _runtime_value(runtime_probe, "pcie_generation_max"),
            "pcie_generation_negotiated": _runtime_value(
                runtime_probe, "pcie_generation_negotiated"
            ),
            "pcie_width_max": _runtime_value(runtime_probe, "pcie_width_max"),
            "pcie_width_negotiated": _runtime_value(runtime_probe, "pcie_width_negotiated"),
            "numa_node": _runtime_value(runtime_probe, "numa_node"),
            "nvlink_present": None,
            "peer_access_matrix": [],
            "commands": detail_commands,
        },
        virtualization={
            "mode": execution_mode,
            "vgpu_profile": None,
            "mig_supported": False if canonical_cell else None,
            "mps_enabled": None,
            "compute_mode": None,
        },
        software={
            "driver": selected.get("driver_version") if selected else None,
            "cuda_toolkit": cuda_release,
            "ptx_isa": detected_ptx_isa,
            "canonical_cuda_toolkit": CANONICAL_CUDA_SERIES,
            "canonical_ptx_isa": CANONICAL_PTX_ISA,
            "commands": tool_commands,
            "compile_targets": target_results,
        },
        features=_feature_states(
            platform_passed=platform_passed,
            compile_passed=compile_passed,
            detected_ptx_isa=detected_ptx_isa,
        ),
        calibration={
            "anchor_version": None,
            "anchor_hash": None,
            "date": None,
            "repetitions": 0,
            "thermal_soak_seconds": 0,
            "status": "unresolved",
        },
        gates=gates,
        contract_sha256=canonical_sha256(contract),
        contract=contract,
        failure_codes=all_failures,
        warnings=warnings,
    )
    return manifest.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an NVIDIA L40S/L20 Ada SM89 cell")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile-id", default="gpu-1x-sm89-ada-cuda133")
    parser.add_argument("--runtime-probe-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-generic-sm89", action="store_true")
    parser.add_argument(
        "--execution-mode", choices=("bare_metal", "passthrough", "vgpu"), default="bare_metal"
    )
    parser.add_argument(
        "--cooling-variant", choices=("passive", "liquid", "unknown"), default="unknown"
    )
    parser.add_argument("--require-platform", action="store_true")
    parser.add_argument("--require-toolchain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = probe(
        device_index=args.device_index,
        artifact_root=args.artifact_root,
        profile_id=args.profile_id,
        runtime_probe_source=args.runtime_probe_source,
        allow_generic_sm89=args.allow_generic_sm89,
        execution_mode=args.execution_mode,
        cooling_variant=args.cooling_variant,
    )
    atomic_write_json(args.output, result)
    if args.require_platform and result["gates"]["platform"]["status"] != "pass":
        raise SystemExit(2)
    if args.require_toolchain and result["gates"]["compile"]["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
