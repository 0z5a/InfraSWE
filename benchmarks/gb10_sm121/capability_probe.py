from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.kernel.gb10 import (
    CANONICAL_CUDA_SERIES,
    CANONICAL_PTX_ISA,
    CAPABILITY_SCHEMA_VERSION,
    FEATURE_CONTRACTS,
    MINIMUM_CUDA_SERIES,
    TARGET_LANES,
    capability_contract_manifest,
    version_tuple,
)
from infraswe.models.gb10 import GB10CapabilityManifest, GB10Gate

PROBE_VERSION = "gb10-sm121-capability-v1"


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
    temporary.replace(path)


def command(argv: list[str], *, timeout_seconds: int = 60) -> dict[str, Any]:
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
        )
        return {
            "argv": resolved,
            "available": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "argv": resolved,
            "available": True,
            "returncode": 124,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "timeout",
        }


def parse_cuda_release(text: str) -> str | None:
    match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def ptx_isa_for_cuda(cuda_release: str | None) -> str | None:
    if cuda_release is None:
        return None
    major_minor = ".".join(cuda_release.split(".")[:2])
    known = {"13.0": "9.0", "13.1": "9.1", "13.2": "9.2", "13.3": "9.3"}
    if major_minor in known:
        return known[major_minor]
    if version_tuple(major_minor) < version_tuple("13.0"):
        return "8.7"
    return CANONICAL_PTX_ISA


def parse_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        index, uuid, name, memory_mib, driver, pci_bus_id, compute_capability = fields
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
                "compute_capability": compute_capability,
                "architecture": "sm" + compute_capability.replace(".", ""),
            }
        )
    return rows


def lane_source(lane: str, ptx_isa: str) -> str:
    return f""".version {ptx_isa}
.target {lane}
.address_size 64

.visible .entry infraswe_{lane}_target_probe()
{{
    ret;
}}
"""


def compile_target_lanes(artifact_root: Path, *, ptx_isa: str | None) -> dict[str, dict[str, Any]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for lane in TARGET_LANES:
        source = artifact_root / f"target-{lane}.ptx"
        cubin = artifact_root / f"target-{lane}.cubin"
        log = artifact_root / f"target-{lane}.ptxas.log"
        if ptx_isa is None:
            source.write_text("// PTX ISA unavailable\n", encoding="utf-8")
            record = {
                "argv": ["ptxas", f"-arch={lane}"],
                "available": shutil.which("ptxas") is not None,
                "returncode": 2,
                "stdout": "",
                "stderr": "CUDA/PTX version could not be determined",
            }
        else:
            source.write_text(lane_source(lane, ptx_isa), encoding="utf-8")
            record = command(
                ["ptxas", f"-arch={lane}", "--warn-on-spills", str(source), "-o", str(cubin)]
            )
        log.write_text(record["stdout"] + record["stderr"], encoding="utf-8")
        passed = record["returncode"] == 0 and cubin.is_file() and cubin.stat().st_size > 0
        results[lane] = {
            "status": "accepted" if passed else "rejected",
            "passed": passed,
            "source_path": source.name,
            "source_sha256": sha256_file(source),
            "cubin_path": cubin.name if cubin.exists() else None,
            "cubin_sha256": sha256_file(cubin) if cubin.exists() else None,
            "log_path": log.name,
            "log_sha256": sha256_file(log),
            "command": record,
        }
    return results


def compile_and_run_runtime_probe(source: Path, artifact_root: Path) -> dict[str, Any]:
    binary = artifact_root / "cuda-runtime-attributes"
    build = command(["nvcc", "-std=c++17", str(source), "-lcuda", "-o", str(binary)])
    result: dict[str, Any] = {"build": build, "run": None, "parsed": {}}
    if build["returncode"] != 0:
        return result
    run = command([str(binary)])
    result["run"] = run
    if run["returncode"] == 0:
        try:
            result["parsed"] = json.loads(run["stdout"])
        except json.JSONDecodeError:
            result["parsed"] = {"available": False, "error": "invalid probe JSON"}
    return result


def _attribute_value(runtime_probe: dict[str, Any], name: str) -> int | None:
    parsed = runtime_probe.get("parsed", runtime_probe)
    item = parsed.get("attributes", {}).get(name, {}) if isinstance(parsed, dict) else {}
    value = item.get("value") if isinstance(item, dict) else None
    return int(value) if isinstance(value, int) else None


def _feature_states(
    *, platform_passed: bool, compile_passed: bool, ptx_isa: str | None, attributes: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    states = {}
    for feature_id, contract in FEATURE_CONTRACTS.items():
        reasons = []
        if not platform_passed:
            state = "unsupported"
            reasons.append("GB10/AArch64 platform identity was not established")
        elif contract.required_target and not compile_passed:
            state = "misconfigured"
            reasons.append("required SM121 compiler lane is unavailable")
        elif contract.minimum_ptx_isa and (
            ptx_isa is None or version_tuple(ptx_isa) < version_tuple(contract.minimum_ptx_isa)
        ):
            state = "toolchain_too_old"
            reasons.append(f"feature requires PTX {contract.minimum_ptx_isa}+")
        else:
            state = "pending_evidence"
            reasons.append(
                "platform capability is eligible; native task evidence is still required"
            )
        if feature_id == "GB10-UMA-001":
            required = {
                "unified_addressing",
                "pageable_memory_access",
                "host_page_table_coherence",
            }
            observed = {name for name in required if _attribute_value(attributes, name) == 1}
            if platform_passed and observed == required:
                state = "pending_evidence"
            elif platform_passed:
                state = "unresolved"
                reasons.append("required full-UMA runtime attributes were not all observed")
        if feature_id == "GB10-ROCE-001":
            state = "not_applicable"
            reasons = ["single-node probe; scale-out is scored separately"]
        states[feature_id] = {
            "namespace": contract.namespace,
            "phase": contract.phase,
            "required_target": contract.required_target,
            "state": state,
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
) -> dict[str, Any]:
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows = parse_gpu_rows(query["stdout"]) if query["returncode"] == 0 else []
    selected = next((row for row in gpu_rows if row["index"] == device_index), None)
    machine = platform.machine().lower()
    hardware_passed = bool(
        selected
        and selected["compute_capability"] == "12.1"
        and re.search(r"\bGB10\b", str(selected["name"]), flags=re.IGNORECASE)
        and machine in {"aarch64", "arm64"}
    )
    platform_failures = [] if hardware_passed else ["GB10_PLATFORM_IDENTITY_NOT_ESTABLISHED"]

    tool_commands = {
        "nvcc_version": command(["nvcc", "--version"]),
        "ptxas_version": command(["ptxas", "--version"]),
        "cuobjdump_version": command(["cuobjdump", "--version"]),
        "nvdisasm_version": command(["nvdisasm", "--version"]),
        "nvcc_gpu_arch": command(["nvcc", "--list-gpu-arch"]),
        "nvcc_gpu_code": command(["nvcc", "--list-gpu-code"]),
    }
    detected_versions = {
        name: parse_cuda_release(record["stdout"] + record["stderr"])
        for name, record in tool_commands.items()
        if name.endswith("_version")
    }
    cuda_release = detected_versions.get("nvcc_version")
    ptx_isa = ptx_isa_for_cuda(cuda_release)
    lane_results = compile_target_lanes(artifact_root / "target-lanes", ptx_isa=ptx_isa)
    tools_available = all(
        tool_commands[name]["returncode"] == 0
        for name in ("nvcc_version", "ptxas_version", "cuobjdump_version", "nvdisasm_version")
    )
    minimum_version = bool(
        cuda_release and version_tuple(cuda_release[:4]) >= version_tuple(MINIMUM_CUDA_SERIES)
    )
    lanes_passed = all(result["passed"] for result in lane_results.values())
    compile_passed = tools_available and minimum_version and lanes_passed
    compile_failures = []
    if not tools_available:
        compile_failures.append("CUDA_BINARY_UTILITIES_MISSING")
    if not minimum_version:
        compile_failures.append("CUDA_13_0_MINIMUM_NOT_ESTABLISHED")
    if not lanes_passed:
        compile_failures.append("SM121_TARGET_LANES_NOT_ESTABLISHED")

    runtime_probe = compile_and_run_runtime_probe(runtime_probe_source, artifact_root)
    runtime_attributes = runtime_probe.get("parsed", {})
    topology_commands = {
        "lscpu": command(["lscpu", "--json"]),
        "nvidia_topology": command(["nvidia-smi", "topo", "-m"]),
        "ibv_devices": command(["ibv_devices"]),
        "ibv_devinfo": command(["ibv_devinfo"]),
        "network": command(["ip", "-json", "-brief", "link"]),
    }
    canonical = bool(cuda_release and cuda_release.startswith(CANONICAL_CUDA_SERIES))
    warnings = []
    if compile_passed and not canonical:
        warnings.append(
            f"minimum CUDA {cuda_release} validation passed; canonical CUDA "
            f"{CANONICAL_CUDA_SERIES}/PTX {CANONICAL_PTX_ISA} is not installed"
        )
    if _attribute_value(runtime_attributes, "gpudirect_rdma_supported") == 1:
        warnings.append("GPUDirect RDMA was reported; the RoCE task must still capability-gate it")

    gates = {
        "platform": GB10Gate(
            status="pass" if hardware_passed else "fail",
            evidence={"host_arch": machine, "selected_gpu": selected},
            failure_codes=platform_failures,
        ),
        "compile": GB10Gate(
            status="pass" if compile_passed else "fail",
            evidence={
                "cuda_release": cuda_release,
                "ptx_isa": ptx_isa,
                "canonical": canonical,
                "target_acceptance": {
                    lane: result["passed"] for lane, result in lane_results.items()
                },
            },
            failure_codes=compile_failures,
        ),
        "native_feature": GB10Gate(
            status="unresolved",
            evidence={"reason": "per-task PTX/SASS and runtime evidence has not been supplied"},
            failure_codes=["GB10_NATIVE_TASK_EVIDENCE_PENDING"],
        ),
        "correctness_liveness": GB10Gate(
            status="unresolved",
            evidence={"reason": "task workloads have not been executed"},
            failure_codes=["GB10_TASK_EXECUTION_PENDING"],
        ),
        "performance": GB10Gate(
            status="unresolved",
            evidence={"reason": "unprofiled steady-state samples have not been captured"},
            failure_codes=["GB10_PERFORMANCE_EVIDENCE_PENDING"],
        ),
    }
    failures = sorted({code for gate in gates.values() for code in gate.failure_codes})
    if hardware_passed and compile_passed:
        status = "ready"
    elif hardware_passed:
        status = "partial"
    else:
        status = "not_ready"
    contract = capability_contract_manifest()
    fingerprint_payload = {
        "profile_id": profile_id,
        "selected_gpu": selected,
        "host_arch": machine,
        "cuda_release": cuda_release,
        "ptx_isa": ptx_isa,
        "target_acceptance": {lane: result["passed"] for lane, result in lane_results.items()},
        "runtime_attributes": runtime_attributes,
        "contract_sha256": canonical_sha256(contract),
    }
    manifest = GB10CapabilityManifest(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        generated_at=utc_now(),
        probe_version=PROBE_VERSION,
        profile_id=profile_id,
        status=status,
        capability_fingerprint=canonical_sha256(fingerprint_payload),
        hardware={
            "host_arch": machine,
            "cpu_count": os.cpu_count(),
            "selected_gpu": selected,
            "all_gpus": gpu_rows,
            "query": query,
        },
        toolchain={
            "minimum_cuda_series": MINIMUM_CUDA_SERIES,
            "canonical_cuda_series": CANONICAL_CUDA_SERIES,
            "canonical_ptx_isa": CANONICAL_PTX_ISA,
            "detected_cuda_release": cuda_release,
            "detected_ptx_isa": ptx_isa,
            "canonical": canonical,
            "commands": tool_commands,
            "targets": lane_results,
        },
        runtime_attributes=runtime_probe,
        topology=topology_commands,
        features=_feature_states(
            platform_passed=hardware_passed,
            compile_passed=compile_passed,
            ptx_isa=ptx_isa,
            attributes=runtime_attributes,
        ),
        gates=gates,
        contract_sha256=canonical_sha256(contract),
        contract=contract,
        failure_codes=failures,
        warnings=warnings,
    )
    return manifest.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a GB10/AArch64/SM121 compiler cell")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile-id", default="gpu-1x-sm121-gb10-cuda130")
    parser.add_argument("--runtime-probe-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    )
    atomic_write_json(args.output, result)
    if args.require_platform and result["gates"]["platform"]["status"] != "pass":
        raise SystemExit(2)
    if args.require_toolchain and result["gates"]["compile"]["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
