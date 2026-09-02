from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.kernel.blackwell import (
    CAPABILITY_SCHEMA_VERSION,
    FEATURE_CONTRACTS,
    SCORE_NAMESPACES,
    STABLE_CUDA_SERIES,
    STABLE_PTX_ISA,
    TARGET_LANES,
    feature_contract_manifest,
)

PROBE_VERSION = "b200-sm100-capability-v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def command(argv: list[str], *, timeout_seconds: int = 30) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {
            "argv": argv,
            "available": False,
            "returncode": 127,
            "stdout": "",
            "stderr": "command not found",
        }
    resolved = [executable, *argv[1:]]
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
    except subprocess.TimeoutExpired as exception:
        return {
            "argv": resolved,
            "available": True,
            "returncode": 124,
            "stdout": exception.stdout or "",
            "stderr": exception.stderr or "timeout",
        }


def parse_cuda_release(text: str) -> str | None:
    match = re.search(r"\brelease\s+([0-9]+(?:\.[0-9]+)+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def parse_gpu_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        index, uuid, name, memory_mib, driver, pci_bus_id, compute_capability = fields
        try:
            parsed_index = int(index)
            memory_bytes = int(float(memory_mib) * 1024**2)
        except ValueError:
            continue
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


def cuda_driver_attributes(device_ordinal: int) -> dict[str, Any]:
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exception:
        return {"available": False, "error": f"OSError: {exception}", "attributes": {}}
    init_returncode = int(driver.cuInit(0))
    device = ctypes.c_int()
    device_returncode = int(driver.cuDeviceGet(ctypes.byref(device), device_ordinal))
    attributes: dict[str, Any] = {}
    for name, identifier in {
        "tensor_map_access_supported": 127,
        "fabric_handle_supported": 128,
        "multicast_supported": 132,
    }.items():
        value = ctypes.c_int()
        returncode = int(driver.cuDeviceGetAttribute(ctypes.byref(value), identifier, device))
        attributes[name] = {
            "attribute_id": identifier,
            "returncode": returncode,
            "value": value.value if returncode == 0 else None,
        }
    return {
        "available": init_returncode == 0 and device_returncode == 0,
        "cu_init_returncode": init_returncode,
        "cu_device_get_returncode": device_returncode,
        "attributes": attributes,
    }


def lane_source(lane: str) -> str:
    return f""".version {STABLE_PTX_ISA}
.target {lane}
.address_size 64

.visible .entry infraswe_{lane}_target_probe()
{{
    ret;
}}
"""


def compile_target_lanes(artifact_root: Path) -> dict[str, dict[str, Any]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    ptxas = shutil.which("ptxas")
    results: dict[str, dict[str, Any]] = {}
    for lane in TARGET_LANES:
        source = artifact_root / f"target-{lane}.ptx"
        cubin = artifact_root / f"target-{lane}.cubin"
        log = artifact_root / f"target-{lane}.ptxas.log"
        source.write_text(lane_source(lane), encoding="utf-8")
        if ptxas is None:
            record = {
                "argv": ["ptxas", f"-arch={lane}", str(source), "-o", str(cubin)],
                "available": False,
                "returncode": 127,
                "stdout": "",
                "stderr": "command not found",
            }
        else:
            record = command(
                [ptxas, f"-arch={lane}", "--warn-on-spills", str(source), "-o", str(cubin)]
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


def feature_capabilities(
    *, hardware_status: str, target_results: dict[str, dict[str, Any]], visible_gpu_count: int
) -> dict[str, dict[str, Any]]:
    features = {}
    for feature_id, contract in FEATURE_CONTRACTS.items():
        target_accepted = bool(target_results[contract.required_target]["passed"])
        if contract.phase == "preview-disabled":
            support_state = "unknown"
            reason = "PTX 9.4 is not part of the frozen CUDA 13.3 / PTX 9.3 baseline"
        elif hardware_status == "unsupported":
            support_state = "unsupported"
            reason = "selected device is not a B200 compute capability 10.0 device"
        elif hardware_status != "supported":
            support_state = "unknown"
            reason = "B200 hardware identity could not be established"
        elif not target_accepted:
            support_state = "misconfigured"
            reason = f"ptxas rejected required target {contract.required_target}"
        else:
            support_state = "supported"
            reason = "B200 identity and required compiler target were both observed"
        runtime_available: bool | None = True
        if contract.runtime_scope == "multi_gpu":
            runtime_available = visible_gpu_count >= 2
        if contract.runtime_scope == "compile_only":
            runtime_available = None
        features[feature_id] = {
            "namespace": contract.namespace,
            "phase": contract.phase,
            "required_target": contract.required_target,
            "runtime_scope": contract.runtime_scope,
            "architecture_eligible": (
                True
                if hardware_status == "supported"
                else False
                if hardware_status == "unsupported"
                else None
            ),
            "target_compiler_accepted": target_accepted,
            "support_state": support_state,
            "runtime_available": runtime_available,
            "reason": reason,
        }
    return features


def probe(
    *, device_index: int, leased_gpu_count: int, artifact_root: Path, profile_id: str
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
    if selected is None:
        hardware_status = "unknown"
    elif selected["compute_capability"] == "10.0" and re.search(
        r"\bB200\b", selected["name"], flags=re.IGNORECASE
    ):
        hardware_status = "supported"
    else:
        hardware_status = "unsupported"

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
    lane_results = compile_target_lanes(artifact_root)
    core_tool_versions = [
        detected_versions.get(name)
        for name in ("nvcc_version", "ptxas_version", "cuobjdump_version", "nvdisasm_version")
    ]
    tools_available = all(
        tool_commands[name]["returncode"] == 0
        for name in ("nvcc_version", "ptxas_version", "cuobjdump_version", "nvdisasm_version")
    )
    versions_match = all(
        version is not None and version.startswith(STABLE_CUDA_SERIES)
        for version in core_tool_versions
    )
    targets_passed = all(item["passed"] for item in lane_results.values())
    if tools_available and versions_match and targets_passed:
        toolchain_status = "supported"
    elif any(record["available"] for record in tool_commands.values()):
        toolchain_status = "misconfigured"
    else:
        toolchain_status = "failed"

    topology = command(["nvidia-smi", "topo", "-m"])
    driver_attributes = cuda_driver_attributes(device_index)
    contract = feature_contract_manifest()
    features = feature_capabilities(
        hardware_status=hardware_status,
        target_results=lane_results,
        visible_gpu_count=leased_gpu_count,
    )
    failures = []
    if hardware_status != "supported":
        failures.append("B200_HARDWARE_IDENTITY_NOT_ESTABLISHED")
    if toolchain_status != "supported":
        failures.append("CUDA_13_3_SM100_TOOLCHAIN_NOT_ESTABLISHED")
    status = "ready" if not failures else "not_ready"
    capability_fingerprint = canonical_sha256(
        {
            "profile_id": profile_id,
            "selected_gpu": selected,
            "required_cuda_series": STABLE_CUDA_SERIES,
            "required_ptx_isa": STABLE_PTX_ISA,
            "detected_versions": detected_versions,
            "target_acceptance": {lane: result["passed"] for lane, result in lane_results.items()},
            "contract_sha256": canonical_sha256(contract),
        }
    )
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "probe_version": PROBE_VERSION,
        "profile_id": profile_id,
        "device_index": device_index,
        "status": status,
        "capability_fingerprint": capability_fingerprint,
        "hardware": {
            "status": hardware_status,
            "selected_gpu": selected,
            "visible_gpu_count": len(gpu_rows),
            "leased_gpu_count": leased_gpu_count,
            "all_gpus": gpu_rows,
            "query": query,
            "driver_attributes": driver_attributes,
            "topology": topology,
        },
        "toolchain": {
            "status": toolchain_status,
            "required_cuda_series": STABLE_CUDA_SERIES,
            "required_ptx_isa": STABLE_PTX_ISA,
            "detected_versions": detected_versions,
            "commands": tool_commands,
            "targets": lane_results,
        },
        "features": features,
        "score_namespaces": {
            namespace: {
                "status": "disabled" if namespace == "PTX-Preview" else "pending_evidence",
                "leaderboard_score_100": None,
            }
            for namespace in SCORE_NAMESPACES
        },
        "contract_sha256": canonical_sha256(contract),
        "contract": contract,
        "failure_codes": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a CUDA 13.3 B200/SM100 compiler cell")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--leased-gpu-count", type=int, default=1)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile-id", default="gpu-1x-sm100-b200-cuda133")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-b200", action="store_true")
    parser.add_argument("--require-toolchain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = probe(
        device_index=args.device_index,
        leased_gpu_count=args.leased_gpu_count,
        artifact_root=args.artifact_root,
        profile_id=args.profile_id,
    )
    atomic_write_json(args.output, result)
    if args.require_b200 and result["hardware"]["status"] != "supported":
        raise SystemExit(2)
    if args.require_toolchain and result["toolchain"]["status"] != "supported":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
