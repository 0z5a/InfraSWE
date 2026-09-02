from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl
from bench_utils import (
    atomic_write_json,
    choose_repetitions,
    hardware_manifest,
    paired_blocks,
    profiler_evidence,
    sha256_file,
    tensor_correctness,
    utc_now,
)

ROWS = 4096
COLS = 4096
BLOCK_M = 32
BLOCK_N = 128


@triton.jit
def tma_copy_add_kernel(
    source,
    destination,
    rows,
    columns,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    source_descriptor = tl.make_tensor_descriptor(
        source,
        shape=[rows, columns],
        strides=[columns, 1],
        block_shape=[block_m, block_n],
    )
    destination_descriptor = tl.make_tensor_descriptor(
        destination,
        shape=[rows, columns],
        strides=[columns, 1],
        block_shape=[block_m, block_n],
    )
    row_offset = tl.program_id(0) * block_m
    column_offset = tl.program_id(1) * block_n
    tile = source_descriptor.load([row_offset, column_offset])
    destination_descriptor.store([row_offset, column_offset], tile + 1.0)


def triton_allocator(size: int, alignment: int, stream: int | None):
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.uint8)


def command(*args: str) -> dict[str, Any]:
    completed = subprocess.run(args, text=True, capture_output=True, check=False)
    return {
        "argv": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def cuda_driver_attributes() -> dict[str, Any]:
    driver = ctypes.CDLL("libcuda.so.1")
    init_returncode = driver.cuInit(0)
    device = ctypes.c_int()
    device_returncode = driver.cuDeviceGet(ctypes.byref(device), 0)
    attributes = {}
    for name, identifier in {
        "tensor_map_access_supported": 127,
        "fabric_handle_supported": 128,
        "multicast_supported": 132,
    }.items():
        value = ctypes.c_int()
        returncode = driver.cuDeviceGetAttribute(ctypes.byref(value), identifier, device)
        attributes[name] = {
            "attribute_id": identifier,
            "returncode": returncode,
            "value": value.value,
        }
    return {
        "cu_init_returncode": init_returncode,
        "cu_device_get_returncode": device_returncode,
        "attributes": attributes,
    }


def copy_compiler_artifacts(cache_root: Path, artifact_root: Path) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for suffix in (".ttgir", ".ptx", ".cubin"):
        for index, source in enumerate(sorted(cache_root.rglob(f"*{suffix}")), start=1):
            destination = artifact_root / f"triton-{index}{suffix}"
            shutil.copy2(source, destination)
            copied.append(
                {
                    "path": destination.name,
                    "source_cache_path": str(source),
                    "sha256": sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )

    ptx_lines = []
    ttgir_lines = []
    sass_lines = []
    sass_artifacts = []
    for path in sorted(artifact_root.glob("*.ptx")):
        text = path.read_text(encoding="utf-8", errors="replace")
        ptx_lines.extend(
            line.strip()
            for line in text.splitlines()
            if re.search(r"cp\.async\.bulk\.tensor|tensormap", line, re.IGNORECASE)
        )
    for path in sorted(artifact_root.glob("*.ttgir")):
        text = path.read_text(encoding="utf-8", errors="replace")
        ttgir_lines.extend(
            line.strip()
            for line in text.splitlines()
            if re.search(r"\btma\b|async_tma|tensor_memory", line, re.IGNORECASE)
        )
    for cubin in sorted(artifact_root.glob("*.cubin")):
        dump = command("cuobjdump", "--dump-sass", str(cubin))
        destination = cubin.with_suffix(".sass.txt")
        destination.write_text(dump["stdout"] + dump["stderr"], encoding="utf-8")
        sass_artifacts.append(
            {
                "path": destination.name,
                "sha256": sha256_file(destination),
                "returncode": dump["returncode"],
                "size_bytes": destination.stat().st_size,
            }
        )
        sass_lines.extend(
            line.strip()
            for line in dump["stdout"].splitlines()
            if re.search(r"\bUTMA(?:LDG|STG)?\b", line, re.IGNORECASE)
        )
    return {
        "cache_root": str(cache_root),
        "artifacts": copied,
        "sass_artifacts": sass_artifacts,
        "ptx_instruction_lines": ptx_lines[:64],
        "ttgir_tma_lines": ttgir_lines[:64],
        "sass_instruction_lines": sass_lines[:64],
        "instruction_gate_passed": bool(ptx_lines or sass_lines),
    }


def run_tma(
    *, replay_index: int, blocks: int, min_timed_span_ms: float, artifact_root: Path
) -> dict[str, Any]:
    triton.set_allocator(triton_allocator)
    source = torch.randn((ROWS, COLS), device="cuda", dtype=torch.bfloat16)
    reference_output = torch.empty_like(source)
    candidate_output = torch.empty_like(source)
    grid = (ROWS // BLOCK_M, COLS // BLOCK_N)

    def reference():
        torch.add(source, 1.0, out=reference_output)
        return reference_output

    def candidate():
        tma_copy_add_kernel[grid](
            source,
            candidate_output,
            ROWS,
            COLS,
            BLOCK_M,
            BLOCK_N,
            num_warps=4,
        )
        return candidate_output

    reference()
    candidate()
    torch.cuda.synchronize()
    errors = tensor_correctness(reference_output, candidate_output)
    original_value = float(candidate_output[0, 0].item())
    source[0, 0].add_(2.0)
    candidate()
    torch.cuda.synchronize()
    dynamic_value = float(candidate_output[0, 0].item())
    reference()
    torch.cuda.synchronize()
    errors_after_change = tensor_correctness(reference_output, candidate_output)
    correctness = {
        **errors_after_change,
        "initial_errors": errors,
        "dynamic_input_changes_output": dynamic_value != original_value,
    }
    correctness["passed"] = bool(
        correctness["max_abs_error"] == 0.0
        and correctness["dynamic_input_changes_output"]
    )

    reference_repetitions, reference_pilot_us = choose_repetitions(
        reference, min_timed_span_ms=min_timed_span_ms
    )
    candidate_repetitions, candidate_pilot_us = choose_repetitions(
        candidate, min_timed_span_ms=min_timed_span_ms
    )
    measurements = paired_blocks(
        reference=reference,
        candidate=candidate,
        reference_repetitions=reference_repetitions,
        candidate_repetitions=candidate_repetitions,
        blocks=blocks,
        seed=91_000 + replay_index,
    )
    compiler = copy_compiler_artifacts(
        Path(os.environ["TRITON_CACHE_DIR"]), artifact_root / "tma"
    )
    profile = profiler_evidence(candidate)
    return {
        "status": (
            "passed"
            if correctness["passed"]
            and profile.get("captured")
            and compiler["instruction_gate_passed"]
            else "failed"
        ),
        "driver_attribute": cuda_driver_attributes()["attributes"][
            "tensor_map_access_supported"
        ],
        "shape": [ROWS, COLS],
        "dtype": "bfloat16",
        "block_shape": [BLOCK_M, BLOCK_N],
        "correctness": correctness,
        "measurement": {
            "blocks": measurements,
            "reference_repetitions": reference_repetitions,
            "candidate_repetitions": candidate_repetitions,
            "reference_pilot_us": reference_pilot_us,
            "candidate_pilot_us": candidate_pilot_us,
            "reference": "torch.add-out",
            "candidate": "triton-tensor-descriptor-copy-add",
        },
        "profiler": profile,
        "compiler_evidence": compiler,
    }


def multimem_ptx_source() -> str:
    return """.version 8.7
.target sm_90
.address_size 64

.visible .entry multimem_load_reduce_probe(
    .param .u64 multimem_address,
    .param .u64 output_address
)
{
    .reg .b64 address;
    .reg .b64 output;
    .reg .b32 value;
    ld.param.u64 address, [multimem_address];
    ld.param.u64 output, [output_address];
    multimem.ld_reduce.relaxed.sys.global.add.u32 value, [address];
    st.global.u32 [output], value;
    ret;
}
"""


def probe_multimem(artifact_root: Path) -> dict[str, Any]:
    root = artifact_root / "multimem"
    root.mkdir(parents=True, exist_ok=True)
    ptx = root / "multimem-probe.ptx"
    cubin = root / "multimem-probe.cubin"
    ptx.write_text(multimem_ptx_source(), encoding="utf-8")
    assembly = command(
        "ptxas", "--verbose", "--gpu-name=sm_90", str(ptx), "--output-file", str(cubin)
    )
    log = root / "ptxas.txt"
    log.write_text(assembly["stdout"] + assembly["stderr"], encoding="utf-8")
    sass = command("cuobjdump", "--dump-sass", str(cubin)) if cubin.exists() else None
    sass_path = root / "multimem-probe.sass.txt"
    if sass is not None:
        sass_path.write_text(sass["stdout"] + sass["stderr"], encoding="utf-8")
    sass_instruction_lines = (
        [
            line.strip()
            for line in sass["stdout"].splitlines()
            if re.search(r"\b(?:LDGMC|STGMC|REDGMC)\b", line, re.IGNORECASE)
        ]
        if sass is not None
        else []
    )

    driver = cuda_driver_attributes()
    multicast = driver["attributes"]["multicast_supported"]
    topology = command("nvidia-smi", "topo", "-m")
    nvlink = command("nvidia-smi", "nvlink", "-s")
    runtime_available = multicast["returncode"] == 0 and multicast["value"] == 1
    return {
        "status": "runtime_available" if runtime_available else "topology_unavailable",
        "execution_attempted": False,
        "execution_reason": (
            "runtime implementation requires a valid CUDA multicast mapping"
            if runtime_available
            else "CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED is 0; launching multimem.* "
            "against an ordinary pointer would be undefined behavior"
        ),
        "driver": driver,
        "visible_cuda_device_count": torch.cuda.device_count(),
        "topology": topology,
        "nvlink_status": nvlink,
        "toolchain_compile": {
            **assembly,
            "passed": assembly["returncode"] == 0 and cubin.exists(),
            "ptx_path": str(ptx.relative_to(artifact_root)),
            "ptx_sha256": sha256_file(ptx),
            "ptxas_log_path": str(log.relative_to(artifact_root)),
            "ptxas_log_sha256": sha256_file(log),
            "cubin_path": str(cubin.relative_to(artifact_root)) if cubin.exists() else None,
            "cubin_sha256": sha256_file(cubin) if cubin.exists() else None,
            "sass_path": (
                str(sass_path.relative_to(artifact_root)) if sass_path.exists() else None
            ),
            "sass_sha256": sha256_file(sass_path) if sass_path.exists() else None,
            "sass_returncode": sass["returncode"] if sass is not None else None,
            "sass_instruction_lines": sass_instruction_lines,
            "instruction_gate_passed": bool(sass_instruction_lines),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--min-timed-span-ms", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "schema_version": "0.3",
        "evidence_kind": "architecture-feature",
        "benchmark": "h200-sm90-feature-supplement-v1",
        "replay_index": args.replay_index,
        "started_at": utc_now(),
        "hardware": hardware_manifest(),
        "protocol": {
            "fresh_process": True,
            "paired_order": "randomized-ABBA-or-BAAB",
            "blocks": args.blocks,
            "min_timed_span_ms": args.min_timed_span_ms,
            "timer": "evaluator-owned-cuda-events",
            "multimem_undefined_behavior_forbidden": True,
        },
        "status": "running",
        "features": {},
    }
    try:
        payload["features"]["tma"] = run_tma(
            replay_index=args.replay_index,
            blocks=args.blocks,
            min_timed_span_ms=args.min_timed_span_ms,
            artifact_root=args.artifact_root,
        )
        payload["features"]["multimem"] = probe_multimem(args.artifact_root)
        payload["status"] = (
            "passed"
            if payload["features"]["tma"]["status"] == "passed"
            and payload["features"]["multimem"]["toolchain_compile"][
                "instruction_gate_passed"
            ]
            else "failed"
        )
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["finished_at"] = utc_now()
    atomic_write_json(args.output, payload)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
