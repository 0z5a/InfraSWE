#!/usr/bin/env python3
"""H100 probe for the exact Liger program-id widening mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@triton.jit
def _boundary_base(
    output,
    stride,
    rows_per_program,
    target_program: tl.constexpr,
    backward: tl.constexpr,
):
    program_id = tl.program_id(0)
    if program_id == target_program:
        if backward:
            row_start = program_id * rows_per_program
            offset = row_start * stride
        else:
            offset = program_id * stride
        tl.store(output, offset)


@triton.jit
def _boundary_head(
    output,
    stride,
    rows_per_program,
    target_program: tl.constexpr,
    backward: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    if program_id == target_program:
        if backward:
            row_start = program_id * rows_per_program
            offset = row_start * stride
        else:
            offset = program_id * stride
        tl.store(output, offset)


@triton.jit
def _common_base(output, stride, n_elements: tl.constexpr):
    program_id = tl.program_id(0)
    if program_id < n_elements:
        tl.store(output + program_id, program_id * stride)


@triton.jit
def _common_head(output, stride, n_elements: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    if program_id < n_elements:
        tl.store(output + program_id, program_id * stride)


def _cache_files(cache_root: Path) -> set[str]:
    if not cache_root.exists():
        return set()
    return {str(path.relative_to(cache_root)) for path in cache_root.rglob("*") if path.is_file()}


def _time_launch(function, output: torch.Tensor, stride: int, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function[(output.numel(),)](output, stride, output.numel())
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _paired_latencies(output: torch.Tensor, stride: int) -> dict[str, Any]:
    pairs: list[dict[str, float | str]] = []
    for pair in range(7):
        order = ("base", "head") if pair % 2 == 0 else ("head", "base")
        values: dict[str, float] = {}
        for variant in order:
            function = _common_base if variant == "base" else _common_head
            values[variant] = _time_launch(function, output, stride, repeats=200)
        pairs.append({"first": order[0], **values})
    base = [float(item["base"]) for item in pairs]
    head = [float(item["head"]) for item in pairs]
    return {
        "pairs": pairs,
        "base_median_us": statistics.median(base),
        "head_median_us": statistics.median(head),
        "head_over_base_median_ratio": statistics.median(head) / statistics.median(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rms-source", type=Path, required=True)
    parser.add_argument("--head-rms-source", type=Path, required=True)
    parser.add_argument("--base-rope-source", type=Path, required=True)
    parser.add_argument("--head-rope-source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    sources = {
        "base_rms": args.base_rms_source.read_text(encoding="utf-8"),
        "head_rms": args.head_rms_source.read_text(encoding="utf-8"),
        "base_rope": args.base_rope_source.read_text(encoding="utf-8"),
        "head_rope": args.head_rope_source.read_text(encoding="utf-8"),
    }
    source_contract = {
        "base_rms_cast_count": sources["base_rms"].count("tl.program_id(0).to(tl.int64)"),
        "head_rms_cast_count": sources["head_rms"].count("tl.program_id(0).to(tl.int64)"),
        "base_rope_cast_count": sources["base_rope"].count("tl.program_id(0).to(tl.int64)"),
        "head_rope_cast_count": sources["head_rope"].count("tl.program_id(0).to(tl.int64)"),
    }
    if source_contract != {
        "base_rms_cast_count": 0,
        "head_rms_cast_count": 2,
        "base_rope_cast_count": 0,
        "head_rope_cast_count": 1,
    }:
        raise ValueError(f"unexpected exact-source contract: {source_contract}")

    target_program = 174_763
    scenarios = [
        {
            "name": "rms-forward-and-rope-row-offset",
            "stride": 12_288,
            "rows_per_program": 1,
            "backward": False,
        },
        {
            "name": "rms-backward-row-block-offset",
            "stride": 768,
            "rows_per_program": 16,
            "backward": True,
        },
    ]
    boundary_results: list[dict[str, Any]] = []
    for scenario in scenarios:
        outputs: dict[str, int] = {}
        for variant, function in (("base", _boundary_base), ("head", _boundary_head)):
            output = torch.full((1,), -1, dtype=torch.int64, device="cuda")
            function[(target_program + 1,)](
                output,
                scenario["stride"],
                scenario["rows_per_program"],
                target_program=target_program,
                backward=scenario["backward"],
            )
            outputs[variant] = int(output.item())
        expected = target_program * int(scenario["rows_per_program"]) * int(scenario["stride"])
        boundary_results.append(
            {
                **scenario,
                "target_program": target_program,
                "expected": expected,
                "base": outputs["base"],
                "head": outputs["head"],
                "base_wraps": outputs["base"] != expected,
                "head_matches": outputs["head"] == expected,
            }
        )

    common_n = 65_536
    common_stride = 384
    common_base = torch.empty(common_n, dtype=torch.int64, device="cuda")
    common_head = torch.empty_like(common_base)
    _common_base[(common_n,)](common_base, common_stride, common_n)
    _common_head[(common_n,)](common_head, common_stride, common_n)
    torch.cuda.synchronize()
    cache_after_precompile = _cache_files(args.cache_root)
    common_equal = torch.equal(common_base, common_head)
    timing_output = torch.empty(common_n, dtype=torch.int64, device="cuda")
    latencies = _paired_latencies(timing_output, common_stride)
    torch.cuda.synchronize()
    cache_after_timing = _cache_files(args.cache_root)
    steady_new_files = sorted(cache_after_timing - cache_after_precompile)

    failure_codes: list[str] = []
    if not all(item["base_wraps"] and item["head_matches"] for item in boundary_results):
        failure_codes.append("LIGER_INT64_BOUNDARY_MECHANISM_FAILED")
    if not common_equal:
        failure_codes.append("LIGER_COMMON_RANGE_OUTPUT_CHANGED")
    if latencies["head_over_base_median_ratio"] > 1.03:
        failure_codes.append("LIGER_COMMON_RANGE_LATENCY_REGRESSION_GT_3PCT")
    if steady_new_files:
        failure_codes.append("LIGER_STEADY_STATE_COMPILATION_DETECTED")

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "liger-program-id-int64-h100-v1",
        "case_id": "liger-pr-804",
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "facts": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "source_contract": source_contract,
            "boundary_results": boundary_results,
            "common_range_outputs_equal": common_equal,
            "latency": latencies,
            "cache_files_after_precompile": len(cache_after_precompile),
            "steady_new_cache_files": steady_new_files,
            "steady_state_compile_seconds": 0.0 if not steady_new_files else None,
        },
        "source_identity": {name: _digest(source) for name, source in sources.items()},
        "environment": {
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "cuda_version": torch.version.cuda,
        },
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": _digest(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
