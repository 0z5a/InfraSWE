#!/usr/bin/env python3
"""Precompiled A/B probe for the pointer-width change in vLLM PR #53038."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def offset_i32_kernel(rows, output, stride, n_elements: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    row = tl.load(rows + offsets, mask=mask)
    value = row * stride
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def offset_i64_kernel(rows, output, stride, n_elements: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    row = tl.load(rows + offsets, mask=mask).to(tl.int64)
    value = row * stride
    tl.store(output + offsets, value, mask=mask)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_files() -> set[str]:
    root = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _launch(kernel: Any, rows: torch.Tensor, output: torch.Tensor) -> None:
    block = 256
    kernel[(triton.cdiv(rows.numel(), block),)](
        rows, output, 12_288, n_elements=rows.numel(), block=block
    )


def _elapsed_ms(kernel: Any, rows: torch.Tensor, output: torch.Tensor, launches: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        _launch(kernel, rows, output)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / launches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--launches", type=int, default=300)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    boundary_row = 174_763
    stride = 12_288
    expected = boundary_row * stride
    boundary = torch.tensor([boundary_row], dtype=torch.int32, device="cuda")
    base_boundary = torch.empty(1, dtype=torch.int64, device="cuda")
    head_boundary = torch.empty_like(base_boundary)
    rows = (torch.arange(args.elements, dtype=torch.int32, device="cuda") % 8192).contiguous()
    output = torch.empty(args.elements, dtype=torch.int64, device="cuda")

    cache_before = _cache_files()
    compile_started = time.perf_counter()
    _launch(offset_i32_kernel, boundary, base_boundary)
    _launch(offset_i64_kernel, boundary, head_boundary)
    _launch(offset_i32_kernel, rows, output)
    _launch(offset_i64_kernel, rows, output)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    cache_after_precompile = _cache_files()

    base_ms: list[float] = []
    head_ms: list[float] = []
    for replay in range(7):
        order = (
            ((offset_i32_kernel, base_ms), (offset_i64_kernel, head_ms))
            if replay % 2 == 0
            else ((offset_i64_kernel, head_ms), (offset_i32_kernel, base_ms))
        )
        for kernel, samples in order:
            samples.append(_elapsed_ms(kernel, rows, output, args.launches))
    torch.cuda.synchronize()
    cache_after_timing = _cache_files()

    changed_kernel_paths = [
        "vllm/lora/ops/triton_ops/lora_expand_op.py",
        "vllm/lora/ops/triton_ops/lora_expand_fp8_op.py",
        "vllm/lora/ops/triton_ops/lora_shrink_op.py",
        "vllm/lora/ops/triton_ops/lora_shrink_fp8_op.py",
    ]
    test_text = (args.head_root / "tests/lora/test_punica_ops.py").read_text(encoding="utf-8")
    test_tail = test_text[test_text.index("def test_kernels_large_token_count") :]
    base_value = int(base_boundary.item())
    head_value = int(head_boundary.item())
    base_median = statistics.median(base_ms)
    head_median = statistics.median(head_ms)
    payload = {
        "schema_version": "0.5",
        "probe": "vllm-lora-int64-pointer-width-r1",
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "triton": triton.__version__,
        },
        "boundary_arithmetic": {
            "row": boundary_row,
            "stride": stride,
            "expected_int64": expected,
            "base_int32_result": base_value,
            "head_int64_result": head_value,
            "base_matches": base_value == expected,
            "head_matches": head_value == expected,
        },
        "precompile": {
            "compile_seconds": compile_seconds,
            "new_cache_files": len(cache_after_precompile - cache_before),
            "steady_state_new_cache_files": len(cache_after_timing - cache_after_precompile),
        },
        "steady_latency_ms": {
            "elements": args.elements,
            "launches_per_replay": args.launches,
            "replays": 7,
            "base_samples": base_ms,
            "head_samples": head_ms,
            "base_median": base_median,
            "head_median": head_median,
            "head_to_base_ratio": head_median / base_median,
        },
        "coverage": {
            "changed_kernel_paths": changed_kernel_paths,
            "large_test_dtype": "bfloat16" if "torch.bfloat16" in test_tail else "unknown",
            "fp8_path_directly_exercised": any(
                token in test_tail for token in ("float8", "fp8", "_fp8")
            ),
        },
        "source_sha256": {
            "base_expand": _sha256(args.base_root / changed_kernel_paths[0]),
            "head_expand": _sha256(args.head_root / changed_kernel_paths[0]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
