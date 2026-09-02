#!/usr/bin/env python3
"""H100 probe for Megatron PR 5608 runtime batch-size specialization."""

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
def _get_base(
    input_tensor,
    output_tensor,
    pos_on_device,
    input_batch_size: tl.constexpr,
    output_batch_size: tl.constexpr,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(pos_on_device)
    copy_size = input_batch_size - pos
    if pid < copy_size and pid < output_batch_size:
        input_idx = pos + pid
        if input_idx < input_batch_size:
            offsets = tl.arange(0, block_size)
            mask = offsets < row_size
            value = tl.load(
                input_tensor + input_idx * row_size + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(output_tensor + pid * row_size + offsets, value, mask=mask)


@triton.jit
def _get_head(
    input_tensor,
    output_tensor,
    pos_on_device,
    input_batch_size,
    output_batch_size,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(pos_on_device)
    copy_size = input_batch_size - pos
    if pid < copy_size and pid < output_batch_size:
        input_idx = pos + pid
        if input_idx < input_batch_size:
            offsets = tl.arange(0, block_size)
            mask = offsets < row_size
            value = tl.load(
                input_tensor + input_idx * row_size + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(output_tensor + pid * row_size + offsets, value, mask=mask)


@triton.jit
def _merge_base(
    tensor_a,
    tensor_b,
    output_tensor,
    pos_on_device,
    tensor_b_batch_size: tl.constexpr,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
    output_batch_size: tl.constexpr,
    is_inplace: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(pos_on_device)
    if pid < pos:
        if not is_inplace:
            offsets = tl.arange(0, block_size)
            mask = offsets < row_size
            value = tl.load(tensor_a + pid * row_size + offsets, mask=mask, other=0.0)
            tl.store(output_tensor + pid * row_size + offsets, value, mask=mask)
    elif pid < pos + tensor_b_batch_size and pid < output_batch_size:
        tensor_b_idx = pid - pos
        if tensor_b_idx < tensor_b_batch_size:
            offsets = tl.arange(0, block_size)
            mask = offsets < row_size
            value = tl.load(
                tensor_b + tensor_b_idx * row_size + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(output_tensor + pid * row_size + offsets, value, mask=mask)


@triton.jit
def _merge_head(
    tensor_a,
    tensor_b,
    output_tensor,
    pos_on_device,
    tensor_b_batch_size,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
    output_batch_size,
    is_inplace: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(pos_on_device)
    if pid < pos:
        if not is_inplace:
            offsets = tl.arange(0, block_size)
            mask = offsets < row_size
            value = tl.load(tensor_a + pid * row_size + offsets, mask=mask, other=0.0)
            tl.store(output_tensor + pid * row_size + offsets, value, mask=mask)
    elif pid < pos + tensor_b_batch_size and pid < output_batch_size:
        tensor_b_idx = pid - pos
        if tensor_b_idx < tensor_b_batch_size:
            offsets = tl.arange(0, block_size)
            mask = offsets < row_size
            value = tl.load(
                tensor_b + tensor_b_idx * row_size + offsets,
                mask=mask,
                other=0.0,
            )
            tl.store(output_tensor + pid * row_size + offsets, value, mask=mask)


def _cache_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _launch_get(variant: str, input_batch: int, output_batch: int, row_size: int) -> bool:
    function = _get_base if variant == "base" else _get_head
    source = torch.arange(input_batch * row_size, device="cuda", dtype=torch.float32).view(
        input_batch, row_size
    )
    output = torch.full((output_batch, row_size), -1.0, device="cuda")
    pos = torch.tensor([input_batch // 3], device="cuda", dtype=torch.int64)
    function[(input_batch,)](
        source,
        output,
        pos,
        input_batch,
        output_batch,
        row_size,
        triton.next_power_of_2(row_size),
    )
    torch.cuda.synchronize()
    expected = torch.full_like(output, -1.0)
    copy = min(input_batch - int(pos.item()), output_batch)
    expected[:copy] = source[int(pos.item()) : int(pos.item()) + copy]
    return torch.equal(output, expected)


def _launch_merge(variant: str, output_batch: int, tensor_b_batch: int, row_size: int) -> bool:
    function = _merge_base if variant == "base" else _merge_head
    tensor_a = torch.arange(output_batch * row_size, device="cuda", dtype=torch.float32).view(
        output_batch, row_size
    )
    tensor_b = -torch.arange(
        1, tensor_b_batch * row_size + 1, device="cuda", dtype=torch.float32
    ).view(tensor_b_batch, row_size)
    output = torch.full_like(tensor_a, 777.0)
    pos_value = max(1, output_batch // 4)
    pos = torch.tensor([pos_value], device="cuda", dtype=torch.int64)
    function[(output_batch,)](
        tensor_a,
        tensor_b,
        output,
        pos,
        tensor_b_batch,
        row_size,
        triton.next_power_of_2(row_size),
        output_batch,
        False,
    )
    torch.cuda.synchronize()
    expected = torch.full_like(output, 777.0)
    expected[:pos_value] = tensor_a[:pos_value]
    copy = min(tensor_b_batch, output_batch - pos_value)
    expected[pos_value : pos_value + copy] = tensor_b[:copy]
    return torch.equal(output, expected)


def _compile_matrix(variant: str, cache_root: Path) -> dict[str, Any]:
    shapes = [(8, 6), (16, 9), (32, 13), (64, 17)]
    correctness: list[dict[str, Any]] = []
    first_input, first_output = shapes[0]
    first_ok = _launch_get(variant, first_input, first_output, 96)
    first_merge_ok = _launch_merge(variant, first_output, first_input // 2, 96)
    after_first = _cache_files(cache_root)
    correctness.append(
        {
            "input_batch": first_input,
            "output_batch": first_output,
            "get_ok": first_ok,
            "merge_ok": first_merge_ok,
        }
    )
    for input_batch, output_batch in shapes[1:]:
        correctness.append(
            {
                "input_batch": input_batch,
                "output_batch": output_batch,
                "get_ok": _launch_get(variant, input_batch, output_batch, 96),
                "merge_ok": _launch_merge(variant, output_batch, max(1, input_batch // 2), 96),
            }
        )
    after_variation = _cache_files(cache_root)
    row_boundary_ok = _launch_get(variant, 16, 9, 129) and _launch_merge(variant, 9, 8, 129)
    return {
        "correctness": correctness,
        "all_runtime_shape_outputs_match": all(
            item["get_ok"] and item["merge_ok"] for item in correctness
        ),
        "row_size_specialization_output_matches": row_boundary_ok,
        "cache_files_after_first_shape": len(after_first),
        "cache_files_after_runtime_shape_variation": len(after_variation),
        "new_cache_files_for_runtime_shape_variation": sorted(after_variation - after_first),
        "new_compiled_kernels_for_runtime_shape_variation": sorted(
            path for path in after_variation - after_first if path.endswith(".cubin")
        ),
    }


def _timed_get(function, repeats: int = 2_000) -> float:
    input_batch, output_batch, row_size = 64, 17, 96
    source = torch.randn(input_batch, row_size, device="cuda")
    output = torch.empty(output_batch, row_size, device="cuda")
    pos = torch.tensor([7], device="cuda", dtype=torch.int64)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function[(input_batch,)](
            source,
            output,
            pos,
            input_batch,
            output_batch,
            row_size,
            128,
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _paired_latency() -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for index in range(7):
        order = ("base", "head") if index % 2 == 0 else ("head", "base")
        values: dict[str, float] = {}
        for variant in order:
            values[variant] = _timed_get(_get_base if variant == "base" else _get_head)
        pairs.append({"first": order[0], **values})
    base = [item["base"] for item in pairs]
    head = [item["head"] for item in pairs]
    return {
        "pairs": pairs,
        "base_median_us": statistics.median(base),
        "head_median_us": statistics.median(head),
        "head_over_base_median_ratio": statistics.median(head) / statistics.median(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--head-source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    base_source = args.base_source.read_text(encoding="utf-8")
    head_source = args.head_source.read_text(encoding="utf-8")
    names = (
        "INPUT_BATCH_SIZE: tl.constexpr",
        "OUTPUT_BATCH_SIZE: tl.constexpr",
        "TENSOR_B_BATCH_SIZE: tl.constexpr",
    )
    source_contract = {
        "base_runtime_batch_constexpr_count": sum(base_source.count(name) for name in names),
        "head_runtime_batch_constexpr_count": sum(head_source.count(name) for name in names),
    }
    if source_contract != {
        "base_runtime_batch_constexpr_count": 4,
        "head_runtime_batch_constexpr_count": 0,
    }:
        raise ValueError(f"unexpected exact-source contract: {source_contract}")

    base_matrix = _compile_matrix("base", args.cache_root)
    head_matrix = _compile_matrix("head", args.cache_root)
    cache_after_precompile = _cache_files(args.cache_root)
    latency = _paired_latency()
    cache_after_timing = _cache_files(args.cache_root)
    steady_new_files = sorted(cache_after_timing - cache_after_precompile)

    failure_codes: list[str] = []
    if (
        not base_matrix["all_runtime_shape_outputs_match"]
        or not head_matrix["all_runtime_shape_outputs_match"]
    ):
        failure_codes.append("MEGATRON_RUNTIME_BATCH_OUTPUT_MISMATCH")
    if (
        not base_matrix["row_size_specialization_output_matches"]
        or not head_matrix["row_size_specialization_output_matches"]
    ):
        failure_codes.append("MEGATRON_ROW_SIZE_SPECIALIZATION_OUTPUT_MISMATCH")
    base_new_kernels = len(base_matrix["new_compiled_kernels_for_runtime_shape_variation"])
    head_new_kernels = len(head_matrix["new_compiled_kernels_for_runtime_shape_variation"])
    if head_new_kernels >= base_new_kernels:
        failure_codes.append("MEGATRON_RUNTIME_BATCH_RECOMPILE_NOT_REDUCED")
    if base_new_kernels == 0:
        failure_codes.append("MEGATRON_BASE_RECOMPILE_CONTROL_NOT_OBSERVED")
    if latency["head_over_base_median_ratio"] > 1.03:
        failure_codes.append("MEGATRON_STEADY_LATENCY_REGRESSION_GT_3PCT")
    if steady_new_files:
        failure_codes.append("MEGATRON_STEADY_STATE_COMPILATION_DETECTED")

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "megatron-runtime-batch-size-h100-v1",
        "case_id": "megatron-pr-5608",
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "facts": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "source_contract": source_contract,
            "base": base_matrix,
            "head": head_matrix,
            "runtime_shape_compile_reduction": {
                "base_new_compiled_kernels": base_new_kernels,
                "head_new_compiled_kernels": head_new_kernels,
                "head_over_base_ratio": (
                    head_new_kernels / base_new_kernels if base_new_kernels else None
                ),
            },
            "latency": latency,
            "cache_files_after_precompile": len(cache_after_precompile),
            "steady_new_cache_files": steady_new_files,
            "steady_state_compile_seconds": 0.0 if not steady_new_files else None,
        },
        "source_identity": {
            "base_source_sha256": _digest(base_source),
            "head_source_sha256": _digest(head_source),
        },
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
