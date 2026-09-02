#!/usr/bin/env python3
"""Precompiled stride-correctness and speed probe for vLLM PR 8200."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.cpp_extension import _get_build_directory, load_inline

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXTENSION_NAME = "infraswe_vllm_cache_layout_pr8200_sm80_v1"

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template <bool STRIDE_AWARE>
__global__ void reshape_cache_kernel(
    const at::Half* __restrict__ key, const at::Half* __restrict__ value,
    at::Half* __restrict__ key_cache, at::Half* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const int64_t key_stride,
    const int64_t value_stride, const int block_stride, const int page_stride,
    const int head_stride, const int num_heads, const int head_size,
    const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;
  for (int i = threadIdx.x; i < num_heads * head_size; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    int64_t target;
    if constexpr (STRIDE_AWARE) {
      target = block_idx * block_stride + block_offset * page_stride +
               head_idx * head_stride + head_offset;
    } else {
      target = block_idx * block_stride +
               block_offset * num_heads * head_size + head_idx * head_size +
               head_offset;
    }
    key_cache[target] = key[src_key_idx];
    value_cache[target] = value[src_value_idx];
  }
}

void run_variant(torch::Tensor key, torch::Tensor value,
                 torch::Tensor key_cache, torch::Tensor value_cache,
                 torch::Tensor slot_mapping, bool stride_aware) {
  TORCH_CHECK(key.scalar_type() == torch::kFloat16);
  TORCH_CHECK(value.scalar_type() == torch::kFloat16);
  TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16);
  TORCH_CHECK(value_cache.scalar_type() == torch::kFloat16);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));
  const int num_heads = key.size(1);
  const int head_size = key.size(2);
  const int block_size = key_cache.size(1);
  const int block_stride = key_cache.stride(0);
  const int page_stride = key_cache.stride(1);
  const int head_stride = key_cache.stride(2);
  const int threads = min(num_heads * head_size, 512);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (stride_aware) {
    reshape_cache_kernel<true><<<key.size(0), threads, 0, stream>>>(
        key.data_ptr<at::Half>(), value.data_ptr<at::Half>(),
        key_cache.data_ptr<at::Half>(), value_cache.data_ptr<at::Half>(),
        slot_mapping.data_ptr<int64_t>(), key.stride(0), value.stride(0),
        block_stride, page_stride, head_stride, num_heads, head_size,
        block_size);
  } else {
    reshape_cache_kernel<false><<<key.size(0), threads, 0, stream>>>(
        key.data_ptr<at::Half>(), value.data_ptr<at::Half>(),
        key_cache.data_ptr<at::Half>(), value_cache.data_ptr<at::Half>(),
        slot_mapping.data_ptr<int64_t>(), key.stride(0), value.stride(0),
        block_stride, page_stride, head_stride, num_heads, head_size,
        block_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run_variant", &run_variant);
}
"""


def _load_extension() -> tuple[object, dict[str, object]]:
    build_directory = Path(_get_build_directory(EXTENSION_NAME, verbose=False))
    cache_hit_before = any(build_directory.glob(f"{EXTENSION_NAME}*.so"))
    started = time.perf_counter()
    extension = load_inline(
        name=EXTENSION_NAME,
        cpp_sources="",
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    return extension, {
        "cache_hit_before": cache_hit_before,
        "seconds": time.perf_counter() - started,
        "build_directory": str(build_directory),
    }


def _cache(
    layout: str,
    *,
    blocks: int,
    block_size: int,
    heads: int,
    head_size: int,
) -> torch.Tensor:
    if layout == "NHD":
        return torch.full(
            (blocks, block_size, heads, head_size),
            -7,
            dtype=torch.float16,
            device="cuda",
        )
    physical = torch.full(
        (blocks, heads, block_size, head_size),
        -7,
        dtype=torch.float16,
        device="cuda",
    )
    return physical.permute(0, 2, 1, 3)


def _oracle(
    key: torch.Tensor,
    value: torch.Tensor,
    key_layout: str,
    value_layout: str,
    slots: torch.Tensor,
    *,
    blocks: int,
    block_size: int,
    heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_key = _cache(
        key_layout,
        blocks=blocks,
        block_size=block_size,
        heads=heads,
        head_size=head_size,
    )
    expected_value = _cache(
        value_layout,
        blocks=blocks,
        block_size=block_size,
        heads=heads,
        head_size=head_size,
    )
    for token, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        block = slot // block_size
        offset = slot % block_size
        expected_key[block, offset].copy_(key[token])
        expected_value[block, offset].copy_(value[token])
    return expected_key, expected_value


def _correctness_case(
    extension: object,
    *,
    variant: str,
    key_layout: str,
    value_layout: str,
) -> dict[str, object]:
    blocks, block_size, heads, head_size = 4, 8, 3, 10
    tokens = 6
    key = torch.arange(tokens * heads * head_size, dtype=torch.float16, device="cuda").reshape(
        tokens, heads, head_size
    )
    value = key + 1000
    slots = torch.tensor([0, 9, -1, 19, 26, 31], dtype=torch.int64, device="cuda")
    key_cache = _cache(
        key_layout,
        blocks=blocks,
        block_size=block_size,
        heads=heads,
        head_size=head_size,
    )
    value_cache = _cache(
        value_layout,
        blocks=blocks,
        block_size=block_size,
        heads=heads,
        head_size=head_size,
    )
    expected_key, expected_value = _oracle(
        key,
        value,
        key_layout,
        value_layout,
        slots,
        blocks=blocks,
        block_size=block_size,
        heads=heads,
        head_size=head_size,
    )
    extension.run_variant(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        variant == "head",
    )
    torch.cuda.synchronize()
    key_matches = torch.equal(key_cache, expected_key)
    value_matches = torch.equal(value_cache, expected_value)
    return {
        "variant": variant,
        "key_layout": key_layout,
        "value_layout": value_layout,
        "key_strides": list(key_cache.stride()),
        "value_strides": list(value_cache.stride()),
        "key_matches_oracle": key_matches,
        "value_matches_oracle": value_matches,
        "matches_oracle": key_matches and value_matches,
        "padded_slot_preserved": slots.tolist().count(-1) == 1,
    }


def _timed(
    extension: object,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    stride_aware: bool,
    iterations: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        extension.run_variant(key, value, key_cache, value_cache, slots, stride_aware)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def _performance(extension: object, iterations: int) -> dict[str, object]:
    blocks, block_size, heads, head_size = 128, 16, 32, 128
    tokens = 1024
    key = torch.randn((tokens, heads, head_size), dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    slots = torch.arange(tokens, dtype=torch.int64, device="cuda").remainder(blocks * block_size)
    results: dict[str, object] = {}
    for layout in ("NHD", "HND"):
        key_cache = _cache(
            layout,
            blocks=blocks,
            block_size=block_size,
            heads=heads,
            head_size=head_size,
        )
        value_cache = _cache(
            layout,
            blocks=blocks,
            block_size=block_size,
            heads=heads,
            head_size=head_size,
        )
        variants = (False, True) if layout == "NHD" else (True,)
        for stride_aware in variants:
            name = "head" if stride_aware else "base"
            for _ in range(5):
                extension.run_variant(key, value, key_cache, value_cache, slots, stride_aware)
            torch.cuda.synchronize()
            samples = _timed(
                extension,
                key,
                value,
                key_cache,
                value_cache,
                slots,
                stride_aware=stride_aware,
                iterations=iterations,
            )
            results[f"{layout.lower()}_{name}"] = {
                "median_us": statistics.median(samples),
                "p95_us": sorted(samples)[int(0.95 * (len(samples) - 1))],
            }
    base = results["nhd_base"]["median_us"]
    head = results["nhd_head"]["median_us"]
    return {
        "iterations": iterations,
        "tokens": tokens,
        "heads": heads,
        "head_size": head_size,
        "timings": results,
        "nhd_head_over_base": head / base,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precompile", choices=("auto", "off"), default="auto")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    started = time.perf_counter()
    precompile: dict[str, object] | None = None
    cold_load: dict[str, object] | None = None
    if args.precompile == "auto":
        extension, precompile = _load_extension()
    else:
        extension, cold_load = _load_extension()

    cases = [
        _correctness_case(
            extension,
            variant=variant,
            key_layout=key_layout,
            value_layout=value_layout,
        )
        for variant, key_layout, value_layout in (
            ("base", "NHD", "NHD"),
            ("head", "NHD", "NHD"),
            ("base", "HND", "HND"),
            ("head", "HND", "HND"),
            ("head", "HND", "NHD"),
            ("head", "NHD", "HND"),
        )
    ]
    performance = _performance(extension, args.iterations)
    head_source = (args.head_root / "csrc/cache_kernels.cu").read_text(encoding="utf-8")
    independent_stride_guard = (
        "key_cache.stride(1) == value_cache.stride(1)" in head_source
        and "key_cache.stride(2) == value_cache.stride(2)" in head_source
    )
    head_failures = [
        item for item in cases if item["variant"] == "head" and not item["matches_oracle"]
    ]
    failure_codes: list[str] = []
    if head_failures:
        failure_codes.append("KEY_VALUE_CACHE_STRIDES_NOT_INDEPENDENT_OR_VALIDATED")

    material = {
        "schema_version": "0.5",
        "probe": "vllm-cache-layout-pr8200-v1",
        "case_id": "vllm-pr-8200",
        "status": "fail" if failure_codes else "pass",
        "failure_codes": failure_codes,
        "phases": {
            "precompile": precompile,
            "cold_start": {"extension_load": cold_load},
            "steady_state": {
                "compile_seconds": 0.0,
                "timed_iterations_per_case": args.iterations,
            },
        },
        "correctness": cases,
        "performance": performance,
        "static_contracts": {
            "independent_key_value_stride_guard_present": independent_stride_guard,
        },
        "source_identity": {
            "head_cache_kernels_sha256": canonical_sha256(head_source),
            "embedded_probe_cuda_sha256": canonical_sha256(CUDA_SOURCE),
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
