#!/usr/bin/env python3
"""Precompiled correctness, lifetime, and speed probe for vLLM PR 11531."""

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

EXTENSION_NAME = "infraswe_vllm_swap_pr11531_sm80_v1"

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void paged_copy(T* __restrict__ dst, const T* __restrict__ src,
                           const int64_t* src_to_dst, const int num_pages,
                           const int num_elements_per_page) {
  const int64_t src_page_idx = src_to_dst[blockIdx.x << 1];
  const int64_t dst_page_idx = src_to_dst[(blockIdx.x << 1) | 1];
  const int64_t src_page_offset = src_page_idx * num_elements_per_page;
  const int64_t dst_page_offset = dst_page_idx * num_elements_per_page;
  for (int i = threadIdx.x; i < num_elements_per_page; i += blockDim.x) {
    dst[dst_page_offset + i] = src[src_page_offset + i];
  }
}

__global__ void delay_kernel(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

void* kernel_pointer(torch::Tensor tensor) {
  if (tensor.is_cuda()) {
    return tensor.data_ptr();
  }
  TORCH_CHECK(tensor.device().is_cpu() && tensor.is_pinned(),
              "CPU tensors must be pinned");
  void* pointer = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&pointer, tensor.data_ptr(), 0));
  return pointer;
}

void swap_fast(torch::Tensor src, torch::Tensor dst, torch::Tensor mapping) {
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous() && mapping.is_contiguous());
  TORCH_CHECK(src.scalar_type() == dst.scalar_type());
  TORCH_CHECK(src.sizes() == dst.sizes());
  TORCH_CHECK(mapping.scalar_type() == torch::kInt64 && mapping.size(1) == 2);
  const auto device = src.is_cuda() ? src.device() : dst.device();
  c10::cuda::CUDAGuard device_guard(device);
  const int64_t num_blocks = mapping.size(0);
  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  paged_copy<int64_t><<<num_blocks, 1024, 0, stream>>>(
      static_cast<int64_t*>(kernel_pointer(dst)),
      static_cast<const int64_t*>(kernel_pointer(src)),
      static_cast<const int64_t*>(kernel_pointer(mapping)), num_blocks,
      block_size_in_bytes / 8);
}

void swap_slow(torch::Tensor src, torch::Tensor dst, torch::Tensor mapping) {
  TORCH_CHECK(mapping.device().is_cpu());
  TORCH_CHECK(mapping.scalar_type() == torch::kInt64 && mapping.size(1) == 2);
  const auto device = src.is_cuda() ? src.device() : dst.device();
  c10::cuda::CUDAGuard device_guard(device);
  cudaMemcpyKind kind;
  if (src.is_cuda() && dst.is_cuda()) {
    kind = cudaMemcpyDeviceToDevice;
  } else if (src.is_cuda()) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyHostToDevice;
  }
  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const auto* pairs = mapping.data_ptr<int64_t>();
  auto* src_bytes = static_cast<char*>(src.data_ptr());
  auto* dst_bytes = static_cast<char*>(dst.data_ptr());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (int64_t i = 0; i < mapping.size(0); ++i) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst_bytes + pairs[2 * i + 1] * block_size_in_bytes,
        src_bytes + pairs[2 * i] * block_size_in_bytes,
        block_size_in_bytes, kind, stream));
  }
}

void enqueue_delay(unsigned long long cycles) {
  delay_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(cycles);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("swap_fast", &swap_fast);
  module.def("swap_slow", &swap_slow);
  module.def("enqueue_delay", &enqueue_delay);
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
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    return extension, {
        "cache_hit_before": cache_hit_before,
        "seconds": elapsed,
        "build_directory": str(build_directory),
    }


def _mapping(pairs: list[tuple[int, int]], *, pinned: bool = True) -> torch.Tensor:
    return (
        torch.tensor(pairs, dtype=torch.int64).pin_memory()
        if pinned
        else torch.tensor(pairs, dtype=torch.int64, device="cuda")
    )


def _expected(src: torch.Tensor, dst: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    result = dst.clone()
    for source, target in pairs:
        result[target].copy_(src[source])
    return result


def _correctness_case(
    extension: object,
    *,
    dtype: torch.dtype,
    elements: int,
    source_device: str,
    target_device: str,
) -> dict[str, object]:
    pages = 6
    pairs = [(0, 4), (2, 1), (5, 3)]
    host_source = torch.arange(pages * elements, dtype=torch.float32).reshape(pages, elements)
    if dtype == torch.uint8:
        host_source = host_source.remainder(251)
    host_source = host_source.to(dtype=dtype)
    src = host_source.to("cuda") if source_device == "cuda" else host_source.pin_memory()
    host_target = torch.full((pages, elements), 17, dtype=dtype)
    dst = host_target.to("cuda") if target_device == "cuda" else host_target.pin_memory()
    expected = _expected(host_source, host_target, pairs)
    mapping = _mapping(pairs)
    extension.swap_fast(src, dst, mapping)
    torch.cuda.synchronize()
    actual = dst.cpu()
    matches = torch.equal(actual, expected)
    differing_elements = int(torch.count_nonzero(actual != expected).item())
    return {
        "dtype": str(dtype),
        "elements_per_page": elements,
        "bytes_per_page": elements * torch.empty((), dtype=dtype).element_size(),
        "source_device": source_device,
        "target_device": target_device,
        "matches_oracle": matches,
        "differing_elements": differing_elements,
    }


def _mapping_lifetime_case(extension: object, delay_cycles: int) -> dict[str, object]:
    pages = 4
    elements = 128
    src = torch.stack(
        [torch.full((elements,), page + 1, dtype=torch.int64) for page in range(pages)]
    ).cuda()
    intended = [(0, 0)]
    overwritten = [(2, 3)]

    fast_dst = torch.zeros_like(src)
    fast_mapping = _mapping(intended)
    fast_stream = torch.cuda.Stream()
    with torch.cuda.stream(fast_stream):
        extension.enqueue_delay(delay_cycles)
        extension.swap_fast(src, fast_dst, fast_mapping)
    fast_mapping.copy_(torch.tensor(overwritten, dtype=torch.int64))
    fast_stream.synchronize()

    slow_dst = torch.zeros_like(src)
    slow_mapping = _mapping(intended)
    slow_stream = torch.cuda.Stream()
    with torch.cuda.stream(slow_stream):
        extension.enqueue_delay(delay_cycles)
        extension.swap_slow(src, slow_dst, slow_mapping)
    slow_mapping.copy_(torch.tensor(overwritten, dtype=torch.int64))
    slow_stream.synchronize()

    intended_expected = _expected(src, torch.zeros_like(src), intended)
    overwritten_expected = _expected(src, torch.zeros_like(src), overwritten)
    return {
        "delay_cycles": delay_cycles,
        "fast_matches_intended_mapping": torch.equal(fast_dst, intended_expected),
        "fast_matches_overwritten_mapping": torch.equal(fast_dst, overwritten_expected),
        "slow_matches_intended_mapping": torch.equal(slow_dst, intended_expected),
        "reused_mapping_race_reproduced": (
            torch.equal(fast_dst, overwritten_expected)
            and not torch.equal(fast_dst, intended_expected)
            and torch.equal(slow_dst, intended_expected)
        ),
    }


def _timed(
    function: object,
    src: torch.Tensor,
    dst: torch.Tensor,
    mapping: torch.Tensor,
    *,
    iterations: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function(src, dst, mapping)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def _performance(extension: object, iterations: int) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    pages = 512
    elements = 8192
    src = torch.randn((pages, elements), dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    for mapping_count in (1, 8, 64, 256):
        pairs = [(index, pages - 1 - index) for index in range(mapping_count)]
        mapping = _mapping(pairs)
        for _ in range(5):
            extension.swap_fast(src, dst, mapping)
            extension.swap_slow(src, dst, mapping)
        torch.cuda.synchronize()
        fast = _timed(extension.swap_fast, src, dst, mapping, iterations=iterations)
        slow = _timed(extension.swap_slow, src, dst, mapping, iterations=iterations)
        fast_median = statistics.median(fast)
        slow_median = statistics.median(slow)
        results.append(
            {
                "mapping_count": mapping_count,
                "bytes_per_page": elements * src.element_size(),
                "iterations": iterations,
                "fast_median_us": fast_median,
                "slow_median_us": slow_median,
                "fast_p95_us": sorted(fast)[int(0.95 * (len(fast) - 1))],
                "slow_p95_us": sorted(slow)[int(0.95 * (len(slow) - 1))],
                "speedup": slow_median / fast_median,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precompile", choices=("auto", "off"), default="auto")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--delay-cycles", type=int, default=200_000_000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    started = time.perf_counter()
    precompile: dict[str, object] | None = None
    cold_start: dict[str, object] | None = None
    if args.precompile == "auto":
        extension, precompile = _load_extension()
    else:
        extension, cold_start = _load_extension()

    cold_started = time.perf_counter()
    cold_src = torch.zeros((2, 8), dtype=torch.int64, device="cuda")
    cold_dst = torch.zeros_like(cold_src)
    cold_mapping = _mapping([(0, 1)])
    extension.swap_fast(
        cold_src,
        cold_dst,
        cold_mapping,
    )
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - cold_started

    correctness = [
        _correctness_case(
            extension,
            dtype=dtype,
            elements=elements,
            source_device=source_device,
            target_device=target_device,
        )
        for dtype, elements, source_device, target_device in (
            (torch.float16, 128, "cuda", "cuda"),
            (torch.float16, 3, "cuda", "cuda"),
            (torch.uint8, 7, "cuda", "cuda"),
            (torch.float32, 128, "cpu", "cuda"),
            (torch.float32, 128, "cuda", "cpu"),
        )
    ]
    lifetime = _mapping_lifetime_case(extension, args.delay_cycles)
    performance = _performance(extension, args.iterations)

    head_source_path = args.head_root / "csrc/cache_kernels.cu"
    head_source = head_source_path.read_text(encoding="utf-8")
    empty_mapping_guard_present = "if (num_blocks == 0)" in head_source
    failed_correctness = [item for item in correctness if not item["matches_oracle"]]
    failure_codes: list[str] = []
    if failed_correctness:
        failure_codes.append("PAGE_COPY_DROPS_NON_INT64_TAIL_BYTES")
    if lifetime["reused_mapping_race_reproduced"]:
        failure_codes.append("ASYNC_PINNED_MAPPING_BUFFER_LIFETIME_RACE")
    if not empty_mapping_guard_present:
        failure_codes.append("ZERO_BLOCK_KERNEL_LAUNCH_UNGUARDED")

    material = {
        "schema_version": "0.5",
        "probe": "vllm-swap-blocks-pr11531-v1",
        "case_id": "vllm-pr-11531",
        "status": "fail" if failure_codes else "pass",
        "failure_codes": failure_codes,
        "phases": {
            "precompile": precompile,
            "cold_start": {
                "extension_load": cold_start,
                "first_launch_seconds": cold_seconds,
            },
            "steady_state": {
                "compile_seconds": 0.0,
                "timed_iterations_per_case": args.iterations,
            },
        },
        "correctness": correctness,
        "mapping_lifetime": lifetime,
        "performance": performance,
        "static_contracts": {
            "empty_mapping_guard_present": empty_mapping_guard_present,
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
