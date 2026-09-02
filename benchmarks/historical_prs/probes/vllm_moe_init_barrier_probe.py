#!/usr/bin/env python3
"""Precompiled base/head probe for the shared-count initialization race in vLLM #13140."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> run_variant(torch::Tensor topk_ids,
                                       int64_t num_experts,
                                       int64_t block_size,
                                       bool with_initialization_barrier);
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <bool WITH_INITIALIZATION_BARRIER>
__global__ void sgl_moe_align_block_size_kernel(
    int32_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];
  __shared__ int32_t local_offsets[256];

  const int warp_id = threadIdx.x / 32;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  if constexpr (WITH_INITIALIZATION_BARRIER) {
    __syncthreads();
  }

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      int expert_count = shared_counts[warp_idx][expert_offset];
      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
    local_offsets[threadIdx.x] = cumsum[threadIdx.x];
  }

  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

std::vector<torch::Tensor> run_variant(torch::Tensor topk_ids,
                                       int64_t num_experts,
                                       int64_t block_size,
                                       bool with_initialization_barrier) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
  TORCH_CHECK(num_experts > 0 && num_experts <= 256, "num_experts must be 1..256");
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  const int64_t max_padded = topk_ids.numel() + num_experts * (block_size - 1);
  auto sorted = torch::full({max_padded}, topk_ids.numel(), options);
  auto experts = torch::full({CEILDIV(max_padded, block_size)}, -1, options);
  auto total = torch::zeros({1}, options);
  auto cumsum = torch::empty({num_experts + 1}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (with_initialization_barrier) {
    sgl_moe_align_block_size_kernel<true><<<1, 1024, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), sorted.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(), total.data_ptr<int32_t>(), num_experts,
        block_size, topk_ids.numel(), cumsum.data_ptr<int32_t>());
  } else {
    sgl_moe_align_block_size_kernel<false><<<1, 1024, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), sorted.data_ptr<int32_t>(),
        experts.data_ptr<int32_t>(), total.data_ptr<int32_t>(), num_experts,
        block_size, topk_ids.numel(), cumsum.data_ptr<int32_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sorted, experts, total};
}
"""


def _expected(ids: list[int], num_experts: int, block_size: int) -> tuple[int, list[list[int]]]:
    positions = [[] for _ in range(num_experts)]
    for position, expert in enumerate(ids):
        positions[expert].append(position)
    total = sum(((len(items) + block_size - 1) // block_size) * block_size for items in positions)
    return total, positions


def _run_once(
    extension: Any,
    ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    with_barrier: bool,
) -> tuple[bool, str]:
    sorted_ids, expert_ids, total_tensor = extension.run_variant(
        ids, num_experts, block_size, with_barrier
    )
    torch.cuda.synchronize()
    source_ids = ids.cpu().tolist()
    expected_total, positions = _expected(source_ids, num_experts, block_size)
    total = int(total_tensor.item())
    if total != expected_total:
        return False, f"padded total {total} != {expected_total}"
    sorted_cpu = sorted_ids[:total].cpu().tolist()
    expert_cpu = expert_ids[: total // block_size].cpu().tolist()
    offset = 0
    for expert, expected_positions in enumerate(positions):
        padded = ((len(expected_positions) + block_size - 1) // block_size) * block_size
        segment = sorted_cpu[offset : offset + padded]
        actual = [position for position in segment if position != len(source_ids)]
        if Counter(actual) != Counter(expected_positions):
            return False, f"expert {expert} positions differ"
        labels = expert_cpu[offset // block_size : (offset + padded) // block_size]
        if labels != [expert] * (padded // block_size):
            return False, f"expert {expert} block labels differ"
        offset += padded
    return True, "matches CPU grouping, padding, and block labels"


def _timed_launches(extension: Any, ids: torch.Tensor, with_barrier: bool, count: int) -> float:
    for _ in range(10):
        extension.run_variant(ids, 256, 32, with_barrier)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(count):
        extension.run_variant(ids, 256, 32, with_barrier)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--timing-iterations", type=int, default=500)
    parser.add_argument("--variant", choices=("base", "head", "both"), default="both")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/workspace/.cache/infraswe/torch_extensions"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability(0)
    identity = json.dumps(
        {
            "cuda_source": CUDA_SOURCE,
            "cpp_source": CPP_SOURCE,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "capability": capability,
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(identity).hexdigest()
    module_name = f"infraswe_moe_barrier_{digest[:16]}"
    args.cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(args.cache_root)
    module_dir = args.cache_root / module_name
    cache_hit = any(module_dir.glob("*.so")) if module_dir.exists() else False

    precompile_started = time.perf_counter()
    extension = load_inline(
        name=module_name,
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=["run_variant"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )
    load_seconds = time.perf_counter() - precompile_started

    cases = {
        "one-last-expert": torch.tensor([255], dtype=torch.int32, device="cuda"),
        "one-mid-expert": torch.tensor([127], dtype=torch.int32, device="cuda"),
        "late-warps": torch.tensor(
            [255, 247, 239, 231, 223, 215, 207, 199], dtype=torch.int32, device="cuda"
        ),
        "random-small": torch.randint(0, 256, (257,), dtype=torch.int32, device="cuda"),
        "random-large": torch.randint(0, 256, (32768,), dtype=torch.int32, device="cuda"),
    }

    cold_started = time.perf_counter()
    cold_with_barrier = args.variant != "base"
    extension.run_variant(cases["one-last-expert"], 256, 32, cold_with_barrier)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - cold_started

    results: list[dict[str, Any]] = []
    variants = (
        (("base", False), ("head", True))
        if args.variant == "both"
        else ((args.variant, args.variant == "head"),)
    )
    for name, ids in cases.items():
        for variant, with_barrier in variants:
            failures: Counter[str] = Counter()
            passed = 0
            for _ in range(args.replays):
                ok, details = _run_once(extension, ids, 256, 32, with_barrier)
                if ok:
                    passed += 1
                else:
                    failures[details] += 1
            results.append(
                {
                    "case": name,
                    "variant": variant,
                    "replays": args.replays,
                    "passed": passed,
                    "failed": args.replays - passed,
                    "failure_examples": dict(failures.most_common(5)),
                }
            )

    timing: dict[str, Any] | None = None
    if not args.skip_timing:
        timing_ids = cases["random-large"]
        base_seconds = _timed_launches(extension, timing_ids, False, args.timing_iterations)
        head_seconds = _timed_launches(extension, timing_ids, True, args.timing_iterations)
        timing = {
            "iterations": args.timing_iterations,
            "base_seconds_per_call": base_seconds,
            "head_seconds_per_call": head_seconds,
            "head_over_base": head_seconds / base_seconds,
        }
    head_failures = sum(item["failed"] for item in results if item["variant"] == "head")
    base_failures = sum(item["failed"] for item in results if item["variant"] == "base")
    payload = {
        "schema_version": "0.5",
        "probe": "vllm-moe-initialization-barrier-v1",
        "case_id": "vllm-pr-13140",
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precompile": {
            "switch": "auto",
            "action": (
                "reuse-precompiled-artifact" if cache_hit else "precompile-before-timed-cases"
            ),
            "cache_key_sha256": "sha256:" + digest,
            "cache_hit": cache_hit,
            "module_name": module_name,
            "load_seconds": load_seconds,
            "precompile_seconds": 0.0 if cache_hit else load_seconds,
        },
        "cold_start_seconds": cold_seconds,
        "steady_state_compile_seconds": 0.0,
        "timing": timing,
        "results": results,
        "base_failures": base_failures,
        "head_failures": head_failures,
        "candidate_failure_codes": (
            ["CUDA_SHARED_COUNT_INITIALIZATION_RACE_REPRODUCED"]
            if base_failures and not head_failures
            else []
        ),
        "status": "pass" if head_failures == 0 else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if head_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
