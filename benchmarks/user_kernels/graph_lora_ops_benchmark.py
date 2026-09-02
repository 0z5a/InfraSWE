#!/usr/bin/env python3
"""Fresh-process CUDA Graph benchmark for SGLang graph LoRA B expansion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


@dataclass(frozen=True)
class Case:
    case_id: str
    tokens: int
    slots: int
    rank: int
    slice_dims: tuple[int, ...]
    dtype: torch.dtype
    weight: float


CASES = (
    Case("control-slots1-fp16", 4, 1, 16, (4096,), torch.float16, 0.10),
    Case("control-slots3-bf16", 16, 3, 32, (4096,), torch.bfloat16, 0.15),
    Case("decode-slots4-r16-fp16", 4, 4, 16, (4096,), torch.float16, 0.20),
    Case("batch-slots4-r32-bf16", 32, 4, 32, (4096,), torch.bfloat16, 0.20),
    Case("qkv-slots4-r32-fp16", 16, 4, 32, (4096, 1024, 1024), torch.float16, 0.15),
    Case("wide-slots8-r64-bf16", 64, 8, 64, (8192,), torch.bfloat16, 0.20),
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("graph_lora_ops_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offsets(slice_dims: tuple[int, ...]) -> torch.Tensor:
    values = [0]
    for size in slice_dims:
        values.append(values[-1] + size)
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _make_inputs(case: Case, seed: int, *, output_dtype: torch.dtype | None = None):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    input_dim = case.rank * len(case.slice_dims)
    total_output_dim = sum(case.slice_dims)
    inputs = 0.05 * torch.randn(
        case.tokens,
        input_dim,
        dtype=case.dtype,
        device="cuda",
        generator=generator,
    )
    weights = 0.05 * torch.randn(
        case.slots,
        total_output_dim,
        case.rank,
        dtype=case.dtype,
        device="cuda",
        generator=generator,
    )
    indices = torch.arange(case.tokens, device="cuda", dtype=torch.int32) % case.slots
    permutation = torch.randperm(case.tokens, device="cuda", generator=generator)
    indices = indices[permutation].contiguous()
    seg_lens = torch.ones(case.tokens, dtype=torch.int32, device="cuda")
    base_output = 0.02 * torch.randn(
        case.tokens,
        total_output_dim,
        dtype=output_dtype or case.dtype,
        device="cuda",
        generator=generator,
    )
    return {
        "inputs": inputs,
        "weights": weights,
        "weight_indices": indices,
        "seg_lens": seg_lens,
        "slice_offsets": _offsets(case.slice_dims),
        "base_output": base_output,
    }


def _call(module: ModuleType, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return module.sgemm_lora_b_graph_fwd(
        tensors["inputs"],
        tensors["weights"],
        tensors["weight_indices"],
        tensors["seg_lens"],
        tensors["slice_offsets"],
        tensors["base_output"],
    )


def _oracle(case: Case, tensors: dict[str, torch.Tensor], base: torch.Tensor) -> torch.Tensor:
    output = base.float().clone()
    inputs = tensors["inputs"].float()
    weights = tensors["weights"].float()
    offsets = tensors["slice_offsets"].tolist()
    for lora_idx in range(case.slots):
        mask = (tensors["weight_indices"] == lora_idx).unsqueeze(1)
        masked = torch.where(mask, inputs, 0)
        for slice_idx, (start, end) in enumerate(pairwise(offsets)):
            input_start = slice_idx * case.rank
            input_end = input_start + case.rank
            output[:, start:end].add_(
                torch.mm(masked[:, input_start:input_end], weights[lora_idx, start:end].t())
            )
    return output


def _compare(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> dict[str, Any]:
    actual_f32 = actual.float()
    difference = (actual_f32 - expected).abs()
    atol = 0.025 if dtype == torch.float16 else 0.08
    rtol = 0.015 if dtype == torch.float16 else 0.04
    return {
        "passed": bool(torch.allclose(actual_f32, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "atol": atol,
        "rtol": rtol,
        "all_finite": bool(torch.isfinite(actual_f32).all().item()),
    }


def _capture(module: ModuleType, tensors: dict[str, torch.Tensor]):
    base_seed = tensors["base_output"].clone()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(3):
            tensors["base_output"].copy_(base_seed)
            _call(module, tensors)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    tensors["base_output"].copy_(base_seed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        output = _call(module, tensors)
    torch.cuda.synchronize()
    return graph, output, base_seed


def _correctness_case(module: ModuleType, case: Case, seed: int) -> dict[str, Any]:
    eager = _make_inputs(case, seed)
    eager_seed = eager["base_output"].clone()
    expected = _oracle(case, eager, eager_seed)
    with torch.inference_mode():
        output = _call(module, eager)
    eager_result = _compare(output, expected, case.dtype)
    eager_result["base_output_alias"] = output.data_ptr() == eager["base_output"].data_ptr()

    captured = _make_inputs(case, seed + 10_000)
    graph, graph_output, graph_seed = _capture(module, captured)
    captured["base_output"].copy_(graph_seed)
    graph.replay()
    torch.cuda.synchronize()
    first = _compare(graph_output, _oracle(case, captured, graph_seed), case.dtype)

    changed = _make_inputs(case, seed + 20_000)
    for name in ("inputs", "weights", "weight_indices", "seg_lens", "base_output"):
        captured[name].copy_(changed[name])
    changed_seed = changed["base_output"].clone()
    graph.replay()
    torch.cuda.synchronize()
    dynamic = _compare(graph_output, _oracle(case, captured, changed_seed), case.dtype)
    return {
        "case_id": case.case_id,
        "eager": eager_result,
        "cuda_graph_first_replay": first,
        "cuda_graph_dynamic_replay": dynamic,
    }


def _special_contracts(module: ModuleType, seed: int) -> dict[str, Any]:
    empty_inputs = torch.randn(3, 16, dtype=torch.float16, device="cuda")
    empty_weights = torch.empty(0, 128, 16, dtype=torch.float16, device="cuda")
    empty = module.sgemm_lora_b_graph_fwd(
        empty_inputs,
        empty_weights,
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        torch.ones(3, dtype=torch.int32, device="cuda"),
        torch.tensor([0, 128], dtype=torch.int32),
    )

    mixed_case = Case("mixed-output-dtype", 4, 4, 16, (256,), torch.float16, 0)
    mixed = _make_inputs(mixed_case, seed + 30_000, output_dtype=torch.float32)
    mixed_seed = mixed["base_output"].clone()
    mixed_expected = _oracle(mixed_case, mixed, mixed_seed)
    with torch.inference_mode():
        mixed_output = _call(module, mixed)
    mixed_result = _compare(mixed_output, mixed_expected, torch.float32)

    grad_case = Case("gradient", 3, 2, 8, (64,), torch.float32, 0)
    grad = _make_inputs(grad_case, seed + 40_000)
    grad["inputs"].requires_grad_(True)
    grad["weights"].requires_grad_(True)
    grad_passed = True
    grad_error = None
    try:
        grad_output = module.sgemm_lora_b_graph_fwd(
            grad["inputs"],
            grad["weights"],
            grad["weight_indices"],
            grad["seg_lens"],
            grad["slice_offsets"],
            None,
        )
        grad_output.sum().backward()
        grad_passed = all(tensor.grad is not None for tensor in (grad["inputs"], grad["weights"]))
    except Exception as error:  # pragma: no cover - evidence path
        grad_passed = False
        grad_error = f"{type(error).__name__}: {error}"

    return {
        "empty_weights": {
            "passed": empty.shape == (3, 128) and bool(torch.count_nonzero(empty) == 0),
            "shape": list(empty.shape),
        },
        "mixed_output_dtype": mixed_result,
        "gradient": {"passed": grad_passed, "error": grad_error},
    }


def _benchmark_case(
    module: ModuleType, case: Case, seed: int, samples: int, iterations: int
) -> dict[str, Any]:
    tensors = _make_inputs(case, seed)
    graph, _, base_seed = _capture(module, tensors)
    timings: list[float] = []
    for _ in range(samples):
        tensors["base_output"].copy_(base_seed)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(1000 * start.elapsed_time(end) / iterations)

    memory_inputs = _make_inputs(case, seed + 50_000)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        memory_output = _call(module, memory_inputs)
    torch.cuda.synchronize()
    peak_delta = max(0, torch.cuda.max_memory_allocated() - before)
    del memory_output
    return {
        "case_id": case.case_id,
        "weight": case.weight,
        "shape": {
            "tokens": case.tokens,
            "slots": case.slots,
            "rank": case.rank,
            "slice_dims": list(case.slice_dims),
            "dtype": str(case.dtype).removeprefix("torch."),
        },
        "samples_us": timings,
        "median_us": statistics.median(timings),
        "min_us": min(timings),
        "max_us": max(timings),
        "iterations_per_sample": iterations,
        "eager_peak_temporary_bytes": peak_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--replay-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.cuda.set_device(0)
    module = _load_module(args.module)
    seed = 2_026_090_200 + args.replay_index * 100_000
    correctness = [
        _correctness_case(module, case, seed + index * 1000) for index, case in enumerate(CASES)
    ]
    special = _special_contracts(module, seed)
    performance = [
        _benchmark_case(module, case, seed + index * 1000, args.samples, args.iterations)
        for index, case in enumerate(CASES)
    ]
    torch.cuda.synchronize()

    payload = {
        "schema_version": "0.5",
        "benchmark": "sglang-graph-lora-b-cuda-graph-v1",
        "variant": args.variant,
        "replay_index": args.replay_index,
        "fresh_process": True,
        "source_sha256": _sha256(args.module),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python_cuda_graph": True,
        },
        "correctness": correctness,
        "special_contracts": special,
        "performance": performance,
        "wall_time_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
