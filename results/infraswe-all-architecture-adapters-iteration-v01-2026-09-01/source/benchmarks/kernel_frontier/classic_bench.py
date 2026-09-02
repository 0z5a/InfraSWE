from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import triton
import triton_kernels
from bench_utils import (
    atomic_write_json,
    choose_repetitions,
    device_timer_name,
    hardware_manifest,
    module_evidence,
    paired_blocks,
    profiler_evidence,
    tensor_correctness,
    utc_now,
)


@dataclass(frozen=True)
class ClassicCase:
    case_id: str
    shape: dict[str, int]
    prepare: Callable[[int], tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]]
    semantic_flops: int
    semantic_bytes: int
    tolerance_l2: float = 0.02


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return generator


def vector_add_case(seed: int):
    elements = 16 * 1024 * 1024
    generator = _generator(seed)
    left = torch.randn(elements, device="cuda", dtype=torch.bfloat16, generator=generator)
    right = torch.randn(elements, device="cuda", dtype=torch.bfloat16, generator=generator)
    return lambda: left + right, lambda: triton_kernels.vector_add(left, right)


def softmax_case(seed: int):
    rows, columns = 4096, 4096
    source = torch.randn(
        (rows, columns), device="cuda", dtype=torch.bfloat16, generator=_generator(seed)
    )
    return lambda: torch.softmax(source, dim=-1), lambda: triton_kernels.softmax(source)


def layernorm_case(seed: int):
    rows, columns = 4096, 4096
    generator = _generator(seed)
    source = torch.randn((rows, columns), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(columns, device="cuda", dtype=torch.bfloat16, generator=generator)
    bias = torch.randn(columns, device="cuda", dtype=torch.bfloat16, generator=generator)
    def reference():
        return functional.layer_norm(source, (columns,), weight, bias, 1e-5)

    def candidate():
        return triton_kernels.layernorm(source, weight, bias, 1e-5)

    return reference, candidate


def rmsnorm_case(seed: int):
    rows, columns = 4096, 4096
    generator = _generator(seed)
    source = torch.randn((rows, columns), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(columns, device="cuda", dtype=torch.bfloat16, generator=generator)

    def reference():
        variance = source.float().square().mean(dim=-1, keepdim=True)
        return (source.float() * torch.rsqrt(variance + 1e-6) * weight.float()).to(source.dtype)

    return reference, lambda: triton_kernels.rmsnorm(source, weight, 1e-6)


def swiglu_case(seed: int):
    rows, columns = 8192, 4096
    generator = _generator(seed)
    gate = torch.randn((rows, columns), device="cuda", dtype=torch.bfloat16, generator=generator)
    value = torch.randn((rows, columns), device="cuda", dtype=torch.bfloat16, generator=generator)
    return lambda: functional.silu(gate) * value, lambda: triton_kernels.swiglu(gate, value)


def rope_case(seed: int):
    batch, seqlen, heads, head_dim = 4, 2048, 16, 128
    generator = _generator(seed)
    source = torch.randn(
        (batch, seqlen, heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    angles = torch.randn(
        (seqlen, head_dim // 2),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cosine = torch.cos(angles).to(torch.bfloat16)
    sine = torch.sin(angles).to(torch.bfloat16)

    def reference():
        first, second = source.chunk(2, dim=-1)
        cos_view = cosine[None, :, None, :]
        sin_view = sine[None, :, None, :]
        return torch.cat(
            (first * cos_view - second * sin_view, first * sin_view + second * cos_view),
            dim=-1,
        )

    return reference, lambda: triton_kernels.rope(source, cosine, sine)


def gemm_case(seed: int):
    dimension = 4096
    generator = _generator(seed)
    left = torch.randn(
        (dimension, dimension), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    right = torch.randn(
        (dimension, dimension), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    return lambda: torch.mm(left, right), lambda: triton_kernels.matmul(left, right)


CASES = [
    ClassicCase(
        "vector-add-bf16-16m",
        {"elements": 16 * 1024 * 1024},
        vector_add_case,
        semantic_flops=16 * 1024 * 1024,
        semantic_bytes=3 * 16 * 1024 * 1024 * 2,
    ),
    ClassicCase(
        "softmax-bf16-4096x4096",
        {"rows": 4096, "columns": 4096},
        softmax_case,
        semantic_flops=5 * 4096 * 4096,
        semantic_bytes=2 * 4096 * 4096 * 2,
    ),
    ClassicCase(
        "layernorm-bf16-4096x4096",
        {"rows": 4096, "columns": 4096},
        layernorm_case,
        semantic_flops=8 * 4096 * 4096,
        semantic_bytes=(2 * 4096 * 4096 + 2 * 4096) * 2,
    ),
    ClassicCase(
        "rmsnorm-bf16-4096x4096",
        {"rows": 4096, "columns": 4096},
        rmsnorm_case,
        semantic_flops=5 * 4096 * 4096,
        semantic_bytes=(2 * 4096 * 4096 + 4096) * 2,
    ),
    ClassicCase(
        "swiglu-bf16-8192x4096",
        {"rows": 8192, "columns": 4096},
        swiglu_case,
        semantic_flops=6 * 8192 * 4096,
        semantic_bytes=3 * 8192 * 4096 * 2,
    ),
    ClassicCase(
        "rope-bf16-b4-s2048-h16-d128",
        {"batch": 4, "seqlen": 2048, "heads": 16, "head_dim": 128},
        rope_case,
        semantic_flops=6 * 4 * 2048 * 16 * 128,
        semantic_bytes=(2 * 4 * 2048 * 16 * 128 + 2 * 2048 * 64) * 2,
    ),
    ClassicCase(
        "gemm-bf16-4096-cube",
        {"m": 4096, "n": 4096, "k": 4096},
        gemm_case,
        semantic_flops=2 * 4096**3,
        semantic_bytes=3 * 4096**2 * 2,
        tolerance_l2=0.03,
    ),
]


def run_case(
    case: ClassicCase,
    *,
    replay_index: int,
    blocks: int,
    min_timed_span_ms: float,
) -> dict[str, Any]:
    seed = 50_000 + replay_index * 1000 + CASES.index(case) * 10
    reference, candidate = case.prepare(seed)
    reference_repetitions, reference_pilot_us = choose_repetitions(
        reference, min_timed_span_ms=min_timed_span_ms
    )
    candidate_repetitions, candidate_pilot_us = choose_repetitions(
        candidate, min_timed_span_ms=min_timed_span_ms
    )
    reference_output = reference()
    candidate_output = candidate()
    torch.cuda.synchronize()
    correctness = tensor_correctness(reference_output, candidate_output)
    correctness["passed"] = (
        correctness["relative_l2_error"] <= case.tolerance_l2
        and correctness["cosine_similarity"] >= 0.999
    )
    measurements = paired_blocks(
        reference=reference,
        candidate=candidate,
        reference_repetitions=reference_repetitions,
        candidate_repetitions=candidate_repetitions,
        blocks=blocks,
        seed=seed + 7,
    )
    return {
        "case_id": case.case_id,
        "shape": case.shape,
        "dtype": "bfloat16",
        "work_model": {
            "version": "classic-semantic-v1",
            "semantic_flops": case.semantic_flops,
            "minimum_external_bytes": case.semantic_bytes,
        },
        "correctness": correctness,
        "measurement": {
            "reference": "pytorch-eager",
            "candidate": "triton-fixed-config",
            "reference_repetitions": reference_repetitions,
            "candidate_repetitions": candidate_repetitions,
            "reference_pilot_us": reference_pilot_us,
            "candidate_pilot_us": candidate_pilot_us,
            "blocks": measurements,
        },
        "profiler": profiler_evidence(candidate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--min-timed-span-ms", type=float, default=50.0)
    parser.add_argument("--backend", default="triton-fixed-config")
    parser.add_argument("--implementation-commit", default="infraswe-portable-fixed-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "schema_version": "0.3",
        "benchmark": "classic-kernel-micro-v1",
        "benchmark_kind": "kernel-micro",
        "backend": args.backend,
        "backend_version": triton.__version__,
        "implementation_commit": args.implementation_commit,
        "replay_index": args.replay_index,
        "started_at": utc_now(),
        "hardware": hardware_manifest(),
        "implementation_provenance": [module_evidence(triton_kernels)],
        "protocol": {
            "paired_order": "randomized-ABBA-or-BAAB",
            "blocks": args.blocks,
            "min_timed_span_ms": args.min_timed_span_ms,
            "warmup_calls": 5,
            "fresh_process": True,
            "fresh_allocator_per_replay": True,
            "timer": "evaluator-owned-cuda-events",
            "device_timer": device_timer_name(),
            "completion_fence": "end-event-synchronize",
            "profiler_separate_from_official_timing": True,
        },
        "status": "running",
        "cases": [],
    }
    try:
        for case in CASES:
            payload["cases"].append(
                run_case(
                    case,
                    replay_index=args.replay_index,
                    blocks=args.blocks,
                    min_timed_span_ms=args.min_timed_span_ms,
                )
            )
        payload["status"] = "passed"
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["finished_at"] = utc_now()
    payload["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    payload["case_count"] = len(payload["cases"])
    payload["all_correct"] = bool(payload["cases"]) and all(
        case["correctness"]["passed"] for case in payload["cases"]
    )
    atomic_write_json(args.output, payload)
    if payload["status"] != "passed" or not payload["all_correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
