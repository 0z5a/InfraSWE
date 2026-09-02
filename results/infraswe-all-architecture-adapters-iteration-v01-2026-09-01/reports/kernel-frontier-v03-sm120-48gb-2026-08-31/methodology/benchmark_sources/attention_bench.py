from __future__ import annotations

import argparse
import importlib.metadata
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from bench_utils import (
    atomic_write_json,
    choose_repetitions,
    hardware_manifest,
    module_evidence,
    paired_blocks,
    profiler_evidence,
    tensor_correctness,
    utc_now,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

CASES = [
    {
        "id": "common-b4-s512-h16-d64-noncausal",
        "group": "common",
        "weight": 0.20,
        "batch": 4,
        "seqlen": 512,
        "heads": 16,
        "head_dim": 64,
        "causal": False,
    },
    {
        "id": "common-b2-s1024-h16-d64-causal",
        "group": "common",
        "weight": 0.20,
        "batch": 2,
        "seqlen": 1024,
        "heads": 16,
        "head_dim": 64,
        "causal": True,
    },
    {
        "id": "common-b1-s2048-h16-d128-causal",
        "group": "common",
        "weight": 0.20,
        "batch": 1,
        "seqlen": 2048,
        "heads": 16,
        "head_dim": 128,
        "causal": True,
    },
    {
        "id": "boundary-b3-s1000-h12-d64-causal",
        "group": "boundary_tail",
        "weight": 0.20,
        "batch": 3,
        "seqlen": 1000,
        "heads": 12,
        "head_dim": 64,
        "causal": True,
    },
    {
        "id": "stress-b1-s4096-h8-d128-causal",
        "group": "stress_large",
        "weight": 0.20,
        "batch": 1,
        "seqlen": 4096,
        "heads": 8,
        "head_dim": 128,
        "causal": True,
    },
]


@dataclass(frozen=True)
class Adapter:
    backend: str
    version: str
    prepare: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], Callable[[], torch.Tensor]]
    provenance: list[dict[str, Any]]


def _torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, backend):
    q_bhsd = q.transpose(1, 2)
    k_bhsd = k.transpose(1, 2)
    v_bhsd = v.transpose(1, 2)

    def run() -> torch.Tensor:
        with sdpa_kernel([backend]):
            output = functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=0.0,
                is_causal=causal,
            )
        return output.transpose(1, 2)

    return run


def reference_prepare(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    return _torch_sdpa(q, k, v, causal, SDPBackend.MATH)


def load_adapter(name: str) -> Adapter:
    if name == "torch-sdpa-flash":
        return Adapter(
            backend=name,
            version=torch.__version__,
            prepare=lambda q, k, v, causal: _torch_sdpa(
                q, k, v, causal, SDPBackend.FLASH_ATTENTION
            ),
            provenance=[module_evidence(torch)],
        )
    if name == "torch-sdpa-cudnn":
        return Adapter(
            backend=name,
            version=torch.__version__,
            prepare=lambda q, k, v, causal: _torch_sdpa(
                q, k, v, causal, SDPBackend.CUDNN_ATTENTION
            ),
            provenance=[module_evidence(torch)],
        )
    if name == "fa1":
        import flash_attn.flash_attn_interface as interface
        import flash_attn_cuda

        def prepare(q, k, v, causal):
            batch, seqlen, _, _ = q.shape
            cu_seqlens = torch.arange(
                0,
                (batch + 1) * seqlen,
                step=seqlen,
                dtype=torch.int32,
                device=q.device,
            )
            q_flat = q.reshape(-1, *q.shape[2:])
            k_flat = k.reshape(-1, *k.shape[2:])
            v_flat = v.reshape(-1, *v.shape[2:])

            def run():
                output = interface.flash_attn_unpadded_func(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens,
                    cu_seqlens,
                    seqlen,
                    seqlen,
                    0.0,
                    softmax_scale=None,
                    causal=causal,
                )
                return output.reshape_as(q)

            return run

        return Adapter(
            backend=name,
            version=importlib.metadata.version("flash-attn"),
            prepare=prepare,
            provenance=[module_evidence(interface), module_evidence(flash_attn_cuda)],
        )
    if name == "fa2":
        import flash_attn
        import flash_attn_2_cuda
        from flash_attn import flash_attn_func

        def prepare(q, k, v, causal):
            return lambda: flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)

        return Adapter(
            backend=name,
            version=importlib.metadata.version("flash-attn"),
            prepare=prepare,
            provenance=[module_evidence(flash_attn), module_evidence(flash_attn_2_cuda)],
        )
    if name == "fa3":
        import flash_attn_3
        from flash_attn_3 import _C as extension
        from flash_attn_3 import flash_attn_interface as interface

        def prepare(q, k, v, causal):
            def run():
                output = interface.flash_attn_func(q, k, v, causal=causal)
                return output[0] if isinstance(output, tuple) else output

            return run

        return Adapter(
            backend=name,
            version=importlib.metadata.version("flash-attn-3"),
            prepare=prepare,
            provenance=[
                module_evidence(flash_attn_3),
                module_evidence(interface),
                module_evidence(extension),
            ],
        )
    if name == "fa4":
        import flash_attn.cute as cute
        from flash_attn.cute import flash_attn_func

        def prepare(q, k, v, causal):
            def run():
                output = flash_attn_func(q, k, v, causal=causal)
                return output[0] if isinstance(output, tuple) else output

            return run

        return Adapter(
            backend=name,
            version=importlib.metadata.version("flash-attn-4"),
            prepare=prepare,
            provenance=[module_evidence(cute, tree=True)],
        )
    if name == "garbage-slow-fa4-waste64":
        import flash_attn.cute as cute
        import garbage_kernels

        return Adapter(
            backend=name,
            version="negative-control-v1",
            prepare=garbage_kernels.slow_fa4_prepare,
            provenance=[
                module_evidence(garbage_kernels),
                module_evidence(cute, tree=True),
            ],
        )
    mediocre_waste_passes = {
        "mediocre-fa4-waste0": 0,
        "mediocre-fa4-waste2": 2,
        "mediocre-fa4-waste4": 4,
        "mediocre-fa4-waste8": 8,
        "mediocre-fa4-waste12": 12,
        "mediocre-fa4-waste16": 16,
        "mediocre-fa4-waste24": 24,
        "mediocre-fa4-waste32": 32,
        "mediocre-fa4-waste48": 48,
        "mediocre-fa4-waste96": 96,
        "mediocre-fa4-waste128": 128,
        "mediocre-fa4-waste160": 160,
        "mediocre-fa4-waste192": 192,
        "mediocre-fa4-waste256": 256,
        "mediocre-fa4-waste384": 384,
        "mediocre-fa4-waste512": 512,
        "mediocre-fa4-waste768": 768,
        "mediocre-fa4-waste1024": 1024,
    }
    if name in mediocre_waste_passes:
        import flash_attn.cute as cute
        import garbage_kernels

        passes = mediocre_waste_passes[name]
        return Adapter(
            backend=name,
            version="controlled-degradation-v1",
            prepare=garbage_kernels.make_fa4_waste_prepare(passes),
            provenance=[
                module_evidence(garbage_kernels),
                module_evidence(cute, tree=True),
            ],
        )
    if name == "garbage-zero-triton":
        import garbage_kernels

        return Adapter(
            backend=name,
            version="negative-control-v1",
            prepare=garbage_kernels.zero_prepare,
            provenance=[module_evidence(garbage_kernels)],
        )
    if name == "garbage-cache-copy":
        import flash_attn.cute as cute
        import garbage_kernels

        cache = garbage_kernels.CachedAnswerAdapter()
        return Adapter(
            backend=name,
            version="negative-control-v1",
            prepare=cache.prepare,
            provenance=[
                module_evidence(garbage_kernels),
                module_evidence(cute, tree=True),
            ],
        )
    if name == "fa-garbage-math-fallback":
        import garbage_kernels

        return Adapter(
            backend=name,
            version="negative-control-v1",
            prepare=lambda q, k, v, causal: _torch_sdpa(q, k, v, causal, SDPBackend.MATH),
            provenance=[module_evidence(garbage_kernels), module_evidence(torch)],
        )
    raise ValueError(f"unknown attention backend: {name}")


def make_qkv(case: dict[str, Any], seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (case["batch"], case["seqlen"], case["heads"], case["head_dim"])
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    return q, k, v


def work_model(case: dict[str, Any]) -> dict[str, Any]:
    batch = case["batch"]
    seqlen = case["seqlen"]
    heads = case["heads"]
    head_dim = case["head_dim"]
    attention_fraction = 0.5 if case["causal"] else 1.0
    semantic_flops = 4 * batch * heads * seqlen * seqlen * head_dim * attention_fraction
    semantic_bytes = 4 * batch * seqlen * heads * head_dim * 2
    return {
        "version": "attention-fwd-semantic-v1",
        "semantic_flops": int(semantic_flops),
        "minimum_external_bytes": int(semantic_bytes),
        "known_omissions": ["softmax scalar operations", "scheduler and launch floors"],
    }


def run_case(
    adapter: Adapter,
    case: dict[str, Any],
    *,
    replay_index: int,
    blocks: int,
    min_timed_span_ms: float,
) -> dict[str, Any]:
    seed = 80_000 + replay_index * 10_000 + CASES.index(case) * 100
    q, k, v = make_qkv(case, seed)
    reference = reference_prepare(q, k, v, case["causal"])
    candidate = adapter.prepare(q, k, v, case["causal"])
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
        correctness["max_abs_error"] <= 0.05
        and correctness["relative_l2_error"] <= 0.03
        and correctness["cosine_similarity"] >= 0.999
    )
    # A one-element perturbation can legitimately disappear after BF16 rounding,
    # and changing causal token zero cannot affect its single-key attention row.
    # Negating every query is deterministic, preserves the declared shape/value
    # domain, and gives the anti-cache probe enough signal on every causal case.
    perturbed_q = q.neg()
    perturbed_candidate = adapter.prepare(perturbed_q, k, v, case["causal"])()
    torch.cuda.synchronize()
    correctness["dynamic_input_changes_output"] = bool(
        not torch.equal(candidate_output, perturbed_candidate)
    )
    blocks_payload = paired_blocks(
        reference=reference,
        candidate=candidate,
        reference_repetitions=reference_repetitions,
        candidate_repetitions=candidate_repetitions,
        blocks=blocks,
        seed=seed + 99,
    )
    return {
        "case_id": case["id"],
        "case_group": case["group"],
        "weight": case["weight"],
        "shape": {key: case[key] for key in ("batch", "seqlen", "heads", "head_dim")},
        "causal": case["causal"],
        "dtype": "bfloat16",
        "work_model": work_model(case),
        "correctness": correctness,
        "measurement": {
            "reference": "torch-sdpa-math",
            "candidate": adapter.backend,
            "reference_repetitions": reference_repetitions,
            "candidate_repetitions": candidate_repetitions,
            "reference_pilot_us": reference_pilot_us,
            "candidate_pilot_us": candidate_pilot_us,
            "blocks": blocks_payload,
        },
        "profiler": profiler_evidence(candidate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--min-timed-span-ms", type=float, default=50.0)
    parser.add_argument("--implementation-commit", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    payload: dict[str, Any] = {
        "schema_version": "0.3",
        "benchmark": "flash-attention-forward-portfolio-v1",
        "benchmark_kind": "kernel-library",
        "backend": args.backend,
        "implementation_commit": args.implementation_commit,
        "replay_index": args.replay_index,
        "started_at": started_at,
        "protocol": {
            "paired_order": "randomized-ABBA-or-BAAB",
            "blocks": args.blocks,
            "min_timed_span_ms": args.min_timed_span_ms,
            "warmup_calls": 5,
            "fresh_process": True,
            "fresh_allocator_per_replay": True,
            "timer": "evaluator-owned-cuda-events",
            "completion_fence": "end-event-synchronize",
            "profiler_separate_from_official_timing": True,
        },
        "hardware": hardware_manifest(),
        "status": "running",
        "cases": [],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        adapter = load_adapter(args.backend)
        payload["backend_version"] = adapter.version
        payload["implementation_provenance"] = adapter.provenance
        for case in CASES:
            payload["cases"].append(
                run_case(
                    adapter,
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
    payload["peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    payload["case_count"] = len(payload["cases"])
    payload["all_correct"] = bool(payload["cases"]) and all(
        case["correctness"]["passed"] and case["correctness"]["dynamic_input_changes_output"]
        for case in payload["cases"]
    )
    atomic_write_json(args.output, payload)
    if payload["status"] != "passed" or not payload["all_correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
