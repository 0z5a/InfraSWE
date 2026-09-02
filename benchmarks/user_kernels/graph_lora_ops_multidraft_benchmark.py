#!/usr/bin/env python3
"""Multi-Draft contract and paired CUDA Graph benchmark for graph LoRA B."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from graph_lora_ops_benchmark import (
    Case,
    _call,
    _capture,
    _compare,
    _correctness_case,
    _load_module,
    _make_inputs,
    _offsets,
    _oracle,
    _sha256,
    _special_contracts,
)
from graph_lora_ops_paired_benchmark import _time_graph


@dataclass(frozen=True)
class DraftProfile:
    profile_id: str
    project: str
    role: str
    authority: str


@dataclass(frozen=True)
class DraftCase:
    profile_id: str
    contract_tags: tuple[str, ...]
    case: Case


DRAFT_PROFILES = (
    DraftProfile(
        "sglang-runtime-kernel-v1",
        "sglang",
        "primary-host",
        "native-target-contract",
    ),
    DraftProfile(
        "cutlass-cute-kernel-library-v1",
        "cutlass-cute",
        "primary-peer",
        "dense-gemm-contract-proxy",
    ),
    DraftProfile(
        "vllm-kernel-integration-v1",
        "vllm",
        "secondary-host",
        "host-workload-contract-proxy",
    ),
    DraftProfile(
        "megatron-core-training-kernel-host-v1",
        "megatron-core",
        "secondary-host",
        "host-workload-contract-proxy",
    ),
    DraftProfile(
        "deepgemm-moe-gemm-kernel-v1",
        "deepgemm",
        "secondary-peer",
        "dense-gemm-contract-proxy",
    ),
)


CASES = (
    DraftCase(
        "sglang-runtime-kernel-v1",
        ("fallback-boundary", "graph-replay"),
        Case("sglang-slots3-boundary-bf16", 16, 3, 32, (4096,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "sglang-runtime-kernel-v1",
        ("fastpath-boundary", "graph-replay"),
        Case("sglang-slots4-boundary-bf16", 16, 4, 32, (4096,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "sglang-runtime-kernel-v1",
        ("decode", "tiny-m"),
        Case("sglang-decode-tiny-fp16", 1, 4, 16, (4096,), torch.float16, 1.0),
    ),
    DraftCase(
        "sglang-runtime-kernel-v1",
        ("multi-slice", "qkv", "dynamic-replay"),
        Case(
            "sglang-qkv-multislice-fp16",
            16,
            4,
            32,
            (4096, 1024, 1024),
            torch.float16,
            1.0,
        ),
    ),
    DraftCase(
        "sglang-runtime-kernel-v1",
        ("high-slot", "wide-output"),
        Case("sglang-slots16-wide-bf16", 64, 16, 64, (8192,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "cutlass-cute-kernel-library-v1",
        ("tiny-m", "epilogue-alias"),
        Case("cutlass-tiny-m-fp16", 1, 4, 8, (256,), torch.float16, 1.0),
    ),
    DraftCase(
        "cutlass-cute-kernel-library-v1",
        ("skinny-shape", "wide-output"),
        Case("cutlass-skinny-bf16", 8, 4, 16, (12288,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "cutlass-cute-kernel-library-v1",
        ("non-aligned-k", "odd-token"),
        Case("cutlass-nonaligned-k-fp16", 7, 5, 24, (3072,), torch.float16, 1.0),
    ),
    DraftCase(
        "cutlass-cute-kernel-library-v1",
        ("non-aligned-n", "odd-token"),
        Case("cutlass-nonaligned-n-bf16", 9, 5, 40, (3073,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "vllm-kernel-integration-v1",
        ("single-adapter-control", "fallback"),
        Case("vllm-slots1-control-fp16", 4, 1, 16, (4096,), torch.float16, 1.0),
    ),
    DraftCase(
        "vllm-kernel-integration-v1",
        ("adapter-burst", "decode"),
        Case("vllm-adapter-burst-fp16", 16, 8, 16, (4096,), torch.float16, 1.0),
    ),
    DraftCase(
        "vllm-kernel-integration-v1",
        ("qkv", "multi-slice"),
        Case(
            "vllm-qkv-bf16",
            32,
            4,
            32,
            (4096, 1024, 1024),
            torch.bfloat16,
            1.0,
        ),
    ),
    DraftCase(
        "vllm-kernel-integration-v1",
        ("mixed-batch", "high-slot"),
        Case("vllm-mixed-batch-bf16", 96, 8, 32, (4096,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "megatron-core-training-kernel-host-v1",
        ("large-batch", "wide-output"),
        Case("megatron-large-batch-bf16", 128, 4, 64, (8192,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "megatron-core-training-kernel-host-v1",
        ("tensor-parallel-shards", "multi-slice"),
        Case(
            "megatron-tp-slices-fp16",
            64,
            4,
            64,
            (2048, 2048, 2048, 2048),
            torch.float16,
            1.0,
        ),
    ),
    DraftCase(
        "megatron-core-training-kernel-host-v1",
        ("high-rank", "bf16"),
        Case("megatron-high-rank-bf16", 32, 4, 128, (4096,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "megatron-core-training-kernel-host-v1",
        ("fallback", "large-batch"),
        Case("megatron-slots3-control-bf16", 128, 3, 64, (8192,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "deepgemm-moe-gemm-kernel-v1",
        ("tiny-m", "high-slot"),
        Case("deepgemm-tiny-m-fp16", 2, 8, 64, (8192,), torch.float16, 1.0),
    ),
    DraftCase(
        "deepgemm-moe-gemm-kernel-v1",
        ("throughput", "high-slot"),
        Case("deepgemm-throughput-bf16", 128, 8, 64, (8192,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "deepgemm-moe-gemm-kernel-v1",
        ("non-aligned-k", "bf16"),
        Case("deepgemm-nonaligned-k-bf16", 32, 8, 48, (6144,), torch.bfloat16, 1.0),
    ),
    DraftCase(
        "deepgemm-moe-gemm-kernel-v1",
        ("multi-slice", "high-slot"),
        Case(
            "deepgemm-multislice-fp16",
            32,
            8,
            64,
            (4096, 4096),
            torch.float16,
            1.0,
        ),
    ),
)


def _serialize_case(draft_case: DraftCase) -> dict[str, Any]:
    case = draft_case.case
    return {
        "case_id": case.case_id,
        "profile_id": draft_case.profile_id,
        "contract_tags": list(draft_case.contract_tags),
        "shape": {
            "tokens": case.tokens,
            "slots": case.slots,
            "rank": case.rank,
            "slice_dims": list(case.slice_dims),
            "dtype": str(case.dtype).removeprefix("torch."),
        },
    }


def _gradient_slots4(module, seed: int) -> dict[str, Any]:
    case = Case("gradient-slots4", 5, 4, 16, (128,), torch.float32, 0)
    tensors = _make_inputs(case, seed)
    tensors["inputs"].requires_grad_(True)
    tensors["weights"].requires_grad_(True)
    try:
        output = module.sgemm_lora_b_graph_fwd(
            tensors["inputs"],
            tensors["weights"],
            tensors["weight_indices"],
            tensors["seg_lens"],
            tensors["slice_offsets"],
            None,
        )
        output.square().sum().backward()
        gradients = (tensors["inputs"].grad, tensors["weights"].grad)
        passed = all(item is not None and torch.isfinite(item).all() for item in gradients)
        return {"passed": bool(passed), "error": None}
    except Exception as error:  # pragma: no cover - evidence path
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def _base_output_none(module, seed: int) -> dict[str, Any]:
    case = Case("base-output-none", 7, 5, 24, (257,), torch.float16, 0)
    tensors = _make_inputs(case, seed)
    expected = _oracle(
        case,
        tensors,
        torch.zeros_like(tensors["base_output"]),
    )
    with torch.inference_mode():
        output = module.sgemm_lora_b_graph_fwd(
            tensors["inputs"],
            tensors["weights"],
            tensors["weight_indices"],
            tensors["seg_lens"],
            tensors["slice_offsets"],
            None,
        )
    result = _compare(output, expected, case.dtype)
    result["allocated_output"] = output.data_ptr() != tensors["base_output"].data_ptr()
    result["passed"] = bool(result["passed"] and result["allocated_output"])
    return result


def _zero_tokens(module) -> dict[str, Any]:
    output = module.sgemm_lora_b_graph_fwd(
        torch.empty(0, 16, dtype=torch.float16, device="cuda"),
        torch.randn(4, 64, 16, dtype=torch.float16, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        _offsets((64,)),
        None,
    )
    return {
        "passed": output.shape == (0, 64) and bool(torch.isfinite(output).all()),
        "shape": list(output.shape),
    }


def _concurrent_streams(module, seed: int) -> dict[str, Any]:
    cases = (
        Case("stream-a", 13, 4, 24, (513,), torch.float16, 0),
        Case("stream-b", 17, 8, 32, (257, 129), torch.bfloat16, 0),
    )
    tensors = [_make_inputs(case, seed + index * 1000) for index, case in enumerate(cases)]
    expected = [
        _oracle(case, item, item["base_output"].clone())
        for case, item in zip(cases, tensors, strict=True)
    ]
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = []
    for stream, item in zip(streams, tensors, strict=True):
        with torch.cuda.stream(stream), torch.inference_mode():
            outputs.append(_call(module, item))
    for stream in streams:
        stream.synchronize()
    results = [
        _compare(output, oracle, case.dtype)
        for output, oracle, case in zip(outputs, expected, cases, strict=True)
    ]
    return {
        "passed": all(result["passed"] and result["all_finite"] for result in results),
        "streams": results,
    }


def _contract_checks(module, seed: int) -> list[dict[str, Any]]:
    existing = _special_contracts(module, seed)
    checks = [{"name": name, **result} for name, result in existing.items()]
    checks.extend(
        [
            {"name": "gradient_slots4_fallback", **_gradient_slots4(module, seed + 60_000)},
            {"name": "base_output_none", **_base_output_none(module, seed + 70_000)},
            {"name": "zero_tokens", **_zero_tokens(module)},
            {"name": "concurrent_streams", **_concurrent_streams(module, seed + 80_000)},
        ]
    )
    return checks


def _paired_case(
    baseline,
    candidate,
    draft_case: DraftCase,
    seed: int,
    replay_index: int,
    case_index: int,
    samples: int,
    iterations: int,
) -> dict[str, Any]:
    case = draft_case.case
    capture_order = (
        ("baseline", "candidate")
        if (replay_index + case_index) % 2 == 0
        else ("candidate", "baseline")
    )
    artifacts = {}
    modules = {"baseline": baseline, "candidate": candidate}
    for variant in capture_order:
        tensors = _make_inputs(case, seed)
        graph, _, base_seed = _capture(modules[variant], tensors)
        artifacts[variant] = (graph, tensors, base_seed)

    values = {"baseline": [], "candidate": []}
    pairs = []
    for sample_index in range(samples):
        order = (
            ("baseline", "candidate")
            if (replay_index + case_index + sample_index) % 2 == 0
            else ("candidate", "baseline")
        )
        sample = {}
        for variant in order:
            graph, tensors, base_seed = artifacts[variant]
            duration = _time_graph(
                graph,
                tensors["base_output"],
                base_seed,
                iterations,
            )
            values[variant].append(duration)
            sample[variant] = duration
        ratio = sample["candidate"] / sample["baseline"]
        pairs.append(
            {
                "sample_index": sample_index,
                "order": list(order),
                "baseline_us": sample["baseline"],
                "candidate_us": sample["candidate"],
                "candidate_to_baseline": ratio,
                "speedup": 1.0 / ratio,
            }
        )
    return {
        **_serialize_case(draft_case),
        "capture_order": list(capture_order),
        "baseline_median_us": statistics.median(values["baseline"]),
        "candidate_median_us": statistics.median(values["candidate"]),
        "paired_ratio_median": statistics.median(pair["candidate_to_baseline"] for pair in pairs),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.cuda.set_device(0)
    baseline = _load_module(args.baseline)
    candidate = _load_module(args.candidate)
    seed = 2_026_090_300 + args.replay_index * 1_000_000

    correctness = {"baseline": [], "candidate": []}
    for case_index, draft_case in enumerate(CASES):
        for variant, module in (("baseline", baseline), ("candidate", candidate)):
            result = _correctness_case(module, draft_case.case, seed + case_index * 10_000)
            correctness[variant].append({**_serialize_case(draft_case), **result})

    contracts = {
        "baseline": _contract_checks(baseline, seed + 300_000),
        "candidate": _contract_checks(candidate, seed + 400_000),
    }
    performance = []
    for case_index, draft_case in enumerate(CASES):
        performance.append(
            _paired_case(
                baseline,
                candidate,
                draft_case,
                seed + case_index * 10_000,
                args.replay_index,
                case_index,
                args.samples,
                args.iterations,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema_version": "0.5",
        "benchmark": "sglang-graph-lora-multidraft-v1",
        "replay_index": args.replay_index,
        "fresh_process": True,
        "baseline_source_sha256": _sha256(args.baseline),
        "candidate_source_sha256": _sha256(args.candidate),
        "draft_profiles": [asdict(profile) for profile in DRAFT_PROFILES],
        "case_plan": [_serialize_case(case) for case in CASES],
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "samples_per_case": args.samples,
        "iterations_per_sample": args.iterations,
        "correctness": correctness,
        "contracts": contracts,
        "performance": performance,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
