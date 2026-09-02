#!/usr/bin/env python3
"""H100 correctness, gradient, layout, and speed probe for TorchTitan PR 2717."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import triton


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("torchtitan_r6_exact_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eager(
    routed_output: torch.Tensor,
    token_indices_experts_sorted: torch.Tensor,
    top_scores: torch.Tensor,
) -> torch.Tensor:
    token_count, top_k = top_scores.shape
    hidden = routed_output.shape[1]
    routed_output_unsorted = torch.zeros(
        (token_count * top_k, hidden),
        dtype=routed_output.dtype,
        device=routed_output.device,
    )
    routed_output_unsorted[token_indices_experts_sorted] = routed_output
    routed_output_unsorted = routed_output_unsorted.reshape(token_count, top_k, hidden)
    return (
        torch.bmm(top_scores.reshape(-1, 1, top_k), routed_output_unsorted.float())
        .to(routed_output.dtype)
        .squeeze(1)
    )


def _routing_indices(token_count: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    size = token_count * top_k
    token_indices = torch.randperm(size, device="cuda", dtype=torch.int64)
    inverse = torch.empty(size, device="cuda", dtype=torch.int32)
    inverse[token_indices] = torch.arange(size, device="cuda", dtype=torch.int32)
    return token_indices, inverse


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "allclose_atol_2e_2_rtol_2e_2": torch.allclose(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        ),
    }


def _correctness_case(
    module: ModuleType,
    token_count: int,
    top_k: int,
    hidden: int,
) -> dict[str, Any]:
    token_indices, inverse = _routing_indices(token_count, top_k)
    routed_base = torch.randn(
        token_count * top_k,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    routed_head = routed_base.detach().clone().requires_grad_(True)
    scores_base = torch.rand(
        token_count, top_k, device="cuda", dtype=torch.float32, requires_grad=True
    )
    scores_head = scores_base.detach().clone().requires_grad_(True)
    base = _eager(routed_base, token_indices, scores_base)
    head = module.apply_router_scores(routed_head, token_indices, scores_head, inverse)
    forward_error = _error(head, base)
    gradient = torch.randn_like(base)
    base.backward(gradient)
    head.backward(gradient)
    routed_gradient_error = _error(routed_head.grad, routed_base.grad)
    score_gradient_error = _error(scores_head.grad, scores_base.grad)
    return {
        "token_count": token_count,
        "top_k": top_k,
        "hidden": hidden,
        "forward": forward_error,
        "routed_gradient": routed_gradient_error,
        "score_gradient": score_gradient_error,
        "passes": all(
            item["allclose_atol_2e_2_rtol_2e_2"]
            for item in (
                forward_error,
                routed_gradient_error,
                score_gradient_error,
            )
        ),
    }


def _dtype_case(module: ModuleType, dtype: torch.dtype) -> dict[str, Any]:
    token_count, top_k, hidden = 17, 4, 129
    token_indices, inverse = _routing_indices(token_count, top_k)
    routed = torch.randn(token_count * top_k, hidden, device="cuda", dtype=dtype)
    scores = torch.rand(token_count, top_k, device="cuda", dtype=torch.float32)
    expected = _eager(routed, token_indices, scores)
    try:
        actual = module.apply_router_scores(routed, token_indices, scores, inverse)
    except Exception as exc:
        return {
            "dtype": str(dtype),
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {"dtype": str(dtype), "status": "executed", "error": _error(actual, expected)}


def _noncontiguous_case(module: ModuleType) -> dict[str, Any]:
    token_count, top_k, hidden = 19, 4, 97
    token_indices, inverse = _routing_indices(token_count, top_k)
    storage = torch.randn(token_count * top_k, hidden * 2, device="cuda", dtype=torch.bfloat16)
    routed = storage[:, ::2]
    scores = torch.rand(token_count, top_k, device="cuda", dtype=torch.float32)
    expected = _eager(routed, token_indices, scores)
    try:
        actual = module.apply_router_scores(routed, token_indices, scores, inverse)
    except Exception as exc:
        return {
            "input_is_contiguous": routed.is_contiguous(),
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "input_is_contiguous": routed.is_contiguous(),
        "status": "executed",
        "error": _error(actual, expected),
    }


def _cache_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


def _timed(function, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _performance(module: ModuleType, cache_root: Path) -> dict[str, Any]:
    token_count, top_k, hidden = 1024, 8, 4096
    token_indices, inverse = _routing_indices(token_count, top_k)
    routed = torch.randn(token_count * top_k, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.rand(token_count, top_k, device="cuda", dtype=torch.float32)

    def base_call():
        return _eager(routed, token_indices, scores)

    def head_call():
        return module.apply_router_scores(routed, token_indices, scores, inverse)

    base_call()
    head_call()
    torch.cuda.synchronize()
    cache_after_precompile = _cache_files(cache_root)
    pairs: list[dict[str, Any]] = []
    for index in range(7):
        order = ("base", "head") if index % 2 == 0 else ("head", "base")
        values: dict[str, float] = {}
        for variant in order:
            values[variant] = _timed(base_call if variant == "base" else head_call, 30)
        pairs.append({"first": order[0], **values})
    cache_after_timing = _cache_files(cache_root)
    base = [item["base"] for item in pairs]
    head = [item["head"] for item in pairs]

    torch.cuda.reset_peak_memory_stats()
    base_call()
    torch.cuda.synchronize()
    base_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    head_call()
    torch.cuda.synchronize()
    head_peak = torch.cuda.max_memory_allocated()
    return {
        "shape": {"token_count": token_count, "top_k": top_k, "hidden": hidden},
        "pairs": pairs,
        "base_median_us": statistics.median(base),
        "head_median_us": statistics.median(head),
        "head_over_base_median_ratio": statistics.median(head) / statistics.median(base),
        "base_peak_allocated_bytes": base_peak,
        "head_peak_allocated_bytes": head_peak,
        "cache_files_after_precompile": len(cache_after_precompile),
        "steady_new_cache_files": sorted(cache_after_timing - cache_after_precompile),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-kernels-source", type=Path, required=True)
    parser.add_argument("--head-kernels-source", type=Path, required=True)
    parser.add_argument("--base-moe-source", type=Path, required=True)
    parser.add_argument("--head-moe-source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    sources = {
        "base_kernels": args.base_kernels_source.read_text(encoding="utf-8"),
        "head_kernels": args.head_kernels_source.read_text(encoding="utf-8"),
        "base_moe": args.base_moe_source.read_text(encoding="utf-8"),
        "head_moe": args.head_moe_source.read_text(encoding="utf-8"),
    }
    source_contract = {
        "base_has_eager_bmm": "torch.bmm(" in sources["base_moe"],
        "head_calls_apply_router_scores": "apply_router_scores(" in sources["head_moe"],
        "head_kernel_count": sources["head_kernels"].count("def _apply_router_scores_"),
        "head_has_custom_autograd": "class _ApplyRouterScoresFunction" in sources["head_kernels"],
        "head_has_bfloat16_output_cast": "acc.to(tl.bfloat16)" in sources["head_kernels"],
        "head_has_contiguity_guard": "is_contiguous" in sources["head_kernels"],
        "head_has_bfloat16_runtime_guard": any(
            guard in sources["head_kernels"]
            for guard in (
                "assert routed_output.dtype == torch.bfloat16",
                "if routed_output.dtype != torch.bfloat16",
                "if routed_output.dtype is not torch.bfloat16",
            )
        ),
    }
    if not all(
        (
            source_contract["base_has_eager_bmm"],
            source_contract["head_calls_apply_router_scores"],
            source_contract["head_kernel_count"] == 3,
            source_contract["head_has_custom_autograd"],
            source_contract["head_has_bfloat16_output_cast"],
        )
    ):
        raise ValueError(f"unexpected exact-source contract: {source_contract}")

    module = _load_module(args.head_kernels_source)
    correctness = [
        _correctness_case(module, 7, 2, 96),
        _correctness_case(module, 32, 4, 128),
        _correctness_case(module, 65, 4, 257),
        _correctness_case(module, 129, 8, 1024),
    ]
    fp16 = _dtype_case(module, torch.float16)
    noncontiguous = _noncontiguous_case(module)
    performance = _performance(module, args.cache_root)

    failure_codes: list[str] = []
    if not all(item["passes"] for item in correctness):
        failure_codes.append("TORCHTITAN_ROUTER_FORWARD_OR_GRADIENT_MISMATCH")
    if not source_contract["head_has_bfloat16_runtime_guard"]:
        failure_codes.append("TORCHTITAN_ROUTER_BFLOAT16_RUNTIME_GATE_MISSING")
    if not source_contract["head_has_contiguity_guard"]:
        failure_codes.append("TORCHTITAN_ROUTER_CONTIGUITY_GATE_MISSING")
    if (
        noncontiguous["status"] == "executed"
        and not noncontiguous["error"]["allclose_atol_2e_2_rtol_2e_2"]
    ):
        failure_codes.append("TORCHTITAN_ROUTER_NONCONTIGUOUS_SILENT_MISMATCH")
    if performance["head_over_base_median_ratio"] >= 1.0:
        failure_codes.append("TORCHTITAN_ROUTER_PERFORMANCE_CLAIM_NOT_OBSERVED")
    if performance["steady_new_cache_files"]:
        failure_codes.append("TORCHTITAN_ROUTER_STEADY_COMPILATION_DETECTED")

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "torchtitan-fused-router-h100-v1",
        "case_id": "torchtitan-pr-2717",
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "facts": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "source_contract": source_contract,
            "correctness": correctness,
            "fp16_behavior": fp16,
            "noncontiguous_behavior": noncontiguous,
            "performance": performance,
            "steady_state_compile_seconds": (
                0.0 if not performance["steady_new_cache_files"] else None
            ),
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
