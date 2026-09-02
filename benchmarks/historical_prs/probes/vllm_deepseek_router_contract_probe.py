#!/usr/bin/env python3
"""Exact-function probe for vLLM #14027 router precision and renormalize semantics."""

from __future__ import annotations

import argparse
import ast
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch

FUNCTIONS = {"vllm_topk_softmax", "dispatch_topk_func", "fused_topk", "grouped_topk"}


class ReferenceOps:
    @staticmethod
    def topk_softmax(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
    ) -> None:
        del token_expert_indices
        weights, indices = torch.topk(
            torch.softmax(gating_output, dim=-1),
            k=topk_weights.shape[1],
            dim=-1,
            sorted=False,
        )
        topk_weights.copy_(weights)
        topk_ids.copy_(indices.to(torch.int32))


def _load_functions(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS:
            node.decorator_list = []
            selected.append(node)
    missing = FUNCTIONS - {
        node.name for node in selected if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if missing:
        raise RuntimeError(f"missing exact functions in {path}: {sorted(missing)}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {
        "torch": torch,
        "Tuple": tuple,
        "Optional": Optional,
        "Callable": Callable,
        "ops": ReferenceOps(),
        "is_rocm_aiter_moe_enabled": lambda: False,
        "rocm_aiter_topk_softmax": None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_calls(function: Callable[[], object], iterations: int) -> float:
    for _ in range(10):
        function()
    _sync()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    _sync()
    return (time.perf_counter() - started) / iterations


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected.double()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--head-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(14027)
    base = _load_functions(args.base_source)
    head = _load_functions(args.head_source)
    hidden = torch.randn(32, 64, dtype=torch.float16, device=device)
    gating = torch.randn(32, 64, dtype=torch.float16, device=device) * 4

    fused_cases: list[dict[str, Any]] = []
    candidate_failures: list[str] = []
    for renormalize in (False, True):
        base_weights, base_ids = base["fused_topk"](hidden, gating, 8, renormalize)
        head_weights, head_ids = head["fused_topk"](hidden, gating, 8, renormalize)
        base_sums = base_weights.sum(dim=-1)
        head_sums = head_weights.sum(dim=-1)
        base_sum_error = _max_error(base_sums, torch.ones_like(base_sums))
        head_sum_error = _max_error(head_sums, torch.ones_like(head_sums))
        ids_equal = bool(torch.equal(base_ids, head_ids))
        status = "pass"
        if renormalize and head_sum_error > 1e-5:
            status = "fail"
            candidate_failures.append("CALLER_CONTRACT_PARAMETER_IGNORED:renormalize")
        fused_cases.append(
            {
                "renormalize": renormalize,
                "status": status,
                "base_sum_to_one_max_error": base_sum_error,
                "head_sum_to_one_max_error": head_sum_error,
                "base_head_ids_equal": ids_equal,
            }
        )

    grouped_cases: list[dict[str, Any]] = []
    for scoring_func in ("softmax", "sigmoid"):
        kwargs = {
            "topk": 4,
            "renormalize": True,
            "num_expert_group": 4,
            "topk_group": 2,
            "scoring_func": scoring_func,
        }
        base_weights, base_ids = base["grouped_topk"](hidden, gating, **kwargs)
        head_weights, head_ids = head["grouped_topk"](hidden, gating, **kwargs)
        reference_scores = (
            torch.softmax(gating.double(), dim=-1)
            if scoring_func == "softmax"
            else gating.double().sigmoid()
        )
        group_scores = reference_scores.view(32, 4, -1).max(dim=-1).values
        groups = torch.topk(group_scores, k=2, dim=-1, sorted=False).indices
        mask = torch.zeros_like(group_scores).scatter(1, groups, 1).bool()
        mask = mask.unsqueeze(-1).expand(32, 4, 16).reshape(32, 64)
        selected_scores = reference_scores.masked_fill(~mask, float("-inf"))
        reference_weights, reference_ids = torch.topk(selected_scores, k=4, dim=-1, sorted=False)
        reference_weights /= reference_weights.sum(dim=-1, keepdim=True)
        base_map = {
            int(index): float(weight)
            for index, weight in zip(base_ids[0].tolist(), base_weights[0].tolist(), strict=True)
        }
        head_map = {
            int(index): float(weight)
            for index, weight in zip(head_ids[0].tolist(), head_weights[0].tolist(), strict=True)
        }
        reference_map = {
            int(index): float(weight)
            for index, weight in zip(
                reference_ids[0].tolist(), reference_weights[0].tolist(), strict=True
            )
        }
        base_error = max(
            abs(base_map.get(key, 0.0) - value) for key, value in reference_map.items()
        )
        head_error = max(
            abs(head_map.get(key, 0.0) - value) for key, value in reference_map.items()
        )
        grouped_cases.append(
            {
                "scoring_func": scoring_func,
                "base_first_row_max_error_vs_fp64": base_error,
                "head_first_row_max_error_vs_fp64": head_error,
                "head_not_worse": head_error <= base_error + 1e-7,
                "status": "pass" if head_error <= base_error + 1e-7 else "fail",
            }
        )

    cold_started = time.perf_counter()
    head["fused_topk"](hidden, gating, 8, True)
    _sync()
    cold_seconds = time.perf_counter() - cold_started
    steady_seconds = _time_calls(
        lambda: head["fused_topk"](hidden, gating, 8, True), args.iterations
    )
    payload = {
        "schema_version": "0.5",
        "probe": "vllm-deepseek-router-contract-v1",
        "case_id": "vllm-pr-14027",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "compilation_path": "not-required",
        "compile_seconds": 0.0,
        "precompile_seconds": 0.0,
        "cold_start_seconds": cold_seconds,
        "steady_state_seconds_per_call": steady_seconds,
        "steady_state_compile_seconds": 0.0,
        "fused_topk_cases": fused_cases,
        "grouped_topk_cases": grouped_cases,
        "candidate_failure_codes": sorted(set(candidate_failures)),
        "status": "fail" if candidate_failures else "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if candidate_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
