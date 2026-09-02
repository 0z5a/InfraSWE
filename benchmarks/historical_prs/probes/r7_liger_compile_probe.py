#!/usr/bin/env python3
"""H100 torch.compile and eager timing probe for Liger PR 1328."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _loss_fn(module: ModuleType) -> Callable[..., torch.Tensor]:
    function = module.LigerFusedLinearCrossEntropyFunction

    def loss_fn(x: torch.Tensor, weight: torch.Tensor, target: torch.Tensor):
        loss, _, _, _ = function.apply(
            x,
            weight,
            target,
            None,
            None,
            -100,
            0.0,
            0.0,
            "mean",
            None,
            False,
            torch.float32,
            False,
            False,
            False,
        )
        return loss

    return loss_fn


def _inputs(rows: int, hidden: int, vocab: int, dtype: torch.dtype, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype, generator=generator)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=dtype, generator=generator)
    target = torch.randint(0, vocab, (rows,), device="cuda", generator=generator)
    return x, weight, target


def _invoke(
    function: Callable[..., torch.Tensor],
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    x = source[0].clone().requires_grad_(True)
    weight = source[1].clone().requires_grad_(True)
    target = source[2]
    loss = function(x, weight, target)
    loss.backward()
    torch.cuda.synchronize()
    return {
        "loss": float(loss.detach()),
        "input_grad": x.grad.detach().clone(),
        "weight_grad": weight.grad.detach().clone(),
    }


def _reference(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    x = source[0].clone().requires_grad_(True)
    weight = source[1].clone().requires_grad_(True)
    target = source[2]
    logits = torch.nn.functional.linear(x, weight).float()
    loss = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    loss.backward()
    torch.cuda.synchronize()
    return {
        "loss": float(loss.detach()),
        "input_grad": x.grad.detach().clone(),
        "weight_grad": weight.grad.detach().clone(),
    }


def _error(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["loss_abs"] = abs(actual["loss"] - expected["loss"])
    for name in ("input_grad", "weight_grad"):
        delta = (actual[name].float() - expected[name].float()).abs()
        result[f"{name}_max_abs"] = float(delta.max())
        result[f"{name}_allclose"] = torch.allclose(
            actual[name].float(), expected[name].float(), atol=1e-3, rtol=1e-2
        )
    result["all_pass"] = (
        result["loss_abs"] <= 1e-3
        and result["input_grad_allclose"]
        and result["weight_grad_allclose"]
    )
    return result


def _counter_snapshot() -> dict[str, dict[str, int]]:
    from torch._dynamo.utils import counters

    return {
        group: {key: int(value) for key, value in values.items()}
        for group, values in counters.items()
        if values
    }


def _compile_case(
    module: ModuleType,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    *,
    fullgraph: bool,
    seed: int,
) -> dict[str, Any]:
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    source = _inputs(*shape, dtype, seed)
    oracle = _reference(source)
    eager = _invoke(_loss_fn(module), source)
    eager_vs_oracle = _error(eager, oracle)
    compiled = torch.compile(_loss_fn(module), fullgraph=fullgraph)
    started = time.perf_counter()
    try:
        actual = _invoke(compiled, source)
    except Exception as exc:
        torch._dynamo.reset()
        return {
            "status": "fail",
            "shape": list(shape),
            "dtype": str(dtype),
            "fullgraph": fullgraph,
            "duration_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc)[-6000:],
            "eager_vs_oracle": eager_vs_oracle,
            "dynamo_counters": _counter_snapshot(),
        }
    first_duration = time.perf_counter() - started
    counters_after_first = _counter_snapshot()
    cache_root = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor"))
    before_files = (
        {str(path.relative_to(cache_root)) for path in cache_root.rglob("*") if path.is_file()}
        if cache_root.exists()
        else set()
    )
    for _ in range(5):
        _invoke(compiled, source)
    after_files = (
        {str(path.relative_to(cache_root)) for path in cache_root.rglob("*") if path.is_file()}
        if cache_root.exists()
        else set()
    )
    counters_after_steady = _counter_snapshot()
    return {
        "status": "pass",
        "shape": list(shape),
        "dtype": str(dtype),
        "fullgraph": fullgraph,
        "first_compile_and_run_seconds": first_duration,
        "error": _error(actual, eager),
        "eager_vs_oracle": eager_vs_oracle,
        "compiled_vs_oracle": _error(actual, oracle),
        "dynamo_counters_after_first": counters_after_first,
        "dynamo_counters_after_steady": counters_after_steady,
        "new_cache_files_during_steady": sorted(after_files - before_files),
        "steady_new_cache_file_count": len(after_files - before_files),
    }


def _timed_call(
    function: Callable[..., torch.Tensor],
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    iterations: int,
) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        _invoke(function, source)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e6 / iterations


def _paired_eager_timing(base: ModuleType, head: ModuleType) -> dict[str, Any]:
    source = _inputs(128, 256, 512, torch.bfloat16, 9017)
    base_fn = _loss_fn(base)
    head_fn = _loss_fn(head)
    for _ in range(8):
        _invoke(base_fn, source)
        _invoke(head_fn, source)
    pairs: list[dict[str, float]] = []
    for pair in range(7):
        order = (("base", base_fn), ("head", head_fn))
        if pair % 2:
            order = tuple(reversed(order))
        values: dict[str, float] = {}
        for name, function in order:
            values[name] = _timed_call(function, source, 64)
        pairs.append(values)
    base_values = [item["base"] for item in pairs]
    head_values = [item["head"] for item in pairs]
    base_median = statistics.median(base_values)
    head_median = statistics.median(head_values)
    return {
        "pairs_us": pairs,
        "base_median_us": base_median,
        "head_median_us": head_median,
        "head_over_base_median_ratio": head_median / base_median,
        "within_three_percent": head_median / base_median <= 1.03,
        "pair_count": 7,
        "iterations_per_measurement": 64,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--selection-sha", required=True)
    parser.add_argument("--test-plan-sha", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    sys.path.insert(0, str(args.head_root / "src"))
    relative = Path("src/liger_kernel/ops/fused_linear_cross_entropy.py")
    base_path = args.base_root / relative
    head_path = args.head_root / relative
    base = _load(base_path, "infraswe_r7_liger_base")
    head = _load(head_path, "infraswe_r7_liger_head")

    target_shape = (8, 16, 32)
    compile_results = {
        "base_default": _compile_case(base, target_shape, torch.bfloat16, fullgraph=False, seed=42),
        "head_default": _compile_case(head, target_shape, torch.bfloat16, fullgraph=False, seed=42),
        "head_boundary_bf16": _compile_case(
            head, (1, 17, 33), torch.bfloat16, fullgraph=False, seed=43
        ),
        "head_boundary_fp16": _compile_case(
            head, (9, 31, 65), torch.float16, fullgraph=False, seed=44
        ),
        "head_fullgraph": _compile_case(
            head, target_shape, torch.bfloat16, fullgraph=True, seed=42
        ),
    }
    timing = _paired_eager_timing(base, head)
    graph_breaks = (
        compile_results["head_default"]
        .get("dynamo_counters_after_steady", {})
        .get("graph_break", {})
    )
    facts = {
        "device": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compile_results": compile_results,
        "base_target_failure_reproduced": compile_results["base_default"]["status"] == "fail",
        "base_failure_mentions_addmm_dtype_out": any(
            token in compile_results["base_default"].get("error", "")
            for token in ("addmm.dtype_out", "tuned_addmm", "out_dtype")
        ),
        "base_eager_matches_oracle": compile_results["base_default"]["eager_vs_oracle"]["all_pass"],
        "head_default_all_cases_pass": all(
            compile_results[name]["status"] == "pass"
            and compile_results[name]["error"]["all_pass"]
            and compile_results[name]["compiled_vs_oracle"]["all_pass"]
            for name in ("head_default", "head_boundary_bf16", "head_boundary_fp16")
        ),
        "head_fullgraph_pass": (
            compile_results["head_fullgraph"]["status"] == "pass"
            and compile_results["head_fullgraph"]["error"]["all_pass"]
            and compile_results["head_fullgraph"]["compiled_vs_oracle"]["all_pass"]
        ),
        "head_default_graph_break_count": sum(graph_breaks.values()),
        "head_default_zero_graph_breaks": not graph_breaks,
        "head_default_zero_steady_compile_artifacts": (
            compile_results["head_default"].get("steady_new_cache_file_count") == 0
        ),
        "eager_timing": timing,
        "source_identity": {
            "base_source_sha256": _digest(base_path.read_text(encoding="utf-8")),
            "head_source_sha256": _digest(head_path.read_text(encoding="utf-8")),
        },
    }
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "probe": "r7-liger-h100-torch-compile-v1",
        "case_id": "liger-pr-1328",
        "project": "liger-kernel",
        "status": "pass",
        "failure_codes": [],
        "facts": facts,
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "selection_lock_sha256": args.selection_sha,
        "test_plan_sha256": args.test_plan_sha,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": _digest(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
