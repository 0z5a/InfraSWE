#!/usr/bin/env python3
"""Measure metadata selection separately from explicitly selected precompile planning."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from infraswe.draft.candidate_registry import (
    build_default_candidate_registry,
    plan_candidate_activation,
    resolve_default_candidates,
)
from infraswe.draft.defaults import build_default_catalog
from infraswe.draft.resolver import resolve_draft
from infraswe.models.candidates import (
    CandidateSelectionRequest,
    DefaultCandidateRegistry,
)
from infraswe.models.draft import DraftCandidate

SELECTION_P95_BUDGET_SECONDS = 0.001
ACTIVATION_P95_BUDGET_SECONDS = 0.0005
DEFAULT_DRAFT_P95_BUDGET_SECONDS = 0.003
CANDIDATE_MODULE_ROOTS = {
    "aiter",
    "deep_gemm",
    "flash_attn",
    "flashinfer",
    "liger_kernel",
    "sglang",
    "transformer_engine",
    "vllm",
    "xformers",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _expanded_registry(
    registry: DefaultCandidateRegistry, extra_candidates: int
) -> DefaultCandidateRegistry:
    payload = registry.model_dump(mode="json")
    template = registry.candidates["torch_eager"]
    for index in range(extra_candidates):
        candidate_id = f"synthetic_unused_{index:06d}"
        payload["candidates"][candidate_id] = template.model_copy(
            update={"id": candidate_id, "display_name": f"Synthetic unused {index}"}
        ).model_dump(mode="json")
    payload["registry_version"] = f"{registry.registry_version}+{extra_candidates}-unused"
    return DefaultCandidateRegistry.model_validate(payload)


def _measure_registry(
    registry: DefaultCandidateRegistry,
    requests: list[CandidateSelectionRequest],
    iterations: int,
) -> dict[str, object]:
    # Materialize the registry digest once. The measured loop must not serialize the full pool.
    resolve_default_candidates(requests[0], registry=registry)
    selection_times: list[float] = []
    activation_times: list[float] = []
    last_resolution = None
    for index in range(iterations):
        request = requests[index % len(requests)]
        started = time.perf_counter_ns()
        resolution = resolve_default_candidates(request, registry=registry)
        selection_times.append((time.perf_counter_ns() - started) / 1e9)
        started = time.perf_counter_ns()
        plan_candidate_activation(resolution, registry=registry)
        activation_times.append((time.perf_counter_ns() - started) / 1e9)
        last_resolution = resolution
    assert last_resolution is not None
    default_plan = plan_candidate_activation(last_resolution, registry=registry)
    return {
        "candidate_count": len(registry.candidates),
        "selection_seconds": _summarize(selection_times),
        "activation_plan_seconds": _summarize(activation_times),
        "selection_compile_seconds": default_plan.selection_compile_seconds,
        "default_activated_candidate_count": len(default_plan.actions),
        "inactive_candidate_count": default_plan.inactive_candidate_count,
        "timed_benchmark_started": default_plan.timed_benchmark_started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--synthetic-extra-candidates", type=int, default=0)
    parser.add_argument("--enforce-budgets", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.synthetic_extra_candidates < 0:
        parser.error("--synthetic-extra-candidates cannot be negative")

    build_default_candidate_registry.cache_clear()
    build_default_catalog.cache_clear()
    cold_started = time.perf_counter()
    registry = build_default_candidate_registry()
    registry_load_seconds = time.perf_counter() - cold_started
    requests = [
        CandidateSelectionRequest(
            operator_family="attention-inference-decode",
            phase="inference",
            backend="cuda",
            requested_primary_host="vllm",
        ),
        CandidateSelectionRequest(
            operator_family="grouped-moe-gemm",
            phase="inference",
            backend="cuda",
            requested_primary_host="sglang",
        ),
        CandidateSelectionRequest(
            operator_family="training-fused-ops",
            phase="training",
            backend="triton",
            requested_primary_host="torchtitan",
        ),
        CandidateSelectionRequest(
            operator_family="generic",
            phase="generic",
            backend="rocm",
        ),
    ]
    modules_before = set(sys.modules)
    base_measurement = _measure_registry(registry, requests, args.iterations)
    modules_after = set(sys.modules)
    imported_candidate_modules = sorted(
        module
        for module in modules_after - modules_before
        if module.partition(".")[0] in CANDIDATE_MODULE_ROOTS
    )

    draft_candidate = DraftCandidate(
        kind="generated",
        revision="sha256:" + "a" * 64,
        intent="integrate",
        implementation_kind="cuda-native",
        entrypoints=["flashinfer.decode"],
    )
    cold_started = time.perf_counter_ns()
    resolve_draft(candidate=draft_candidate)
    default_draft_cold_seconds = (time.perf_counter_ns() - cold_started) / 1e9
    draft_iterations = min(args.iterations, 10_000)
    draft_times: list[float] = []
    for _ in range(draft_iterations):
        started = time.perf_counter_ns()
        resolve_draft(candidate=draft_candidate)
        draft_times.append((time.perf_counter_ns() - started) / 1e9)

    scaled_measurement = None
    if args.synthetic_extra_candidates:
        scaled_registry = _expanded_registry(registry, args.synthetic_extra_candidates)
        scaled_measurement = _measure_registry(scaled_registry, requests, args.iterations)

    base_selection = base_measurement["selection_seconds"]
    base_activation = base_measurement["activation_plan_seconds"]
    assert isinstance(base_selection, dict)
    assert isinstance(base_activation, dict)
    default_draft_summary = _summarize(draft_times)
    violations: list[str] = []
    if base_selection["p95"] > SELECTION_P95_BUDGET_SECONDS:
        violations.append("BASE_SELECTION_P95_EXCEEDED")
    if base_activation["p95"] > ACTIVATION_P95_BUDGET_SECONDS:
        violations.append("BASE_ACTIVATION_P95_EXCEEDED")
    if default_draft_summary["p95"] > DEFAULT_DRAFT_P95_BUDGET_SECONDS:
        violations.append("DEFAULT_DRAFT_P95_EXCEEDED")
    if imported_candidate_modules:
        violations.append("CANDIDATE_IMPORT_DURING_SELECTION")
    if base_measurement["default_activated_candidate_count"] != 1:
        violations.append("DEFAULT_ACTIVATION_NOT_SINGLE")
    if scaled_measurement is not None:
        scaled_selection = scaled_measurement["selection_seconds"]
        scaled_activation = scaled_measurement["activation_plan_seconds"]
        assert isinstance(scaled_selection, dict)
        assert isinstance(scaled_activation, dict)
        if scaled_selection["p95"] > SELECTION_P95_BUDGET_SECONDS:
            violations.append("SCALED_SELECTION_P95_EXCEEDED")
        if scaled_activation["p95"] > ACTIVATION_P95_BUDGET_SECONDS:
            violations.append("SCALED_ACTIVATION_P95_EXCEEDED")
    payload = {
        "schema_version": "0.5",
        "benchmark": "default-candidate-metadata-resolution-v2",
        "iterations": args.iterations,
        "rule_count": len(registry.rules),
        "registry_load_seconds": registry_load_seconds,
        "base_registry": base_measurement,
        "scaled_registry": scaled_measurement,
        "default_draft_resolution": {
            "cold_seconds": default_draft_cold_seconds,
            "iterations": draft_iterations,
            "steady_seconds": default_draft_summary,
        },
        "candidate_imports_during_selection": imported_candidate_modules,
        "budgets_seconds": {
            "selection_p95": SELECTION_P95_BUDGET_SECONDS,
            "activation_p95": ACTIVATION_P95_BUDGET_SECONDS,
            "default_draft_p95": DEFAULT_DRAFT_P95_BUDGET_SECONDS,
        },
        "budget_passed": not violations,
        "budget_violations": violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if args.enforce_budgets and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
