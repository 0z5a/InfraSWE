from __future__ import annotations

import math

import pytest

from infraswe.models.score import ScoreComponent
from infraswe.models.task import TaskPackage
from infraswe.scoring.deployability import (
    build_v04_score,
    score_cell_efficiency,
    score_concurrent_stability,
    score_kernel_reuse,
    score_maintainability,
    weighted_geometric,
)


def v04_task_payload() -> dict:
    return {
        "schema_version": "0.4",
        "task": {
            "id": "rmsnorm-runtime-replacement-v2",
            "title": "RMSNorm production-path replacement",
            "track": "serving-runtime",
            "repository": "fixture",
            "base_commit": "abc",
            "kind": "benchmark-replacement",
            "implementation_level": "integrated",
        },
        "replay": {"count": 7, "require_all": True},
        "semantic_contract": {
            "path": "semantic-contract.json",
            "sha256": "sha256:semantic",
        },
        "backend_profile": {
            "id": "gpu-1x-sm80",
            "adapter": "cuda",
            "benchmark_cell_id": "sha256:cell",
        },
        "certification": {
            "hidden_correctness_required": 1.0,
            "silent_fallback_rate_max": 0.0,
            "fresh_replays": 7,
            "require_all": True,
        },
        "concurrency": {
            "protocol_id": "serving-load-normalized-v1",
            "reference_saturation_anchor": "local-reference",
            "load_ratios": [0.25, 0.5, 0.8, 1.0, 1.2],
            "minimum_completed_requests_per_cell": 1000,
            "burst_or_soak": "required",
            "request_mix_sha256": "sha256:mix",
        },
        "reuse_contract": {
            "sha256": "sha256:reuse",
            "expected_variant_budget": 8,
            "max_variant_budget": 32,
            "require_case_to_implementation_map": True,
            "compile_cache_observability": "required-if-applicable",
        },
        "maintainability": {
            "probe_set_sha256": "sha256:probes",
            "require_capability_contract": True,
            "require_structured_failure_codes": True,
            "build_profiles": ["current-cell", "compatibility-cell"],
        },
        "efficiency": {
            "work_model_id": "rmsnorm-bf16-v2",
            "regime": "memory-bound",
            "work_model_confidence_min": "high",
            "calibration_manifest_sha256": "sha256:calibration",
            "traffic_amplification_budget": 1.15,
        },
        "evidence": {
            "minimum_grade_for_deployability": "E2-system-trace",
            "minimum_grade_for_cell_efficiency": "E3-kernel-counter",
            "collectors": ["runtime", "system-trace", "kernel-counter-final-cases"],
        },
        "scoring": {
            "deployability_template": "deployability-v0.4",
            "cell_artifact_template": "cell-artifact-memory-v0.4",
            "absolute_latency_global_ranking": "forbidden",
            "raw_peak_performance_in_cross_cell_score": False,
        },
    }


def load_cells() -> list[dict]:
    return [
        {
            "regime": regime,
            "completed_requests": 1200,
            "slo_goodput_ratio": 0.9,
            "tail_score": 0.8,
            "replay_jitter_score": 0.95,
            "resource_stability_score": 0.98,
            "fairness_score": 1.0,
        }
        for regime in ("light", "normal", "knee", "saturation", "overload", "burst_or_soak")
    ]


def test_v04_task_contract_is_coherent_and_forbids_post_hoc_formula_changes() -> None:
    task = TaskPackage.model_validate(v04_task_payload())
    assert task.task.kind == "benchmark-replacement"
    assert task.scoring.absolute_latency_global_ranking == "forbidden"

    payload = v04_task_payload()
    payload["scoring"].pop("absolute_latency_global_ranking")
    with pytest.raises(ValueError, match="forbids absolute latency"):
        TaskPackage.model_validate(payload)

    payload = v04_task_payload()
    payload["reuse_contract"]["expected_variant_budget"] = 64
    with pytest.raises(ValueError, match="expected_variant_budget"):
        TaskPackage.model_validate(payload)


def test_three_replays_are_diagnostic_but_five_are_official() -> None:
    diagnostic = score_concurrent_stability(load_cells(), fresh_process_replays=3)
    assert diagnostic.component.status == "diagnostic"
    assert diagnostic.component.value is None
    assert diagnostic.raw_metrics and diagnostic.raw_metrics["diagnostic_estimate"] > 0

    official = score_concurrent_stability(load_cells(), fresh_process_replays=5)
    assert official.component.status == "scored"
    assert official.component.confidence == "medium"
    assert official.component.value and 0 < official.component.value < 1


def test_concurrency_hard_failure_is_not_compensated() -> None:
    cells = load_cells()
    cells[3]["queue_unbounded"] = True
    result = score_concurrent_stability(cells, fresh_process_replays=7)
    assert result.component.value == 0
    assert "CONCURRENCY_QUEUE_GROWTH:saturation" in result.failure_codes


def test_reuse_scores_runtime_variants_and_rejects_silent_fallback() -> None:
    within_budget = score_kernel_reuse(
        coverage=0.9,
        observed_variants=8,
        expected_variant_budget=8,
        max_variant_budget=32,
        compile_reuse=0.8,
        port_reuse=0.7,
    )
    exploded = score_kernel_reuse(
        coverage=0.9,
        observed_variants=40,
        expected_variant_budget=8,
        max_variant_budget=32,
        compile_reuse=0.8,
        port_reuse=0.7,
    )
    fallback = score_kernel_reuse(
        coverage=1.0,
        observed_variants=1,
        expected_variant_budget=8,
        max_variant_budget=32,
        compile_reuse=1.0,
        port_reuse=1.0,
        silent_fallback_rate=0.01,
    )
    assert within_budget.component.value and within_budget.component.value > 0
    assert exploded.component.value == 0
    assert "REUSE_VARIANT_BUDGET_EXCEEDED" in exploded.failure_codes
    assert fallback.component.value == 0
    assert "REUSE_SILENT_FALLBACK" in fallback.failure_codes


def test_memory_efficiency_penalizes_traffic_amplification() -> None:
    common = {
        "work_model_id": "rmsnorm-bf16-v2",
        "regime": "memory-bound",
        "work_model": {"minimum_external_bytes": 1e9, "semantic_flops": 1e9},
        "calibration": {
            "launch_floor_us": 1.0,
            "compute_tflops": 10.0,
            "memory_bandwidth_gbps": 100.0,
        },
        "candidate_time_seconds": 0.02,
        "traffic_amplification_budget": 1.0,
        "counter_evidence_available": True,
    }
    efficient = score_cell_efficiency(actual_memory_bytes=1e9, **common)
    amplified = score_cell_efficiency(actual_memory_bytes=4e9, **common)
    assert efficient.status == "scored"
    assert amplified.status == "scored"
    assert efficient.useful_memory_band_efficiency.value is not None
    assert amplified.useful_memory_band_efficiency.value is not None
    assert (
        amplified.useful_memory_band_efficiency.value
        < efficient.useful_memory_band_efficiency.value
    )
    assert amplified.raw["traffic_amplification_raw"] == 4.0


def test_missing_counter_is_unresolved_not_zero() -> None:
    efficiency = score_cell_efficiency(
        work_model_id="x",
        regime="memory-bound",
        work_model={},
        calibration={},
        candidate_time_seconds=1.0,
        actual_memory_bytes=None,
        traffic_amplification_budget=1.0,
        counter_evidence_available=False,
    )
    assert efficiency.status == "unresolved"
    assert efficiency.sol_efficiency.value is None
    assert efficiency.sol_efficiency.status == "unresolved"


def test_deployability_uses_frozen_geometric_formula_and_component_floors() -> None:
    concurrency = score_concurrent_stability(load_cells(), fresh_process_replays=7)
    reuse = score_kernel_reuse(
        coverage=0.9,
        observed_variants=8,
        expected_variant_budget=8,
        max_variant_budget=32,
        compile_reuse=0.8,
        port_reuse=0.7,
    )
    maintainability = score_maintainability(
        contract=0.9,
        locality=0.8,
        tests=0.85,
        build=0.9,
    )
    result = build_v04_score(
        hard_gate_status="pass",
        benchmark_cell_id="sha256:cell",
        evidence_grade="E2-system-trace",
        concurrent_stability=concurrency,
        kernel_reuse=reuse,
        maintainability=maintainability,
        raw_metrics={"latency_us": {"p50": 10}},
    )
    assert result.deployability and result.deployability.status == "scored"
    expected = 100 * weighted_geometric(
        {
            "concurrent_stability": float(concurrency.component.value),
            "kernel_reuse": float(reuse.component.value),
            "maintainability": float(maintainability.component.value),
        },
        {"concurrent_stability": 0.45, "kernel_reuse": 0.30, "maintainability": 0.25},
    )
    assert math.isclose(result.deployability.score_100 or 0, expected)
    assert result.raw_metrics["latency_us"]["p50"] == 10

    low_maintainability = score_maintainability(
        contract=0.4,
        locality=0.4,
        tests=0.4,
        build=0.4,
    )
    floored = build_v04_score(
        hard_gate_status="pass",
        benchmark_cell_id="sha256:cell",
        evidence_grade="E2-system-trace",
        concurrent_stability=concurrency,
        kernel_reuse=reuse,
        maintainability=low_maintainability,
    )
    assert floored.deployability and floored.deployability.status == "not_deployable"
    assert floored.leaderboard_effective_deployability_100 == 0


def test_failed_certification_nulls_deployability_and_sets_effective_zero() -> None:
    concurrency = score_concurrent_stability(load_cells(), fresh_process_replays=7)
    reuse = score_kernel_reuse(
        coverage=1.0,
        observed_variants=1,
        expected_variant_budget=8,
        max_variant_budget=32,
        compile_reuse=1.0,
        port_reuse=1.0,
    )
    maintainability = score_maintainability(contract=1, locality=1, tests=1, build=1)
    result = build_v04_score(
        hard_gate_status="fail",
        benchmark_cell_id="sha256:cell",
        evidence_grade="E4-sealed",
        concurrent_stability=concurrency,
        kernel_reuse=reuse,
        maintainability=maintainability,
        additional_failure_codes=["SILENT_FALLBACK"],
    )
    assert result.infra_cert == "fail"
    assert result.deployability and result.deployability.score_100 is None
    assert result.leaderboard_effective_deployability_100 == 0


def test_component_status_cannot_smuggle_zero_for_na_or_unresolved() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        ScoreComponent(
            status="not_applicable",
            value=0,
            formula_version="x",
            confidence="not_applicable",
        )
    with pytest.raises(ValueError, match="require a value"):
        ScoreComponent(status="scored", value=None, formula_version="x", confidence="high")
