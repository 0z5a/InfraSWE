from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from infraswe.models.score import (
    CellArtifactScore,
    CellEfficiencyScore,
    DeployabilityScore,
    EvidenceGrade,
    ScoreComponent,
    ScoreResult,
)

DEPLOYABILITY_WEIGHTS = {
    "concurrent_stability": 0.45,
    "kernel_reuse": 0.30,
    "maintainability": 0.25,
}
DEPLOYABILITY_FLOORS = {
    "concurrent_stability": 0.60,
    "kernel_reuse": 0.55,
    "maintainability": 0.50,
}
LOAD_CELL_WEIGHTS = {
    "light": 0.10,
    "normal": 0.20,
    "knee": 0.25,
    "saturation": 0.25,
    "overload": 0.10,
    "burst_or_soak": 0.10,
}
CELL_TEMPLATE_WEIGHTS = {
    "cell-artifact-mixed-v0.4": {
        "concurrent_stability": 0.35,
        "kernel_reuse": 0.20,
        "maintainability": 0.20,
        "sol_efficiency": 0.15,
        "memory_band_efficiency": 0.10,
    },
    "cell-artifact-memory-v0.4": {
        "concurrent_stability": 0.30,
        "kernel_reuse": 0.20,
        "maintainability": 0.20,
        "sol_efficiency": 0.10,
        "memory_band_efficiency": 0.20,
    },
    "cell-artifact-compute-v0.4": {
        "concurrent_stability": 0.30,
        "kernel_reuse": 0.20,
        "maintainability": 0.20,
        "sol_efficiency": 0.25,
        "memory_band_efficiency": 0.05,
    },
    "cell-artifact-distributed-v0.4": {
        "concurrent_stability": 0.35,
        "kernel_reuse": 0.15,
        "maintainability": 0.15,
        "communication_sol": 0.20,
        "memory_or_link_band_efficiency": 0.15,
    },
}
EVIDENCE_GRADE_ORDER: dict[str, int] = {
    "legacy-framework-trace": 0,
    "E0-runtime": 1,
    "E1-framework": 2,
    "E2-system-trace": 3,
    "E3-kernel-counter": 4,
    "E4-sealed": 5,
}


@dataclass(frozen=True)
class DimensionResult:
    component: ScoreComponent
    failure_codes: tuple[str, ...] = ()
    raw_metrics: Mapping[str, Any] | None = None


def _unit(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def weighted_geometric(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    if set(values) != set(weights):
        raise ValueError("component values must exactly match the frozen formula template")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("formula weights must sum to one")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("formula weights cannot be negative")
    checked = {name: _unit(value, name) for name, value in values.items()}
    if any(value == 0 for value in checked.values()):
        return 0.0
    return math.exp(sum(weights[name] * math.log(value) for name, value in checked.items()))


def _component(
    *,
    status: Literal["scored", "not_applicable", "unresolved", "diagnostic"],
    value: float | None,
    formula: str,
    digests: Sequence[str] = (),
    confidence: Literal["low", "medium", "high", "not_applicable"] = "high",
    reason: str | None = None,
) -> ScoreComponent:
    return ScoreComponent(
        status=status,
        value=value,
        formula_version=formula,
        input_evidence_digests=list(digests),
        confidence=confidence,
        reason=reason,
    )


def score_concurrent_stability(
    load_cells: Sequence[Mapping[str, Any]],
    *,
    fresh_process_replays: int,
    evidence_digests: Sequence[str] = (),
) -> DimensionResult:
    """Score a frozen normalized load ladder without comparing absolute QPS."""

    if not load_cells:
        return DimensionResult(
            _component(
                status="unresolved",
                value=None,
                formula="concurrent-stability-v0.4",
                digests=evidence_digests,
                confidence="low",
                reason="no load-cell evidence",
            ),
            ("CONCURRENCY_LOAD_CELLS_MISSING",),
        )
    observed_regimes = {str(cell["regime"]) for cell in load_cells}
    expected_regimes = set(LOAD_CELL_WEIGHTS)
    if observed_regimes != expected_regimes:
        raise ValueError("load regimes must exactly match the frozen v0.4 load ladder")

    cell_scores: dict[str, float] = {}
    failures: list[str] = []
    exploratory_tail: list[str] = []
    for cell in load_cells:
        regime = str(cell["regime"])
        completed = int(cell.get("completed_requests", 0))
        if completed < 1000:
            exploratory_tail.append(regime)
        hard_failures = {
            "deadlock": "CONCURRENCY_DEADLOCK",
            "livelock": "CONCURRENCY_LIVELOCK",
            "queue_unbounded": "CONCURRENCY_QUEUE_GROWTH",
            "memory_growth_limit_exceeded": "CONCURRENCY_MEMORY_GROWTH",
            "silent_fallback": "CONCURRENCY_SILENT_FALLBACK",
            "error_drop_rate_above_max": "CONCURRENCY_ERROR_DROP",
        }
        cell_failed = False
        for field, code in hard_failures.items():
            if bool(cell.get(field, False)):
                failures.append(f"{code}:{regime}")
                cell_failed = True
        factors = {
            "goodput": _unit(cell["slo_goodput_ratio"], f"{regime}.slo_goodput_ratio"),
            "tail": _unit(cell["tail_score"], f"{regime}.tail_score"),
            "jitter": _unit(cell["replay_jitter_score"], f"{regime}.replay_jitter_score"),
            "resource": _unit(
                cell["resource_stability_score"], f"{regime}.resource_stability_score"
            ),
            "fairness": _unit(cell["fairness_score"], f"{regime}.fairness_score"),
        }
        cell_scores[regime] = (
            0.0
            if cell_failed
            else weighted_geometric(
                factors,
                {"goodput": 0.35, "tail": 0.25, "jitter": 0.15, "resource": 0.15, "fairness": 0.10},
            )
        )
    estimate = weighted_geometric(cell_scores, LOAD_CELL_WEIGHTS)
    raw = {
        "fresh_process_replays": fresh_process_replays,
        "load_cell_scores": cell_scores,
        "diagnostic_estimate": estimate,
        "p99_exploratory_regimes": exploratory_tail,
    }
    if fresh_process_replays < 5:
        return DimensionResult(
            _component(
                status="diagnostic",
                value=None,
                formula="concurrent-stability-v0.4",
                digests=evidence_digests,
                confidence="low",
                reason="at least five fresh-process replays are required for an official score",
            ),
            tuple(sorted({*failures, "CONCURRENCY_REPLAY_COUNT_DIAGNOSTIC"})),
            raw,
        )
    return DimensionResult(
        _component(
            status="scored",
            value=estimate,
            formula="concurrent-stability-v0.4",
            digests=evidence_digests,
            confidence="high" if fresh_process_replays >= 7 else "medium",
        ),
        tuple(sorted(set(failures))),
        raw,
    )


def score_kernel_reuse(
    *,
    coverage: float,
    observed_variants: int,
    expected_variant_budget: int,
    max_variant_budget: int,
    compile_reuse: float,
    port_reuse: float,
    silent_fallback_rate: float = 0.0,
    evidence_digests: Sequence[str] = (),
) -> DimensionResult:
    if expected_variant_budget < 1 or max_variant_budget < expected_variant_budget:
        raise ValueError("variant budgets are invalid")
    if observed_variants < 0:
        raise ValueError("observed_variants cannot be negative")
    if observed_variants <= expected_variant_budget:
        variant = 1.0
    elif observed_variants >= max_variant_budget:
        variant = 0.0
    else:
        variant = (max_variant_budget - observed_variants) / (
            max_variant_budget - expected_variant_budget
        )
    failures: list[str] = []
    if observed_variants > max_variant_budget:
        failures.append("REUSE_VARIANT_BUDGET_EXCEEDED")
    if silent_fallback_rate > 0:
        failures.append("REUSE_SILENT_FALLBACK")
    values = {
        "coverage": 0.0 if silent_fallback_rate > 0 else _unit(coverage, "coverage"),
        "variant": variant,
        "compile": _unit(compile_reuse, "compile_reuse"),
        "port": _unit(port_reuse, "port_reuse"),
    }
    score = weighted_geometric(
        values,
        {"coverage": 0.45, "variant": 0.20, "compile": 0.20, "port": 0.15},
    )
    return DimensionResult(
        _component(
            status="scored",
            value=score,
            formula="kernel-reuse-v0.4",
            digests=evidence_digests,
        ),
        tuple(failures),
        {"components": values, "observed_variants": observed_variants},
    )


def score_maintainability(
    *,
    contract: float,
    locality: float,
    tests: float,
    build: float,
    evidence_digests: Sequence[str] = (),
) -> DimensionResult:
    values = {
        "contract": _unit(contract, "contract"),
        "locality": _unit(locality, "locality"),
        "tests": _unit(tests, "tests"),
        "build": _unit(build, "build"),
    }
    score = weighted_geometric(
        values,
        {"contract": 0.30, "locality": 0.20, "tests": 0.30, "build": 0.20},
    )
    return DimensionResult(
        _component(
            status="scored",
            value=score,
            formula="maintainability-v0.4",
            digests=evidence_digests,
        ),
        raw_metrics={"components": values},
    )


def _empty_efficiency_component(status: str, reason: str) -> ScoreComponent:
    confidence = "not_applicable" if status == "not_applicable" else "low"
    return _component(
        status=status,  # type: ignore[arg-type]
        value=None,
        formula="cell-efficiency-v0.4",
        confidence=confidence,  # type: ignore[arg-type]
        reason=reason,
    )


def score_cell_efficiency(
    *,
    work_model_id: str,
    regime: str,
    work_model: Mapping[str, Any],
    calibration: Mapping[str, Any],
    candidate_time_seconds: float,
    actual_memory_bytes: float | None,
    traffic_amplification_budget: float,
    counter_evidence_available: bool,
    counter_confidence: Literal["low", "medium", "high"] = "high",
    beyond_sol_tolerance: float = 0.03,
    evidence_digests: Sequence[str] = (),
) -> CellEfficiencyScore:
    if regime == "utility/no-efficiency-score":
        empty = _empty_efficiency_component("not_applicable", "utility workload")
        return CellEfficiencyScore(
            status="not_applicable",
            work_model_id=work_model_id,
            sol_efficiency=empty,
            useful_memory_band_efficiency=empty,
            physical_memory_band_efficiency=empty,
            traffic_amplification=empty,
            counter_confidence="not_applicable",
        )
    if not counter_evidence_available:
        empty = _empty_efficiency_component(
            "unresolved", "E3 kernel-counter evidence is unavailable"
        )
        return CellEfficiencyScore(
            status="unresolved",
            work_model_id=work_model_id,
            sol_efficiency=empty,
            useful_memory_band_efficiency=empty,
            physical_memory_band_efficiency=empty,
            traffic_amplification=empty,
            counter_confidence="low",
        )
    if candidate_time_seconds <= 0:
        raise ValueError("candidate_time_seconds must be positive")
    minimum_bytes = float(work_model.get("minimum_external_bytes", 0.0))
    semantic_flops = float(work_model.get("semantic_flops", 0.0))
    launch_seconds = float(calibration.get("launch_floor_us", 0.0)) * 1e-6
    compute_tflops = float(calibration.get("compute_tflops", 0.0))
    memory_gbps = float(calibration.get("memory_bandwidth_gbps", 0.0))
    link_bytes = float(work_model.get("minimum_link_bytes", 0.0))
    link_gbps = float(calibration.get("link_bandwidth_gbps", 0.0))
    lower_bounds = [launch_seconds]
    if semantic_flops > 0 and compute_tflops > 0:
        lower_bounds.append(semantic_flops / (compute_tflops * 1e12))
    if minimum_bytes > 0 and memory_gbps > 0:
        lower_bounds.append(minimum_bytes / (memory_gbps * 1e9))
    if link_bytes > 0 and link_gbps > 0:
        lower_bounds.append(link_bytes / (link_gbps * 1e9))
    if not lower_bounds or max(lower_bounds) <= 0:
        raise ValueError("calibration does not define an applicable SOL lower bound")
    sol_raw = max(lower_bounds) / candidate_time_seconds
    sol_quarantined = sol_raw > 1 + beyond_sol_tolerance
    sol_component = _component(
        status="unresolved" if sol_quarantined else "scored",
        value=None if sol_quarantined else min(1.0, sol_raw),
        formula="sol-efficiency-v0.4",
        digests=evidence_digests,
        confidence=counter_confidence,
        reason="observed efficiency exceeds calibrated SOL tolerance" if sol_quarantined else None,
    )
    if minimum_bytes <= 0 or memory_gbps <= 0 or actual_memory_bytes is None:
        memory_component = _empty_efficiency_component(
            "unresolved", "memory work model or physical traffic counter is missing"
        )
        physical_component = _empty_efficiency_component(
            "unresolved", "physical traffic counter is missing"
        )
        amplification_component = _empty_efficiency_component(
            "unresolved", "physical traffic counter is missing"
        )
        status = "unresolved"
        raw = {"sol_efficiency_raw": sol_raw}
    else:
        if actual_memory_bytes < minimum_bytes:
            raise ValueError("actual_memory_bytes cannot be below semantic minimum bytes")
        useful_ratio = (minimum_bytes / candidate_time_seconds) / (memory_gbps * 1e9)
        physical_ratio = (actual_memory_bytes / candidate_time_seconds) / (memory_gbps * 1e9)
        amplification = actual_memory_bytes / minimum_bytes
        amplification_score = min(1.0, traffic_amplification_budget / amplification)
        memory_score = math.sqrt(min(1.0, useful_ratio) * amplification_score)
        memory_component = _component(
            status="scored",
            value=memory_score,
            formula="memory-band-efficiency-v0.4",
            digests=evidence_digests,
            confidence=counter_confidence,
        )
        physical_component = _component(
            status="scored",
            value=min(1.0, physical_ratio),
            formula="physical-memory-band-efficiency-v0.4",
            digests=evidence_digests,
            confidence=counter_confidence,
        )
        amplification_component = _component(
            status="scored",
            value=amplification_score,
            formula="traffic-amplification-v0.4",
            digests=evidence_digests,
            confidence=counter_confidence,
        )
        status = (
            "unresolved"
            if sol_quarantined
            else ("diagnostic" if counter_confidence == "low" else "scored")
        )
        raw = {
            "sol_efficiency_raw": sol_raw,
            "useful_memory_band_efficiency_raw": useful_ratio,
            "physical_memory_band_efficiency_raw": physical_ratio,
            "traffic_amplification_raw": amplification,
            "minimum_external_bytes": minimum_bytes,
            "actual_memory_bytes": actual_memory_bytes,
            "candidate_time_seconds": candidate_time_seconds,
        }
    return CellEfficiencyScore(
        status=status,
        work_model_id=work_model_id,
        sol_efficiency=sol_component,
        useful_memory_band_efficiency=memory_component,
        physical_memory_band_efficiency=physical_component,
        traffic_amplification=amplification_component,
        raw=raw,
        counter_confidence=counter_confidence,
    )


def _grade_at_least(observed: EvidenceGrade, required: str) -> bool:
    return EVIDENCE_GRADE_ORDER[observed] >= EVIDENCE_GRADE_ORDER[required]


def build_v04_score(
    *,
    hard_gate_status: Literal["pass", "fail", "unresolved"],
    benchmark_cell_id: str,
    evidence_grade: EvidenceGrade,
    concurrent_stability: DimensionResult,
    kernel_reuse: DimensionResult,
    maintainability: DimensionResult,
    cell_efficiency: CellEfficiencyScore | None = None,
    cell_artifact_template: str | None = None,
    raw_metrics: Mapping[str, Any] | None = None,
    additional_failure_codes: Sequence[str] = (),
) -> ScoreResult:
    dimensions = {
        "concurrent_stability": concurrent_stability.component,
        "kernel_reuse": kernel_reuse.component,
        "maintainability": maintainability.component,
    }
    failures = sorted(
        {
            *concurrent_stability.failure_codes,
            *kernel_reuse.failure_codes,
            *maintainability.failure_codes,
            *additional_failure_codes,
        }
    )
    if hard_gate_status == "fail":
        deployability = DeployabilityScore(
            status="unresolved",
            score_100=None,
            components=dimensions,
            component_floors=DEPLOYABILITY_FLOORS,
        )
        return ScoreResult(
            schema_version="0.4",
            infra_cert="fail",
            disposition="invalid",
            benchmark_cell_id=benchmark_cell_id,
            evidence_grade=evidence_grade,
            deployability=deployability,
            leaderboard_effective_deployability_100=0.0,
            cell_efficiency=cell_efficiency,
            raw_metrics=dict(raw_metrics or {}),
            failure_codes=failures,
        )
    official_components = all(component.status == "scored" for component in dimensions.values())
    evidence_sufficient = _grade_at_least(evidence_grade, "E2-system-trace")
    if hard_gate_status == "unresolved" or not official_components or not evidence_sufficient:
        if not evidence_sufficient:
            failures.append("DEPLOYABILITY_EVIDENCE_GRADE_BELOW_E2")
        deployability = DeployabilityScore(
            status="unresolved",
            score_100=None,
            components=dimensions,
            component_floors=DEPLOYABILITY_FLOORS,
        )
        return ScoreResult(
            schema_version="0.4",
            infra_cert="unresolved" if hard_gate_status == "unresolved" else "pass",
            disposition="partial",
            benchmark_cell_id=benchmark_cell_id,
            evidence_grade=evidence_grade,
            deployability=deployability,
            leaderboard_effective_deployability_100=None,
            cell_efficiency=cell_efficiency,
            raw_metrics=dict(raw_metrics or {}),
            failure_codes=sorted(set(failures)),
        )
    numeric = {name: float(component.value) for name, component in dimensions.items()}
    score_100 = 100 * weighted_geometric(numeric, DEPLOYABILITY_WEIGHTS)
    below_floor = [name for name, floor in DEPLOYABILITY_FLOORS.items() if numeric[name] < floor]
    deployability_status = "not_deployable" if below_floor else "scored"
    for name in below_floor:
        failures.append(f"DEPLOYABILITY_COMPONENT_FLOOR:{name}")
    deployability = DeployabilityScore(
        status=deployability_status,
        score_100=score_100,
        components=dimensions,
        component_floors=DEPLOYABILITY_FLOORS,
    )

    cell_artifact = None
    if cell_artifact_template:
        if cell_artifact_template not in CELL_TEMPLATE_WEIGHTS:
            raise ValueError(f"unknown frozen cell artifact template: {cell_artifact_template}")
        template = CELL_TEMPLATE_WEIGHTS[cell_artifact_template]
        if cell_artifact_template == "cell-artifact-distributed-v0.4":
            cell_artifact = CellArtifactScore(
                status="unresolved",
                formula_template_id=cell_artifact_template,
                components=dimensions,
            )
            failures.append("DISTRIBUTED_EFFICIENCY_COMPONENTS_UNRESOLVED")
        elif cell_efficiency and cell_efficiency.status == "scored":
            cell_components = {
                **dimensions,
                "sol_efficiency": cell_efficiency.sol_efficiency,
                "memory_band_efficiency": cell_efficiency.useful_memory_band_efficiency,
            }
            cell_values = {
                name: float(component.value) for name, component in cell_components.items()
            }
            cell_score = 100 * weighted_geometric(cell_values, template)
            cell_artifact = CellArtifactScore(
                status="not_deployable" if below_floor else "scored",
                formula_template_id=cell_artifact_template,
                score_100=cell_score,
                components=cell_components,
            )
        else:
            cell_artifact = CellArtifactScore(
                status="unresolved",
                formula_template_id=cell_artifact_template,
                components=dimensions,
            )
    return ScoreResult(
        schema_version="0.4",
        infra_cert="pass",
        disposition="valid" if deployability_status == "scored" else "partial",
        benchmark_cell_id=benchmark_cell_id,
        evidence_grade=evidence_grade,
        deployability=deployability,
        leaderboard_effective_deployability_100=(
            score_100 if deployability_status == "scored" else 0.0
        ),
        cell_efficiency=cell_efficiency,
        cell_artifact=cell_artifact,
        raw_metrics=dict(raw_metrics or {}),
        failure_codes=sorted(set(failures)),
    )
