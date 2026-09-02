from __future__ import annotations

from collections.abc import Mapping, Sequence

from infraswe.models.project_score import ProjectScoreComponent
from infraswe.models.score import CellArtifactScore, ScoreComponent
from infraswe.models.system_paths import (
    SystemPathInfraCertEvidence,
    SystemPathInfraCertResult,
    SystemPathLoadCell,
)
from infraswe.scoring.deployability import DimensionResult, weighted_geometric
from infraswe.scoring.project_fit import ProjectDimensionResult

SYSTEM_PATH_LOAD_WEIGHTS = {
    "light": 0.10,
    "normal": 0.20,
    "knee": 0.25,
    "saturation": 0.25,
    "overload": 0.10,
    "soak": 0.10,
}
SYSTEM_PATH_CELL_WEIGHTS = {
    "goodput": 0.30,
    "tail": 0.20,
    "jitter": 0.15,
    "overlap_progress": 0.15,
    "resource": 0.10,
    "fairness": 0.10,
}
MEMORY_TIERING_CELL_ARTIFACT_WEIGHTS = {
    "concurrent_stability": 0.25,
    "implementation_reuse": 0.15,
    "maintainability": 0.15,
    "service_performance_attainment": 0.20,
    "device_residency_attainment": 0.15,
    "transfer_efficiency": 0.10,
}

_COMMON_HARD_CHECKS = {
    "correctness_passed": "INCORRECT_RESULT",
    "progress_passed": "PROGRESS_FAILURE",
    "lifecycle_quiescent": "TEARDOWN_NOT_QUIESCENT",
    "bounded_resources": "UNBOUNDED_RESOURCE_GROWTH",
    "fallback_policy_respected": "SILENT_FALLBACK",
}
_COMMUNICATION_HARD_CHECKS = {
    "collective_order_consistent": "COLLECTIVE_ORDER_VIOLATION",
    "rank_divergence_absent": "RANK_DIVERGENCE",
    "deadlock_absent": "DEADLOCK",
}
_MEMORY_HARD_CHECKS = {
    "residency_state_valid": "RESIDENCY_STATE_VIOLATION",
    "version_token_valid": "VERSION_TOKEN_MISMATCH",
    "consumer_visibility_valid": "CONSUMER_VISIBILITY_VIOLATION",
    "isolation_valid": "ISOLATION_VIOLATION",
    "use_after_free_absent": "USE_AFTER_FREE",
    "partial_copy_absent": "PARTIAL_COPY_VISIBLE",
    "stale_or_lost_update_absent": "STALE_OR_LOST_UPDATE",
    "prefetch_queue_bounded": "UNBOUNDED_PREFETCH_QUEUE",
    "host_memory_leak_absent": "HOST_MEMORY_LEAK",
    "pageable_fallback_explicit": "IMPLICIT_PAGEABLE_FALLBACK",
}


def evaluate_system_path_infra_cert(
    evidence: SystemPathInfraCertEvidence,
) -> SystemPathInfraCertResult:
    checks = dict(_COMMON_HARD_CHECKS)
    checks.update(
        _COMMUNICATION_HARD_CHECKS
        if evidence.domain == "distributed-communication"
        else _MEMORY_HARD_CHECKS
    )
    missing = sorted(name for name in checks if getattr(evidence, name) is None)
    failures = sorted(
        failure_code for name, failure_code in checks.items() if getattr(evidence, name) is False
    )
    status = "fail" if failures else "unresolved" if missing else "pass"
    return SystemPathInfraCertResult(
        domain=evidence.domain,
        status=status,
        failure_codes=failures,
        missing_checks=missing,
        evidence_digests=evidence.evidence_digests,
    )


def _score_component(
    *,
    status: str,
    value: float | None,
    formula: str,
    evidence_digests: Sequence[str],
    confidence: str,
    reason: str | None = None,
) -> ScoreComponent:
    return ScoreComponent(
        status=status,  # type: ignore[arg-type]
        value=value,
        formula_version=formula,
        input_evidence_digests=list(evidence_digests),
        confidence=confidence,  # type: ignore[arg-type]
        reason=reason,
    )


def score_system_path_concurrent_stability(
    load_cells: Sequence[SystemPathLoadCell],
    *,
    fresh_process_replays: int,
) -> DimensionResult:
    if not load_cells:
        return DimensionResult(
            _score_component(
                status="unresolved",
                value=None,
                formula="system-path-concurrent-stability-v1",
                evidence_digests=(),
                confidence="low",
                reason="no system-path load cells",
            ),
            ("SYSTEM_PATH_LOAD_CELLS_MISSING",),
        )
    domains = {cell.domain for cell in load_cells}
    protocols = {cell.protocol_id for cell in load_cells}
    if len(domains) != 1 or len(protocols) != 1:
        raise ValueError("one C result cannot mix domains or concurrency protocols")
    by_regime = {cell.regime: cell for cell in load_cells}
    if len(by_regime) != len(load_cells) or set(by_regime) != set(SYSTEM_PATH_LOAD_WEIGHTS):
        raise ValueError("system-path load cells must exactly cover the frozen load ladder")

    failures: list[str] = []
    cell_scores: dict[str, float] = {}
    exploratory: list[str] = []
    evidence_digests: set[str] = set()
    for regime, cell in sorted(by_regime.items()):
        evidence_digests.update(cell.evidence_digests)
        if cell.p99_status == "exploratory":
            exploratory.append(regime)
        factors = {
            "goodput": cell.goodput_score,
            "tail": cell.tail_score,
            "jitter": cell.jitter_score,
            "overlap_progress": cell.overlap_progress_score,
            "resource": cell.resource_stability_score,
            "fairness": cell.fairness_score,
        }
        if cell.hard_gate_failure_codes:
            failures.extend(f"{code}:{regime}" for code in cell.hard_gate_failure_codes)
            cell_scores[regime] = 0.0
        else:
            cell_scores[regime] = weighted_geometric(factors, SYSTEM_PATH_CELL_WEIGHTS)
    estimate = weighted_geometric(cell_scores, SYSTEM_PATH_LOAD_WEIGHTS)
    raw = {
        "domain": next(iter(domains)),
        "protocol_id": next(iter(protocols)),
        "fresh_process_replays": fresh_process_replays,
        "load_cell_scores": cell_scores,
        "p99_exploratory_regimes": exploratory,
        "diagnostic_estimate": estimate,
    }
    if fresh_process_replays < 5:
        return DimensionResult(
            _score_component(
                status="diagnostic",
                value=None,
                formula="system-path-concurrent-stability-v1",
                evidence_digests=sorted(evidence_digests),
                confidence="low",
                reason="official C requires at least five fresh-process replays",
            ),
            tuple(sorted({*failures, "SYSTEM_PATH_REPLAY_COUNT_DIAGNOSTIC"})),
            raw,
        )
    return DimensionResult(
        _score_component(
            status="scored",
            value=estimate,
            formula="system-path-concurrent-stability-v1",
            evidence_digests=sorted(evidence_digests),
            confidence="high" if fresh_process_replays >= 7 else "medium",
        ),
        tuple(sorted(set(failures))),
        raw,
    )


def project_operational_projection(
    concurrent_stability: DimensionResult,
) -> ProjectDimensionResult:
    source = concurrent_stability.component
    if source.status != "scored":
        return ProjectDimensionResult(
            ProjectScoreComponent(
                status="unresolved",
                value=None,
                formula_version="operational-fit-concurrent-stability-identity-v1",
                input_evidence_digests=source.input_evidence_digests,
                confidence="low",
                failure_codes=["CONCURRENT_STABILITY_NOT_SCORED"],
                reason="O is an identity projection and cannot outlive C",
            )
        )
    return ProjectDimensionResult(
        ProjectScoreComponent(
            status="scored",
            value=source.value,
            formula_version="operational-fit-concurrent-stability-identity-v1",
            input_evidence_digests=source.input_evidence_digests,
            confidence=source.confidence,
        ),
        {"concurrent_stability": float(source.value)},
    )


def target_attainment(
    value: float,
    *,
    direction: str,
    target: float,
    zero_limit: float,
) -> float:
    if direction == "higher":
        if target <= zero_limit:
            raise ValueError("higher-is-better target must exceed its zero limit")
        return min(1.0, max(0.0, (value - zero_limit) / (target - zero_limit)))
    if direction == "lower":
        if zero_limit <= target:
            raise ValueError("lower-is-better zero limit must exceed its target")
        return min(1.0, max(0.0, (zero_limit - value) / (zero_limit - target)))
    raise ValueError("direction must be higher or lower")


def score_attainment_portfolio(
    values: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    return weighted_geometric(values, weights)


def score_system_path_implementation_reuse(
    *,
    coverage: float,
    family_reuse: float,
    cache_reuse: float,
    observed_variants: int,
    expected_variant_budget: int,
    maximum_variant_budget: int,
    evidence_digests: Sequence[str] = (),
) -> DimensionResult:
    if expected_variant_budget < 1 or maximum_variant_budget <= expected_variant_budget:
        raise ValueError("system-path variant budgets are invalid")
    if observed_variants < 0:
        raise ValueError("observed variants cannot be negative")
    if observed_variants <= expected_variant_budget:
        variant = 1.0
    elif observed_variants >= maximum_variant_budget:
        variant = 0.0
    else:
        variant = (maximum_variant_budget - observed_variants) / (
            maximum_variant_budget - expected_variant_budget
        )
    values = {
        "coverage": coverage,
        "family": family_reuse,
        "cache": cache_reuse,
        "variant": variant,
    }
    score = weighted_geometric(
        values,
        {"coverage": 0.40, "family": 0.25, "cache": 0.20, "variant": 0.15},
    )
    failures = (
        ("SYSTEM_PATH_VARIANT_BUDGET_EXCEEDED",)
        if observed_variants > maximum_variant_budget
        else ()
    )
    return DimensionResult(
        _score_component(
            status="scored",
            value=score,
            formula="system-path-implementation-reuse-v0.5.1",
            evidence_digests=evidence_digests,
            confidence="high",
        ),
        failures,
        {
            "components": values,
            "observed_variants": observed_variants,
            "portability_component_present": False,
        },
    )


def score_transfer_efficiency(
    *,
    useful_bandwidth_attainment: float,
    actual_transfer_bytes: float,
    semantic_useful_bytes: float,
    traffic_amplification_budget: float,
    stall_attainment: float,
) -> float:
    if semantic_useful_bytes <= 0 or actual_transfer_bytes < semantic_useful_bytes:
        raise ValueError("transfer bytes must preserve the semantic-useful lower bound")
    if traffic_amplification_budget < 1:
        raise ValueError("traffic amplification budget cannot be below one")
    amplification = actual_transfer_bytes / semantic_useful_bytes
    traffic_score = min(1.0, traffic_amplification_budget / amplification)
    return weighted_geometric(
        {
            "useful_bandwidth": useful_bandwidth_attainment,
            "traffic": traffic_score,
            "stall": stall_attainment,
        },
        {"useful_bandwidth": 0.35, "traffic": 0.35, "stall": 0.30},
    )


def build_memory_tiering_cell_artifact(
    *,
    concurrent_stability: DimensionResult,
    implementation_reuse: DimensionResult,
    maintainability: DimensionResult,
    service_performance_attainment: float | None,
    device_residency_attainment: float | None,
    transfer_efficiency: float | None,
    evidence_digests: Sequence[str] = (),
) -> CellArtifactScore:
    base = {
        "concurrent_stability": concurrent_stability.component,
        "implementation_reuse": implementation_reuse.component,
        "maintainability": maintainability.component,
    }
    attainment = {
        "service_performance_attainment": service_performance_attainment,
        "device_residency_attainment": device_residency_attainment,
        "transfer_efficiency": transfer_efficiency,
    }
    components = dict(base)
    for name, value in attainment.items():
        components[name] = _score_component(
            status="scored" if value is not None else "unresolved",
            value=value,
            formula=f"{name}-v0.5.2",
            evidence_digests=evidence_digests,
            confidence="high" if value is not None else "low",
            reason=None if value is not None else "required attainment evidence is missing",
        )
    if any(component.status != "scored" for component in components.values()):
        return CellArtifactScore(
            status="unresolved",
            formula_template_id="cell-artifact-memory-tiering-v0.5.2",
            components=components,
        )
    values = {name: float(component.value) for name, component in components.items()}
    score = 100 * weighted_geometric(values, MEMORY_TIERING_CELL_ARTIFACT_WEIGHTS)
    return CellArtifactScore(
        status="scored",
        formula_template_id="cell-artifact-memory-tiering-v0.5.2",
        score_100=score,
        components=components,
    )
