from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from infraswe.models.score import EvidenceGrade, ScoreComponent
from infraswe.models.training import (
    TrainingCertification,
    TrainingComparability,
    TrainingEvidenceBundle,
    TrainingRawMetric,
    TrainingResult,
    TrainingScoreInput,
)
from infraswe.scoring.deployability import (
    DimensionResult,
    build_v04_score,
    score_cell_efficiency,
    score_concurrent_stability,
    score_kernel_reuse,
    score_maintainability,
    weighted_geometric,
)

TRAINING_PROFILER_TO_V04: dict[str, EvidenceGrade] = {
    "G0": "E0-runtime",
    "G1": "E1-framework",
    # A PyTorch profiler trace is framework evidence. v0.4 E2 requires a system timeline.
    "G2": "E1-framework",
    "G3": "E2-system-trace",
    "G4": "E3-kernel-counter",
}


def training_profiler_to_v04(grade: str, *, sealed: bool = False) -> EvidenceGrade:
    try:
        mapped = TRAINING_PROFILER_TO_V04[grade]
    except KeyError as error:
        raise ValueError(f"unknown training profiler grade: {grade}") from error
    if sealed:
        if grade != "G4":
            raise ValueError("sealed E4 training evidence requires a G4 profiler pack")
        return "E4-sealed"
    return mapped


def _unresolved_dimension(name: str, reason: str, digests: list[str]) -> DimensionResult:
    return DimensionResult(
        component=ScoreComponent(
            status="unresolved",
            value=None,
            formula_version=f"{name}-v0.4",
            input_evidence_digests=digests,
            confidence="low",
            reason=reason,
        ),
        failure_codes=(f"TRAIN_{name.upper().replace('-', '_')}_EVIDENCE_MISSING",),
    )


def _score_reuse(score_input: TrainingScoreInput) -> DimensionResult:
    evidence = score_input.reuse
    if evidence is None:
        return _unresolved_dimension(
            "kernel-reuse",
            "training graph/kernel reuse evidence is missing",
            score_input.evidence_digests,
        )
    semantic_coverage = weighted_geometric(
        {
            "shape": evidence.shape_coverage,
            "dtype": evidence.dtype_coverage,
            "layout": evidence.layout_coverage,
        },
        {"shape": 0.5, "dtype": 0.2, "layout": 0.3},
    )
    compile_reuse = math.sqrt(evidence.dispatcher_reuse * evidence.compile_cache_reuse)
    return score_kernel_reuse(
        coverage=semantic_coverage,
        observed_variants=evidence.observed_variants,
        expected_variant_budget=evidence.expected_variant_budget,
        max_variant_budget=evidence.max_variant_budget,
        compile_reuse=compile_reuse,
        port_reuse=evidence.portability_reuse,
        silent_fallback_rate=evidence.silent_fallback_rate,
        evidence_digests=score_input.evidence_digests,
    )


def _score_maintainability(score_input: TrainingScoreInput) -> DimensionResult:
    evidence = score_input.maintainability
    if evidence is None:
        return _unresolved_dimension(
            "maintainability",
            "structured training maintainability probes are missing",
            score_input.evidence_digests,
        )
    return score_maintainability(
        contract=evidence.contract,
        locality=evidence.locality,
        tests=evidence.tests,
        build=evidence.build_reproducibility,
        evidence_digests=score_input.evidence_digests,
    )


def _raw_metrics(payload: Mapping[str, Any]) -> dict[str, TrainingRawMetric]:
    rendered: dict[str, TrainingRawMetric] = {}
    for name, value in payload.items():
        if isinstance(value, Mapping) and (
            "value" in value or "reason" in value or "unit" in value
        ):
            rendered[name] = TrainingRawMetric.model_validate(dict(value))
        else:
            rendered[name] = TrainingRawMetric(value=value)
    return rendered


def build_training_result(
    *,
    bundle: TrainingEvidenceBundle,
    certification: TrainingCertification,
    score_input: TrainingScoreInput,
    comparability: TrainingComparability,
) -> TrainingResult:
    """Issue a training result through the frozen v0.4 C/U/M score envelope."""

    if bundle.hardware_cell_id != score_input.benchmark_cell_id:
        raise ValueError("training evidence and score input use different hardware cells")
    if comparability.hardware_cell_id != bundle.hardware_cell_id:
        raise ValueError("comparability key and evidence use different hardware cells")

    concurrent = score_concurrent_stability(
        score_input.load_cells,
        fresh_process_replays=score_input.fresh_process_replays,
        evidence_digests=score_input.evidence_digests,
    )
    reuse = _score_reuse(score_input)
    maintainability = _score_maintainability(score_input)
    cell_efficiency = None
    if score_input.cell_artifact_template:
        efficiency = score_input.cell_efficiency
        if efficiency is None:
            cell_efficiency = score_cell_efficiency(
                work_model_id=comparability.work_model_id,
                regime="mixed",
                work_model={},
                calibration={},
                candidate_time_seconds=1.0,
                actual_memory_bytes=None,
                traffic_amplification_budget=1.0,
                counter_evidence_available=False,
            )
        else:
            if efficiency.work_model_id != comparability.work_model_id:
                raise ValueError(
                    "cell-efficiency evidence and comparability use different work models"
                )
            cell_efficiency = score_cell_efficiency(
                work_model_id=efficiency.work_model_id,
                regime=efficiency.regime,
                work_model=efficiency.work_model,
                calibration=efficiency.calibration,
                candidate_time_seconds=efficiency.candidate_time_seconds,
                actual_memory_bytes=efficiency.actual_memory_bytes,
                traffic_amplification_budget=efficiency.traffic_amplification_budget,
                counter_evidence_available=(
                    efficiency.counter_evidence_available and score_input.profiler_grade == "G4"
                ),
                counter_confidence=efficiency.counter_confidence,
                evidence_digests=efficiency.evidence_digests,
            )
    v04_score = build_v04_score(
        hard_gate_status=certification.status,
        benchmark_cell_id=score_input.benchmark_cell_id,
        evidence_grade=training_profiler_to_v04(
            score_input.profiler_grade, sealed=score_input.sealed
        ),
        concurrent_stability=concurrent,
        kernel_reuse=reuse,
        maintainability=maintainability,
        cell_efficiency=cell_efficiency,
        cell_artifact_template=score_input.cell_artifact_template,
        raw_metrics=score_input.raw_metrics,
        additional_failure_codes=certification.failure_codes,
    )
    raw_payload = dict(score_input.raw_metrics)
    raw_payload.setdefault(
        "fresh_process_replays",
        {"value": score_input.fresh_process_replays, "unit": "processes"},
    )
    return TrainingResult(
        task_id=bundle.task_id,
        algorithm=bundle.algorithm,
        optimizer=bundle.optimizer,
        framework_stack_id=bundle.framework_stack_id,
        hardware_cell_id=bundle.hardware_cell_id,
        implementation_bundle_id=bundle.implementation_bundle_id,
        training_cert=certification,
        v04_score=v04_score,
        profiler_grade=score_input.profiler_grade,
        raw_metrics=_raw_metrics(raw_payload),
        comparability=comparability,
    )
