from __future__ import annotations

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.judge import (
    JudgeCalibrationReport,
    JudgeCell,
    JudgeDriftSentinel,
    JudgeProfile,
    JudgeRubric,
    JudgeTrustCard,
)


def audit_calibration(
    profile: JudgeProfile,
    report: JudgeCalibrationReport,
) -> list[str]:
    failures: list[str] = []
    if report.profile_id != profile.profile_id:
        failures.append("JUDGE_CALIBRATION_PROFILE_MISMATCH")
    if report.calibration_set_sha256 != profile.calibration_set_sha256:
        failures.append("JUDGE_CALIBRATION_SET_MISMATCH")
    if report.domain not in profile.supported_domains:
        failures.append("JUDGE_CALIBRATION_OUT_OF_DOMAIN")
    if canonical_sha256(report) != profile.calibration_report_sha256:
        failures.append("JUDGE_CALIBRATION_REPORT_DIGEST_MISMATCH")
    if report.status != "pass":
        failures.append("JUDGE_CALIBRATION_REPORTED_FAIL")

    metrics = report.metrics
    floors = profile.floors
    lower_bound_checks = {
        "WEIGHTED_KAPPA": (
            metrics.weighted_kappa_lower_95ci,
            floors.weighted_kappa_lower_95ci,
        ),
        "TEST_RETEST": (metrics.test_retest_agreement, floors.test_retest_agreement),
        "POSITION_CONSISTENCY": (
            metrics.position_consistency,
            floors.position_consistency,
        ),
        "SEMANTIC_EQUIVALENCE": (
            metrics.semantic_equivalence_invariance,
            floors.semantic_equivalence_invariance,
        ),
        "EVIDENCE_GROUNDING": (
            metrics.evidence_grounding_precision,
            floors.evidence_grounding_precision,
        ),
        "SCHEMA_VALID_RATE": (metrics.schema_valid_rate, floors.schema_valid_rate),
        "IN_DOMAIN_COVERAGE": (
            metrics.in_domain_non_abstain_coverage,
            floors.in_domain_non_abstain_coverage,
        ),
    }
    for name, (observed, required) in lower_bound_checks.items():
        if observed < required:
            failures.append(f"JUDGE_CALIBRATION_{name}_BELOW_FLOOR")
    upper_bound_checks = {
        "HARD_VIOLATION_FALSE_PASS": (
            metrics.hard_violation_false_pass_rate,
            floors.hard_violation_false_pass_rate,
        ),
        "PROMPT_INJECTION_ASR": (
            metrics.prompt_injection_asr,
            floors.prompt_injection_asr,
        ),
    }
    for name, (observed, maximum) in upper_bound_checks.items():
        if observed > maximum:
            failures.append(f"JUDGE_CALIBRATION_{name}_ABOVE_FLOOR")
    return sorted(set(failures))


def audit_drift(profile: JudgeProfile, sentinel: JudgeDriftSentinel) -> list[str]:
    failures: list[str] = []
    if sentinel.profile_id != profile.profile_id:
        failures.append("JUDGE_DRIFT_PROFILE_MISMATCH")
    if sentinel.model_identity_sha256 != canonical_sha256(
        [member.model.model_dump(mode="json") for member in profile.panel]
    ):
        failures.append("JUDGE_DRIFT_MODEL_IDENTITY_MISMATCH")
    if canonical_sha256(sentinel) != profile.drift_sentinel_sha256:
        failures.append("JUDGE_DRIFT_SENTINEL_DIGEST_MISMATCH")
    if sentinel.status != "pass":
        failures.append("JUDGE_DRIFTED")
    if not sentinel.checks or not all(sentinel.checks.values()):
        failures.append("JUDGE_DRIFT_SENTINEL_CHECK_FAILED")
    return sorted(set(failures))


def audit_profile_eligibility(
    profile: JudgeProfile,
    *,
    calibration: JudgeCalibrationReport | None = None,
    drift: JudgeDriftSentinel | None = None,
) -> list[str]:
    """Return fail-closed official eligibility failures for a sealed profile."""

    if profile.authority != "bounded-score":
        return []
    failures: list[str] = []
    if calibration is None:
        failures.append("JUDGE_CALIBRATION_MISSING")
    else:
        failures.extend(audit_calibration(profile, calibration))
    if drift is None:
        failures.append("JUDGE_DRIFT_SENTINEL_MISSING")
    else:
        failures.extend(audit_drift(profile, drift))
    return sorted(set(failures))


def _audit_rubric_binding(profile: JudgeProfile, rubric: JudgeRubric) -> list[str]:
    failures: list[str] = []
    if rubric.domain not in profile.supported_domains:
        failures.append("JUDGE_RUBRIC_OUT_OF_DOMAIN")
    for component in {item.owner_component for item in rubric.criteria}:
        observed = sum(
            item.weight_within_component
            for item in rubric.criteria
            if item.owner_component == component and item.owner_type == "semantic-judge"
        )
        profile_cap = profile.component_judge_weight_caps.get(component, 0.0)
        if observed > profile_cap + 1e-12:
            failures.append(f"JUDGE_RUBRIC_{component}_WEIGHT_EXCEEDS_PROFILE_CAP")
    return failures


def build_judge_cell(
    profile: JudgeProfile,
    rubric: JudgeRubric,
    calibration: JudgeCalibrationReport,
    drift: JudgeDriftSentinel,
) -> JudgeCell:
    failures = [
        *audit_profile_eligibility(profile, calibration=calibration, drift=drift),
        *_audit_rubric_binding(profile, rubric),
    ]
    if failures:
        raise ValueError("Judge Cell is not eligible: " + ", ".join(sorted(set(failures))))

    ownership = [
        {
            "criterion_id": item.criterion_id,
            "owner_component": item.owner_component,
            "owner_type": item.owner_type,
            "weight_within_component": item.weight_within_component,
        }
        for item in rubric.criteria
    ]
    material = {
        "schema_version": "0.5.3",
        "profile_id": profile.profile_id,
        "benchmark_season": profile.benchmark_season,
        "profile_sha256": canonical_sha256(profile),
        "rubric_sha256": canonical_sha256(rubric),
        "criterion_ownership_sha256": canonical_sha256(ownership),
        "panel_sha256": canonical_sha256(
            [member.model_dump(mode="json") for member in profile.panel]
        ),
        "aggregation_policy_sha256": canonical_sha256(profile.aggregation),
        "calibration_report_sha256": canonical_sha256(calibration),
        "drift_sentinel_sha256": canonical_sha256(drift),
        "security_policy_sha256": canonical_sha256(profile.security),
    }
    return JudgeCell.model_validate({**material, "judge_cell_sha256": canonical_sha256(material)})


def audit_judge_cell(cell: JudgeCell) -> list[str]:
    material = cell.model_dump(mode="json", exclude={"judge_cell_sha256"})
    if cell.judge_cell_sha256 != canonical_sha256(material):
        return ["JUDGE_CELL_DIGEST_MISMATCH"]
    return []


def build_trust_card(
    profile: JudgeProfile,
    *,
    domain: str,
    calibration: JudgeCalibrationReport | None = None,
    drift: JudgeDriftSentinel | None = None,
    cell: JudgeCell | None = None,
) -> JudgeTrustCard:
    failures = audit_profile_eligibility(
        profile,
        calibration=calibration,
        drift=drift,
    )
    if profile.authority != "bounded-score":
        failures.append("JUDGE_PROFILE_NOT_BOUNDED_SCORE")
    if any(not member.model.pinned for member in profile.panel):
        status = "unpinned"
        failures.append("JUDGE_MODEL_IDENTITY_UNPINNED")
    elif drift is not None and drift.status != "pass":
        status = "drifted"
    elif failures or cell is None:
        status = "fail"
        if cell is None:
            failures.append("JUDGE_CELL_MISSING")
    else:
        status = "pass"
    return JudgeTrustCard(
        status=status,
        judge_cell_sha256=cell.judge_cell_sha256 if status == "pass" and cell else None,
        profile_id=profile.profile_id,
        domain=domain,
        calibration_report_sha256=(
            canonical_sha256(calibration) if calibration is not None else None
        ),
        drift_sentinel_sha256=(canonical_sha256(drift) if drift is not None else None),
        metrics=calibration.metrics if calibration is not None else None,
        failure_codes=sorted(set(failures)),
    )
