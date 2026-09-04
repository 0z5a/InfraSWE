from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from infraswe.models.history import (
    BlindEvaluationEvidence,
    HistoricalCalibrationCase,
    HistoricalCalibrationReport,
    HistoricalGroundTruth,
    HistoricalPRCandidate,
    HistoricalPredictionLock,
    HistoricalPredictionMaterial,
)
from infraswe.policy import (
    HISTORICAL_SCORE_BAND_PREDICTION_POLICY_ID,
    LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID,
    MERGE_ACCEPT_SCORE_FLOOR_100,
    project_fit_decision_band,
)

OUTCOME_KEYS = {
    "actual_outcome",
    "closed_at",
    "conclusion",
    "ground_truth",
    "merge_commit_sha",
    "merged",
    "merged_at",
    "review_decision",
    "state",
}


def canonical_sha256(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_outcome_free(value: Any, path: str = "$") -> None:
    """Reject outcome-bearing fields before any evaluator sees a blind case."""

    if isinstance(value, dict):
        leaked = OUTCOME_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"ground-truth leakage at {path}: {sorted(leaked)}")
        for key, item in value.items():
            assert_outcome_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_outcome_free(item, f"{path}[{index}]")


def compile_prediction(
    candidate: HistoricalPRCandidate,
    evidence: BlindEvaluationEvidence,
    *,
    frozen_at: datetime | None = None,
    policy_id: Literal[
        "historical-merge-prediction-v0.5-r1",
        "historical-merge-prediction-v0.5-r2-polarized",
        "historical-merge-prediction-v0.1-score-bands",
    ] = "historical-merge-prediction-v0.5-r1",
) -> HistoricalPredictionMaterial:
    if evidence.case_id != candidate.case_id:
        raise ValueError("candidate/evidence case mismatch")
    if evidence.candidate_sha256 != canonical_sha256(candidate):
        raise ValueError("candidate digest does not match blind evidence")

    if evidence.infrastructure_failure_codes:
        predicted = "abstain"
        decision = "unresolved"
        confidence = "not-applicable"
        rationale = sorted(set(evidence.infrastructure_failure_codes))
    elif evidence.candidate_failure_codes:
        predicted = "not-merged"
        hard_failure = any(
            code.startswith("HARD_POLICY_") for code in evidence.candidate_failure_codes
        )
        polarized = policy_id in {
            LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID,
            HISTORICAL_SCORE_BAND_PREDICTION_POLICY_ID,
        }
        decision = "reject" if hard_failure or polarized else "revise"
        confidence = "high" if hard_failure or polarized else "medium"
        rationale = sorted(set(evidence.candidate_failure_codes))
    else:
        passed_categories = {check.category for check in evidence.checks if check.status == "pass"}
        required = {"checkout", "static"}
        substantive = {"build", "unit", "gpu"}
        has_substantive = bool(passed_categories.intersection(substantive))
        if required.issubset(passed_categories) and has_substantive:
            if policy_id == "historical-merge-prediction-v0.5-r1":
                predicted = "merged"
                decision = "accept_with_scope"
                confidence = "medium" if evidence.stage in {"F0-contract", "F1-smoke"} else "high"
                rationale = ["BLIND_MACHINE_CHECKS_PASSED"]
            elif evidence.project_fit_score_100 is None:
                predicted = "abstain"
                decision = "unresolved"
                confidence = "not-applicable"
                rationale = ["HISTORICAL_PROJECT_FIT_SCORE_MISSING"]
            elif policy_id == LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID:
                accepted = evidence.project_fit_score_100 >= MERGE_ACCEPT_SCORE_FLOOR_100
                predicted = "merged" if accepted else "not-merged"
                decision = "accept_with_scope" if accepted else "reject"
                confidence = "medium" if evidence.stage in {"F0-contract", "F1-smoke"} else "high"
                rationale = [
                    "BLIND_MACHINE_CHECKS_PASSED",
                    "MERGE_SCORE_AT_LEAST_85" if accepted else "PROJECT_FIT_BELOW_MERGE_FLOOR_85",
                ]
            else:
                score_band = project_fit_decision_band(evidence.project_fit_score_100)
                predicted = "merged" if score_band == "accept" else "not-merged"
                decision = "accept_with_scope" if score_band == "accept" else score_band
                confidence = "medium" if evidence.stage in {"F0-contract", "F1-smoke"} else "high"
                rationale = [
                    "BLIND_MACHINE_CHECKS_PASSED",
                    {
                        "accept": "PROJECT_FIT_ACCEPT_BAND_ABOVE_65",
                        "check": "PROJECT_FIT_CHECK_BAND_50_TO_65",
                        "reject": "PROJECT_FIT_REJECT_BAND_BELOW_50",
                    }[score_band],
                ]
        else:
            predicted = "abstain"
            decision = "unresolved"
            confidence = "not-applicable"
            rationale = ["BLIND_SUBSTANTIVE_EVIDENCE_INSUFFICIENT"]

    return HistoricalPredictionMaterial(
        case_id=candidate.case_id,
        candidate_sha256=evidence.candidate_sha256,
        evidence_sha256=canonical_sha256(evidence),
        prediction_policy_id=policy_id,
        predicted_outcome=predicted,
        mergeability_decision=decision,
        score_100=evidence.project_fit_score_100,
        confidence=confidence,
        rationale_codes=rationale,
        frozen_at=frozen_at or datetime.now(UTC),
    )


def freeze_prediction(material: HistoricalPredictionMaterial) -> HistoricalPredictionLock:
    return HistoricalPredictionLock(material=material, lock_sha256=canonical_sha256(material))


def audit_prediction_lock(lock: HistoricalPredictionLock) -> bool:
    return lock.lock_sha256 == canonical_sha256(lock.material)


def join_revealed_case(
    lock: HistoricalPredictionLock,
    truth: HistoricalGroundTruth,
) -> HistoricalCalibrationCase:
    if not audit_prediction_lock(lock):
        raise ValueError("prediction lock digest mismatch")
    if truth.prediction_lock_sha256 != lock.lock_sha256:
        raise ValueError("ground truth is not bound to this prediction lock")
    if truth.case_id != lock.material.case_id:
        raise ValueError("prediction/ground-truth case mismatch")
    if truth.observed_at < lock.material.frozen_at:
        raise ValueError("ground truth was observed before the prediction was frozen")
    actual = "merged" if truth.merged else "not-merged"
    predicted = lock.material.predicted_outcome
    correct = None if predicted == "abstain" else predicted == actual
    return HistoricalCalibrationCase(
        case_id=truth.case_id,
        predicted_outcome=predicted,
        actual_outcome=actual,
        correct=correct,
        prediction_lock_sha256=lock.lock_sha256,
        ground_truth_sha256=canonical_sha256(truth),
    )


def build_calibration_report(
    cases: list[HistoricalCalibrationCase],
) -> HistoricalCalibrationReport:
    covered = [case for case in cases if case.correct is not None]
    correct = [case for case in covered if case.correct]
    confusion = {
        "predicted_merged_actual_merged": 0,
        "predicted_merged_actual_not_merged": 0,
        "predicted_not_merged_actual_merged": 0,
        "predicted_not_merged_actual_not_merged": 0,
        "abstained_actual_merged": 0,
        "abstained_actual_not_merged": 0,
    }
    for case in cases:
        predicted = case.predicted_outcome.replace("-", "_")
        actual = case.actual_outcome.replace("-", "_")
        key = (
            f"abstained_actual_{actual}"
            if predicted == "abstain"
            else f"predicted_{predicted}_actual_{actual}"
        )
        confusion[key] += 1
    return HistoricalCalibrationReport(
        cases=cases,
        total_cases=len(cases),
        covered_cases=len(covered),
        abstained_cases=len(cases) - len(covered),
        correct_cases=len(correct),
        accuracy=len(correct) / len(covered) if covered else None,
        confusion=confusion,
    )
