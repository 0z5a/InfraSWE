from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import (
    DecisionLabel,
    DecisionPlaneModel,
    canonical_sha256,
)


class CalibrationCase(DecisionPlaneModel):
    case_id: str
    oracle_label: DecisionLabel
    p_accept: float = Field(ge=0, le=1)
    non_accept_label: Literal["check", "reject"]


class CalibrationPoint(DecisionPlaneModel):
    threshold: float = Field(ge=0, le=1)
    eligible_cases: int = Field(ge=1)
    exact_matches: int = Field(ge=0)
    predicted_accept_cases: int = Field(ge=0)
    oracle_accept_cases: int = Field(ge=0)
    accept_true_positives: int = Field(ge=0)
    accuracy3: float = Field(ge=0, le=1)
    precision_accept: float | None = Field(default=None, ge=0, le=1)
    recall_accept: float | None = Field(default=None, ge=0, le=1)


class CalibrationProfileMaterial(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    profile_id: str
    policy_digest: Digest
    population_digest: Digest
    purpose: Literal["confidence-reporting", "evidence-routing"]
    recall_floor: float = Field(ge=0, le=1)
    precision_target: float | None = Field(default=None, ge=0, le=1)
    accuracy_target: float | None = Field(default=None, ge=0, le=1)
    points: list[CalibrationPoint] = Field(min_length=1)
    max_precision_at_recall_floor: float | None = Field(default=None, ge=0, le=1)
    max_precision_threshold: float | None = Field(default=None, ge=0, le=1)
    max_accuracy_at_recall_floor: float | None = Field(default=None, ge=0, le=1)
    max_accuracy_threshold: float | None = Field(default=None, ge=0, le=1)
    selected_threshold: float | None = Field(default=None, ge=0, le=1)
    target_reachable: bool
    generated_at: datetime

    @model_validator(mode="after")
    def profile_is_frozen_and_non_decisional(self) -> CalibrationProfileMaterial:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.target_reachable != (self.selected_threshold is not None):
            raise ValueError(
                "reachable profiles must select a threshold and unreachable profiles cannot"
            )
        thresholds = [point.threshold for point in self.points]
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("calibration profile thresholds must be unique")
        if self.selected_threshold is not None and self.selected_threshold not in thresholds:
            raise ValueError("selected threshold must be one of the measured points")
        if (self.max_precision_at_recall_floor is None) != (self.max_precision_threshold is None):
            raise ValueError("maximum precision and its threshold must be reported together")
        if (self.max_accuracy_at_recall_floor is None) != (self.max_accuracy_threshold is None):
            raise ValueError("maximum accuracy and its threshold must be reported together")
        return self


class CalibrationProfile(DecisionPlaneModel):
    material: CalibrationProfileMaterial
    profile_sha256: Digest


def sweep_accept_thresholds(
    cases: list[CalibrationCase], thresholds: list[float]
) -> list[CalibrationPoint]:
    if not cases:
        raise ValueError("calibration requires at least one case")
    if not thresholds:
        raise ValueError("calibration requires at least one threshold")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("calibration thresholds must be unique")

    points: list[CalibrationPoint] = []
    for threshold in sorted(thresholds):
        if not 0 <= threshold <= 1:
            raise ValueError("calibration thresholds must be between zero and one")
        predictions = [
            "accept" if case.p_accept >= threshold else case.non_accept_label for case in cases
        ]
        exact = sum(
            prediction == case.oracle_label
            for prediction, case in zip(predictions, cases, strict=True)
        )
        predicted_accept = sum(prediction == "accept" for prediction in predictions)
        oracle_accept = sum(case.oracle_label == "accept" for case in cases)
        true_positives = sum(
            prediction == "accept" and case.oracle_label == "accept"
            for prediction, case in zip(predictions, cases, strict=True)
        )
        points.append(
            CalibrationPoint(
                threshold=threshold,
                eligible_cases=len(cases),
                exact_matches=exact,
                predicted_accept_cases=predicted_accept,
                oracle_accept_cases=oracle_accept,
                accept_true_positives=true_positives,
                accuracy3=exact / len(cases),
                precision_accept=(true_positives / predicted_accept if predicted_accept else None),
                recall_accept=(true_positives / oracle_accept if oracle_accept else None),
            )
        )
    return points


def build_calibration_profile(
    *,
    profile_id: str,
    policy_digest: Digest,
    population_digest: Digest,
    purpose: Literal["confidence-reporting", "evidence-routing"],
    cases: list[CalibrationCase],
    thresholds: list[float],
    recall_floor: float,
    generated_at: datetime,
    precision_target: float | None = None,
    accuracy_target: float | None = None,
) -> CalibrationProfile:
    points = sweep_accept_thresholds(cases, thresholds)
    recall_feasible = [
        point
        for point in points
        if point.recall_accept is not None and point.recall_accept >= recall_floor
    ]
    precision_max = max(
        recall_feasible,
        key=lambda point: (point.precision_accept or 0, point.accuracy3, point.threshold),
        default=None,
    )
    accuracy_max = max(
        recall_feasible,
        key=lambda point: (point.accuracy3, point.precision_accept or 0, point.threshold),
        default=None,
    )
    target_feasible = [
        point
        for point in recall_feasible
        if (precision_target is None or (point.precision_accept or 0) >= precision_target)
        and (accuracy_target is None or point.accuracy3 >= accuracy_target)
    ]
    selected = max(
        target_feasible,
        key=lambda point: (point.precision_accept or 0, point.accuracy3, point.threshold),
        default=None,
    )
    material = CalibrationProfileMaterial(
        profile_id=profile_id,
        policy_digest=policy_digest,
        population_digest=population_digest,
        purpose=purpose,
        recall_floor=recall_floor,
        precision_target=precision_target,
        accuracy_target=accuracy_target,
        points=points,
        max_precision_at_recall_floor=(
            precision_max.precision_accept if precision_max is not None else None
        ),
        max_precision_threshold=precision_max.threshold if precision_max is not None else None,
        max_accuracy_at_recall_floor=(accuracy_max.accuracy3 if accuracy_max is not None else None),
        max_accuracy_threshold=accuracy_max.threshold if accuracy_max is not None else None,
        selected_threshold=selected.threshold if selected is not None else None,
        target_reachable=selected is not None,
        generated_at=generated_at,
    )
    return CalibrationProfile(material=material, profile_sha256=canonical_sha256(material))


def audit_calibration_profile(profile: CalibrationProfile) -> list[str]:
    if profile.profile_sha256 != canonical_sha256(profile.material):
        return ["calibration profile digest mismatch"]
    return []
