from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from fractions import Fraction
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

DecisionLabel = Literal["accept", "check", "reject"]
LabelKind = Literal[
    "upstream_outcome",
    "contract_acceptability",
    "executable_verifier_outcome",
]
EvaluationTrack = Literal[
    "prequential_campaign_result",
    "frozen_policy_holdout_result",
    "historical_diagnostic_replay_result",
]


class DecisionPlaneModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def canonical_sha256(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


class PRCaseIdentity(DecisionPlaneModel):
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    pr_number: int = Field(ge=1)
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    prediction_at: datetime
    label_schema_version: str = Field(min_length=1)
    issue_or_patch_family: str | None = None

    @model_validator(mode="after")
    def identity_is_versioned(self) -> PRCaseIdentity:
        _aware(self.prediction_at, "prediction_at")
        if self.base_sha == self.head_sha:
            raise ValueError("base_sha and head_sha must differ")
        return self


class PolicyIdentity(DecisionPlaneModel):
    policy_digest: Digest
    retrieval_index_digest: Digest
    project_profile_digest: Digest
    decision_profile_digest: Digest
    calibration_profile_digest: Digest | None = None


class DecisionMicroscores(DecisionPlaneModel):
    """Optional nested diagnostics; never peers of overall_score_100."""

    project_fit_100: float | None = Field(default=None, ge=0, le=100)
    benchmark_trust_100: float | None = Field(default=None, ge=0, le=100)
    status: Literal["non-official"] = "non-official"

    @model_validator(mode="after")
    def official_adapter_is_required(self) -> DecisionMicroscores:
        if self.project_fit_100 is not None or self.benchmark_trust_100 is not None:
            raise ValueError("formal microscores require a qualified EvidencePack adapter")
        return self


class DecisionPrediction(DecisionPlaneModel):
    label: DecisionLabel
    overall_score_100: float = Field(ge=0, le=100)
    decision_basis: Literal[
        "blocking-obligation",
        "unresolved-obligation",
        "overall-score-band",
    ] = "overall-score-band"
    p_accept_calibrated: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)
    blocking_obligations: list[str] = Field(default_factory=list)
    missing_obligations: list[str] = Field(default_factory=list)
    microscores: DecisionMicroscores | None = None

    @model_validator(mode="after")
    def decision_respects_ordered_gates(self) -> DecisionPrediction:
        if self.decision_basis == "blocking-obligation":
            if (
                self.label != "reject"
                or not self.blocking_obligations
                or not self.evidence_refs
                or self.missing_obligations
            ):
                raise ValueError(
                    "blocking-obligation decisions require Reject, blocking evidence, and no "
                    "unresolved obligations"
                )
        elif self.decision_basis == "unresolved-obligation":
            if self.label != "check" or not self.missing_obligations or self.blocking_obligations:
                raise ValueError(
                    "unresolved-obligation decisions require Check, named missing obligations, "
                    "and no blocking violations"
                )
        else:
            expected = label_for_overall_score(self.overall_score_100)
            if self.label != expected:
                raise ValueError(
                    f"label {self.label!r} does not match frozen overall score band {expected!r}"
                )
            if self.blocking_obligations:
                raise ValueError("overall-score-band decisions cannot carry blocking obligations")
            if self.label != "check" and self.missing_obligations:
                raise ValueError(
                    "resolved overall-score-band decisions cannot carry unresolved obligations"
                )
        if self.label == "accept" and (self.blocking_obligations or self.missing_obligations):
            raise ValueError("Accept cannot carry blocking or unresolved obligations")
        if self.label == "check" and not self.missing_obligations:
            raise ValueError("Check is reserved for explicitly named unresolved obligations")
        return self


def label_for_overall_score(score: float) -> DecisionLabel:
    if score < 50:
        return "reject"
    if score <= 65:
        return "check"
    return "accept"


class MetricContract(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    contract_id: str
    accuracy3_minimum: float = Field(ge=0, le=1)
    recall_accept_minimum: float = Field(ge=0, le=1)
    precision_accept_minimum: float | None = Field(default=None, ge=0, le=1)
    require_nonzero_support: bool = True
    evaluation_track: EvaluationTrack
    confidence_method: Literal["point-estimate", "wilson-one-sided"] = "point-estimate"
    confidence_alpha: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def confidence_contract_is_explicit(self) -> MetricContract:
        if self.confidence_method == "point-estimate" and self.confidence_alpha is not None:
            raise ValueError("point-estimate contracts cannot set confidence_alpha")
        if self.confidence_method != "point-estimate" and self.confidence_alpha is None:
            raise ValueError("confidence-bound contracts require confidence_alpha")
        return self


BASELINE_95_99_CONTRACT = MetricContract(
    contract_id="pr-decision-accuracy95-recall99-v0.6.1",
    accuracy3_minimum=0.95,
    recall_accept_minimum=0.99,
    evaluation_track="frozen_policy_holdout_result",
)

PRECISION_95_99_95_CONTRACT = MetricContract(
    contract_id="pr-decision-accuracy95-recall99-precision95-v0.6.1",
    accuracy3_minimum=0.95,
    recall_accept_minimum=0.99,
    precision_accept_minimum=0.95,
    evaluation_track="frozen_policy_holdout_result",
)

# Separately versioned: historical two-gate/precision-95 contracts remain unchanged.
STRICT_95_99_99_CONTRACT = MetricContract(
    contract_id="pr-decision-accuracy95-recall99-precision99-v0.6.1",
    accuracy3_minimum=0.95,
    recall_accept_minimum=0.99,
    precision_accept_minimum=0.99,
    evaluation_track="frozen_policy_holdout_result",
)


class DecisionCountDelta(DecisionPlaneModel):
    recovered_old_fn: int = Field(ge=0)
    introduced_new_fn: int = Field(ge=0)
    removed_old_fp: int = Field(ge=0)
    introduced_new_fp: int = Field(ge=0)

    @property
    def net_accept_true_positives(self) -> int:
        return self.recovered_old_fn - self.introduced_new_fn

    @property
    def net_accept_false_positives_removed(self) -> int:
        return self.removed_old_fp - self.introduced_new_fp


class IntegerErrorBudget(DecisionPlaneModel):
    eligible_cases: int = Field(ge=0)
    oracle_accept_cases: int = Field(ge=0)
    required_exact_matches: int = Field(ge=0)
    required_accept_true_positives: int = Field(ge=0)
    maximum_accept_false_positives: int | None = Field(default=None, ge=0)


def minimum_successes(total: int, minimum: float) -> int:
    if total < 0:
        raise ValueError("total cannot be negative")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be between zero and one")
    return math.ceil(total * Fraction(str(minimum)))


def integer_error_budget(
    contract: MetricContract,
    *,
    eligible_cases: int,
    oracle_accept_cases: int,
) -> IntegerErrorBudget:
    if not 0 <= oracle_accept_cases <= eligible_cases:
        raise ValueError("oracle Accept count must be within the eligible population")
    required_tp = minimum_successes(oracle_accept_cases, contract.recall_accept_minimum)
    maximum_fp = None
    if contract.precision_accept_minimum:
        precision = Fraction(str(contract.precision_accept_minimum))
        maximum_fp = math.floor(required_tp * (1 - precision) / precision)
    return IntegerErrorBudget(
        eligible_cases=eligible_cases,
        oracle_accept_cases=oracle_accept_cases,
        required_exact_matches=minimum_successes(eligible_cases, contract.accuracy3_minimum),
        required_accept_true_positives=required_tp,
        maximum_accept_false_positives=maximum_fp,
    )
