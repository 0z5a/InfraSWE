from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal

from pydantic import Field, model_validator

from infraswe.pr_decision.contracts import (
    DecisionLabel,
    DecisionPlaneModel,
    IntegerErrorBudget,
    MetricContract,
    integer_error_budget,
)


class DecisionEvaluationCase(DecisionPlaneModel):
    case_id: str = Field(min_length=1)
    predicted_label: DecisionLabel
    oracle_label: DecisionLabel
    valid: bool = True
    invalid_reason: str | None = None

    @model_validator(mode="after")
    def invalid_case_is_explained(self) -> DecisionEvaluationCase:
        if self.valid and self.invalid_reason is not None:
            raise ValueError("valid cases cannot have invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid cases require invalid_reason")
        return self


class ConfusionMatrix3(DecisionPlaneModel):
    labels: tuple[DecisionLabel, DecisionLabel, DecisionLabel] = (
        "accept",
        "check",
        "reject",
    )
    counts: dict[DecisionLabel, dict[DecisionLabel, int]]


class ReliabilityMetrics(DecisionPlaneModel):
    total_cases: int = Field(ge=0)
    eligible_cases: int = Field(ge=0)
    invalid_cases: int = Field(ge=0)
    exact_matches: int = Field(ge=0)
    predicted_accept_cases: int = Field(ge=0)
    oracle_accept_cases: int = Field(ge=0)
    accept_true_positives: int = Field(ge=0)
    accept_false_positives: int = Field(ge=0)
    accept_false_negatives: int = Field(ge=0)
    accuracy3: float | None = Field(default=None, ge=0, le=1)
    precision_accept: float | None = Field(default=None, ge=0, le=1)
    recall_accept: float | None = Field(default=None, ge=0, le=1)
    confusion: ConfusionMatrix3


class MetricGateResult(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    evaluation_scope: Literal["numerical-diagnostic-only"] = "numerical-diagnostic-only"
    release_authorized: Literal[False] = False
    contract: MetricContract
    metrics: ReliabilityMetrics
    integer_budget: IntegerErrorBudget
    accuracy_value_for_gate: float | None = Field(default=None, ge=0, le=1)
    recall_accept_value_for_gate: float | None = Field(default=None, ge=0, le=1)
    precision_accept_value_for_gate: float | None = Field(default=None, ge=0, le=1)
    accuracy_passed: bool
    recall_accept_passed: bool
    precision_accept_passed: bool | None
    support_passed: bool
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


def _wilson_lower_bound(successes: int, total: int, alpha: float) -> float | None:
    if total == 0:
        return None
    z = NormalDist().inv_cdf(1 - alpha)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    return max(0.0, (center - margin) / denominator)


def _metric_value(successes: int, total: int, contract: MetricContract) -> float | None:
    if total == 0:
        return None
    if contract.confidence_method == "point-estimate":
        return successes / total
    assert contract.confidence_alpha is not None
    return _wilson_lower_bound(successes, total, contract.confidence_alpha)


def _confusion(cases: list[DecisionEvaluationCase]) -> ConfusionMatrix3:
    labels: tuple[DecisionLabel, DecisionLabel, DecisionLabel] = ("accept", "check", "reject")
    counts: dict[DecisionLabel, dict[DecisionLabel, int]] = {
        actual: {predicted: 0 for predicted in labels} for actual in labels
    }
    for case in cases:
        counts[case.oracle_label][case.predicted_label] += 1
    return ConfusionMatrix3(counts=counts)


def evaluate_release_gate(
    cases: list[DecisionEvaluationCase], contract: MetricContract
) -> MetricGateResult:
    """Compute numerical gates, not holdout provenance or permission to publish.

    Caller-declared invalid rows remain visible in the diagnostic denominators,
    but cannot produce a passing result without independent eligibility review.
    No attestor is implicit in a contract's requested evaluation_track.
    """
    cases = [DecisionEvaluationCase.model_validate(case.model_dump()) for case in cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate evaluation case ids")
    eligible = [case for case in cases if case.valid]
    exact = sum(case.predicted_label == case.oracle_label for case in eligible)
    predicted_accept = sum(case.predicted_label == "accept" for case in eligible)
    oracle_accept = sum(case.oracle_label == "accept" for case in eligible)
    true_positives = sum(
        case.predicted_label == "accept" and case.oracle_label == "accept" for case in eligible
    )
    false_positives = sum(
        case.predicted_label == "accept" and case.oracle_label != "accept" for case in eligible
    )
    false_negatives = sum(
        case.predicted_label != "accept" and case.oracle_label == "accept" for case in eligible
    )
    metrics = ReliabilityMetrics(
        total_cases=len(cases),
        eligible_cases=len(eligible),
        invalid_cases=len(cases) - len(eligible),
        exact_matches=exact,
        predicted_accept_cases=predicted_accept,
        oracle_accept_cases=oracle_accept,
        accept_true_positives=true_positives,
        accept_false_positives=false_positives,
        accept_false_negatives=false_negatives,
        accuracy3=(exact / len(eligible) if eligible else None),
        precision_accept=(true_positives / predicted_accept if predicted_accept else None),
        recall_accept=(true_positives / oracle_accept if oracle_accept else None),
        confusion=_confusion(eligible),
    )
    support_passed = not contract.require_nonzero_support or bool(eligible and oracle_accept)
    accuracy_value = _metric_value(exact, len(eligible), contract)
    recall_value = _metric_value(true_positives, oracle_accept, contract)
    precision_value = _metric_value(true_positives, predicted_accept, contract)
    accuracy_passed = accuracy_value is not None and accuracy_value >= contract.accuracy3_minimum
    recall_passed = recall_value is not None and recall_value >= contract.recall_accept_minimum
    precision_passed = None
    if contract.precision_accept_minimum is not None:
        precision_passed = (
            precision_value is not None and precision_value >= contract.precision_accept_minimum
        )

    failures: list[str] = []
    if len(eligible) != len(cases):
        failures.append("unverified eligibility exclusions: independent ledger required")
    if not support_passed:
        failures.append("nonzero eligible and oracle-Accept support is required")
    if not accuracy_passed:
        failures.append("Accuracy3 gate failed")
    if not recall_passed:
        failures.append("Accept recall gate failed")
    if precision_passed is False:
        failures.append("Accept precision gate failed")
    passed = not failures
    return MetricGateResult(
        contract=contract,
        metrics=metrics,
        integer_budget=integer_error_budget(
            contract,
            eligible_cases=len(eligible),
            oracle_accept_cases=oracle_accept,
        ),
        accuracy_value_for_gate=accuracy_value,
        recall_accept_value_for_gate=recall_value,
        precision_accept_value_for_gate=precision_value,
        accuracy_passed=accuracy_passed,
        recall_accept_passed=recall_passed,
        precision_accept_passed=precision_passed,
        support_passed=support_passed,
        passed=passed,
        failure_reasons=failures,
    )
