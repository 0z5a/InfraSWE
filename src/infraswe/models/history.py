from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import DefaultDraftProject, Digest
from infraswe.policy import (
    HISTORICAL_SCORE_BAND_PREDICTION_POLICY_ID,
    LEGACY_HISTORICAL_POLARIZED_ORACLE_POLICY_ID,
    LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID,
    MERGE_ACCEPT_SCORE_FLOOR_100,
    PROJECT_FIT_ACCEPT_ABOVE_100,
)


class HistoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HistoricalPRCandidate(HistoryModel):
    """Outcome-free PR metadata that is safe to expose to the evaluator."""

    schema_version: Literal["0.5"] = "0.5"
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,99}$")
    project: DefaultDraftProject
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    pull_number: int = Field(ge=1)
    title: str
    created_at: datetime
    base_ref: str
    base_tip_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_derivation: Literal[
        "base-ref-oid-path-parity",
        "first-pr-commit-first-parent-path-parity",
    ]
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    pr_commit_shas: list[str] = Field(min_length=1)
    changed_files: int = Field(ge=1)
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    paths: list[str] = Field(min_length=1)
    acquisition_query: str
    selection_policy_id: str
    outcome_fields_requested: Literal[False] = False

    @model_validator(mode="after")
    def file_count_is_coherent(self) -> HistoricalPRCandidate:
        if self.base_sha == self.head_sha:
            raise ValueError("historical PR base and head must differ")
        invalid_shas = [
            sha
            for sha in self.pr_commit_shas
            if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha)
        ]
        if invalid_shas:
            raise ValueError("historical PR commit SHAs must be lowercase 40-hex")
        if self.pr_commit_shas[-1] != self.head_sha:
            raise ValueError("the final PR commit must match head_sha")
        if len(self.paths) > self.changed_files:
            raise ValueError("listed paths cannot exceed changed_files")
        invalid_paths = [
            path for path in self.paths if path.startswith(("/", "\\")) or ".." in path.split("/")
        ]
        if invalid_paths:
            raise ValueError("historical PR paths must be repository-relative")
        return self


HistoricalCheckStatus = Literal["pass", "fail", "unresolved", "not-applicable"]
HistoricalExplainablePolicyId = Literal[
    "historical-explainable-agent-v0.5-r2",
    "historical-explainable-agent-v0.5-r3",
    "historical-explainable-agent-v0.5-r4",
    "historical-explainable-agent-v0.5-r5-polarized",
]


class HistoricalCheckResult(HistoryModel):
    name: str
    category: Literal["checkout", "static", "build", "unit", "gpu", "policy"]
    status: HistoricalCheckStatus
    command: list[str] = Field(default_factory=list)
    return_code: int | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    stdout_sha256: Digest | None = None
    stderr_sha256: Digest | None = None
    failure_code: str | None = None
    details: str | None = None

    @model_validator(mode="after")
    def failure_is_explained(self) -> HistoricalCheckResult:
        if self.status in {"fail", "unresolved"} and self.failure_code is None:
            raise ValueError(f"{self.status} historical checks require a failure_code")
        if self.status == "not-applicable" and self.return_code is not None:
            raise ValueError("not-applicable checks cannot have a return code")
        return self


class BlindEvaluationEvidence(HistoryModel):
    schema_version: Literal["0.5"] = "0.5"
    case_id: str
    candidate_sha256: Digest
    test_plan_sha256: Digest
    environment_sha256: Digest
    started_at: datetime
    finished_at: datetime
    stage: Literal["F0-contract", "F1-smoke", "F2-affected", "F3-verify"]
    checks: list[HistoricalCheckResult] = Field(min_length=1)
    candidate_failure_codes: list[str] = Field(default_factory=list)
    infrastructure_failure_codes: list[str] = Field(default_factory=list)
    raw_evidence_digests: list[Digest] = Field(default_factory=list)
    project_fit_score_100: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def evidence_is_coherent(self) -> BlindEvaluationEvidence:
        if self.finished_at < self.started_at:
            raise ValueError("blind evaluation cannot finish before it starts")
        failed = {check.failure_code for check in self.checks if check.status == "fail"}
        unresolved = {check.failure_code for check in self.checks if check.status == "unresolved"}
        if not failed.issubset(set(self.candidate_failure_codes)):
            raise ValueError("failed checks must be represented in candidate_failure_codes")
        if not unresolved.issubset(set(self.infrastructure_failure_codes)):
            raise ValueError(
                "unresolved checks must be represented in infrastructure_failure_codes"
            )
        return self


class HistoricalPredictionMaterial(HistoryModel):
    schema_version: Literal["0.5"] = "0.5"
    case_id: str
    candidate_sha256: Digest
    evidence_sha256: Digest
    prediction_policy_id: Literal[
        "historical-merge-prediction-v0.5-r1",
        "historical-merge-prediction-v0.5-r2-polarized",
        "historical-merge-prediction-v0.1-score-bands",
    ] = "historical-merge-prediction-v0.5-r1"
    predicted_outcome: Literal["merged", "not-merged", "abstain"]
    mergeability_decision: Literal[
        "accept", "accept_with_scope", "check", "revise", "reject", "unresolved"
    ]
    score_100: float | None = Field(default=None, ge=0, le=100)
    confidence: Literal["low", "medium", "high", "not-applicable"]
    rationale_codes: list[str] = Field(default_factory=list)
    frozen_at: datetime

    @model_validator(mode="after")
    def prediction_matches_decision(self) -> HistoricalPredictionMaterial:
        allowed = {
            "merged": {"accept", "accept_with_scope"},
            "not-merged": {"check", "revise", "reject"},
            "abstain": {"unresolved"},
        }
        if self.mergeability_decision not in allowed[self.predicted_outcome]:
            raise ValueError("historical prediction and mergeability decision disagree")
        if self.predicted_outcome == "abstain" and self.confidence != "not-applicable":
            raise ValueError("abstentions require not-applicable confidence")
        if self.prediction_policy_id in {
            LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID,
            HISTORICAL_SCORE_BAND_PREDICTION_POLICY_ID,
        }:
            if self.mergeability_decision == "revise":
                raise ValueError("blind polarized predictions cannot claim active-review REVISE")
            if self.predicted_outcome == "merged":
                if self.prediction_policy_id == LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID:
                    eligible = (
                        self.score_100 is not None
                        and self.score_100 >= MERGE_ACCEPT_SCORE_FLOOR_100
                    )
                    message = "legacy polarized merged predictions require score_100 >= 85"
                else:
                    eligible = (
                        self.score_100 is not None and self.score_100 > PROJECT_FIT_ACCEPT_ABOVE_100
                    )
                    message = "score-band merged predictions require score_100 > 65"
                if not eligible:
                    raise ValueError(message)
        return self


class HistoricalPredictionLock(HistoryModel):
    material: HistoricalPredictionMaterial
    lock_sha256: Digest


class HistoricalHeuristicObservation(HistoryModel):
    """One explicit yes/no/unresolved question used by the historical agent."""

    rule_id: str
    question: str
    status: HistoricalCheckStatus
    blocking: bool
    evidence: list[str] = Field(default_factory=list)
    counterevidence: list[str] = Field(default_factory=list)
    conclusion: str
    failure_code: str | None = None

    @model_validator(mode="after")
    def nonpassing_observations_are_explained(self) -> HistoricalHeuristicObservation:
        if self.status in {"fail", "unresolved"} and not self.failure_code:
            raise ValueError("fail/unresolved heuristic observations require a failure_code")
        if self.status in {"pass", "not-applicable"} and self.failure_code is not None:
            raise ValueError("pass/not-applicable observations cannot carry a failure_code")
        if not self.conclusion.strip():
            raise ValueError("heuristic observations require a human-readable conclusion")
        return self


class HistoricalExplainableJudgmentMaterial(HistoryModel):
    """Frozen rule trace; deliberately contains no learned or weighted score."""

    schema_version: Literal["0.5"] = "0.5"
    policy_id: HistoricalExplainablePolicyId = "historical-explainable-agent-v0.5-r2"
    case_id: str
    candidate_sha256: Digest
    test_plan_sha256: Digest
    evidence_sha256: Digest
    observations: list[HistoricalHeuristicObservation] = Field(min_length=1)
    decision: Literal["accept_with_scope", "revise", "reject", "unresolved"]
    rationale_codes: list[str]
    narrative: str
    frozen_at: datetime

    @model_validator(mode="after")
    def decision_matches_ordered_rules(self) -> HistoricalExplainableJudgmentMaterial:
        blocking_unresolved = any(
            item.blocking and item.status == "unresolved" for item in self.observations
        )
        blocking_failures = [
            item for item in self.observations if item.blocking and item.status == "fail"
        ]
        hard_failure = any(
            (item.failure_code or "").startswith("HARD_POLICY_") for item in blocking_failures
        )
        polarized = self.policy_id == "historical-explainable-agent-v0.5-r5-polarized"
        expected = (
            "unresolved"
            if blocking_unresolved
            else "reject"
            if hard_failure or (polarized and blocking_failures)
            else "revise"
            if blocking_failures
            else "accept_with_scope"
        )
        if self.decision != expected:
            raise ValueError("explainable judgment does not match the ordered rule trace")
        expected_codes = sorted(
            {item.failure_code for item in self.observations if item.blocking and item.failure_code}
        )
        if self.rationale_codes != expected_codes:
            raise ValueError("rationale_codes do not match blocking observations")
        if not self.narrative.strip():
            raise ValueError("explainable judgments require a human-readable narrative")
        return self


class HistoricalExplainableJudgmentLock(HistoryModel):
    material: HistoricalExplainableJudgmentMaterial
    lock_sha256: Digest


class HistoricalGroundTruth(HistoryModel):
    schema_version: Literal["0.5"] = "0.5"
    case_id: str
    repository: str
    pull_number: int = Field(ge=1)
    state: Literal["open", "closed"]
    merged: bool
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    merge_commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    html_url: str
    observed_at: datetime
    prediction_lock_sha256: Digest
    api_response_sha256: Digest

    @model_validator(mode="after")
    def merge_metadata_is_coherent(self) -> HistoricalGroundTruth:
        if self.merged and (self.merged_at is None or self.merge_commit_sha is None):
            raise ValueError("merged ground truth requires merge timestamp and commit")
        if not self.merged and self.merged_at is not None:
            raise ValueError("unmerged ground truth cannot carry merged_at")
        return self


class HistoricalReviewActivitySnapshot(HistoryModel):
    """Post-lock review activity used to derive the narrow CHECK oracle."""

    schema_version: Literal["0.5.1"] = "0.5.1"
    case_id: str
    observed_at: datetime
    last_activity_at: datetime
    last_human_review_at: datetime | None = None
    current_head_human_non_author_review_count: int = Field(default=0, ge=0)
    total_human_non_author_review_count: int = Field(default=0, ge=0)
    pending_human_review_request: bool = False
    api_response_digests: list[Digest] = Field(default_factory=list)

    @model_validator(mode="after")
    def activity_is_coherent(self) -> HistoricalReviewActivitySnapshot:
        timestamps = [self.observed_at, self.last_activity_at]
        if self.last_human_review_at is not None:
            timestamps.append(self.last_human_review_at)
        if any(value.tzinfo is None or value.utcoffset() is None for value in timestamps):
            raise ValueError("historical review timestamps must be timezone-aware")
        if self.last_activity_at > self.observed_at:
            raise ValueError("last activity cannot follow observation")
        if (
            self.last_human_review_at is not None
            and self.last_human_review_at > self.last_activity_at
        ):
            raise ValueError("last activity cannot predate the last human review")
        if (
            self.current_head_human_non_author_review_count
            > self.total_human_non_author_review_count
        ):
            raise ValueError("current-head review count cannot exceed total review count")
        if (
            self.current_head_human_non_author_review_count > 0
            and self.last_human_review_at is None
        ):
            raise ValueError("current-head human reviews require a review timestamp")
        return self


class HistoricalPolarizedDecisionOracle(HistoryModel):
    """Post-reveal decision label and merged-score calibration invariant."""

    schema_version: Literal["0.5.1"] = "0.5.1"
    policy_id: Literal[
        "historical-polarized-oracle-v0.5.1",
        "historical-score-band-oracle-v0.1",
    ] = "historical-score-band-oracle-v0.1"
    case_id: str
    decision: Literal["accept", "check", "reject", "revise"]
    machine_score_100: float | None = Field(default=None, ge=0, le=100)
    merged_score_floor_100: float = Field(default=65, ge=65, le=85)
    merged_score_floor_satisfied: bool | None = None
    pr_age_days: float = Field(ge=0)
    review_idle_days: float | None = Field(default=None, ge=0)
    ground_truth_sha256: Digest
    review_activity_sha256: Digest
    rationale_codes: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def merged_floor_is_only_attached_to_accept(self) -> HistoricalPolarizedDecisionOracle:
        if self.decision == "accept" and self.merged_score_floor_satisfied is None:
            raise ValueError("accepted merge oracle requires a score-floor result")
        if self.decision != "accept" and self.merged_score_floor_satisfied is not None:
            raise ValueError("non-merge oracle cannot carry a merged score-floor result")
        legacy = self.policy_id == LEGACY_HISTORICAL_POLARIZED_ORACLE_POLICY_ID
        expected_threshold = (
            MERGE_ACCEPT_SCORE_FLOOR_100 if legacy else PROJECT_FIT_ACCEPT_ABOVE_100
        )
        if self.merged_score_floor_100 != expected_threshold:
            raise ValueError("merged score threshold disagrees with oracle policy")
        expected_floor = self.machine_score_100 is not None and (
            self.machine_score_100 >= self.merged_score_floor_100
            if legacy
            else self.machine_score_100 > self.merged_score_floor_100
        )
        if self.decision == "accept" and self.merged_score_floor_satisfied != expected_floor:
            raise ValueError("merged score-floor flag disagrees with machine score")
        return self


class HistoricalCalibrationCase(HistoryModel):
    case_id: str
    predicted_outcome: Literal["merged", "not-merged", "abstain"]
    actual_outcome: Literal["merged", "not-merged"]
    correct: bool | None
    prediction_lock_sha256: Digest
    ground_truth_sha256: Digest

    @model_validator(mode="after")
    def correctness_matches_abstention(self) -> HistoricalCalibrationCase:
        if self.predicted_outcome == "abstain" and self.correct is not None:
            raise ValueError("abstentions cannot be marked correct or incorrect")
        if self.predicted_outcome != "abstain" and self.correct is None:
            raise ValueError("covered calibration cases require correctness")
        return self


class HistoricalCalibrationReport(HistoryModel):
    schema_version: Literal["0.5"] = "0.5"
    protocol_id: Literal["historical-pr-blind-calibration-v0.5-r1"] = (
        "historical-pr-blind-calibration-v0.5-r1"
    )
    cases: list[HistoricalCalibrationCase] = Field(min_length=1)
    total_cases: int = Field(ge=1)
    covered_cases: int = Field(ge=0)
    abstained_cases: int = Field(ge=0)
    correct_cases: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    confusion: dict[str, int]

    @model_validator(mode="after")
    def aggregates_match_cases(self) -> HistoricalCalibrationReport:
        covered = [case for case in self.cases if case.correct is not None]
        correct = [case for case in covered if case.correct]
        if self.total_cases != len(self.cases):
            raise ValueError("total_cases does not match cases")
        if self.covered_cases != len(covered):
            raise ValueError("covered_cases does not match cases")
        if self.abstained_cases != self.total_cases - self.covered_cases:
            raise ValueError("abstained_cases does not match cases")
        if self.correct_cases != len(correct):
            raise ValueError("correct_cases does not match cases")
        expected_accuracy = len(correct) / len(covered) if covered else None
        if self.accuracy != expected_accuracy:
            raise ValueError("accuracy does not match covered cases")
        return self


class HistoricalReviewFeedbackItem(HistoryModel):
    feedback_id: str
    source: Literal["review", "review-comment", "issue-comment"]
    author: str
    author_association: str
    is_bot: bool
    review_state: str | None = None
    path: str | None = None
    line: int | None = Field(default=None, ge=1)
    body: str
    html_url: str
    created_at: datetime


class HistoricalReviewEvidence(HistoryModel):
    """Post-lock reviewer evidence for closed, unmerged historical PRs."""

    schema_version: Literal["0.5"] = "0.5"
    case_id: str
    repository: str
    pull_number: int = Field(ge=1)
    prediction_lock_sha256: Digest
    ground_truth_sha256: Digest
    observed_at: datetime
    closed_unmerged: Literal[True] = True
    feedback: list[HistoricalReviewFeedbackItem]
    human_feedback_count: int = Field(ge=0)
    machine_eligible_for_feedback_audit: bool
    eligibility_reasons: list[str] = Field(default_factory=list)
    api_response_digests: list[Digest] = Field(min_length=2)

    @model_validator(mode="after")
    def review_counts_are_coherent(self) -> HistoricalReviewEvidence:
        human_count = sum(not item.is_bot for item in self.feedback)
        if self.human_feedback_count != human_count:
            raise ValueError("human_feedback_count does not match feedback items")
        if self.machine_eligible_for_feedback_audit != bool(self.eligibility_reasons):
            raise ValueError("review feedback eligibility and reasons disagree")
        return self


class HistoricalJudgmentReviewEvidence(HistoryModel):
    """Reviewer evidence revealed only after an explainable judgment lock."""

    schema_version: Literal["0.5"] = "0.5"
    case_id: str
    repository: str
    pull_number: int = Field(ge=1)
    judgment_lock_sha256: Digest
    observed_at: datetime
    state: Literal["closed"] = "closed"
    merged: Literal[False] = False
    pr_author: str
    feedback: list[HistoricalReviewFeedbackItem]
    human_non_author_review_count: int = Field(ge=0)
    explicit_human_non_author_feedback_count: int = Field(ge=0)
    strict_feedback_audit_eligible: bool
    eligibility_reasons: list[str] = Field(default_factory=list)
    api_response_digests: list[Digest] = Field(min_length=4)

    @model_validator(mode="after")
    def strict_eligibility_is_coherent(self) -> HistoricalJudgmentReviewEvidence:
        expected = (
            self.human_non_author_review_count > 0
            and self.explicit_human_non_author_feedback_count > 0
        )
        if self.strict_feedback_audit_eligible != expected:
            raise ValueError("strict reviewer eligibility counts and flag disagree")
        if expected != bool(self.eligibility_reasons):
            raise ValueError("strict reviewer eligibility reasons and flag disagree")
        return self


class HistoricalReviewFinalityEvidence(HistoryModel):
    """Temporal audit separating historical review from attributable closure feedback."""

    schema_version: Literal["0.5"] = "0.5"
    case_id: str
    repository: str
    pull_number: int = Field(ge=1)
    judgment_lock_sha256: Digest
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    head_committed_at: datetime
    closed_at: datetime
    closed_by: str
    final_head_inline_feedback_ids: list[str] = Field(default_factory=list)
    post_head_human_feedback_ids: list[str] = Field(default_factory=list)
    final_head_changes_requested_review_ids: list[str] = Field(default_factory=list)
    explicit_close_feedback_ids: list[str] = Field(default_factory=list)
    stale_close_feedback_ids: list[str] = Field(default_factory=list)
    final_head_feedback_eligible: bool
    closure_reason_attributable: bool
    close_classification: Literal[
        "final-head-changes-requested",
        "explicit-human-technical-close",
        "stale-inactivity",
        "author-close-without-explicit-reason",
        "other-unattributed-close",
    ]
    calibration_eligible: bool
    eligibility_reasons: list[str] = Field(default_factory=list)
    api_response_digests: list[Digest] = Field(min_length=6)

    @model_validator(mode="after")
    def finality_flags_are_coherent(self) -> HistoricalReviewFinalityEvidence:
        expected_final = bool(
            self.final_head_inline_feedback_ids
            or self.post_head_human_feedback_ids
            or self.final_head_changes_requested_review_ids
        )
        if self.final_head_feedback_eligible != expected_final:
            raise ValueError("final-head feedback flag does not match temporal evidence")
        expected_attribution = self.close_classification in {
            "final-head-changes-requested",
            "explicit-human-technical-close",
        }
        if self.closure_reason_attributable != expected_attribution:
            raise ValueError("closure attribution flag does not match classification")
        expected_eligible = expected_final and expected_attribution
        if self.calibration_eligible != expected_eligible:
            raise ValueError("calibration eligibility does not match finality requirements")
        if self.calibration_eligible != bool(self.eligibility_reasons):
            raise ValueError("calibration eligibility reasons disagree")
        return self
