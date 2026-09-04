from __future__ import annotations

from typing import Literal

from infraswe.history.blind import canonical_sha256
from infraswe.models.history import (
    HistoricalGroundTruth,
    HistoricalPolarizedDecisionOracle,
    HistoricalPRCandidate,
    HistoricalReviewActivitySnapshot,
)
from infraswe.policy import (
    CHECK_ACTIVITY_MAX_IDLE_DAYS,
    CHECK_NEW_PR_MAX_AGE_DAYS,
    HISTORICAL_POLARIZED_ORACLE_POLICY_ID,
    LEGACY_HISTORICAL_POLARIZED_ORACLE_POLICY_ID,
    MERGE_ACCEPT_SCORE_FLOOR_100,
    PROJECT_FIT_ACCEPT_ABOVE_100,
    STALE_REVIEWED_OPEN_MIN_AGE_DAYS,
)


def compile_polarized_oracle(
    candidate: HistoricalPRCandidate,
    truth: HistoricalGroundTruth,
    review: HistoricalReviewActivitySnapshot,
    *,
    machine_score_100: float | None,
    policy_id: Literal[
        "historical-polarized-oracle-v0.5.1",
        "historical-score-band-oracle-v0.1",
    ] = HISTORICAL_POLARIZED_ORACLE_POLICY_ID,
) -> HistoricalPolarizedDecisionOracle:
    """Compile a post-lock oracle without feeding outcome data back into the judge."""

    if candidate.case_id != truth.case_id or candidate.case_id != review.case_id:
        raise ValueError("candidate, ground truth, and review activity must share a case_id")
    if candidate.repository != truth.repository or candidate.pull_number != truth.pull_number:
        raise ValueError("candidate and ground truth identify different pull requests")
    if truth.observed_at < candidate.created_at:
        raise ValueError("ground truth cannot predate PR creation")
    if review.observed_at > truth.observed_at:
        raise ValueError("review activity cannot be observed after ground truth")
    if review.last_activity_at < candidate.created_at:
        raise ValueError("review activity cannot predate PR creation")

    age_days = (truth.observed_at - candidate.created_at).total_seconds() / 86_400
    activity_idle_days = (truth.observed_at - review.last_activity_at).total_seconds() / 86_400
    review_idle_days = (
        (truth.observed_at - review.last_human_review_at).total_seconds() / 86_400
        if review.last_human_review_at is not None
        else None
    )
    common = {
        "policy_id": policy_id,
        "case_id": candidate.case_id,
        "machine_score_100": machine_score_100,
        "pr_age_days": age_days,
        "review_idle_days": review_idle_days,
        "ground_truth_sha256": canonical_sha256(truth),
        "review_activity_sha256": canonical_sha256(review),
    }

    if truth.merged:
        if truth.state != "closed":
            raise ValueError("merged pull requests must be closed")
        legacy = policy_id == LEGACY_HISTORICAL_POLARIZED_ORACLE_POLICY_ID
        threshold = MERGE_ACCEPT_SCORE_FLOOR_100 if legacy else PROJECT_FIT_ACCEPT_ABOVE_100
        floor_satisfied = machine_score_100 is not None and (
            machine_score_100 >= threshold if legacy else machine_score_100 > threshold
        )
        return HistoricalPolarizedDecisionOracle(
            **common,
            decision="accept",
            merged_score_floor_100=threshold,
            merged_score_floor_satisfied=floor_satisfied,
            rationale_codes=[
                (
                    "MERGED_PR_SCORE_AT_LEAST_85"
                    if legacy and floor_satisfied
                    else "MERGED_PR_SCORE_BELOW_85_OR_MISSING"
                    if legacy
                    else "MERGED_PR_SCORE_ABOVE_65"
                    if floor_satisfied
                    else "MERGED_PR_SCORE_AT_MOST_65_OR_MISSING"
                )
            ],
        )

    if truth.state == "closed":
        return HistoricalPolarizedDecisionOracle(
            **common,
            decision="reject",
            rationale_codes=["CLOSED_UNMERGED_REJECT_ORACLE"],
        )

    current_head_review_is_recent = (
        review.current_head_human_non_author_review_count > 0
        and review_idle_days is not None
        and review_idle_days <= CHECK_ACTIVITY_MAX_IDLE_DAYS
    )
    pending_review_is_recent = (
        review.pending_human_review_request and activity_idle_days <= CHECK_ACTIVITY_MAX_IDLE_DAYS
    )
    if age_days <= CHECK_NEW_PR_MAX_AGE_DAYS and (
        current_head_review_is_recent or pending_review_is_recent
    ):
        return HistoricalPolarizedDecisionOracle(
            **common,
            decision="check",
            rationale_codes=["ACTIVE_NEW_PR_REVIEW_CHECK_ORACLE"],
        )

    if (
        age_days >= STALE_REVIEWED_OPEN_MIN_AGE_DAYS
        and review.total_human_non_author_review_count > 0
    ):
        reason = "STALE_REVIEWED_OPEN_REJECT_ORACLE"
    else:
        reason = "OPEN_PR_NOT_ACTIVE_NEW_REVIEW_REJECT_ORACLE"
    return HistoricalPolarizedDecisionOracle(
        **common,
        decision="reject",
        rationale_codes=[reason],
    )


def polarized_oracle_matches_machine(
    oracle: HistoricalPolarizedDecisionOracle,
    machine_decision: Literal[
        "accept", "accept_with_scope", "check", "reject", "unresolved", "revise"
    ],
) -> bool:
    normalized = (
        "accept" if machine_decision in {"accept", "accept_with_scope"} else machine_decision
    )
    if normalized == "revise":
        normalized = "check"
    if normalized != oracle.decision:
        return False
    return oracle.decision != "accept" or oracle.merged_score_floor_satisfied is True
