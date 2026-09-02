from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from infraswe.history.blind import (
    assert_outcome_free,
    audit_prediction_lock,
    build_calibration_report,
    canonical_sha256,
    compile_prediction,
    freeze_prediction,
    join_revealed_case,
)
from infraswe.history.oracle import (
    compile_polarized_oracle,
    polarized_oracle_matches_machine,
)
from infraswe.models.history import (
    BlindEvaluationEvidence,
    HistoricalCheckResult,
    HistoricalGroundTruth,
    HistoricalPRCandidate,
    HistoricalReviewActivitySnapshot,
    HistoricalReviewEvidence,
    HistoricalReviewFeedbackItem,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def candidate() -> HistoricalPRCandidate:
    return HistoricalPRCandidate(
        case_id="vllm-pr-123",
        project="vllm",
        repository="vllm-project/vllm",
        pull_number=123,
        title="kernel correction",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        base_ref="main",
        base_tip_sha="9" * 40,
        base_sha="a" * 40,
        base_derivation="first-pr-commit-first-parent-path-parity",
        head_sha="b" * 40,
        pr_commit_shas=["b" * 40],
        changed_files=2,
        additions=10,
        deletions=3,
        paths=["csrc/op.cu", "tests/test_op.py"],
        acquisition_query="outcome-free GraphQL",
        selection_policy_id="fixed-seed-v1",
    )


def evidence(
    *,
    failed: bool = False,
    unresolved: bool = False,
    score_100: float | None = 90,
) -> BlindEvaluationEvidence:
    item = candidate()
    status = "unresolved" if unresolved else "fail" if failed else "pass"
    code = "REMOTE_RUNTIME_UNAVAILABLE" if unresolved else "UNIT_TEST_FAILED" if failed else None
    checks = [
        HistoricalCheckResult(name="checkout", category="checkout", status="pass"),
        HistoricalCheckResult(name="diff-check", category="static", status="pass"),
        HistoricalCheckResult(
            name="affected-test",
            category="unit",
            status=status,
            return_code=1 if failed else 0 if not unresolved else None,
            failure_code=code,
        ),
    ]
    return BlindEvaluationEvidence(
        case_id=item.case_id,
        candidate_sha256=canonical_sha256(item),
        test_plan_sha256="sha256:" + "c" * 64,
        environment_sha256="sha256:" + "d" * 64,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        stage="F1-smoke",
        checks=checks,
        candidate_failure_codes=[code] if failed and code else [],
        infrastructure_failure_codes=[code] if unresolved and code else [],
        project_fit_score_100=score_100,
    )


def truth(lock_sha256: str, *, merged: bool) -> HistoricalGroundTruth:
    return HistoricalGroundTruth(
        case_id="vllm-pr-123",
        repository="vllm-project/vllm",
        pull_number=123,
        state="closed",
        merged=merged,
        merged_at=NOW + timedelta(seconds=3) if merged else None,
        closed_at=NOW + timedelta(seconds=3),
        merge_commit_sha="e" * 40 if merged else None,
        html_url="https://github.com/vllm-project/vllm/pull/123",
        observed_at=NOW + timedelta(seconds=4),
        prediction_lock_sha256=lock_sha256,
        api_response_sha256="sha256:" + "f" * 64,
    )


def test_outcome_fields_are_rejected_recursively() -> None:
    assert_outcome_free(candidate().model_dump(mode="json"))
    with pytest.raises(ValueError, match="ground-truth leakage"):
        assert_outcome_free({"candidate": {"merged": True}})


def test_passing_machine_evidence_is_frozen_before_reveal() -> None:
    item = candidate()
    material = compile_prediction(item, evidence(), frozen_at=NOW + timedelta(seconds=2))
    lock = freeze_prediction(material)
    assert material.predicted_outcome == "merged"
    assert audit_prediction_lock(lock)
    case = join_revealed_case(lock, truth(lock.lock_sha256, merged=True))
    assert case.correct is True


def test_candidate_failure_predicts_not_merged() -> None:
    item = candidate()
    material = compile_prediction(item, evidence(failed=True), frozen_at=NOW + timedelta(seconds=2))
    assert material.predicted_outcome == "not-merged"
    assert material.mergeability_decision == "revise"


def test_polarized_prediction_requires_85_and_legacy_r1_remains_replayable() -> None:
    item = candidate()
    below = compile_prediction(
        item,
        evidence(score_100=84.99),
        frozen_at=NOW,
        policy_id="historical-merge-prediction-v0.5-r2-polarized",
    )
    assert below.predicted_outcome == "not-merged"
    assert below.mergeability_decision == "reject"
    missing = compile_prediction(
        item,
        evidence(score_100=None),
        frozen_at=NOW,
        policy_id="historical-merge-prediction-v0.5-r2-polarized",
    )
    assert missing.predicted_outcome == "abstain"
    legacy = compile_prediction(
        item,
        evidence(score_100=None),
        frozen_at=NOW,
        policy_id="historical-merge-prediction-v0.5-r1",
    )
    assert legacy.predicted_outcome == "merged"


def test_infrastructure_failure_abstains() -> None:
    item = candidate()
    material = compile_prediction(
        item, evidence(unresolved=True), frozen_at=NOW + timedelta(seconds=2)
    )
    assert material.predicted_outcome == "abstain"
    assert material.mergeability_decision == "unresolved"


def test_reveal_cannot_predate_prediction() -> None:
    item = candidate()
    lock = freeze_prediction(
        compile_prediction(item, evidence(), frozen_at=NOW + timedelta(seconds=5))
    )
    revealed = truth(lock.lock_sha256, merged=True).model_copy(
        update={"observed_at": NOW + timedelta(seconds=4)}
    )
    with pytest.raises(ValueError, match="before the prediction"):
        join_revealed_case(lock, revealed)


def test_calibration_uses_coverage_and_confusion() -> None:
    item = candidate()
    positive = freeze_prediction(
        compile_prediction(item, evidence(), frozen_at=NOW + timedelta(seconds=2))
    )
    abstention = freeze_prediction(
        compile_prediction(item, evidence(unresolved=True), frozen_at=NOW + timedelta(seconds=2))
    )
    cases = [
        join_revealed_case(positive, truth(positive.lock_sha256, merged=True)),
        join_revealed_case(abstention, truth(abstention.lock_sha256, merged=False)),
    ]
    report = build_calibration_report(cases)
    assert report.covered_cases == 1
    assert report.abstained_cases == 1
    assert report.accuracy == 1.0
    assert report.confusion["predicted_merged_actual_merged"] == 1
    assert report.confusion["abstained_actual_not_merged"] == 1


def test_post_lock_review_evidence_requires_consistent_human_count() -> None:
    item = HistoricalReviewFeedbackItem(
        feedback_id="review-comment:1",
        source="review-comment",
        author="maintainer",
        author_association="MEMBER",
        is_bot=False,
        path="csrc/op.cu",
        line=12,
        body="The stream belongs to the caller; do not replace it here.",
        html_url="https://github.com/vllm-project/vllm/pull/123#discussion_r1",
        created_at=NOW,
    )
    evidence = HistoricalReviewEvidence(
        case_id="vllm-pr-123",
        repository="vllm-project/vllm",
        pull_number=123,
        prediction_lock_sha256="sha256:" + "a" * 64,
        ground_truth_sha256="sha256:" + "b" * 64,
        observed_at=NOW,
        feedback=[item],
        human_feedback_count=1,
        machine_eligible_for_feedback_audit=True,
        eligibility_reasons=["HUMAN_INLINE_TECHNICAL_REVIEW"],
        api_response_digests=["sha256:" + "c" * 64, "sha256:" + "d" * 64],
    )
    assert evidence.closed_unmerged is True
    invalid = evidence.model_dump()
    invalid["human_feedback_count"] = 0
    with pytest.raises(ValueError, match="human_feedback_count"):
        HistoricalReviewEvidence.model_validate(invalid)


def test_polarized_oracle_reserves_check_for_active_new_review() -> None:
    observed = NOW + timedelta(seconds=4)
    item = candidate().model_copy(update={"created_at": NOW - timedelta(days=10)})
    open_truth = truth("sha256:" + "a" * 64, merged=False).model_copy(
        update={"state": "open", "closed_at": None}
    )
    review = HistoricalReviewActivitySnapshot(
        case_id=item.case_id,
        observed_at=observed,
        last_activity_at=NOW - timedelta(days=1),
        last_human_review_at=NOW - timedelta(days=1),
        current_head_human_non_author_review_count=1,
        total_human_non_author_review_count=1,
    )
    oracle = compile_polarized_oracle(item, open_truth, review, machine_score_100=80)
    assert oracle.decision == "check"
    assert polarized_oracle_matches_machine(oracle, "check")
    assert polarized_oracle_matches_machine(oracle, "revise")


def test_polarized_oracle_treats_stale_reviewed_open_as_reject() -> None:
    observed = NOW + timedelta(seconds=4)
    item = candidate().model_copy(update={"created_at": NOW - timedelta(days=180)})
    open_truth = truth("sha256:" + "a" * 64, merged=False).model_copy(
        update={"state": "open", "closed_at": None}
    )
    review = HistoricalReviewActivitySnapshot(
        case_id=item.case_id,
        observed_at=observed,
        last_activity_at=NOW - timedelta(days=60),
        last_human_review_at=NOW - timedelta(days=60),
        total_human_non_author_review_count=2,
    )
    oracle = compile_polarized_oracle(item, open_truth, review, machine_score_100=80)
    assert oracle.decision == "reject"
    assert oracle.rationale_codes == ["STALE_REVIEWED_OPEN_REJECT_ORACLE"]


@pytest.mark.parametrize(("score", "expected"), [(84.99, False), (85.0, True)])
def test_merged_oracle_requires_machine_score_at_least_85(score: float, expected: bool) -> None:
    item = candidate()
    merged_truth = truth("sha256:" + "a" * 64, merged=True)
    review = HistoricalReviewActivitySnapshot(
        case_id=item.case_id,
        observed_at=merged_truth.observed_at,
        last_activity_at=NOW + timedelta(seconds=3),
    )
    oracle = compile_polarized_oracle(
        item,
        merged_truth,
        review,
        machine_score_100=score,
    )
    assert oracle.decision == "accept"
    assert oracle.merged_score_floor_satisfied is expected
    assert polarized_oracle_matches_machine(oracle, "accept") is expected
