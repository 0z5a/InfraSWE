from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from infraswe.cli import app
from infraswe.pr_decision.calibration import (
    CalibrationCase,
    audit_calibration_profile,
    build_calibration_profile,
)
from infraswe.pr_decision.cascade import (
    CascadeRecallBudget,
    CascadeStageBudget,
    CorrectionProposal,
    apply_bidirectional_cascade,
    count_accept_corrections,
)
from infraswe.pr_decision.contracts import (
    BASELINE_95_99_CONTRACT,
    PRECISION_95_99_95_CONTRACT,
    DecisionPrediction,
    PolicyIdentity,
    PRCaseIdentity,
    canonical_sha256,
    integer_error_budget,
)
from infraswe.pr_decision.errorbook import (
    ErrorAuditOnly,
    ErrorRecord,
    audit_errorbook,
    seal_errorbook,
)
from infraswe.pr_decision.evidence import EvidenceClaim, EvidenceRequest
from infraswe.pr_decision.label_vault import (
    LabelVaultRecord,
    audit_label_vault,
    build_learning_record,
    seal_label_vault,
)
from infraswe.pr_decision.obligations import Obligation, ObligationMap, ordered_obligations
from infraswe.pr_decision.precedent import (
    DecisionPrecedent,
    PrecedentQuery,
    retrieve_contrastive_precedents,
)
from infraswe.pr_decision.project_router import (
    ProjectDecisionProfile,
    route_project_profile,
)
from infraswe.pr_decision.release_gate import (
    DecisionEvaluationCase,
    evaluate_release_gate,
)
from infraswe.pr_decision.report.decision_card import (
    AutomationCoverage,
    DecisionCost,
    DecisionReliabilityCardMaterial,
    audit_reliability_card,
    seal_reliability_card,
)
from infraswe.pr_decision.snapshot import (
    OutcomeBlindSnapshotMaterial,
    audit_snapshot,
    seal_snapshot,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
BASE_SHA = "1" * 40
HEAD_SHA = "2" * 40
NOW = datetime(2026, 9, 5, tzinfo=UTC)


def case_identity() -> PRCaseIdentity:
    return PRCaseIdentity(
        repository="owner/repo",
        pr_number=42,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        prediction_at=NOW,
        label_schema_version="0.6.1",
        issue_or_patch_family="family-1",
    )


def policy_identity() -> PolicyIdentity:
    return PolicyIdentity(
        policy_digest=DIGEST_A,
        retrieval_index_digest=DIGEST_B,
        project_profile_digest=DIGEST_C,
        decision_profile_digest=DIGEST_A,
    )


def claim(
    claim_id: str,
    disposition: str,
    *,
    authority: str = "executable-verifier",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim=f"claim {claim_id}",
        source_ref=f"evidence/{claim_id}.json",
        authority=authority,
        observed_at=NOW,
        head_sha=HEAD_SHA,
        disposition=disposition,
        case_identity_sha256=canonical_sha256(case_identity()),
    )


def test_prediction_uses_frozen_score_bands_and_names_check_uncertainty() -> None:
    assert DecisionPrediction(label="reject", overall_score_100=49).label == "reject"
    assert (
        DecisionPrediction(
            label="check", overall_score_100=65, missing_obligations=["deploy.test"]
        ).label
        == "check"
    )
    assert DecisionPrediction(label="accept", overall_score_100=65.1).label == "accept"
    with pytest.raises(ValidationError, match="frozen overall score band"):
        DecisionPrediction(label="accept", overall_score_100=65)
    with pytest.raises(ValidationError, match="explicitly named unresolved"):
        DecisionPrediction(label="check", overall_score_100=60)
    blocked = DecisionPrediction(
        label="reject",
        overall_score_100=90,
        decision_basis="blocking-obligation",
        evidence_refs=["verifier.fail"],
        blocking_obligations=["deploy.matrix"],
    )
    assert blocked.label == "reject"


def test_outcome_blind_snapshot_rejects_label_leakage_and_wrong_time() -> None:
    material = OutcomeBlindSnapshotMaterial(
        case_identity=case_identity(),
        policy_identity=policy_identity(),
        claims=[claim("source.ok", "supports", authority="source-code")],
        observations={"changed_files": ["src/a.py"]},
    )
    sealed = seal_snapshot(material)
    assert audit_snapshot(sealed) == []
    tampered = sealed.model_copy(update={"snapshot_sha256": DIGEST_A})
    assert audit_snapshot(tampered) == ["snapshot digest mismatch"]

    with pytest.raises(ValidationError, match="label-vault leakage"):
        OutcomeBlindSnapshotMaterial(
            case_identity=case_identity(),
            policy_identity=policy_identity(),
            observations={"nested": {"oracle_label": "accept"}},
        )
    future_claim = claim("future.evidence", "supports").model_copy(
        update={"observed_at": NOW + timedelta(seconds=1)}
    )
    with pytest.raises(ValidationError, match="after prediction_at"):
        OutcomeBlindSnapshotMaterial(
            case_identity=case_identity(),
            policy_identity=policy_identity(),
            claims=[future_claim],
        )


def test_label_vault_is_post_prediction_and_not_policy_gradient_data() -> None:
    label = LabelVaultRecord(
        case_identity=case_identity(),
        oracle_label="accept",
        label_kind="contract_acceptability",
        oracle_source="human/adjudication.json",
        observed_at=NOW + timedelta(hours=1),
        adjudication_status="reviewed",
        decisive_evidence_refs=["review-1"],
    )
    vault = seal_label_vault("vault-1", [label])
    assert audit_label_vault(vault) == []
    learning = build_learning_record(
        snapshot_sha256=DIGEST_A,
        decision_record_sha256=DIGEST_B,
        label=label,
    )
    assert learning.policy_gradient_eligible is False
    assert "external-policy" in learning.allowed_uses
    with pytest.raises(ValueError, match="duplicate"):
        seal_label_vault("duplicate-vault", [label, label])

    with pytest.raises(ValidationError, match="cannot predate"):
        LabelVaultRecord(
            case_identity=case_identity(),
            oracle_label="accept",
            label_kind="upstream_outcome",
            oracle_source="github",
            observed_at=NOW - timedelta(seconds=1),
            adjudication_status="unreviewed",
        )


def test_errorbook_only_contains_structured_current_policy_errors() -> None:
    record = ErrorRecord(
        case_identity=case_identity(),
        policy_identity=policy_identity(),
        prediction=DecisionPrediction(label="reject", overall_score_100=20),
        audit_only=ErrorAuditOnly(
            oracle_label="accept",
            label_kind="contract_acceptability",
            oracle_source="human/adjudication.json",
            adjudication_status="reviewed",
            failure_owner="reasoning",
            reason_code="evidence_present_reasoning_miss",
            decisive_evidence_ref="review-1",
        ),
    )
    book = seal_errorbook("errors-policy-a", DIGEST_A, [record])
    assert audit_errorbook(book) == []
    with pytest.raises(ValueError, match="duplicate"):
        seal_errorbook("duplicates", DIGEST_A, [record, record])
    with pytest.raises(ValueError, match="frozen policy"):
        seal_errorbook("wrong-policy", DIGEST_B, [record])
    with pytest.raises(ValidationError, match="disagreement"):
        ErrorRecord(
            case_identity=case_identity(),
            policy_identity=policy_identity(),
            prediction=DecisionPrediction(label="accept", overall_score_100=90),
            audit_only=record.audit_only,
        )


def test_obligation_map_keeps_order_and_names_missing_evidence() -> None:
    obligations = [
        Obligation(
            obligation_id="perf.raw",
            dimension="performance",
            question="Does raw performance improve?",
            status="satisfied",
            blocking=True,
            evidence_refs=["benchmark-1"],
        ),
        Obligation(
            obligation_id="maint.scope",
            dimension="maintainability",
            question="Is the change maintainable?",
            status="satisfied",
            blocking=True,
            evidence_refs=["source-1"],
        ),
        Obligation(
            obligation_id="deploy.matrix",
            dimension="deployability",
            question="Does the deployment matrix pass?",
            status="unknown",
            blocking=True,
            missing_evidence=["build-or-test"],
        ),
    ]
    obligation_map = ObligationMap(
        case_identity_sha256=canonical_sha256(case_identity()), obligations=obligations
    )
    assert [item.dimension for item in ordered_obligations(obligation_map)] == [
        "maintainability",
        "deployability",
        "performance",
    ]
    assert obligation_map.blocking_unknowns[0].obligation_id == "deploy.matrix"
    with pytest.raises(ValidationError, match="name missing evidence"):
        Obligation(
            obligation_id="deploy.unknown",
            dimension="deployability",
            question="Can it deploy?",
            status="unknown",
            blocking=True,
        )


def precedent(role: str, decision: str, identifier: str, family: str = "safe") -> DecisionPrecedent:
    return DecisionPrecedent(
        precedent_id=identifier,
        repository="owner/repo",
        project="vllm",
        module="scheduler",
        mechanism="batching",
        issue_or_patch_family=family,
        decision=decision,
        role=role,
        decisive_evidence_refs=[f"evidence-{identifier}"],
        observed_at=NOW - timedelta(days=1),
        head_sha=HEAD_SHA,
        split="train",
    )


def test_precedent_retrieval_is_contrastive_and_leak_guarded() -> None:
    query = PrecedentQuery(
        repository="owner/repo",
        project="vllm",
        module="scheduler",
        mechanism="batching",
        prediction_at=NOW,
        split="heldout",
        excluded_families=["leaked-family"],
    )
    records = [
        precedent("historical-accept", "accept", "accept-1"),
        precedent("historical-reject", "reject", "reject-1"),
        precedent("legitimate-exception", "check", "exception-1"),
        precedent("historical-accept", "accept", "leaked", "leaked-family"),
    ]
    bundle = retrieve_contrastive_precedents(query, records, index_digest=DIGEST_A)
    assert [item.precedent_id for item in bundle.accepts] == ["accept-1"]
    assert len(bundle.rejects) == len(bundle.legitimate_exceptions) == 1
    with pytest.raises(ValidationError, match="requires Accept, Reject, and exception"):
        retrieve_contrastive_precedents(query, records[:2], index_digest=DIGEST_A)


def test_project_router_uses_supported_profile_and_shared_fallback() -> None:
    profile = ProjectDecisionProfile(
        profile_id="vllm-profile",
        project="vllm",
        profile_digest=DIGEST_A,
        training_support=100,
        routing_features={"project": "vllm", "module": "scheduler"},
    )
    selected = route_project_profile(
        project="vllm",
        shared_profile_digest=DIGEST_B,
        profiles={"vllm": profile},
        minimum_project_support=50,
    )
    fallback = route_project_profile(
        project="sglang",
        shared_profile_digest=DIGEST_B,
        profiles={"vllm": profile},
        minimum_project_support=50,
    )
    assert selected.route_kind == "project-conditioned"
    assert fallback.route_kind == "shared-fallback"
    with pytest.raises(ValidationError, match="forbidden features"):
        ProjectDecisionProfile(
            profile_id="leaky",
            project="vllm",
            profile_digest=DIGEST_A,
            training_support=100,
            routing_features={"pr_number": 42},
        )


def test_bidirectional_cascade_only_applies_decisive_corrections() -> None:
    initial = DecisionPrediction(label="accept", overall_score_100=90)
    challenger = CorrectionProposal(
        proposal_id="challenge-1",
        direction="accept-challenger",
        from_label="accept",
        revised_prediction=DecisionPrediction(
            label="reject",
            overall_score_100=90,
            decision_basis="blocking-obligation",
            evidence_refs=["verifier.fail"],
            blocking_obligations=["deploy.matrix"],
        ),
        evidence_refs=["verifier.fail"],
        reason="verifier demonstrates a contract violation",
    )
    request = EvidenceRequest(
        state="NEEDS_BUILD_OR_TEST",
        obligation_id="deploy.matrix",
        requested_sources=["build-or-test"],
        reason="missing target platform result",
    )
    blocked = apply_bidirectional_cascade(
        initial=initial,
        proposals=[challenger],
        claims=[claim("verifier.fail", "unknown")],
        evidence_requests=[request],
    )
    assert blocked.final_prediction == initial
    assert blocked.blocked_proposal_ids == ["challenge-1"]
    assert blocked.evidence_requests[0].state == "NEEDS_BUILD_OR_TEST"

    applied = apply_bidirectional_cascade(
        initial=initial,
        proposals=[challenger],
        claims=[claim("verifier.fail", "refutes")],
        case_identity=case_identity(),
        expected_claim_digests={
            "verifier.fail": canonical_sha256(claim("verifier.fail", "refutes"))
        },
    )
    assert applied.final_prediction.label == "reject"
    assert applied.applied_proposal_ids == ["challenge-1"]

    check_proposal = CorrectionProposal(
        proposal_id="check-1",
        direction="accept-challenger",
        from_label="accept",
        revised_prediction=DecisionPrediction(
            label="check",
            overall_score_100=90,
            decision_basis="unresolved-obligation",
            evidence_refs=["matrix.unknown"],
            missing_obligations=["deploy.matrix"],
        ),
        evidence_refs=["matrix.unknown"],
        reason="authoritative matrix evidence is unresolved",
    )
    check_result = apply_bidirectional_cascade(
        initial=initial,
        proposals=[check_proposal],
        claims=[claim("matrix.unknown", "unknown", authority="build-or-test")],
        case_identity=case_identity(),
        expected_claim_digests={
            "matrix.unknown": canonical_sha256(
                claim("matrix.unknown", "unknown", authority="build-or-test")
            )
        },
    )
    assert check_result.final_prediction.label == "check"


def test_cascade_recall_budget_composes_conditional_stage_recall() -> None:
    budget = CascadeRecallBudget(
        stages=[
            CascadeStageBudget(stage_id="initial", minimum_conditional_accept_recall=0.999),
            CascadeStageBudget(stage_id="challenger", minimum_conditional_accept_recall=0.991),
        ],
        minimum_total_accept_recall=0.99,
    )
    assert budget.composed_minimum_accept_recall == pytest.approx(0.990009)
    with pytest.raises(ValidationError, match="exhaust"):
        CascadeRecallBudget(
            stages=[
                CascadeStageBudget(stage_id="a", minimum_conditional_accept_recall=0.99),
                CascadeStageBudget(stage_id="b", minimum_conditional_accept_recall=0.99),
            ],
            minimum_total_accept_recall=0.99,
        )


def test_accept_correction_counts_track_fn_and_fp_separately() -> None:
    delta = count_accept_corrections(
        oracle_by_case={"a": "accept", "b": "reject", "c": "accept", "d": "check"},
        old_by_case={"a": "reject", "b": "accept", "c": "accept", "d": "reject"},
        new_by_case={"a": "accept", "b": "reject", "c": "reject", "d": "accept"},
    )
    assert delta.recovered_old_fn == 1
    assert delta.introduced_new_fn == 1
    assert delta.removed_old_fp == 1
    assert delta.introduced_new_fp == 1


def test_calibration_reports_when_workpoint_is_unreachable() -> None:
    cases = [
        CalibrationCase(
            case_id="a", oracle_label="accept", p_accept=0.9, non_accept_label="reject"
        ),
        CalibrationCase(
            case_id="b", oracle_label="accept", p_accept=0.8, non_accept_label="reject"
        ),
        CalibrationCase(
            case_id="c", oracle_label="reject", p_accept=0.85, non_accept_label="reject"
        ),
    ]
    profile = build_calibration_profile(
        profile_id="unreachable",
        policy_digest=DIGEST_A,
        population_digest=DIGEST_B,
        purpose="evidence-routing",
        cases=cases,
        thresholds=[0.8, 0.85, 0.9],
        recall_floor=0.99,
        precision_target=0.95,
        generated_at=NOW,
    )
    assert profile.material.target_reachable is False
    assert profile.material.selected_threshold is None
    assert profile.material.max_precision_at_recall_floor == pytest.approx(2 / 3)
    assert audit_calibration_profile(profile) == []


def gate_cases_with_five_accept_false_positives() -> list[DecisionEvaluationCase]:
    cases = [
        DecisionEvaluationCase(
            case_id=f"accept-{index}", predicted_label="accept", oracle_label="accept"
        )
        for index in range(50)
    ]
    cases.extend(
        DecisionEvaluationCase(
            case_id=f"reject-correct-{index}", predicted_label="reject", oracle_label="reject"
        )
        for index in range(45)
    )
    cases.extend(
        DecisionEvaluationCase(
            case_id=f"reject-fp-{index}", predicted_label="accept", oracle_label="reject"
        )
        for index in range(5)
    )
    return cases


def test_precision_is_an_independent_optional_hard_gate() -> None:
    cases = gate_cases_with_five_accept_false_positives()
    baseline = evaluate_release_gate(cases, BASELINE_95_99_CONTRACT)
    strict = evaluate_release_gate(cases, PRECISION_95_99_95_CONTRACT)
    assert baseline.passed is True
    assert baseline.metrics.accuracy3 == 0.95
    assert baseline.metrics.recall_accept == 1
    assert strict.passed is False
    assert strict.precision_accept_passed is False
    assert strict.metrics.precision_accept == pytest.approx(50 / 55)
    assert strict.integer_budget.maximum_accept_false_positives == 2


def test_reported_cohort_integer_budget_matches_v061_contract() -> None:
    budget = integer_error_budget(
        PRECISION_95_99_95_CONTRACT,
        eligible_cases=79_948,
        oracle_accept_cases=48_791,
    )
    assert budget.required_exact_matches == 75_951
    assert budget.required_accept_true_positives == 48_304
    assert budget.maximum_accept_false_positives == 2_542


def test_checked_in_metric_contracts_match_code_presets() -> None:
    root = Path(__file__).parents[1]
    baseline = json.loads(
        (root / "profiles/pr-decision-accuracy95-recall99-v0.6.1.json").read_text()
    )
    strict = json.loads(
        (root / "profiles/pr-decision-accuracy95-recall99-precision95-v0.6.1.json").read_text()
    )
    assert baseline == BASELINE_95_99_CONTRACT.model_dump(mode="json")
    assert strict == PRECISION_95_99_95_CONTRACT.model_dump(mode="json")


def test_zero_support_and_unverified_invalid_exclusions_fail_closed() -> None:
    no_accepts = [
        DecisionEvaluationCase(case_id="r", predicted_label="reject", oracle_label="reject")
    ]
    assert evaluate_release_gate(no_accepts, PRECISION_95_99_95_CONTRACT).passed is False
    cases = [
        DecisionEvaluationCase(case_id="a", predicted_label="accept", oracle_label="accept"),
        DecisionEvaluationCase(
            case_id="timeout",
            predicted_label="reject",
            oracle_label="accept",
            valid=False,
            invalid_reason="neutral abandon: hard timeout",
        ),
    ]
    result = evaluate_release_gate(cases, PRECISION_95_99_95_CONTRACT)
    assert result.passed is False
    assert "unverified eligibility" in result.failure_reasons[0]
    assert result.metrics.eligible_cases == 1
    assert result.metrics.invalid_cases == 1


def test_reliability_card_seals_contract_digests_and_cost() -> None:
    gate = evaluate_release_gate(
        [DecisionEvaluationCase(case_id="a", predicted_label="accept", oracle_label="accept")],
        PRECISION_95_99_95_CONTRACT,
    )
    material = DecisionReliabilityCardMaterial(
        card_id="holdout-card-1",
        evaluation_track="frozen_policy_holdout_result",
        policy_digest=DIGEST_A,
        harness_digest=DIGEST_B,
        retrieval_index_digest=DIGEST_C,
        label_vault_digest=DIGEST_A,
        population_digest=DIGEST_B,
        gate_result=gate,
        coverage=AutomationCoverage(
            automatic_cases=1,
            unresolved_cases=0,
            human_assisted_cases=0,
            invalid_or_abandoned_cases=0,
        ),
        cost=DecisionCost(input_tokens=100, output_tokens=10, tool_calls=2),
        statistical_assumptions=["point estimates only; no iid confidence claim"],
        selection_protocol="single frozen policy evaluated once on heldout data",
        attempted_policy_digests=[DIGEST_A],
        generated_at=NOW,
    )
    card = seal_reliability_card(material)
    assert audit_reliability_card(card) == []
    assert card.material.gate_result.passed is True


def test_pr_decision_gate_cli_writes_failed_result(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    output_path = tmp_path / "gate.json"
    cases_path.write_text(
        '[{"case_id":"a","predicted_label":"reject","oracle_label":"accept"}]',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["pr-decision", "gate", str(cases_path), "--output", str(output_path)],
    )
    assert result.exit_code == 2
    assert output_path.is_file()
    assert '"passed": false' in output_path.read_text(encoding="utf-8")
