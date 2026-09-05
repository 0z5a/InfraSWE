from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infraswe.pr_decision.cascade import CorrectionProposal, apply_bidirectional_cascade
from infraswe.pr_decision.contracts import (
    STRICT_95_99_99_CONTRACT,
    DecisionMicroscores,
    DecisionPrediction,
    PolicyIdentity,
    PRCaseIdentity,
    canonical_sha256,
    integer_error_budget,
    minimum_successes,
)
from infraswe.pr_decision.evidence import EvidenceClaim
from infraswe.pr_decision.obligations import Obligation, ObligationMap
from infraswe.pr_decision.project_router import ProjectDecisionProfile
from infraswe.pr_decision.release_gate import DecisionEvaluationCase, evaluate_release_gate
from infraswe.pr_decision.snapshot import (
    OutcomeBlindSnapshotMaterial,
    audit_snapshot,
    seal_snapshot,
)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/historical_prs"))
from decision_v061_shadow import _write_once, guarded_case  # noqa: E402

NOW = datetime(2026, 9, 5, tzinfo=UTC)
DIGEST = "sha256:" + "a" * 64


def identity() -> PRCaseIdentity:
    return PRCaseIdentity(
        repository="owner/repo",
        pr_number=1,
        base_sha="1" * 40,
        head_sha="2" * 40,
        prediction_at=NOW,
        label_schema_version="0.6.1",
    )


def snapshot() -> OutcomeBlindSnapshotMaterial:
    return OutcomeBlindSnapshotMaterial(
        case_identity=identity(),
        policy_identity=PolicyIdentity(
            policy_digest=DIGEST,
            retrieval_index_digest=DIGEST,
            project_profile_digest=DIGEST,
            decision_profile_digest=DIGEST,
        ),
    )


def test_three_independent_gates_and_exact_decimal_budget() -> None:
    # 100% recall + >95% accuracy cannot compensate for <99% precision.
    cases = [
        DecisionEvaluationCase(
            case_id=str(i), predicted_label="accept", oracle_label="accept" if i < 98 else "reject"
        )
        for i in range(100)
    ]
    gate = evaluate_release_gate(cases, STRICT_95_99_99_CONTRACT)
    assert gate.accuracy_passed and gate.recall_accept_passed
    assert not gate.precision_accept_passed and not gate.passed
    assert gate.release_authorized is False
    assert minimum_successes(100, 0.07) == 7
    assert (
        integer_error_budget(
            STRICT_95_99_99_CONTRACT, eligible_cases=100, oracle_accept_cases=100
        ).maximum_accept_false_positives
        == 1
    )
    with pytest.raises(ValueError, match="population"):
        integer_error_budget(STRICT_95_99_99_CONTRACT, eligible_cases=2, oracle_accept_cases=3)
    zero = STRICT_95_99_99_CONTRACT.model_copy(update={"precision_accept_minimum": 0})
    assert (
        integer_error_budget(
            zero, eligible_cases=10, oracle_accept_cases=1
        ).maximum_accept_false_positives
        is None
    )
    stored = json.loads(
        (ROOT / "profiles/pr-decision-accuracy95-recall99-precision99-v0.6.1.json").read_text()
    )
    assert stored == STRICT_95_99_99_CONTRACT.model_dump(mode="json")


def test_duplicate_and_posthoc_invalid_cannot_pass() -> None:
    correct = DecisionEvaluationCase(
        case_id="same", predicted_label="accept", oracle_label="accept"
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_release_gate([correct] * 1000, STRICT_95_99_99_CONTRACT)
    wrong = DecisionEvaluationCase(
        case_id="wrong",
        predicted_label="reject",
        oracle_label="accept",
        valid=False,
        invalid_reason="posthoc_exclusion",
    )
    result = evaluate_release_gate([correct, wrong], STRICT_95_99_99_CONTRACT)
    assert not result.passed and result.metrics.invalid_cases == 1
    assert evaluate_release_gate([correct], STRICT_95_99_99_CONTRACT).release_authorized is False


def test_structural_snapshot_rejects_free_notes_and_unresolved_memory() -> None:
    material = snapshot()
    with pytest.raises(ValueError, match="observation"):
        OutcomeBlindSnapshotMaterial.model_validate(
            {**material.model_dump(), "observations": {"notes": "oracle_label=accept"}}
        )
    with pytest.raises(ValueError, match="resolver"):
        OutcomeBlindSnapshotMaterial.model_validate(
            {**material.model_dump(), "retrieval_memory_refs": ["label-vault.json"]}
        )
    sealed = seal_snapshot(material)
    sealed.material.observations["notes"] = "oracle_label=accept"
    # Even rehashing a mutated object must not bypass structural validation.
    sealed.snapshot_sha256 = canonical_sha256(sealed.material)
    assert audit_snapshot(sealed)
    assert sealed.provenance_authenticated is False


def test_nested_project_outcome_and_unqualified_microscores_rejected() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        ProjectDecisionProfile(
            profile_id="p",
            project="vllm",
            profile_digest=DIGEST,
            training_support=100,
            routing_features={"nested": {"merged": True}},
        )
    with pytest.raises(ValueError, match="EvidencePack"):
        DecisionMicroscores(project_fit_100=99, benchmark_trust_100=99)


def rescue_inputs():
    case = identity()
    evidence = EvidenceClaim(
        claim_id="verified.proof",
        claim="targeted proof",
        source_ref="proof.json",
        authority="executable-verifier",
        observed_at=NOW,
        head_sha=case.head_sha,
        disposition="supports",
        case_identity_sha256=canonical_sha256(case),
    )
    initial = DecisionPrediction(
        label="check", overall_score_100=60, missing_obligations=["deploy.one", "deploy.two"]
    )
    revised = DecisionPrediction(
        label="accept", overall_score_100=80, evidence_refs=[evidence.claim_id]
    )
    proposal = CorrectionProposal(
        proposal_id="rescue",
        direction="reject-rescuer",
        from_label="check",
        revised_prediction=revised,
        evidence_refs=[evidence.claim_id],
        reason="all obligations closed",
    )
    resolved = ObligationMap(
        case_identity_sha256=canonical_sha256(case),
        obligations=[
            Obligation(
                obligation_id=key,
                dimension="deployability",
                question="proven?",
                status="satisfied",
                blocking=True,
                evidence_refs=[evidence.claim_id],
            )
            for key in ("deploy.one", "deploy.two")
        ],
    )
    return dict(
        initial=initial,
        proposals=[proposal],
        claims=[evidence],
        case_identity=case,
        expected_claim_digests={evidence.claim_id: canonical_sha256(evidence)},
        resolved_obligations=resolved,
    )


@pytest.mark.parametrize(
    "defect", ["foreign-head", "foreign-pr", "unbound", "unclosed", "duplicate"]
)
def test_cascade_rejects_unbound_or_incomplete_rescue(defect: str) -> None:
    args = rescue_inputs()
    assert apply_bidirectional_cascade(**args).final_prediction.label == "accept"
    if defect == "foreign-head":
        args["claims"][0].head_sha = "3" * 40
    elif defect == "foreign-pr":
        args["case_identity"].pr_number = 2
    elif defect == "unbound":
        args["expected_claim_digests"] = {}
    elif defect == "unclosed":
        args["resolved_obligations"].obligations.pop()
    else:
        args["claims"].append(args["claims"][0])
        with pytest.raises(ValueError, match="duplicate"):
            apply_bidirectional_cascade(**args)
        return
    assert apply_bidirectional_cascade(**args).final_prediction.label == "check"


def review_case() -> dict:
    return {
        "head_sha": "2" * 40,
        "review_list_complete": True,
        "human_non_author_review_state_counts": {"APPROVED": 1},
        "final_head_human_non_author_review_state_counts": {"APPROVED": 1},
        "human_non_author_reviews": [
            {
                "state": "APPROVED",
                "is_final_head": True,
                "commit_oid": "2" * 40,
                "submitted_at": "2026-09-01T00:00:00Z",
            }
        ],
    }


@pytest.mark.parametrize("defect", ["old-head", "future", "incomplete", "withdrawn"])
def test_final_head_counterfactual_guard_is_nonmutating(defect: str) -> None:
    case = review_case()
    assert guarded_case(case, NOW)["human_non_author_review_state_counts"]["APPROVED"] == 1
    if defect == "old-head":
        case["human_non_author_reviews"][0]["commit_oid"] = "1" * 40
    elif defect == "future":
        case["human_non_author_reviews"][0]["submitted_at"] = "2027-01-01T00:00:00Z"
    elif defect == "incomplete":
        case["review_list_complete"] = False
    else:
        case["human_non_author_reviews"].append(
            {"state": "DISMISSED", "submitted_at": "2026-09-02T00:00:00Z"}
        )
    original = copy.deepcopy(case)
    assert guarded_case(case, NOW)["human_non_author_review_state_counts"]["APPROVED"] == 0
    assert case == original


def test_sidecars_never_overwrite_conflicting_evidence(tmp_path: Path) -> None:
    output = tmp_path / "shadow-lock.json"
    _write_once(output, {"one": 1}, "digest")
    _write_once(output, {"one": 1}, "digest")
    before = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        _write_once(output, {"one": 2}, "digest")
    assert output.read_bytes() == before


@pytest.mark.parametrize("domain", ["training", "inference", "communication"])
def test_campaign_finalization_never_publishes_deletes_or_stops(
    domain: str, tmp_path: Path
) -> None:
    script = ROOT / f"benchmarks/historical_prs/run_{domain}_bulk_campaign.sh"
    text = script.read_text()
    marker = "# Campaign completion is not release qualification"
    assert marker in text
    tail = text[text.index(marker) :]
    credential = tmp_path / "credential-placeholder"
    credential.write_text("must survive")
    for command in ("git", "vastai", "rm"):
        executable = tmp_path / command
        executable.write_text("#!/bin/bash\nexit 97\n")
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + ":" + os.environ["PATH"],
        "INFRASWE_GITHUB_CREDENTIAL_COPY": str(credential),
        "INFRASWE_PUBLISH_ON_COMPLETE": "1",
        "INFRASWE_STOP_INSTANCE_ON_COMPLETE": "1",
    }
    result = subprocess.run(
        ["/bin/bash", "-eu", "-c", tail], env=env, text=True, capture_output=True, timeout=5
    )
    assert result.returncode == 0
    assert credential.read_text() == "must survive"
    assert "pending independent verification" in result.stdout
    assert "git push" not in text and "vastai stop" not in text and "rm -f" not in text


def test_round_orders_shadow_before_reveal_and_audit_after_chain() -> None:
    text = (ROOT / "benchmarks/historical_prs/run_communication_bulk_round.sh").read_text()
    assert text.index("communication.deployment-pause") < text.index('mkdir -p "${group_dir}"')
    assert text.index("decision_v061_shadow.py freeze") < text.index(
        "reveal_training_bulk_group.py"
    )
    assert text.index("decision_v061_shadow.py audit") > text.index(
        "derive_training_bulk_policy_iteration.py"
    )
