from __future__ import annotations

from datetime import UTC, datetime

import pytest

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.task_quality import (
    AcceptanceObligation,
    AlternativeValidSolutionOutcome,
    BaselineDifferential,
    FeasibilityWitness,
    HumanTaskQualificationReview,
    MutationOutcome,
    NegativeControlOutcome,
    ObligationObservation,
    ObligationOracle,
    TaskLeakageAudit,
    TaskRequirement,
    TaskSpecification,
    TaskTarget,
    VerifierFlakinessAudit,
    WitnessReplayResult,
)
from infraswe.task_quality import (
    audit_acceptance_contract,
    audit_task_seal,
    audit_witness_set,
    build_acceptance_contract,
    build_task_seal,
    build_verifier_result,
    build_witness_set,
    qualify_task,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _obligation(
    obligation_id: str,
    bucket: str,
    *,
    severity: str = "hard",
    failure_owner: str = "candidate",
    evidence_owner: str = "pristine-verifier",
    release_gate: bool = False,
) -> AcceptanceObligation:
    return AcceptanceObligation(
        obligation_id=obligation_id,
        source_requirements=["REQ-CORE"] if bucket != "environment-sentinel" else [],
        profile_provenance=(["environment-profile-v1"] if bucket == "environment-sentinel" else []),
        bucket=bucket,
        severity=severity,
        release_gate=release_gate,
        oracle=ObligationOracle(kind="semantic", reference_id=obligation_id.lower()),
        failure_owner=failure_owner,
        evidence_owner=evidence_owner,
        visibility="public-contract-hidden-cases",
        provenance=["task-author-v1"],
    )


def _fixture():
    specification = TaskSpecification(
        task_id="kernel-contract-001",
        task_revision=3,
        change_kind="repair",
        target=TaskTarget(repository="github:org/runtime", revision=_digest("1")),
        change_intent="Repair the fused operation without regressions.",
        requirements=[
            TaskRequirement(
                requirement_id="REQ-CORE",
                statement="The fused operation must preserve all public semantics.",
                visibility="public",
            )
        ],
        allowed_changes=["src/**"],
        forbidden_changes=["verifier/**", "tests/hidden/**"],
        supported_behavior=["bf16 contiguous inputs"],
        unsupported_behavior=["sparse tensors"],
        correctness_tolerance_policy_id="bf16-reference-v1",
        resource_expectation_id="single-gpu-v1",
        public_validation_interface=["python -m tests.public"],
        artifact_policy_sha256=_digest("2"),
        capability_contract_sha256=_digest("3"),
    )
    obligations = [
        _obligation("CO-CORE", "capability-obligation"),
        _obligation("RI-REGRESSION", "regression-invariant"),
        _obligation("NB-BOUNDARY", "negative-boundary"),
        _obligation("MP-NATIVE", "mechanism-proof"),
        _obligation("SL-LIVENESS", "safety-liveness"),
        _obligation(
            "ES-RUNNER",
            "environment-sentinel",
            severity="sentinel",
            failure_owner="infrastructure",
            evidence_owner="environment-sentinel",
        ),
        _obligation(
            "MT-STYLE",
            "maintenance-probe",
            severity="advisory",
            failure_owner="task-author",
            evidence_owner="deterministic-maintenance-probe",
        ),
    ]
    contract = build_acceptance_contract(
        task_id=specification.task_id,
        task_revision=specification.task_revision,
        obligations=obligations,
    )
    hard_ids = [item.obligation_id for item in obligations if item.severity == "hard"]
    witness = FeasibilityWitness(
        witness_id="reference-a",
        kind="reference-patch",
        source_locator="private://qualification/reference-a.patch",
        sha256=_digest("4"),
        target_revision=specification.target.revision,
        build_recipe_sha256=_digest("5"),
        covers=hard_ids,
        known_limitations=[],
        license_status="private-qualification-only",
        reviewer="maintainer-a",
    )
    witness_set = build_witness_set([witness])

    def observation(obligation_id: str, status: str) -> ObligationObservation:
        obligation = next(item for item in obligations if item.obligation_id == obligation_id)
        return ObligationObservation(
            obligation_id=obligation_id,
            bucket=obligation.bucket,
            severity=obligation.severity,
            status=status,
            evidence_refs=["evidence://verifier/" + obligation_id.lower()],
            failure_code=("EXPECTED_BASELINE_GAP" if status == "FAIL" else None),
        )

    baseline = BaselineDifferential(
        task_id=specification.task_id,
        task_revision=specification.task_revision,
        baseline_sha256=specification.target.revision,
        fresh_replays=5,
        stable=True,
        observations=[
            observation(item.obligation_id, "FAIL" if item.obligation_id == "CO-CORE" else "PASS")
            for item in obligations
            if item.severity == "hard" or item.bucket == "environment-sentinel"
        ],
    )
    replay = WitnessReplayResult(
        witness_id=witness.witness_id,
        witness_sha256=witness.sha256,
        fresh_replays=5,
        stable=True,
        observations=[observation(item, "PASS") for item in hard_ids],
    )
    mutations = [
        MutationOutcome(
            mutation_id="drop-output-write",
            patch_sha256=_digest("6"),
            target_obligations=["CO-CORE"],
            weight=2,
            critical=True,
            observed="rejected",
            evidence_refs=["evidence://qualification/mutant/drop-output"],
        ),
        MutationOutcome(
            mutation_id="silent-fallback",
            patch_sha256=_digest("7"),
            target_obligations=["MP-NATIVE"],
            observed="rejected",
            evidence_refs=["evidence://qualification/mutant/fallback"],
        ),
    ]
    controls = [
        NegativeControlOutcome(
            control_id="do-nothing",
            kind="do-nothing",
            observed="rejected",
            evidence_refs=["evidence://qualification/control/do-nothing"],
        )
    ]
    alternative = AlternativeValidSolutionOutcome(
        solution_id="independent-b",
        implementation_sha256=_digest("8"),
        structure_fingerprint_sha256=_digest("9"),
        source="independent-author",
        observed="accepted",
        observations=[observation(item, "PASS") for item in hard_ids],
    )
    flakiness = VerifierFlakinessAudit(
        fresh_replays=7,
        status="STABLE",
        pass_fail_flips=0,
    )
    leakage = TaskLeakageAudit(status="pass")
    mutation_digest = canonical_sha256([item.model_dump(mode="json") for item in mutations])
    alternative_digest = canonical_sha256([alternative.model_dump(mode="json")])
    review = HumanTaskQualificationReview(
        reviewer="maintainer-a",
        reviewed_at=datetime(2026, 9, 2, tzinfo=UTC),
        decision="approve",
        specification_sha256=canonical_sha256(specification),
        contract_sha256=contract.contract_sha256,
        witness_set_sha256=witness_set.witness_set_sha256,
        mutation_suite_sha256=mutation_digest,
        alternative_solution_set_sha256=alternative_digest,
        artifact_policy_sha256=specification.artifact_policy_sha256,
        capability_policy_sha256=specification.capability_contract_sha256,
        rationale="The verifier is independent, stable, and broad enough for official use.",
    )
    return {
        "specification": specification,
        "contract": contract,
        "witness_set": witness_set,
        "baseline": baseline,
        "witness_replays": [replay],
        "mutations": mutations,
        "negative_controls": controls,
        "alternatives": [alternative],
        "flakiness": flakiness,
        "leakage": leakage,
        "review": review,
    }


def test_task_tri_contract_qualifies_and_seals() -> None:
    fixture = _fixture()
    report = qualify_task(**fixture)
    assert report.status == "QUALIFIED"
    assert report.coverage.status == "pass"
    assert report.trust.status == "pass"
    assert report.candidate_score_effect is False
    seal = build_task_seal(
        specification=fixture["specification"],
        contract=fixture["contract"],
        witness_set=fixture["witness_set"],
        qualification=report,
        review=fixture["review"],
        verifier_bundle_sha256=_digest("a"),
        capability_policy_sha256=_digest("3"),
        capability_registry_sha256=_digest("b"),
        resource_envelope_sha256=_digest("c"),
        topology_contract_sha256=_digest("d"),
        benchmark_cell_policy_sha256=_digest("e"),
        runner_selection_policy_sha256=_digest("f"),
        benchmark_season="2026q3",
    )
    assert audit_task_seal(seal) == []
    assert seal.task_seal_sha256.startswith("sha256:")


def test_missing_human_review_never_silently_qualifies() -> None:
    fixture = _fixture()
    fixture["review"] = None
    report = qualify_task(**fixture)
    assert report.status == "REVIEW_REQUIRED"
    assert report.trust.status == "pass"
    assert report.failure_codes == []


def test_mutant_survival_and_witness_digest_tampering_make_task_ineligible() -> None:
    fixture = _fixture()
    fixture["mutations"][0] = fixture["mutations"][0].model_copy(update={"observed": "survived"})
    report = qualify_task(**fixture)
    assert report.status == "INELIGIBLE"
    assert "CRITICAL_MUTANT_SURVIVED:drop-output-write" in report.failure_codes

    witness_set = fixture["witness_set"].model_copy(update={"witness_set_sha256": _digest("0")})
    failures = audit_witness_set(fixture["specification"], fixture["contract"], witness_set)
    assert "WITNESS_SET_DIGEST_MISMATCH" in failures


def test_contract_audit_rejects_unmapped_scoring_requirement() -> None:
    fixture = _fixture()
    specification = fixture["specification"].model_copy(
        update={
            "requirements": [
                *fixture["specification"].requirements,
                TaskRequirement(
                    requirement_id="REQ-EXTRA",
                    statement="A second scoring requirement must also be verifiably covered.",
                    visibility="public",
                ),
            ]
        }
    )
    failures = audit_acceptance_contract(specification, fixture["contract"])
    assert "SCORING_REQUIREMENT_UNMAPPED:REQ-EXTRA" in failures


def _seal_and_observations():
    fixture = _fixture()
    report = qualify_task(**fixture)
    seal = build_task_seal(
        specification=fixture["specification"],
        contract=fixture["contract"],
        witness_set=fixture["witness_set"],
        qualification=report,
        review=fixture["review"],
        verifier_bundle_sha256=_digest("a"),
        capability_policy_sha256=_digest("3"),
        capability_registry_sha256=_digest("b"),
        resource_envelope_sha256=_digest("c"),
        topology_contract_sha256=_digest("d"),
        benchmark_cell_policy_sha256=_digest("e"),
        runner_selection_policy_sha256=_digest("f"),
        benchmark_season="2026q3",
    )
    observations = [
        ObligationObservation(
            obligation_id=item.obligation_id,
            bucket=item.bucket,
            severity=item.severity,
            status="PASS",
            evidence_refs=["evidence://verifier/" + item.obligation_id.lower()],
        )
        for item in fixture["contract"].obligations
    ]
    refs = {ref for item in observations for ref in item.evidence_refs}
    return fixture, seal, observations, refs


def test_verifier_result_exact_coverage_and_environment_ownership() -> None:
    fixture, seal, observations, refs = _seal_and_observations()
    passed = build_verifier_result(
        seal=seal,
        contract=fixture["contract"],
        candidate_sha256=_digest("a"),
        observations=observations,
        sealed_evidence_refs=refs,
    )
    assert passed.result_status == "VALID_PASS"
    assert passed.infra_cert == 1

    failed = [
        item.model_copy(update={"status": "FAIL", "failure_code": "CANDIDATE_OUTPUT_MISMATCH"})
        if item.obligation_id == "CO-CORE"
        else item
        for item in observations
    ]
    result = build_verifier_result(
        seal=seal,
        contract=fixture["contract"],
        candidate_sha256=_digest("b"),
        observations=failed,
        sealed_evidence_refs=refs,
    )
    assert result.result_status == "VALID_FAIL"
    assert result.infra_cert == 0

    infra_invalid = [
        item.model_copy(update={"status": "INFRA_INVALID", "failure_code": "DEVICE_LOST"})
        if item.obligation_id == "ES-RUNNER"
        else item
        for item in failed
    ]
    result = build_verifier_result(
        seal=seal,
        contract=fixture["contract"],
        candidate_sha256=_digest("c"),
        observations=infra_invalid,
        sealed_evidence_refs=refs,
    )
    assert result.result_status == "INFRA_INVALID"
    assert result.infra_cert is None


def test_verifier_result_rejects_missing_unknown_and_unsealed_evidence() -> None:
    fixture, seal, observations, refs = _seal_and_observations()
    with pytest.raises(ValueError, match="missing mandatory obligation"):
        build_verifier_result(
            seal=seal,
            contract=fixture["contract"],
            candidate_sha256=_digest("a"),
            observations=observations[:-1],
        )
    with pytest.raises(ValueError, match="not present in sealed EvidencePack"):
        build_verifier_result(
            seal=seal,
            contract=fixture["contract"],
            candidate_sha256=_digest("a"),
            observations=observations,
            sealed_evidence_refs=refs - {next(iter(refs))},
        )
