from __future__ import annotations

import pytest

from infraswe.history.triage import CaseContractTriageEvidence, classify_case_contract


def evidence(**updates: object) -> CaseContractTriageEvidence:
    values: dict[str, object] = {
        "contract_satisfied": False,
        "evidence_complete": True,
        "primary_claim_demonstrated": True,
        "remediation_scope": "single-site",
        "closure_test": "frozen-probe",
        "residual_failure_families": 1,
    }
    values.update(updates)
    return CaseContractTriageEvidence(**values)  # type: ignore[arg-type]


def test_satisfied_contract_accepts_and_incomplete_evidence_abstains() -> None:
    accepted = classify_case_contract(
        evidence(
            contract_satisfied=True,
            primary_claim_demonstrated=True,
            remediation_scope="none",
            closure_test="existing",
            residual_failure_families=0,
        )
    )
    unresolved = classify_case_contract(evidence(evidence_complete=False))
    assert accepted.decision == "accept_with_scope"
    assert unresolved.decision == "unresolved"


def test_check_requires_a_valid_direction_bounded_repair_and_closure_test() -> None:
    result = classify_case_contract(evidence())
    assert result.decision == "check"
    assert result.rationale_codes == ("PRIMARY_DIRECTION_VALID_BOUNDED_REPAIR_WITH_CLOSURE_TEST",)


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"primary_claim_demonstrated": False}, "PRIMARY_CLAIM_NOT_DEMONSTRATED"),
        ({"semantic_noop": True}, "CENTRAL_PATCH_IS_SEMANTIC_NOOP"),
        ({"baseline_regression": True}, "HEAD_REGRESSES_BASELINE"),
        ({"safety_or_integrity_failure": True}, "SAFETY_OR_INTEGRITY_FAILURE"),
        ({"design_change_required": True}, "REMEDIATION_REQUIRES_DESIGN_CHANGE"),
        ({"remediation_scope": "cross-cutting"}, "REMEDIATION_SCOPE_NOT_BOUNDED"),
        ({"closure_test": "missing"}, "EXECUTABLE_CLOSURE_TEST_MISSING"),
        ({"residual_failure_families": 2}, "RESIDUAL_FAILURE_FAMILIES_NOT_CLOSED"),
    ],
)
def test_rejects_when_repairability_is_not_demonstrated(
    updates: dict[str, object], code: str
) -> None:
    result = classify_case_contract(evidence(**updates))
    assert result.decision == "reject"
    assert code in result.rationale_codes


def test_r10_lessons_separate_deepgemm_liger_and_vllm() -> None:
    deepgemm = classify_case_contract(
        evidence(primary_claim_demonstrated=False, semantic_noop=True)
    )
    liger = classify_case_contract(evidence())
    vllm = classify_case_contract(
        evidence(
            remediation_scope="cross-cutting",
            design_change_required=True,
            residual_failure_families=3,
        )
    )
    assert deepgemm.decision == "reject"
    assert liger.decision == "check"
    assert vllm.decision == "reject"


def test_satisfied_contract_cannot_retain_failures() -> None:
    with pytest.raises(ValueError, match="satisfied contract"):
        evidence(contract_satisfied=True)
