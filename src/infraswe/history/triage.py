from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TriageDecision = Literal["accept_with_scope", "revise", "reject", "unresolved"]
RemediationScope = Literal["none", "single-site", "bounded-multi-site", "cross-cutting", "unknown"]
ClosureTest = Literal["existing", "frozen-probe", "missing"]


@dataclass(frozen=True, slots=True)
class CaseContractTriageEvidence:
    """Outcome-free evidence used to separate revise from reject.

    The classifier is intentionally downstream of a case-specific contract. It does not infer
    mergeability from PR metadata and it never consumes review, CI, state, or merge outcomes.
    """

    contract_satisfied: bool
    evidence_complete: bool
    primary_claim_demonstrated: bool
    remediation_scope: RemediationScope = "none"
    closure_test: ClosureTest = "missing"
    semantic_noop: bool = False
    baseline_regression: bool = False
    safety_or_integrity_failure: bool = False
    design_change_required: bool = False
    residual_failure_families: int = 0

    def __post_init__(self) -> None:
        if self.residual_failure_families < 0:
            raise ValueError("residual_failure_families cannot be negative")
        if self.contract_satisfied and self.residual_failure_families:
            raise ValueError("a satisfied contract cannot retain residual failure families")


@dataclass(frozen=True, slots=True)
class CaseContractTriageResult:
    decision: TriageDecision
    rationale_codes: tuple[str, ...]


def classify_case_contract(evidence: CaseContractTriageEvidence) -> CaseContractTriageResult:
    """Classify a frozen exact-contract result without outcome or reviewer leakage.

    ``revise`` is a narrow repairability claim: the patch must demonstrate its primary direction,
    leave no regression or integrity hazard, have a bounded remediation surface, and have an
    executable closure test. A semantic no-op, false primary claim, broad residual invariant, or
    design-level replacement is ``reject``. Missing execution evidence is ``unresolved``.
    """

    if not evidence.evidence_complete:
        return CaseContractTriageResult(
            decision="unresolved",
            rationale_codes=("CASE_CONTRACT_EVIDENCE_INCOMPLETE",),
        )
    if evidence.contract_satisfied:
        return CaseContractTriageResult(
            decision="accept_with_scope",
            rationale_codes=("CASE_CONTRACT_SATISFIED",),
        )

    reject_codes: list[str] = []
    if not evidence.primary_claim_demonstrated:
        reject_codes.append("PRIMARY_CLAIM_NOT_DEMONSTRATED")
    if evidence.semantic_noop:
        reject_codes.append("CENTRAL_PATCH_IS_SEMANTIC_NOOP")
    if evidence.baseline_regression:
        reject_codes.append("HEAD_REGRESSES_BASELINE")
    if evidence.safety_or_integrity_failure:
        reject_codes.append("SAFETY_OR_INTEGRITY_FAILURE")
    if evidence.design_change_required:
        reject_codes.append("REMEDIATION_REQUIRES_DESIGN_CHANGE")
    if evidence.remediation_scope in {"cross-cutting", "unknown", "none"}:
        reject_codes.append("REMEDIATION_SCOPE_NOT_BOUNDED")
    if evidence.closure_test == "missing":
        reject_codes.append("EXECUTABLE_CLOSURE_TEST_MISSING")
    if evidence.residual_failure_families > 1:
        reject_codes.append("RESIDUAL_FAILURE_FAMILIES_NOT_CLOSED")
    if reject_codes:
        return CaseContractTriageResult(
            decision="reject",
            rationale_codes=tuple(sorted(set(reject_codes))),
        )

    return CaseContractTriageResult(
        decision="revise",
        rationale_codes=("PRIMARY_DIRECTION_VALID_BOUNDED_REPAIR_WITH_CLOSURE_TEST",),
    )
