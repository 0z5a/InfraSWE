from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from infraswe.pr_decision.contracts import (
    DecisionCountDelta,
    DecisionLabel,
    DecisionPlaneModel,
    DecisionPrediction,
)
from infraswe.pr_decision.evidence import (
    EvidenceClaim,
    EvidenceRequest,
    decisive_refutation,
)

CorrectionDirection = Literal["accept-challenger", "reject-rescuer"]


class CascadeStageBudget(DecisionPlaneModel):
    stage_id: str
    minimum_conditional_accept_recall: float = Field(ge=0, le=1)


class CascadeRecallBudget(DecisionPlaneModel):
    stages: list[CascadeStageBudget] = Field(min_length=1)
    minimum_total_accept_recall: float = Field(ge=0, le=1)

    @property
    def composed_minimum_accept_recall(self) -> float:
        result = 1.0
        for stage in self.stages:
            result *= stage.minimum_conditional_accept_recall
        return result

    @model_validator(mode="after")
    def stages_preserve_total_recall_budget(self) -> CascadeRecallBudget:
        if self.composed_minimum_accept_recall < self.minimum_total_accept_recall:
            raise ValueError("cascade stages exhaust the frozen Accept recall budget")
        return self


class CorrectionProposal(DecisionPlaneModel):
    proposal_id: str
    direction: CorrectionDirection
    from_label: DecisionLabel
    revised_prediction: DecisionPrediction
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def direction_matches_transition(self) -> CorrectionProposal:
        target = self.revised_prediction.label
        if self.direction == "accept-challenger":
            if self.from_label != "accept" or target not in {"check", "reject"}:
                raise ValueError("AcceptChallenger must move Accept to Check or Reject")
        elif self.from_label not in {"check", "reject"} or target != "accept":
            raise ValueError("RejectRescuer must move Check or Reject to Accept")
        if not self.evidence_refs:
            raise ValueError("correction proposals require decisive evidence references")
        if not set(self.evidence_refs).issubset(self.revised_prediction.evidence_refs):
            raise ValueError("proposal evidence must be bound into the revised prediction")
        return self


class CascadeResult(DecisionPlaneModel):
    initial_prediction: DecisionPrediction
    final_prediction: DecisionPrediction
    applied_proposal_ids: list[str] = Field(default_factory=list)
    blocked_proposal_ids: list[str] = Field(default_factory=list)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)


def _decisive_support(claims: list[EvidenceClaim], refs: set[str]) -> bool:
    return any(
        claim.claim_id in refs
        and claim.disposition == "supports"
        and claim.authority
        in {"source-code", "build-or-test", "executable-verifier", "trusted-meter", "human-review"}
        for claim in claims
    )


def _decisive_uncertainty(claims: list[EvidenceClaim], refs: set[str]) -> bool:
    return any(
        claim.claim_id in refs
        and claim.disposition == "unknown"
        and claim.authority
        in {"source-code", "build-or-test", "executable-verifier", "human-review"}
        for claim in claims
    )


def apply_bidirectional_cascade(
    *,
    initial: DecisionPrediction,
    proposals: list[CorrectionProposal],
    claims: list[EvidenceClaim],
    evidence_requests: list[EvidenceRequest] | None = None,
) -> CascadeResult:
    """Apply only evidence-backed corrections; unsupported doubt remains an internal request."""

    current = initial
    applied: list[str] = []
    blocked: list[str] = []
    requests = list(evidence_requests or [])
    claim_ids = {claim.claim_id for claim in claims}

    for proposal in proposals:
        refs = set(proposal.evidence_refs)
        if proposal.from_label != current.label or not refs.issubset(claim_ids):
            blocked.append(proposal.proposal_id)
            continue
        if proposal.direction == "reject-rescuer":
            decisive = _decisive_support(claims, refs)
        elif proposal.revised_prediction.label == "check":
            decisive = _decisive_uncertainty(claims, refs)
        else:
            decisive = decisive_refutation(claims, refs)
        if not decisive:
            blocked.append(proposal.proposal_id)
            continue
        current = proposal.revised_prediction
        applied.append(proposal.proposal_id)

    return CascadeResult(
        initial_prediction=initial,
        final_prediction=current,
        applied_proposal_ids=applied,
        blocked_proposal_ids=blocked,
        evidence_requests=requests,
    )


def count_accept_corrections(
    *,
    oracle_by_case: dict[str, DecisionLabel],
    old_by_case: dict[str, DecisionLabel],
    new_by_case: dict[str, DecisionLabel],
) -> DecisionCountDelta:
    case_ids = set(oracle_by_case)
    if set(old_by_case) != case_ids or set(new_by_case) != case_ids:
        raise ValueError("oracle, old, and new predictions must cover identical case ids")

    recovered_old_fn = 0
    introduced_new_fn = 0
    removed_old_fp = 0
    introduced_new_fp = 0
    for case_id in case_ids:
        oracle = oracle_by_case[case_id]
        old = old_by_case[case_id]
        new = new_by_case[case_id]
        if oracle == "accept":
            recovered_old_fn += int(old != "accept" and new == "accept")
            introduced_new_fn += int(old == "accept" and new != "accept")
        else:
            removed_old_fp += int(old == "accept" and new != "accept")
            introduced_new_fp += int(old != "accept" and new == "accept")
    return DecisionCountDelta(
        recovered_old_fn=recovered_old_fn,
        introduced_new_fn=introduced_new_fn,
        removed_old_fp=removed_old_fp,
        introduced_new_fp=introduced_new_fp,
    )
