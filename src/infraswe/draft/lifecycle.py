from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from infraswe.models.draft import (
    DRAFT_STATE_ORDER,
    DraftSpec,
    DraftState,
    HumanReviewRecord,
    SealedDraft,
    SealMaterial,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def advance_draft_state(draft: DraftSpec, target_state: DraftState) -> DraftSpec:
    """Advance exactly one state without mutating the input Draft."""

    current_index = DRAFT_STATE_ORDER[draft.draft.state]
    target_index = DRAFT_STATE_ORDER[target_state]
    if target_index != current_index + 1:
        raise ValueError("Draft states advance exactly one step; loops use revision events")
    payload = draft.model_dump(mode="json")
    payload["draft"]["state"] = target_state
    return DraftSpec.model_validate(payload)


def _validate_review_matches_draft(draft: DraftSpec, review: HumanReviewRecord) -> None:
    if review.decision != "approve":
        raise ValueError("only an approved human maintainer review can seal a Draft")
    if not all(
        (
            draft.target,
            draft.deployment,
            draft.acceptance_contract,
            draft.retrieval,
            draft.benchmark_loop,
            draft.scoring,
        )
    ):
        raise ValueError("the Draft is incomplete and cannot be sealed")
    assert draft.target is not None
    assert draft.deployment is not None
    assert draft.acceptance_contract is not None
    assert draft.scoring is not None
    review_sha256 = canonical_sha256(review)
    if draft.acceptance_contract.human_review_sha256 != review_sha256:
        raise ValueError("human review digest does not match the Draft review binding")
    expected = {
        "target_profile_sha256": draft.target.project_profile_sha256,
        "acceptance_contract_sha256": draft.acceptance_contract.sha256,
        "probe_set_sha256": draft.acceptance_contract.probe_set_sha256,
        "workload_portfolio_sha256": draft.deployment.workload_portfolio.sha256,
        "formula_template_id": draft.scoring.formula_template_id,
    }
    observed = {
        "target_profile_sha256": review.target_profile_sha256,
        "acceptance_contract_sha256": review.acceptance_contract_sha256,
        "probe_set_sha256": review.probe_set_sha256,
        "workload_portfolio_sha256": review.workload_portfolio_sha256,
        "formula_template_id": review.formula_template_id,
    }
    mismatched = sorted(name for name in expected if expected[name] != observed[name])
    if mismatched:
        raise ValueError("human review does not match Draft: " + ", ".join(mismatched))


def seal_draft(
    draft: DraftSpec,
    review: HumanReviewRecord,
    *,
    performance_target_sha256: str,
    sealed_by: str,
    sealed_at: datetime | None = None,
) -> SealedDraft:
    if draft.draft.state != "D5-fast-loop":
        raise ValueError("only a D5 fast-loop Draft can be sealed")
    _validate_review_matches_draft(draft, review)
    assert draft.target is not None
    assert draft.deployment is not None
    assert draft.acceptance_contract is not None
    assert draft.retrieval is not None
    assert draft.retrieval.precedent_set_sha256 is not None
    assert draft.benchmark_loop is not None
    assert draft.scoring is not None
    material = SealMaterial(
        target_profile_sha256=draft.target.project_profile_sha256,
        target_repository_sha256=draft.target.revision,
        candidate_sha256=draft.candidate.revision,
        precedent_set_sha256=draft.retrieval.precedent_set_sha256,
        acceptance_contract_sha256=draft.acceptance_contract.sha256,
        probe_set_sha256=draft.acceptance_contract.probe_set_sha256,
        workload_portfolio_sha256=draft.deployment.workload_portfolio.sha256,
        performance_target_sha256=performance_target_sha256,
        required_deployment_cell_set_sha256=canonical_sha256(
            sorted(draft.deployment.required_cells)
        ),
        formula_template_id=draft.scoring.formula_template_id,
        benchmark_budget_policy_id=draft.benchmark_loop.benchmark_budget_policy_id,
        evidence_policy_id=draft.benchmark_loop.evidence_policy_id,
        project_season=draft.scoring.project_season,
    )
    timestamp = sealed_at or datetime.now(UTC)
    review_sha256 = canonical_sha256(review)
    unsigned = SealedDraft(
        draft_id=draft.draft.id,
        draft_revision=draft.draft.revision,
        sealed_at=timestamp,
        sealed_by=sealed_by,
        human_review_sha256=review_sha256,
        material=material,
        seal_sha256="sha256:" + "0" * 64,
    )
    canonical = unsigned.model_dump(mode="json", exclude={"seal_sha256"})
    return unsigned.model_copy(update={"seal_sha256": canonical_sha256(canonical)})


def audit_seal(seal: SealedDraft) -> list[str]:
    payload = seal.model_dump(mode="json")
    observed = payload.pop("seal_sha256")
    expected = canonical_sha256(payload)
    return [] if observed == expected else ["DRAFT_SEAL_DIGEST_MISMATCH"]
