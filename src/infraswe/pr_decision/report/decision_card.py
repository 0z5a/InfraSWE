from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import (
    DecisionPlaneModel,
    EvaluationTrack,
    canonical_sha256,
)
from infraswe.pr_decision.release_gate import MetricGateResult, ReliabilityMetrics


class MetricSlice(DecisionPlaneModel):
    slice_id: str
    selector: dict[str, str]
    metrics: ReliabilityMetrics


class AutomationCoverage(DecisionPlaneModel):
    automatic_cases: int = Field(ge=0)
    unresolved_cases: int = Field(ge=0)
    human_assisted_cases: int = Field(ge=0)
    invalid_or_abandoned_cases: int = Field(ge=0)


class DecisionCost(DecisionPlaneModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    gpu_seconds: float = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0, ge=0)


class DecisionReliabilityCardMaterial(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    card_id: str
    evaluation_track: EvaluationTrack
    policy_digest: Digest
    harness_digest: Digest
    retrieval_index_digest: Digest
    calibration_profile_digest: Digest | None = None
    label_vault_digest: Digest
    population_digest: Digest
    gate_result: MetricGateResult
    slices: list[MetricSlice] = Field(default_factory=list)
    coverage: AutomationCoverage
    cost: DecisionCost
    statistical_assumptions: list[str] = Field(min_length=1)
    selection_protocol: str = Field(min_length=1)
    attempted_policy_digests: list[Digest] = Field(min_length=1)
    attempted_calibration_profile_digests: list[Digest] = Field(default_factory=list)
    generated_at: datetime

    @model_validator(mode="after")
    def card_matches_metric_contract(self) -> DecisionReliabilityCardMaterial:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.evaluation_track != self.gate_result.contract.evaluation_track:
            raise ValueError("reliability card track must match its MetricContract")
        if sum(self.coverage.model_dump().values()) != self.gate_result.metrics.total_cases:
            raise ValueError("coverage categories must partition the total population")
        if self.policy_digest not in self.attempted_policy_digests:
            raise ValueError("selected policy must be in the attempted policy ledger")
        if self.calibration_profile_digest is not None and (
            self.calibration_profile_digest not in self.attempted_calibration_profile_digests
        ):
            raise ValueError("selected calibration must be in the attempted calibration ledger")
        return self


class DecisionReliabilityCard(DecisionPlaneModel):
    material: DecisionReliabilityCardMaterial
    card_sha256: Digest


def seal_reliability_card(
    material: DecisionReliabilityCardMaterial,
) -> DecisionReliabilityCard:
    return DecisionReliabilityCard(material=material, card_sha256=canonical_sha256(material))


def audit_reliability_card(card: DecisionReliabilityCard) -> list[str]:
    try:
        DecisionReliabilityCardMaterial.model_validate(card.material.model_dump())
    except ValueError as error:
        return [str(error)]
    if card.card_sha256 != canonical_sha256(card.material):
        return ["decision reliability card digest mismatch"]
    return []
