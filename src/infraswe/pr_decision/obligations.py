from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from infraswe.pr_decision.contracts import DecisionPlaneModel

ObligationDimension = Literal[
    "maintainability",
    "deployability",
    "correctness-safety",
    "performance",
]
ObligationStatus = Literal["satisfied", "violated", "unknown", "not-applicable"]


class Obligation(DecisionPlaneModel):
    obligation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    dimension: ObligationDimension
    question: str = Field(min_length=1)
    status: ObligationStatus
    blocking: bool
    evidence_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def disposition_has_support(self) -> Obligation:
        if self.status in {"satisfied", "violated"} and not self.evidence_refs:
            raise ValueError("resolved obligations require evidence references")
        if self.status == "unknown" and not self.missing_evidence:
            raise ValueError("unknown obligations must name missing evidence")
        if self.status == "not-applicable" and self.blocking:
            raise ValueError("not-applicable obligations cannot be blocking")
        return self


class ObligationMap(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    case_identity_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    obligations: list[Obligation] = Field(min_length=1)

    @model_validator(mode="after")
    def obligation_ids_are_unique(self) -> ObligationMap:
        ids = [item.obligation_id for item in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("obligation ids must be unique")
        return self

    @property
    def blocking_violations(self) -> list[Obligation]:
        return [item for item in self.obligations if item.blocking and item.status == "violated"]

    @property
    def blocking_unknowns(self) -> list[Obligation]:
        return [item for item in self.obligations if item.blocking and item.status == "unknown"]


def ordered_obligations(obligation_map: ObligationMap) -> list[Obligation]:
    """Keep final disposition order stable; correctness/safety is a deployability sub-gate."""

    order = {
        "maintainability": 0,
        "deployability": 1,
        "correctness-safety": 1,
        "performance": 2,
    }
    return sorted(
        obligation_map.obligations,
        key=lambda item: (order[item.dimension], item.obligation_id),
    )
