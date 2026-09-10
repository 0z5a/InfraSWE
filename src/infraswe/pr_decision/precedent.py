from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import DecisionLabel, DecisionPlaneModel

PrecedentRole = Literal["historical-accept", "historical-reject", "legitimate-exception"]
DataSplit = Literal["train", "dev", "calibration", "heldout"]


class DecisionPrecedent(DecisionPlaneModel):
    precedent_id: str
    repository: str
    project: str
    module: str
    mechanism: str
    issue_or_patch_family: str
    decision: DecisionLabel
    role: PrecedentRole
    decisive_evidence_refs: list[str] = Field(min_length=1)
    observed_at: datetime
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: DataSplit

    @model_validator(mode="after")
    def role_matches_decision(self) -> DecisionPrecedent:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.role == "historical-accept" and self.decision != "accept":
            raise ValueError("historical-accept precedents must be Accept")
        if self.role == "historical-reject" and self.decision != "reject":
            raise ValueError("historical-reject precedents must be Reject")
        return self


class PrecedentQuery(DecisionPlaneModel):
    repository: str
    project: str
    module: str
    mechanism: str
    prediction_at: datetime
    split: DataSplit
    allowed_precedent_splits: list[DataSplit] = Field(default_factory=lambda: ["train"])
    excluded_families: list[str] = Field(default_factory=list)
    limit_per_role: int = Field(default=3, ge=1, le=100)

    @model_validator(mode="after")
    def query_is_time_and_split_safe(self) -> PrecedentQuery:
        if self.prediction_at.tzinfo is None or self.prediction_at.utcoffset() is None:
            raise ValueError("prediction_at must be timezone-aware")
        if not self.allowed_precedent_splits:
            raise ValueError("at least one frozen precedent split is required")
        if len(self.allowed_precedent_splits) != len(set(self.allowed_precedent_splits)):
            raise ValueError("frozen precedent splits must be unique")
        if self.split in {"dev", "calibration", "heldout"} and self.split in set(
            self.allowed_precedent_splits
        ):
            raise ValueError("target dev/calibration/heldout labels cannot be precedent memory")
        return self


class ContrastivePrecedentBundle(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    query: PrecedentQuery
    accepts: list[DecisionPrecedent]
    rejects: list[DecisionPrecedent]
    legitimate_exceptions: list[DecisionPrecedent]
    index_digest: Digest

    @model_validator(mode="after")
    def bundle_is_contrastive_and_leak_free(self) -> ContrastivePrecedentBundle:
        if not self.accepts or not self.rejects or not self.legitimate_exceptions:
            raise ValueError(
                "contrastive retrieval requires Accept, Reject, and exception examples"
            )
        records = [*self.accepts, *self.rejects, *self.legitimate_exceptions]
        if any(item.observed_at > self.query.prediction_at for item in records):
            raise ValueError("precedents cannot be observed after prediction_at")
        if any(item.split not in self.query.allowed_precedent_splits for item in records):
            raise ValueError("precedents must come from an explicitly frozen source split")
        if any(item.issue_or_patch_family in self.query.excluded_families for item in records):
            raise ValueError("excluded patch families leaked into precedent retrieval")
        return self


def retrieve_contrastive_precedents(
    query: PrecedentQuery,
    records: list[DecisionPrecedent],
    *,
    index_digest: Digest,
) -> ContrastivePrecedentBundle:
    eligible = [
        item
        for item in records
        if item.observed_at <= query.prediction_at
        and item.split in query.allowed_precedent_splits
        and item.issue_or_patch_family not in query.excluded_families
    ]

    def rank(item: DecisionPrecedent) -> tuple[int, int, int, str]:
        return (
            0 if item.repository == query.repository else 1,
            0 if item.module == query.module else 1,
            0 if item.mechanism == query.mechanism else 1,
            item.precedent_id,
        )

    def take(role: PrecedentRole) -> list[DecisionPrecedent]:
        return sorted((item for item in eligible if item.role == role), key=rank)[
            : query.limit_per_role
        ]

    return ContrastivePrecedentBundle(
        query=query,
        accepts=take("historical-accept"),
        rejects=take("historical-reject"),
        legitimate_exceptions=take("legitimate-exception"),
        index_digest=index_digest,
    )
