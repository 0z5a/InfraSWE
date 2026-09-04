from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from infraswe.pr_decision.contracts import DecisionPlaneModel

EvidenceAuthority = Literal[
    "untrusted-text",
    "source-code",
    "build-or-test",
    "executable-verifier",
    "trusted-meter",
    "human-review",
]
ClaimDisposition = Literal["supports", "refutes", "unknown"]


class EvidenceClaim(DecisionPlaneModel):
    claim_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")
    claim: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    authority: EvidenceAuthority
    observed_at: datetime
    head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    disposition: ClaimDisposition
    counterevidence_refs: list[str] = Field(default_factory=list)
    details: str | None = None

    @model_validator(mode="after")
    def claim_is_provenanced(self) -> EvidenceClaim:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.disposition == "refutes" and self.authority == "untrusted-text":
            raise ValueError("untrusted text cannot independently establish a refutation")
        return self


class EvidenceRequest(DecisionPlaneModel):
    state: Literal[
        "NEEDS_SOURCE_CONTEXT",
        "NEEDS_BUILD_OR_TEST",
        "NEEDS_REVIEW_EVIDENCE",
        "NEEDS_LABEL_AUDIT",
    ]
    obligation_id: str
    requested_sources: list[EvidenceAuthority] = Field(min_length=1)
    reason: str = Field(min_length=1)


def decisive_refutation(claims: list[EvidenceClaim], refs: set[str]) -> bool:
    return any(
        claim.claim_id in refs
        and claim.disposition == "refutes"
        and claim.authority
        in {"source-code", "build-or-test", "executable-verifier", "human-review"}
        for claim in claims
    )
