from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import (
    DecisionLabel,
    DecisionPlaneModel,
    DecisionPrediction,
    LabelKind,
    PolicyIdentity,
    PRCaseIdentity,
    canonical_sha256,
)

ErrorReasonCode = Literal[
    "evidence_missing",
    "evidence_present_reasoning_miss",
    "label_revision_mismatch",
    "oracle_disagreement",
    "scope_contract_mismatch",
    "policy_tradeoff_error",
]
FailureOwner = Literal[
    "acquisition",
    "reasoning",
    "label-contract",
    "oracle",
    "project-profile",
    "decision-policy",
]


class ErrorAuditOnly(DecisionPlaneModel):
    oracle_label: DecisionLabel
    label_kind: LabelKind
    oracle_source: str = Field(min_length=1)
    adjudication_status: Literal["unreviewed", "reviewed", "disputed", "superseded"]
    failure_owner: FailureOwner
    reason_code: ErrorReasonCode
    decisive_evidence_ref: str | None = None

    @model_validator(mode="after")
    def reviewed_error_is_explained(self) -> ErrorAuditOnly:
        if self.adjudication_status == "reviewed" and self.decisive_evidence_ref is None:
            raise ValueError("reviewed errors require a decisive evidence reference")
        return self


class ErrorRecord(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    case_identity: PRCaseIdentity
    policy_identity: PolicyIdentity
    prediction: DecisionPrediction
    audit_only: ErrorAuditOnly

    @model_validator(mode="after")
    def record_is_an_error(self) -> ErrorRecord:
        if self.prediction.label == self.audit_only.oracle_label:
            raise ValueError("ErrorRecord requires a prediction/oracle disagreement")
        return self


class ErrorBook(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    errorbook_id: str
    policy_digest: Digest
    records: list[ErrorRecord]
    errorbook_sha256: Digest

    @model_validator(mode="after")
    def records_match_frozen_policy(self) -> ErrorBook:
        if any(
            record.policy_identity.policy_digest != self.policy_digest for record in self.records
        ):
            raise ValueError("every ErrorBook record must match the frozen policy digest")
        return self


def seal_errorbook(
    errorbook_id: str,
    policy_digest: Digest,
    records: list[ErrorRecord],
) -> ErrorBook:
    if any(record.policy_identity.policy_digest != policy_digest for record in records):
        raise ValueError("every ErrorBook record must match the frozen policy digest")
    identities = [canonical_sha256(record.case_identity) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("ErrorBook cannot contain duplicate case identities")
    material = {
        "schema_version": "0.6.1",
        "errorbook_id": errorbook_id,
        "policy_digest": policy_digest,
        "records": [record.model_dump(mode="json") for record in records],
    }
    return ErrorBook(**material, errorbook_sha256=canonical_sha256(material))


def audit_errorbook(errorbook: ErrorBook) -> list[str]:
    material = errorbook.model_dump(mode="json", exclude={"errorbook_sha256"})
    failures: list[str] = []
    if errorbook.errorbook_sha256 != canonical_sha256(material):
        failures.append("ErrorBook digest mismatch")
    identities = [canonical_sha256(record.case_identity) for record in errorbook.records]
    if len(identities) != len(set(identities)):
        failures.append("ErrorBook contains duplicate case identities")
    return failures
