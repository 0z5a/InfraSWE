from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import (
    DecisionLabel,
    DecisionPlaneModel,
    LabelKind,
    PRCaseIdentity,
    canonical_sha256,
)


class LabelVaultRecord(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    case_identity: PRCaseIdentity
    oracle_label: DecisionLabel
    label_kind: LabelKind
    oracle_source: str = Field(min_length=1)
    observed_at: datetime
    adjudication_status: Literal["unreviewed", "reviewed", "disputed", "superseded"]
    decisive_evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def label_is_post_prediction(self) -> LabelVaultRecord:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.observed_at < self.case_identity.prediction_at:
            raise ValueError("label observation cannot predate the blind prediction")
        if self.adjudication_status == "reviewed" and not self.decisive_evidence_refs:
            raise ValueError("reviewed labels require decisive evidence")
        return self


class LabelVault(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    vault_id: str
    records: list[LabelVaultRecord] = Field(min_length=1)
    vault_sha256: Digest


class LearningRecord(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    case_identity_sha256: Digest
    snapshot_sha256: Digest
    decision_record_sha256: Digest
    label_record_sha256: Digest
    allowed_uses: list[
        Literal["external-policy", "curriculum", "offline-retrieval", "qualitative-audit"]
    ] = Field(min_length=1)
    policy_gradient_eligible: Literal[False] = False


def seal_label_vault(vault_id: str, records: list[LabelVaultRecord]) -> LabelVault:
    identities = [canonical_sha256(record.case_identity) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("label vault cannot contain duplicate case identities")
    material = {
        "schema_version": "0.6.1",
        "vault_id": vault_id,
        "records": [record.model_dump(mode="json") for record in records],
    }
    return LabelVault(**material, vault_sha256=canonical_sha256(material))


def audit_label_vault(vault: LabelVault) -> list[str]:
    material = vault.model_dump(mode="json", exclude={"vault_sha256"})
    failures: list[str] = []
    if vault.vault_sha256 != canonical_sha256(material):
        failures.append("label vault digest mismatch")
    identities = [canonical_sha256(record.case_identity) for record in vault.records]
    if len(identities) != len(set(identities)):
        failures.append("label vault contains duplicate case identities")
    return failures


def build_learning_record(
    *,
    snapshot_sha256: Digest,
    decision_record_sha256: Digest,
    label: LabelVaultRecord,
) -> LearningRecord:
    return LearningRecord(
        case_identity_sha256=canonical_sha256(label.case_identity),
        snapshot_sha256=snapshot_sha256,
        decision_record_sha256=decision_record_sha256,
        label_record_sha256=canonical_sha256(label),
        allowed_uses=["external-policy", "curriculum", "offline-retrieval", "qualitative-audit"],
    )
