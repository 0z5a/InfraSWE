from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import (
    DecisionPlaneModel,
    PolicyIdentity,
    PRCaseIdentity,
    canonical_sha256,
)
from infraswe.pr_decision.evidence import EvidenceClaim

LABEL_ONLY_KEYS = {
    "actual_outcome",
    "closed_at",
    "conclusion",
    "ground_truth",
    "label_vault",
    "merge_commit_sha",
    "merged",
    "merged_at",
    "oracle_decision",
    "oracle_label",
    "review_decision",
    "state",
}


def assert_label_free(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        leaked = LABEL_ONLY_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"label-vault leakage at {path}: {sorted(leaked)}")
        for key, item in value.items():
            assert_label_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_label_free(item, f"{path}[{index}]")


class OutcomeBlindSnapshotMaterial(DecisionPlaneModel):
    schema_version: Literal["0.6.1"] = "0.6.1"
    case_identity: PRCaseIdentity
    policy_identity: PolicyIdentity
    claims: list[EvidenceClaim] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
    retrieval_memory_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def snapshot_is_time_and_label_safe(self) -> OutcomeBlindSnapshotMaterial:
        assert_label_free(self.observations)
        # Only structural observations are accepted by this adapter. Free-form
        # notes need a separately reviewed collector, not another key blacklist.
        if set(self.observations) - {"changed_files"}:
            raise ValueError("unsupported blind observation fields")
        paths = self.observations.get("changed_files", [])
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/")
            for path in paths
        ):
            raise ValueError("changed_files must contain repository-relative paths")
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("blind claim ids must be unique")
        for claim in self.claims:
            if claim.observed_at > self.case_identity.prediction_at:
                raise ValueError("blind claims cannot be observed after prediction_at")
            if claim.head_sha != self.case_identity.head_sha:
                raise ValueError("blind claim head_sha must match the case identity")
        if len(self.retrieval_memory_refs) != len(set(self.retrieval_memory_refs)):
            raise ValueError("retrieval memory references must be unique")
        if self.retrieval_memory_refs:
            raise ValueError(
                "retrieval references require a verified memory resolver (not enabled)"
            )
        return self


class OutcomeBlindSnapshot(DecisionPlaneModel):
    material: OutcomeBlindSnapshotMaterial
    snapshot_sha256: Digest
    # Hash/time checks do not authenticate a collector or establish data isolation.
    provenance_authenticated: Literal[False] = False


def seal_snapshot(material: OutcomeBlindSnapshotMaterial) -> OutcomeBlindSnapshot:
    return OutcomeBlindSnapshot(material=material, snapshot_sha256=canonical_sha256(material))


def audit_snapshot(snapshot: OutcomeBlindSnapshot) -> list[str]:
    failures: list[str] = []
    if snapshot.snapshot_sha256 != canonical_sha256(snapshot.material):
        failures.append("snapshot digest mismatch")
    try:
        OutcomeBlindSnapshotMaterial.model_validate(snapshot.material.model_dump())
    except ValueError as error:
        failures.append(str(error))
    return failures
