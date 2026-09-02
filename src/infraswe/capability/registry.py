from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    PROOF_LEVEL_ORDER,
    CapabilityAttestation,
    CapabilityDefinition,
    CapabilityProbeIdentity,
    CapabilityRegistry,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def build_registry(
    *,
    registry_id: str,
    revision: int,
    definitions: Sequence[CapabilityDefinition],
) -> CapabilityRegistry:
    preliminary = CapabilityRegistry(
        registry_id=registry_id,
        revision=revision,
        definitions=list(definitions),
        registry_sha256=_ZERO_DIGEST,
    )
    failures = _audit_registry_semantics(preliminary)
    if failures:
        raise ValueError("; ".join(failures))
    material = preliminary.model_dump(mode="json", exclude={"registry_sha256"})
    return preliminary.model_copy(update={"registry_sha256": canonical_sha256(material)})


def _audit_registry_semantics(registry: CapabilityRegistry) -> list[str]:
    failures: list[str] = []
    known = {item.capability_id for item in registry.definitions}
    aliases = {
        alias: item.capability_id
        for item in registry.definitions
        for alias in item.relationships.aliases
    }
    for definition in registry.definitions:
        for implied in definition.relationships.implies:
            if implied not in known and implied not in aliases:
                failures.append(
                    "CAPABILITY_IMPLICATION_TARGET_UNKNOWN:"
                    + definition.capability_id
                    + ":"
                    + implied
                )
        for conflict in definition.relationships.conflicts:
            if conflict not in known and conflict not in aliases:
                failures.append(
                    "CAPABILITY_CONFLICT_TARGET_UNKNOWN:"
                    + definition.capability_id
                    + ":"
                    + conflict
                )
    graph = {
        item.capability_id: set(item.relationships.implies) & known for item in registry.definitions
    }

    def has_cycle(node: str, visiting: set[str], visited: set[str]) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(has_cycle(child, visiting, visited) for child in graph[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    visited: set[str] = set()
    if any(has_cycle(node, set(), visited) for node in sorted(graph)):
        failures.append("CAPABILITY_IMPLICATION_CYCLE")
    return failures


def audit_registry(registry: CapabilityRegistry) -> list[str]:
    failures = _audit_registry_semantics(registry)
    material = registry.model_dump(mode="json", exclude={"registry_sha256"})
    if registry.registry_sha256 != canonical_sha256(material):
        failures.append("CAPABILITY_REGISTRY_DIGEST_MISMATCH")
    return sorted(set(failures))


def build_attestation(
    *,
    capability_id: str,
    capability_definition_version: int,
    runner_snapshot_sha256: str,
    status: str,
    proof_level: str | None,
    parameters: dict[str, object],
    probe: CapabilityProbeIdentity,
    evidence_refs: Sequence[str],
    observed_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> CapabilityAttestation:
    preliminary = CapabilityAttestation(
        capability_id=capability_id,
        capability_definition_version=capability_definition_version,
        runner_snapshot_sha256=runner_snapshot_sha256,
        status=status,
        proof_level=proof_level,
        parameters=parameters,
        probe=probe,
        evidence_refs=list(evidence_refs),
        observed_at=observed_at or datetime.now(UTC),
        expires_at=expires_at,
        attestation_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"attestation_sha256"})
    return preliminary.model_copy(update={"attestation_sha256": canonical_sha256(material)})


def audit_attestation(
    attestation: CapabilityAttestation,
    *,
    registry: CapabilityRegistry,
    snapshot_sha256: str,
    now: datetime | None = None,
) -> list[str]:
    failures: list[str] = []
    material = attestation.model_dump(mode="json", exclude={"attestation_sha256"})
    if attestation.attestation_sha256 != canonical_sha256(material):
        failures.append("CAPABILITY_ATTESTATION_DIGEST_MISMATCH")
    if attestation.runner_snapshot_sha256 != snapshot_sha256:
        failures.append("CAPABILITY_ATTESTATION_SNAPSHOT_MISMATCH")
    definitions = {(item.capability_id, item.version): item for item in registry.definitions}
    if (
        attestation.capability_id,
        attestation.capability_definition_version,
    ) not in definitions:
        failures.append("CAPABILITY_ATTESTATION_DEFINITION_UNKNOWN")
    current = now or datetime.now(UTC)
    if attestation.expires_at is not None and current >= attestation.expires_at:
        failures.append("CAPABILITY_ATTESTATION_EXPIRED")
    return failures


def merge_attestations(
    attestations: Sequence[CapabilityAttestation],
    *,
    registry: CapabilityRegistry,
    snapshot_sha256: str,
    now: datetime | None = None,
) -> tuple[dict[str, CapabilityAttestation], list[str]]:
    """Merge trusted facts; conflicting proof levels quarantine the runner."""

    grouped: dict[str, list[CapabilityAttestation]] = {}
    failures: list[str] = []
    for attestation in attestations:
        audit = audit_attestation(
            attestation,
            registry=registry,
            snapshot_sha256=snapshot_sha256,
            now=now,
        )
        if audit:
            failures.extend(audit)
            continue
        grouped.setdefault(attestation.capability_id, []).append(attestation)

    merged: dict[str, CapabilityAttestation] = {}
    for capability_id, values in grouped.items():
        decided = [item for item in values if item.status != "unknown"]
        if not decided:
            merged[capability_id] = sorted(values, key=lambda item: item.attestation_sha256)[0]
            continue
        statuses = {item.status for item in decided}
        selected = max(
            decided,
            key=lambda item: (
                PROOF_LEVEL_ORDER[item.proof_level] if item.proof_level else -1,
                item.attestation_sha256,
            ),
        )
        if len(statuses) > 1 or "contradictory" in statuses:
            failures.append("CAPABILITY_PROOF_CONTRADICTION:" + capability_id)
            material = selected.model_dump(mode="json", exclude={"attestation_sha256"})
            material["status"] = "contradictory"
            material["evidence_refs"] = sorted(
                {ref for item in decided for ref in item.evidence_refs}
            )
            merged[capability_id] = CapabilityAttestation.model_validate(
                {**material, "attestation_sha256": canonical_sha256(material)}
            )
        else:
            merged[capability_id] = selected
    return merged, sorted(set(failures))
