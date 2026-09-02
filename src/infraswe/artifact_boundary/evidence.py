from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.artifact_boundary import (
    ORIGIN_TRUST_ORDER,
    CandidateArtifactManifest,
    EvidenceArtifact,
    EvidencePackManifest,
    ScoreEvidenceBinding,
    TransportEnvelope,
    TransportPayloadEntry,
    TrialSeal,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_transport_envelope(
    manifest: CandidateArtifactManifest,
    *,
    base_revision: str,
    source_environment_id: str,
    destination_policy_id: str,
    encryption: str = "provider-managed",
    controller_signature: str | None = None,
    created_at: datetime | None = None,
) -> TransportEnvelope:
    payload = [
        TransportPayloadEntry(
            logical_name=item.logical_name,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in manifest.artifacts
    ]
    preliminary = TransportEnvelope(
        task_seal_sha256=manifest.task_seal_sha256,
        candidate_artifact_manifest_sha256=manifest.manifest_sha256,
        base_revision=base_revision,
        created_at=created_at or datetime.now(UTC),
        payload=payload,
        source_environment_id=source_environment_id,
        destination_policy_id=destination_policy_id,
        encryption=encryption,
        controller_signature=controller_signature,
        envelope_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"envelope_sha256"})
    return preliminary.model_copy(update={"envelope_sha256": canonical_sha256(material)})


def audit_transport_envelope(
    envelope: TransportEnvelope,
    manifest: CandidateArtifactManifest,
) -> list[str]:
    failures: list[str] = []
    material = envelope.model_dump(mode="json", exclude={"envelope_sha256"})
    if envelope.envelope_sha256 != canonical_sha256(material):
        failures.append("TRANSPORT_ENVELOPE_DIGEST_MISMATCH")
    if envelope.task_seal_sha256 != manifest.task_seal_sha256:
        failures.append("TRANSPORT_TASK_SEAL_MISMATCH")
    if envelope.candidate_artifact_manifest_sha256 != manifest.manifest_sha256:
        failures.append("TRANSPORT_ARTIFACT_MANIFEST_MISMATCH")
    expected = {(item.logical_name, item.sha256, item.size_bytes) for item in manifest.artifacts}
    observed = {(item.logical_name, item.sha256, item.size_bytes) for item in envelope.payload}
    if expected != observed:
        failures.append("TRANSPORT_PAYLOAD_MANIFEST_MISMATCH")
    return failures


def build_trial_seal(
    *,
    task_seal_sha256: str,
    draft_seal_sha256: str,
    artifact_policy_sha256: str,
    cache_policy_sha256: str,
    capability_resolution_sha256: str,
    runner_attestation_sha256: str,
    candidate_artifact_manifest_sha256: str,
    build_environment_sha256: str,
    verifier_environment_sha256: str,
    meter_environment_sha256: str,
    resource_lease_sha256: str,
    benchmark_cell_sha256: str,
    environment_sentinel_policy_sha256: str,
    controller: str = "infraswe-runner-v0.1",
    start_time: datetime | None = None,
) -> TrialSeal:
    preliminary = TrialSeal(
        task_seal_sha256=task_seal_sha256,
        draft_seal_sha256=draft_seal_sha256,
        artifact_policy_sha256=artifact_policy_sha256,
        cache_policy_sha256=cache_policy_sha256,
        capability_resolution_sha256=capability_resolution_sha256,
        runner_attestation_sha256=runner_attestation_sha256,
        candidate_artifact_manifest_sha256=candidate_artifact_manifest_sha256,
        build_environment_sha256=build_environment_sha256,
        verifier_environment_sha256=verifier_environment_sha256,
        meter_environment_sha256=meter_environment_sha256,
        resource_lease_sha256=resource_lease_sha256,
        benchmark_cell_sha256=benchmark_cell_sha256,
        environment_sentinel_policy_sha256=environment_sentinel_policy_sha256,
        start_time=start_time or datetime.now(UTC),
        controller=controller,
        trial_seal_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"trial_seal_sha256"})
    return preliminary.model_copy(update={"trial_seal_sha256": canonical_sha256(material)})


def audit_trial_seal(seal: TrialSeal) -> list[str]:
    material = seal.model_dump(mode="json", exclude={"trial_seal_sha256"})
    return (
        []
        if seal.trial_seal_sha256 == canonical_sha256(material)
        else ["TRIAL_SEAL_DIGEST_MISMATCH"]
    )


def build_evidence_pack(
    *,
    trial_seal: TrialSeal,
    draft_seal_sha256: str,
    candidate_artifact_manifest_sha256: str,
    runner_attestation_sha256: str,
    verifier_result_sha256: str,
    measurement_set_sha256: str,
    score_input_sha256: str,
    artifacts: Sequence[EvidenceArtifact],
) -> EvidencePackManifest:
    if audit_trial_seal(trial_seal):
        raise ValueError("cannot build EvidencePack from an invalid TrialSeal")
    bindings = {
        "draft seal": (draft_seal_sha256, trial_seal.draft_seal_sha256),
        "candidate artifact manifest": (
            candidate_artifact_manifest_sha256,
            trial_seal.candidate_artifact_manifest_sha256,
        ),
        "runner attestation": (
            runner_attestation_sha256,
            trial_seal.runner_attestation_sha256,
        ),
    }
    mismatches = [name for name, (value, sealed) in bindings.items() if value != sealed]
    if mismatches:
        raise ValueError("EvidencePack binding mismatch: " + ", ".join(mismatches))
    preliminary = EvidencePackManifest(
        task_seal_sha256=trial_seal.task_seal_sha256,
        draft_seal_sha256=draft_seal_sha256,
        trial_seal_sha256=trial_seal.trial_seal_sha256,
        candidate_artifact_manifest_sha256=candidate_artifact_manifest_sha256,
        runner_attestation_sha256=runner_attestation_sha256,
        verifier_result_sha256=verifier_result_sha256,
        measurement_set_sha256=measurement_set_sha256,
        score_input_sha256=score_input_sha256,
        artifacts=list(artifacts),
        evidence_pack_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"evidence_pack_sha256"})
    return preliminary.model_copy(update={"evidence_pack_sha256": canonical_sha256(material)})


def audit_evidence_pack(
    pack: EvidencePackManifest,
    *,
    trial_seal: TrialSeal,
    root: Path | None = None,
) -> list[str]:
    failures = audit_trial_seal(trial_seal)
    material = pack.model_dump(mode="json", exclude={"evidence_pack_sha256"})
    if pack.evidence_pack_sha256 != canonical_sha256(material):
        failures.append("EVIDENCE_PACK_DIGEST_MISMATCH")
    bindings = {
        "EVIDENCE_PACK_TASK_SEAL_MISMATCH": (
            pack.task_seal_sha256,
            trial_seal.task_seal_sha256,
        ),
        "EVIDENCE_PACK_DRAFT_SEAL_MISMATCH": (
            pack.draft_seal_sha256,
            trial_seal.draft_seal_sha256,
        ),
        "EVIDENCE_PACK_TRIAL_SEAL_MISMATCH": (
            pack.trial_seal_sha256,
            trial_seal.trial_seal_sha256,
        ),
        "EVIDENCE_PACK_ARTIFACT_MANIFEST_MISMATCH": (
            pack.candidate_artifact_manifest_sha256,
            trial_seal.candidate_artifact_manifest_sha256,
        ),
        "EVIDENCE_PACK_RUNNER_ATTESTATION_MISMATCH": (
            pack.runner_attestation_sha256,
            trial_seal.runner_attestation_sha256,
        ),
    }
    failures.extend(code for code, values in bindings.items() if values[0] != values[1])
    if root is not None:
        resolved = root.resolve(strict=True)
        for artifact in pack.artifacts:
            relative = PurePosixPath(artifact.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                failures.append("EVIDENCE_PATH_UNSAFE:" + artifact.evidence_id)
                continue
            path = resolved.joinpath(*relative.parts)
            if not path.is_file() or path.is_symlink():
                failures.append("EVIDENCE_ARTIFACT_MISSING:" + artifact.evidence_id)
                continue
            payload = path.read_bytes()
            if len(payload) != artifact.size_bytes:
                failures.append("EVIDENCE_ARTIFACT_SIZE_MISMATCH:" + artifact.evidence_id)
            if _sha256_bytes(payload) != artifact.sha256:
                failures.append("EVIDENCE_ARTIFACT_DIGEST_MISMATCH:" + artifact.evidence_id)
    return sorted(set(failures))


def audit_score_binding(
    binding: ScoreEvidenceBinding,
    pack: EvidencePackManifest,
) -> list[str]:
    failures: list[str] = []
    if binding.evidence_pack_sha256 != pack.evidence_pack_sha256:
        failures.append("SCORE_EVIDENCE_PACK_MISMATCH")
    by_id = {item.evidence_id: item for item in pack.artifacts}
    for reference in binding.evidence_refs:
        artifact = by_id.get(reference)
        if artifact is None:
            failures.append("SCORE_EVIDENCE_REF_MISSING:" + reference)
            continue
        if artifact.authority != binding.required_authority:
            failures.append("SCORE_EVIDENCE_AUTHORITY_MISMATCH:" + reference)
        required_trust = {
            "VERIFIER": 2,
            "METER": 3,
            "INFRASTRUCTURE": 4,
            "BOUNDED_JUDGE": 2,
        }[binding.required_authority]
        if ORIGIN_TRUST_ORDER[artifact.origin_trust] < required_trust:
            failures.append("SCORE_EVIDENCE_TRUST_TOO_LOW:" + reference)
    if binding.status == "VALID" and failures:
        failures.append("VALID_SCORE_COMPONENT_HAS_INVALID_EVIDENCE")
    return sorted(set(failures))
