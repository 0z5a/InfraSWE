from __future__ import annotations

from collections.abc import Collection, Mapping

from infraswe.artifact_boundary.collector import audit_candidate_manifest
from infraswe.artifact_boundary.evidence import audit_transport_envelope
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.artifact_boundary import (
    ArtifactBuildPolicy,
    CandidateArtifactManifest,
    PristineApplyResult,
    PristineBase,
    PristineBuildResult,
    TransportEnvelope,
)

_ZERO_DIGEST = "sha256:" + "0" * 64
_FORBIDDEN_INHERITED_ENV = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_CLIENT_SECRET",
    "CUDA_CACHE_PATH",
    "GITHUB_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HOME",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "SSH_AUTH_SOCK",
}


def sanitize_pristine_environment(
    source: Mapping[str, str],
    *,
    allowed_names: Collection[str],
) -> dict[str, str]:
    """Construct a new verifier/build environment without Agent-controlled inheritance."""

    allowed = set(allowed_names) - _FORBIDDEN_INHERITED_ENV
    return {name: source[name] for name in sorted(allowed) if name in source}


def build_pristine_apply_result(
    *,
    pristine_base: PristineBase,
    envelope: TransportEnvelope,
    manifest: CandidateArtifactManifest,
    environment_allowlist_sha256: str,
    patch_applied: bool,
    resulting_source_tree_sha256: str | None = None,
    base_available: bool = True,
    verifier_assets_valid: bool = True,
    infrastructure_available: bool = True,
    network_observed: bool = False,
) -> PristineApplyResult:
    base_material = pristine_base.model_dump(mode="json", exclude={"pristine_base_sha256"})
    if pristine_base.pristine_base_sha256 != canonical_sha256(base_material):
        verifier_assets_valid = False
    transport_failures = audit_transport_envelope(envelope, manifest)
    manifest_failures = audit_candidate_manifest(manifest)
    if not infrastructure_available or transport_failures or manifest_failures:
        status, owner, failure = (
            "INFRA_INVALID",
            "infrastructure",
            "PRISTINE_TRANSPORT_OR_INFRA_INVALID",
        )
    elif not base_available or not verifier_assets_valid:
        status, owner, failure = (
            "BENCHMARK_DEFECT",
            "benchmark",
            "PRISTINE_BASE_OR_VERIFIER_ASSET_INVALID",
        )
    elif not patch_applied:
        status, owner, failure = (
            "APPLY_FAILED",
            "candidate",
            "CANDIDATE_PATCH_APPLY_FAILED",
        )
    elif resulting_source_tree_sha256 is None:
        raise ValueError("successful pristine apply requires resulting source-tree digest")
    else:
        status, owner, failure = "APPLIED", "none", None
    preliminary = PristineApplyResult(
        pristine_base_sha256=pristine_base.pristine_base_sha256,
        transport_envelope_sha256=envelope.envelope_sha256,
        candidate_artifact_manifest_sha256=manifest.manifest_sha256,
        status=status,
        owner=owner,
        failure_code=failure,
        resulting_source_tree_sha256=(
            resulting_source_tree_sha256 if status == "APPLIED" else None
        ),
        environment_allowlist_sha256=environment_allowlist_sha256,
        network_observed=network_observed,
        result_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"result_sha256"})
    return preliminary.model_copy(update={"result_sha256": canonical_sha256(material)})


def audit_pristine_apply_result(result: PristineApplyResult) -> list[str]:
    material = result.model_dump(mode="json", exclude={"result_sha256"})
    return (
        []
        if result.result_sha256 == canonical_sha256(material)
        else ["PRISTINE_APPLY_RESULT_DIGEST_MISMATCH"]
    )


def build_pristine_build_result(
    *,
    apply_result: PristineApplyResult,
    build_policy: ArtifactBuildPolicy,
    build_environment_sha256: str,
    toolchain_sha256: str,
    dependency_lock_sha256: str,
    network_observed: bool,
    build_succeeded: bool,
    output_sha256: str | None,
    log_evidence_ref: str,
    infrastructure_available: bool = True,
    harness_valid: bool = True,
) -> PristineBuildResult:
    if audit_pristine_apply_result(apply_result):
        raise ValueError("PristineApplyResult digest mismatch")
    if apply_result.status != "APPLIED":
        raise ValueError("pristine build requires a successful apply")
    if build_policy.network == "disabled" and network_observed:
        status, owner, failure = (
            "BUILD_FAILED",
            "candidate",
            "UNDECLARED_BUILD_NETWORK_USE",
        )
        recorded_network = True
    elif not infrastructure_available:
        status, owner, failure = "INFRA_INVALID", "infrastructure", "BUILD_INFRA_INVALID"
        recorded_network = network_observed
    elif not harness_valid:
        status, owner, failure = "BENCHMARK_DEFECT", "benchmark", "BUILD_HARNESS_DEFECT"
        recorded_network = network_observed
    elif not build_succeeded:
        status, owner, failure = "BUILD_FAILED", "candidate", "CANDIDATE_BUILD_FAILED"
        recorded_network = network_observed
    elif output_sha256 is None:
        raise ValueError("successful pristine build requires output digest")
    else:
        status, owner, failure = "BUILT", "none", None
        recorded_network = network_observed
    preliminary = PristineBuildResult(
        apply_result_sha256=apply_result.result_sha256,
        build_environment_sha256=build_environment_sha256,
        toolchain_sha256=toolchain_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        network_policy=build_policy.network,
        network_observed=recorded_network,
        cache_policy_sha256=build_policy.cache_policy_sha256,
        status=status,
        owner=owner,
        output_sha256=output_sha256 if status == "BUILT" else None,
        log_evidence_ref=log_evidence_ref,
        failure_code=failure,
        result_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"result_sha256"})
    return preliminary.model_copy(update={"result_sha256": canonical_sha256(material)})


def audit_pristine_build_result(result: PristineBuildResult) -> list[str]:
    material = result.model_dump(mode="json", exclude={"result_sha256"})
    return (
        []
        if result.result_sha256 == canonical_sha256(material)
        else ["PRISTINE_BUILD_RESULT_DIGEST_MISMATCH"]
    )
