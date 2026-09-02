from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

OriginTrust = Literal[
    "T0_AGENT_SELF_REPORTED",
    "T1_COLLECTOR_CAPTURED",
    "T2_PRISTINE_REEXECUTED",
    "T3_TRUSTED_METERED",
    "T4_INFRA_ATTESTED",
]
EvidenceAuthority = Literal[
    "NONE",
    "CANDIDATE_CLAIM",
    "VERIFIER",
    "METER",
    "INFRASTRUCTURE",
    "BOUNDED_JUDGE",
]

ORIGIN_TRUST_ORDER = {
    "T0_AGENT_SELF_REPORTED": 0,
    "T1_COLLECTOR_CAPTURED": 1,
    "T2_PRISTINE_REEXECUTED": 2,
    "T3_TRUSTED_METERED": 3,
    "T4_INFRA_ATTESTED": 4,
}


class ArtifactBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ArtifactAllowRule(ArtifactBoundaryModel):
    logical_name: str
    source_glob: str
    kind: Literal[
        "source-patch",
        "source-file",
        "configuration",
        "declaration",
        "build-recipe",
        "binary",
    ]
    max_bytes: int = Field(gt=0)
    required: bool = True


class ArtifactFilesystemPolicy(ArtifactBoundaryModel):
    symlink_policy: Literal["reject", "reject-escape"] = "reject-escape"
    hardlink_policy: Literal["reject", "flatten-within-root"] = "flatten-within-root"
    special_file_policy: Literal["reject"] = "reject"
    max_total_bytes: int = Field(default=104_857_600, gt=0)
    max_file_count: int = Field(default=10_000, ge=1)
    max_path_length: int = Field(default=240, ge=32)
    secret_policy: Literal["reject", "report"] = "reject"


class ArtifactCanonicalizationPolicy(ArtifactBoundaryModel):
    normalize_patch_headers: bool = True
    preserve_executable_bit: bool = True
    normalize_text_eol: bool = False
    stable_archive_order: Literal[True] = True
    timestamp_policy: Literal["zero"] = "zero"
    uid_gid_policy: Literal["zero"] = "zero"


class ArtifactBuildPolicy(ArtifactBoundaryModel):
    mode: Literal["pristine-rebuild"] = "pristine-rebuild"
    binary_submission: Literal["forbidden", "allowlisted-attested"] = "forbidden"
    network: Literal["disabled", "allowlist"] = "disabled"
    cache_policy_sha256: Digest


class ArtifactEvidencePolicy(ArtifactBoundaryModel):
    candidate_claim_authority: Literal["none"] = "none"
    official_evidence_min_origin_trust: Literal["T2_PRISTINE_REEXECUTED"] = "T2_PRISTINE_REEXECUTED"


class ArtifactPolicy(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    policy_id: str
    collection_roots: list[str] = Field(min_length=1)
    allow: list[ArtifactAllowRule] = Field(min_length=1)
    deny: list[str] = Field(default_factory=list)
    filesystem: ArtifactFilesystemPolicy = Field(default_factory=ArtifactFilesystemPolicy)
    canonicalization: ArtifactCanonicalizationPolicy = Field(
        default_factory=ArtifactCanonicalizationPolicy
    )
    build: ArtifactBuildPolicy
    evidence: ArtifactEvidencePolicy = Field(default_factory=ArtifactEvidencePolicy)
    policy_sha256: Digest

    @model_validator(mode="after")
    def policy_is_unambiguous(self) -> ArtifactPolicy:
        names = [item.logical_name for item in self.allow]
        if len(names) != len(set(names)):
            raise ValueError("artifact allow rules require unique logical names")
        if self.build.binary_submission == "forbidden" and any(
            item.kind == "binary" for item in self.allow
        ):
            raise ValueError("binary allow rule conflicts with forbidden binary submission")
        for item in self.allow:
            path = PurePosixPath(item.logical_name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifact logical names must stay output-relative")
        return self


class WorkspaceFreezeAttestation(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    environment_id: str
    frozen_at: datetime
    filesystem_frozen: bool
    candidate_processes_terminated: bool
    accelerator_work_drained: bool
    mutation_channel_closed: bool
    environment_sha256: Digest
    attestation_sha256: Digest

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> WorkspaceFreezeAttestation:
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() is None:
            raise ValueError("workspace freeze timestamp must be timezone-aware")
        return self


class CandidateArtifactEntry(ArtifactBoundaryModel):
    logical_name: str
    relative_path: str
    kind: Literal[
        "source-patch",
        "source-file",
        "configuration",
        "declaration",
        "build-recipe",
        "binary",
    ]
    media_type: str
    sha256: Digest
    size_bytes: int = Field(ge=0)
    mode: str = Field(pattern=r"^0[0-7]{3}$")
    origin_trust: Literal["T0_AGENT_SELF_REPORTED", "T1_COLLECTOR_CAPTURED"]
    authority: Literal["NONE", "CANDIDATE_CLAIM"] = "NONE"


class CandidateArtifactManifest(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    artifact_policy_id: str
    artifact_policy_sha256: Digest
    task_seal_sha256: Digest
    candidate_id: str
    artifacts: list[CandidateArtifactEntry] = Field(min_length=1)
    collection_environment_sha256: Digest
    freeze_attestation_sha256: Digest
    canonicalizer_version: str
    manifest_sha256: Digest

    @model_validator(mode="after")
    def entries_are_unique(self) -> CandidateArtifactManifest:
        names = [item.logical_name for item in self.artifacts]
        paths = [item.relative_path for item in self.artifacts]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("candidate artifact names and paths must be unique")
        return self


class TransportPayloadEntry(ArtifactBoundaryModel):
    logical_name: str
    sha256: Digest
    size_bytes: int = Field(ge=0)


class TransportEnvelope(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    task_seal_sha256: Digest
    candidate_artifact_manifest_sha256: Digest
    base_revision: Digest
    created_at: datetime
    payload: list[TransportPayloadEntry] = Field(min_length=1)
    source_environment_id: str
    destination_policy_id: str
    encryption: Literal["provider-managed", "controller-managed", "none"]
    controller_signature: str | None = None
    envelope_sha256: Digest

    @model_validator(mode="after")
    def created_at_is_aware(self) -> TransportEnvelope:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("transport envelope timestamp must be timezone-aware")
        names = [item.logical_name for item in self.payload]
        if len(names) != len(set(names)):
            raise ValueError("transport payload names must be unique")
        return self


class PristineBase(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    repository_sha256: Digest
    base_revision: Digest
    submodule_revisions: dict[str, Digest] = Field(default_factory=dict)
    lfs_object_digests: list[Digest] = Field(default_factory=list)
    image_sha256: Digest
    toolchain_sha256: Digest
    dependency_lock_sha256: Digest
    build_script_sha256: Digest
    environment_allowlist_sha256: Digest
    network_policy_id: str
    pristine_base_sha256: Digest


class PristineApplyResult(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    pristine_base_sha256: Digest
    transport_envelope_sha256: Digest
    candidate_artifact_manifest_sha256: Digest
    status: Literal["APPLIED", "APPLY_FAILED", "BENCHMARK_DEFECT", "INFRA_INVALID"]
    owner: Literal["candidate", "benchmark", "infrastructure", "none"]
    failure_code: str | None = None
    resulting_source_tree_sha256: Digest | None = None
    environment_allowlist_sha256: Digest
    network_observed: bool = False
    result_sha256: Digest

    @model_validator(mode="after")
    def apply_status_has_the_right_owner(self) -> PristineApplyResult:
        expected_owner = {
            "APPLIED": "none",
            "APPLY_FAILED": "candidate",
            "BENCHMARK_DEFECT": "benchmark",
            "INFRA_INVALID": "infrastructure",
        }[self.status]
        if self.owner != expected_owner:
            raise ValueError("pristine apply status has the wrong failure owner")
        if self.status == "APPLIED" and (
            self.failure_code is not None or self.resulting_source_tree_sha256 is None
        ):
            raise ValueError("successful pristine apply requires a source digest and no failure")
        if self.status != "APPLIED" and not self.failure_code:
            raise ValueError("failed pristine apply requires a failure code")
        return self


class PristineBuildResult(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    apply_result_sha256: Digest
    build_environment_sha256: Digest
    toolchain_sha256: Digest
    dependency_lock_sha256: Digest
    network_policy: Literal["disabled", "allowlist"]
    network_observed: bool
    cache_policy_sha256: Digest
    status: Literal["BUILT", "BUILD_FAILED", "BENCHMARK_DEFECT", "INFRA_INVALID"]
    owner: Literal["candidate", "benchmark", "infrastructure", "none"]
    output_sha256: Digest | None = None
    log_evidence_ref: str
    failure_code: str | None = None
    result_sha256: Digest

    @model_validator(mode="after")
    def build_policy_and_owner_are_coherent(self) -> PristineBuildResult:
        if self.network_policy == "disabled" and self.network_observed and self.status == "BUILT":
            raise ValueError("successful build cannot violate disabled network policy")
        if self.status == "BUILT" and (
            self.owner != "none" or self.output_sha256 is None or self.failure_code is not None
        ):
            raise ValueError("successful pristine build has invalid owner/output")
        if self.status != "BUILT" and (self.owner == "none" or not self.failure_code):
            raise ValueError("failed pristine build requires an owner and failure code")
        return self


class CacheDeclaration(ArtifactBoundaryModel):
    cache_id: str
    kind: Literal[
        "source",
        "dependency",
        "compiler",
        "build",
        "jit",
        "runtime",
        "page",
        "device",
    ]
    owner: Literal["candidate", "builder", "verifier", "meter", "runner"]
    initial_state: Literal["empty", "sealed-read-only", "prewarmed-profile", "isolated-fresh"]
    reusable_across_candidates: bool = False

    @model_validator(mode="after")
    def candidate_cache_is_never_shared(self) -> CacheDeclaration:
        if self.owner == "candidate" and self.reusable_across_candidates:
            raise ValueError("Candidate-owned cache cannot cross Candidate boundaries")
        return self


class CachePolicy(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    policy_id: str
    declarations: list[CacheDeclaration] = Field(min_length=1)
    paired_order: Literal["counterbalanced", "randomized-pre-registered"]
    policy_sha256: Digest


class EvidenceProducerIdentity(ArtifactBoundaryModel):
    producer_id: str
    role: Literal["pristine-verifier", "trusted-meter", "infra-attestor", "bounded-judge"]
    implementation_sha256: Digest
    image_sha256: Digest
    configuration_sha256: Digest


class OfficialTimingSample(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    sample_id: str
    pair_id: str
    pair_position: Literal["baseline-first", "candidate-first"]
    role: Literal["baseline", "candidate"]
    repetition: int = Field(ge=1)
    work_units: int = Field(ge=1)
    host_elapsed_seconds: float = Field(gt=0)
    device_elapsed_seconds: float | None = Field(default=None, gt=0)
    synchronization: Literal[
        "device-synchronize",
        "event-completion",
        "blocking-contract",
        "distributed-barrier-and-device-sync",
    ]
    completion_counter_before: int = Field(ge=0)
    completion_counter_after: int = Field(ge=0)
    output_consumed_by_trusted_plane: bool
    deferred_error_check_passed: bool
    profiler_active: Literal[False] = False
    origin_trust: Literal["T3_TRUSTED_METERED"] = "T3_TRUSTED_METERED"
    evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def sample_proves_completed_work(self) -> OfficialTimingSample:
        if self.completion_counter_after - self.completion_counter_before != self.work_units:
            raise ValueError("timing sample completion counter does not match work units")
        if not self.output_consumed_by_trusted_plane:
            raise ValueError("timing sample output must be consumed by the trusted plane")
        if not self.deferred_error_check_passed:
            raise ValueError("timing sample must surface deferred execution errors")
        return self


class MeasurementIntegrityReport(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    timer_policy_id: str
    paired_order_policy: Literal["counterbalanced", "randomized-pre-registered"]
    samples: list[OfficialTimingSample] = Field(default_factory=list)
    all_samples_retained: bool
    official_timing_profiled: Literal[False] = False
    status: Literal["PASS", "BENCHMARK_DEFECT", "INFRA_INVALID"]
    failure_codes: list[str] = Field(default_factory=list)
    report_sha256: Digest

    @model_validator(mode="after")
    def integrity_status_matches_samples(self) -> MeasurementIntegrityReport:
        sample_ids = [item.sample_id for item in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("official timing sample ids must be unique")
        if self.status == "PASS" and (not self.all_samples_retained or self.failure_codes):
            raise ValueError("passing timing integrity requires all samples and no failures")
        if self.status != "PASS" and not self.failure_codes:
            raise ValueError("non-passing timing integrity requires failure codes")
        return self


class EvidenceArtifact(ArtifactBoundaryModel):
    evidence_id: str = Field(pattern=r"^evidence://[a-z0-9][a-z0-9._/-]*$")
    relative_path: str
    media_type: str
    sha256: Digest
    size_bytes: int = Field(ge=0)
    origin_trust: OriginTrust
    authority: EvidenceAuthority
    producer: EvidenceProducerIdentity
    sanitized: bool = False

    @model_validator(mode="after")
    def trust_supports_authority(self) -> EvidenceArtifact:
        minimum = {
            "NONE": 0,
            "CANDIDATE_CLAIM": 0,
            "VERIFIER": 2,
            "METER": 3,
            "INFRASTRUCTURE": 4,
            "BOUNDED_JUDGE": 2,
        }[self.authority]
        if ORIGIN_TRUST_ORDER[self.origin_trust] < minimum:
            raise ValueError("evidence origin trust is below its claimed authority")
        expected_roles = {
            "VERIFIER": {"pristine-verifier"},
            "METER": {"trusted-meter"},
            "INFRASTRUCTURE": {"infra-attestor"},
            "BOUNDED_JUDGE": {"bounded-judge"},
        }
        if (
            self.authority in expected_roles
            and self.producer.role not in expected_roles[self.authority]
        ):
            raise ValueError("evidence producer role does not own the claimed authority")
        return self


class TrialSeal(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    task_seal_sha256: Digest
    draft_seal_sha256: Digest
    artifact_policy_sha256: Digest
    cache_policy_sha256: Digest
    capability_resolution_sha256: Digest
    runner_attestation_sha256: Digest
    candidate_artifact_manifest_sha256: Digest
    build_environment_sha256: Digest
    verifier_environment_sha256: Digest
    meter_environment_sha256: Digest
    resource_lease_sha256: Digest
    benchmark_cell_sha256: Digest
    environment_sentinel_policy_sha256: Digest
    start_time: datetime
    controller: str
    trial_seal_sha256: Digest

    @model_validator(mode="after")
    def start_time_is_aware(self) -> TrialSeal:
        if self.start_time.tzinfo is None or self.start_time.utcoffset() is None:
            raise ValueError("TrialSeal start time must be timezone-aware")
        return self


class EvidencePackManifest(ArtifactBoundaryModel):
    schema_version: Literal["0.1"] = "0.1"
    task_seal_sha256: Digest
    draft_seal_sha256: Digest
    trial_seal_sha256: Digest
    candidate_artifact_manifest_sha256: Digest
    runner_attestation_sha256: Digest
    verifier_result_sha256: Digest
    measurement_set_sha256: Digest
    score_input_sha256: Digest
    artifacts: list[EvidenceArtifact] = Field(min_length=1)
    evidence_pack_sha256: Digest

    @model_validator(mode="after")
    def evidence_entries_are_unique(self) -> EvidencePackManifest:
        identifiers = [item.evidence_id for item in self.artifacts]
        paths = [item.relative_path for item in self.artifacts]
        if len(identifiers) != len(set(identifiers)) or len(paths) != len(set(paths)):
            raise ValueError("EvidencePack evidence ids and paths must be unique")
        return self


class ScoreEvidenceBinding(ArtifactBoundaryModel):
    component_id: str
    value: float | None = Field(default=None, ge=0, le=1)
    status: Literal["VALID", "UNRESOLVED", "NOT_APPLICABLE"]
    required_authority: Literal["VERIFIER", "METER", "INFRASTRUCTURE", "BOUNDED_JUDGE"]
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_pack_sha256: Digest

    @model_validator(mode="after")
    def status_controls_value_and_refs(self) -> ScoreEvidenceBinding:
        if self.status == "VALID" and (self.value is None or not self.evidence_refs):
            raise ValueError("valid score component requires a value and EvidenceRefs")
        if self.status != "VALID" and self.value is not None:
            raise ValueError("unresolved/not-applicable component value must be null")
        return self
