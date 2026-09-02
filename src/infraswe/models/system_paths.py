from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import (
    DRAFT_STATE_ORDER,
    Digest,
    DraftAcceptanceContract,
    DraftMetadata,
    DraftTarget,
)

SystemDomain = Literal["distributed-communication", "memory-tiering"]
LoadRegime = Literal["light", "normal", "knee", "saturation", "overload", "soak"]


class SystemPathModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SystemPathCandidate(SystemPathModel):
    domain: SystemDomain
    kind: Literal["git-diff", "source-tree", "package", "generated"]
    revision: Digest
    intent: str
    implementation_kind: Literal[
        "communication-native",
        "communication-plugin",
        "communication-framework-integration",
        "memory-tiering-runtime",
        "framework-offload-integration",
        "native-offload-extension",
        "native-library",
        "mixed-system",
    ]
    entrypoints: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def implementation_matches_domain(self) -> SystemPathCandidate:
        communication = self.implementation_kind.startswith("communication") or (
            self.implementation_kind in {"native-library", "mixed-system"}
        )
        if self.domain == "distributed-communication" and not communication:
            raise ValueError("communication Draft requires a communication implementation kind")
        if self.domain == "memory-tiering" and communication:
            raise ValueError("memory-tiering Draft requires an offload implementation kind")
        return self


class SystemRetrievalBinding(SystemPathModel):
    status: Literal["complete", "partial", "blocked", "stale", "conflicting"]
    anchor_plugin: Literal["communication-v1", "memory-tier-v1"]
    corpus_cutoff: datetime
    snapshot_sha256: Digest
    query_plan_sha256: Digest
    leakage_audit_sha256: Digest
    precedent_set_sha256: Digest | None = None
    trust_card_sha256: Digest | None = None
    human_waiver_sha256: Digest | None = None

    @model_validator(mode="after")
    def official_status_has_a_precedent_set(self) -> SystemRetrievalBinding:
        if self.corpus_cutoff.tzinfo is None or self.corpus_cutoff.utcoffset() is None:
            raise ValueError("retrieval cutoff must be timezone-aware")
        if self.status == "complete" and (
            self.precedent_set_sha256 is None or self.trust_card_sha256 is None
        ):
            raise ValueError("complete retrieval requires precedent-set and trust digests")
        if self.status == "partial" and self.human_waiver_sha256 is None:
            raise ValueError("partial retrieval requires an explicit human waiver")
        return self


class CommunicationSemantics(SystemPathModel):
    ordering_scope: str
    async_completion: Literal["event-backed", "future-backed", "blocking", "profile-defined"]
    error_propagation: Literal["structured"] = "structured"
    collective_order_consistency: Literal["required"] = "required"
    no_rank_local_divergence: Literal[True] = True


class CommunicationLifecycle(SystemPathModel):
    communicator_owner: str
    cache_policy: str
    teardown_phase: str
    repeated_init_destroy: Literal["required"] = "required"
    bounded_resource_growth: Literal[True] = True
    no_silent_fallback: Literal[True] = True


class CommunicationCellIdentity(SystemPathModel):
    cell_id: str
    node_count: int = Field(ge=1)
    accelerators_per_node: int = Field(ge=1)
    accelerator_model: str
    topology_sha256: Digest
    nic_profile: str
    numa_binding: str
    transport: str
    provider: str
    provider_version: str
    runtime_driver: str
    message_size_portfolio_sha256: Digest
    collective_mix_sha256: Digest
    concurrency_protocol_id: str


class CommunicationContract(SystemPathModel):
    layer: Literal[
        "collective-library",
        "transport-runtime",
        "framework-process-group",
        "collective-scheduling",
        "communication-lifecycle",
        "one-sided-runtime",
    ]
    execution_scope: Literal["library", "integrated"]
    providers: list[str] = Field(min_length=1)
    operations: list[str] = Field(min_length=1)
    semantics: CommunicationSemantics
    lifecycle: CommunicationLifecycle
    required_cells: list[CommunicationCellIdentity] = Field(min_length=1)
    optional_cells: list[CommunicationCellIdentity] = Field(default_factory=list)
    message_size_portfolio_sha256: Digest
    concurrency_protocol_id: Literal[
        "communication-concurrency-core-v1",
        "communication-concurrency-recovery-v1",
    ]

    @model_validator(mode="after")
    def cells_are_unique_and_bound_to_portfolio(self) -> CommunicationContract:
        cells = [*self.required_cells, *self.optional_cells]
        identifiers = [cell.cell_id for cell in cells]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("communication cells must have unique ids")
        if any(
            cell.message_size_portfolio_sha256 != self.message_size_portfolio_sha256
            for cell in cells
        ):
            raise ValueError("communication cells must bind the Draft message portfolio")
        if any(cell.provider not in self.providers for cell in cells):
            raise ValueError("communication cell provider must be declared by the contract")
        return self


class SystemPathScoringPolicy(SystemPathModel):
    deployability_template_id: Literal["deployability-v0.4"] = "deployability-v0.4"
    project_fit_template_id: Literal["project-fit-system-path-v0.5.1"] = (
        "project-fit-system-path-v0.5.1"
    )
    concurrent_stability_source: Literal["system-path-load-cells-v1"] = "system-path-load-cells-v1"
    operational_fit_source: Literal["concurrent-stability"] = "concurrent-stability"
    operational_projection_policy: Literal["identity-v1"] = "identity-v1"
    communication_domain_score: Literal["forbidden"] = "forbidden"
    offload_domain_score: Literal["forbidden"] = "forbidden"
    generic_cross_platform_score: Literal["forbidden"] = "forbidden"
    backend_or_tier_count_score: Literal["forbidden"] = "forbidden"
    parent_inherits_child_pure_triton_x: Literal[False] = False


class CommunicationDraftSpec(SystemPathModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    template_id: Literal["communication-path-integration-v1"] = "communication-path-integration-v1"
    draft: DraftMetadata
    target: DraftTarget
    candidate: SystemPathCandidate
    retrieval: SystemRetrievalBinding | None = None
    acceptance_contract: DraftAcceptanceContract | None = None
    communication: CommunicationContract | None = None
    scoring: SystemPathScoringPolicy = Field(default_factory=SystemPathScoringPolicy)

    @model_validator(mode="after")
    def state_and_domain_are_coherent(self) -> CommunicationDraftSpec:
        if self.candidate.domain != "distributed-communication":
            raise ValueError("communication Draft candidate domain is invalid")
        state = DRAFT_STATE_ORDER[self.draft.state]
        if state >= 2 and self.retrieval is None:
            raise ValueError("D2+ communication Draft requires retrieval evidence")
        if state >= 2 and self.retrieval and self.retrieval.status not in {"complete", "partial"}:
            raise ValueError("D2+ communication Draft retrieval is not sealable")
        if state >= 3 and (self.acceptance_contract is None or self.communication is None):
            raise ValueError("D3+ communication Draft requires contracts")
        if state >= 6 and (
            self.acceptance_contract is None or self.acceptance_contract.status != "sealed"
        ):
            raise ValueError("D6+ communication Draft requires a sealed contract")
        return self


class MemoryTier(SystemPathModel):
    id: str
    kind: Literal["device", "host-pinned", "host-pageable", "cxl", "nvme", "remote-memory"]
    numa_node: int | None = Field(default=None, ge=0)
    capacity_bytes: int = Field(gt=0)
    policy: Literal["required", "optional", "explicit-unsupported", "roadmap"]

    @model_validator(mode="after")
    def numa_only_applies_to_host_like_tiers(self) -> MemoryTier:
        if self.numa_node is not None and self.kind not in {
            "host-pinned",
            "host-pageable",
            "cxl",
        }:
            raise ValueError("NUMA identity only applies to host-like memory tiers")
        return self


class ResidencyTransition(SystemPathModel):
    source_state: Literal[
        "UNALLOCATED",
        "DEVICE_RESIDENT",
        "EVICTING",
        "HOST_RESIDENT",
        "PREFETCHING",
        "DEVICE_READY",
        "INVALIDATED",
        "FAILED",
        "FREED",
    ]
    target_state: Literal[
        "DEVICE_RESIDENT",
        "EVICTING",
        "HOST_RESIDENT",
        "PREFETCHING",
        "DEVICE_READY",
        "INVALIDATED",
        "FAILED",
        "FREED",
    ]
    operation: str

    @model_validator(mode="after")
    def transition_changes_state(self) -> ResidencyTransition:
        if self.source_state == self.target_state:
            raise ValueError("residency transitions must change state")
        return self


class MemoryTierCapacity(SystemPathModel):
    device_budget_bytes: int = Field(gt=0)
    host_pinned_budget_bytes: int = Field(ge=0)
    host_pageable_budget_bytes: int = Field(ge=0)
    queue_limit: int = Field(gt=0)
    unbounded_growth: Literal[False] = False


OffloadProfileId = Literal[
    "memory-tiering-offload-runtime-v1",
    "kv-cache-cpu-offload-v1",
    "weight-cpu-offload-v1",
    "training-state-cpu-offload-v1",
    "activation-cpu-offload-v1",
    "checkpoint-staging-cpu-offload-v1",
]
MemoryObjectKind = Literal[
    "kv-cache",
    "weight",
    "training-state",
    "activation",
    "checkpoint-staging",
]


class MemoryTieringContract(SystemPathModel):
    profile_id: OffloadProfileId
    abstract_parent_profile: Literal["memory-tiering-offload-runtime-v1"] = (
        "memory-tiering-offload-runtime-v1"
    )
    object_kind: MemoryObjectKind
    mutability: str
    source_tier_id: str
    destination_tier_id: str
    fallback_tier_id: str | None = None
    tiers: list[MemoryTier] = Field(min_length=2)
    transitions: list[ResidencyTransition] = Field(min_length=1)
    version_token_required: bool
    consumer_waits_for_ready_event: Literal[True] = True
    explicit_fallback: Literal[True] = True
    teardown_quiescence: Literal[True] = True
    request_or_step_isolation: Literal[True] = True
    allocator_owner: str
    residency_owner: str
    copy_stream_owner: str
    capacity: MemoryTierCapacity

    @model_validator(mode="after")
    def profile_tiers_and_versioning_are_coherent(self) -> MemoryTieringContract:
        profile_objects = {
            "kv-cache-cpu-offload-v1": "kv-cache",
            "weight-cpu-offload-v1": "weight",
            "training-state-cpu-offload-v1": "training-state",
            "activation-cpu-offload-v1": "activation",
            "checkpoint-staging-cpu-offload-v1": "checkpoint-staging",
        }
        expected = profile_objects.get(self.profile_id)
        if expected is not None and self.object_kind != expected:
            raise ValueError("offload profile does not match object_kind")
        identifiers = [tier.id for tier in self.tiers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("memory tier ids must be unique")
        required_ids = {self.source_tier_id, self.destination_tier_id}
        if self.fallback_tier_id is not None:
            required_ids.add(self.fallback_tier_id)
        if not required_ids.issubset(set(identifiers)):
            raise ValueError("source/destination/fallback tiers must be declared")
        if self.object_kind in {"kv-cache", "training-state", "checkpoint-staging"} and (
            not self.version_token_required
        ):
            raise ValueError("mutable or durable offload objects require version tokens")
        return self


class OffloadBaseline(SystemPathModel):
    mode: str
    revision: Digest
    outcome: Literal["runnable", "expected-capacity-limit", "unresolved"] = "runnable"


class OffloadBaselineSet(SystemPathModel):
    semantic_baseline: OffloadBaseline
    scoring_baseline: OffloadBaseline
    load_anchor: OffloadBaseline


class MemoryTierCellIdentity(SystemPathModel):
    cell_id: str
    gpu_model: str
    gpu_count: int = Field(ge=1)
    gpu_topology_sha256: Digest
    gpu_memory_bytes: int = Field(gt=0)
    cpu_model: str
    cpu_socket_count: int = Field(ge=1)
    numa_topology_sha256: Digest
    host_memory_bytes: int = Field(gt=0)
    cpu_affinity: str
    numa_policy: str
    interconnect: str
    pinned_policy: Literal["required", "optional", "forbidden"]
    host_page_policy: str
    os_kernel: str
    driver_runtime: str
    framework: str
    allocator: str
    background_load_policy: str


class MemoryTierDeployment(SystemPathModel):
    performance_mode: Literal["fixed-workload", "fixed-device-budget", "capacity-enable"]
    workload_portfolio_sha256: Digest
    required_cells: list[MemoryTierCellIdentity] = Field(min_length=1)
    optional_cells: list[MemoryTierCellIdentity] = Field(default_factory=list)
    service_target_sha256: Digest
    residency_target_sha256: Digest
    transfer_target_sha256: Digest
    concurrency_protocol_id: Literal["memory-tiering-load-normalized-v1"] = (
        "memory-tiering-load-normalized-v1"
    )

    @model_validator(mode="after")
    def cells_are_unique(self) -> MemoryTierDeployment:
        identifiers = [cell.cell_id for cell in [*self.required_cells, *self.optional_cells]]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("memory-tier cells must have unique ids")
        return self


class MemoryTierScoringPolicy(SystemPathScoringPolicy):
    cell_artifact_template_id: Literal["cell-artifact-memory-tiering-v0.5.2"] = (
        "cell-artifact-memory-tiering-v0.5.2"
    )
    cpu_offload_score: Literal["forbidden"] = "forbidden"
    memory_tier_count_score: Literal["forbidden"] = "forbidden"


class MemoryTierDraftSpec(SystemPathModel):
    schema_version: Literal["0.5.2"] = "0.5.2"
    template_id: Literal["memory-tier-integration-v1"] = "memory-tier-integration-v1"
    draft: DraftMetadata
    target: DraftTarget
    candidate: SystemPathCandidate
    retrieval: SystemRetrievalBinding | None = None
    acceptance_contract: DraftAcceptanceContract | None = None
    memory_tiering: MemoryTieringContract | None = None
    baseline_set: OffloadBaselineSet | None = None
    deployment: MemoryTierDeployment | None = None
    scoring: MemoryTierScoringPolicy = Field(default_factory=MemoryTierScoringPolicy)

    @model_validator(mode="after")
    def state_profile_and_domain_are_coherent(self) -> MemoryTierDraftSpec:
        if self.candidate.domain != "memory-tiering":
            raise ValueError("memory-tier Draft candidate domain is invalid")
        state = DRAFT_STATE_ORDER[self.draft.state]
        if state >= 2 and self.retrieval is None:
            raise ValueError("D2+ memory-tier Draft requires retrieval evidence")
        if state >= 2 and self.retrieval and self.retrieval.status not in {"complete", "partial"}:
            raise ValueError("D2+ memory-tier Draft retrieval is not sealable")
        if state >= 3 and any(
            item is None
            for item in (
                self.acceptance_contract,
                self.memory_tiering,
                self.baseline_set,
                self.deployment,
            )
        ):
            raise ValueError("D3+ memory-tier Draft requires contract/baseline/deployment")
        if (
            state >= 6
            and self.memory_tiering is not None
            and self.memory_tiering.profile_id == "memory-tiering-offload-runtime-v1"
        ):
            raise ValueError("abstract memory-tier parent profile is not sealable")
        if state >= 6 and (
            self.acceptance_contract is None or self.acceptance_contract.status != "sealed"
        ):
            raise ValueError("D6+ memory-tier Draft requires a sealed contract")
        if (
            self.deployment is not None
            and self.deployment.performance_mode == "capacity-enable"
            and self.baseline_set is not None
            and self.baseline_set.scoring_baseline.outcome
            not in {"expected-capacity-limit", "runnable"}
        ):
            raise ValueError("capacity-enable requires a resolved scoring baseline outcome")
        return self


class SystemPathInfraCertEvidence(SystemPathModel):
    schema_version: Literal["0.5.2"] = "0.5.2"
    domain: SystemDomain
    evidence_digests: list[Digest] = Field(min_length=1)
    correctness_passed: bool | None
    progress_passed: bool | None
    lifecycle_quiescent: bool | None
    bounded_resources: bool | None
    fallback_policy_respected: bool | None
    collective_order_consistent: bool | None = None
    rank_divergence_absent: bool | None = None
    deadlock_absent: bool | None = None
    residency_state_valid: bool | None = None
    version_token_valid: bool | None = None
    consumer_visibility_valid: bool | None = None
    isolation_valid: bool | None = None
    use_after_free_absent: bool | None = None
    partial_copy_absent: bool | None = None
    stale_or_lost_update_absent: bool | None = None
    prefetch_queue_bounded: bool | None = None
    host_memory_leak_absent: bool | None = None
    pageable_fallback_explicit: bool | None = None

    @model_validator(mode="after")
    def checks_are_owned_by_exactly_one_domain(self) -> SystemPathInfraCertEvidence:
        communication = (
            "collective_order_consistent",
            "rank_divergence_absent",
            "deadlock_absent",
        )
        memory = (
            "residency_state_valid",
            "version_token_valid",
            "consumer_visibility_valid",
            "isolation_valid",
            "use_after_free_absent",
            "partial_copy_absent",
            "stale_or_lost_update_absent",
            "prefetch_queue_bounded",
            "host_memory_leak_absent",
            "pageable_fallback_explicit",
        )
        irrelevant = memory if self.domain == "distributed-communication" else communication
        if any(getattr(self, name) is not None for name in irrelevant):
            raise ValueError("InfraCert evidence cannot mix communication and memory checks")
        return self


class SystemPathInfraCertResult(SystemPathModel):
    schema_version: Literal["0.5.2"] = "0.5.2"
    domain: SystemDomain
    status: Literal["pass", "fail", "unresolved"]
    hard_gate: Literal[True] = True
    failure_codes: list[str] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)
    evidence_digests: list[Digest] = Field(min_length=1)

    @model_validator(mode="after")
    def status_matches_failures_and_missing_checks(self) -> SystemPathInfraCertResult:
        if self.status == "pass" and (self.failure_codes or self.missing_checks):
            raise ValueError("passing InfraCert cannot have failures or missing checks")
        if self.status == "fail" and not self.failure_codes:
            raise ValueError("failed InfraCert requires failure codes")
        if self.status == "unresolved" and not self.missing_checks:
            raise ValueError("unresolved InfraCert requires missing checks")
        return self


class SystemPathLoadCell(SystemPathModel):
    schema_version: Literal["0.5.2"] = "0.5.2"
    domain: SystemDomain
    protocol_id: str
    regime: LoadRegime
    load_ratio: float = Field(gt=0)
    offered_work: int = Field(ge=1)
    completed_work: int = Field(ge=0)
    goodput_score: float = Field(ge=0, le=1)
    tail_score: float = Field(ge=0, le=1)
    jitter_score: float = Field(ge=0, le=1)
    overlap_progress_score: float = Field(ge=0, le=1)
    resource_stability_score: float = Field(ge=0, le=1)
    fairness_score: float = Field(ge=0, le=1)
    p99_status: Literal["official", "exploratory"]
    hard_gate_failure_codes: list[str] = Field(default_factory=list)
    evidence_digests: list[Digest] = Field(min_length=1)

    @model_validator(mode="after")
    def sample_counts_and_protocol_match_domain(self) -> SystemPathLoadCell:
        if self.completed_work > self.offered_work:
            raise ValueError("completed work cannot exceed offered work")
        if self.completed_work < 1000 and self.p99_status != "exploratory":
            raise ValueError("p99 must be exploratory below 1000 completed work items")
        if self.domain == "distributed-communication" and not self.protocol_id.startswith(
            "communication-concurrency-"
        ):
            raise ValueError("communication load cell requires communication protocol")
        if self.domain == "memory-tiering" and self.protocol_id != (
            "memory-tiering-load-normalized-v1"
        ):
            raise ValueError("memory-tier load cell requires memory-tiering protocol")
        return self


class CommunicationEfficiencyCard(SystemPathModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    score_status: Literal["not-a-score"] = "not-a-score"
    cell_identity_sha256: Digest
    latency: dict[str, Any]
    algorithmic_bandwidth: dict[str, Any]
    bus_bandwidth: dict[str, Any]
    overlap: dict[str, Any]
    rank_skew: dict[str, Any]
    lifecycle: dict[str, Any]
    raw_evidence_digests: list[Digest]
    cross_cell_ranking_allowed: Literal[False] = False


class MemoryTieringEfficiencyCard(SystemPathModel):
    schema_version: Literal["0.5.2"] = "0.5.2"
    score_status: Literal["not-a-score"] = "not-a-score"
    cell_identity_sha256: Digest
    service: dict[str, Any]
    residency: dict[str, Any]
    transfer: dict[str, Any]
    host_system: dict[str, Any]
    raw_evidence_digests: list[Digest]
    cross_cell_ranking_allowed: Literal[False] = False


class CompositeSystemPathPolicy(SystemPathModel):
    domains: tuple[Literal["memory-tiering"], Literal["distributed-communication"]] = (
        "memory-tiering",
        "distributed-communication",
    )
    concurrent_stability_aggregation_count: Literal[1] = 1
    protocol_id: Literal["composite-runtime-load-v1"] = "composite-runtime-load-v1"
    shared_resource_stall_weighted_once: Literal[True] = True
    domain_tags_are_diagnostic_only: Literal[True] = True


class SystemDraftProfile(SystemPathModel):
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+-v[0-9]+$")
    domain: SystemDomain
    template_id: Literal[
        "communication-path-integration-v1",
        "memory-tier-integration-v1",
    ]
    status: Literal["proposed", "human-reviewed"] = "proposed"
    sealable: bool
    layer: (
        Literal[
            "collective-library",
            "transport-runtime",
            "framework-process-group",
            "collective-scheduling",
            "communication-lifecycle",
            "one-sided-runtime",
        ]
        | None
    ) = None
    abstract_parent_profile: Literal["memory-tiering-offload-runtime-v1"] | None = None
    object_kind: MemoryObjectKind | None = None
    anchor_plugin: Literal["communication-v1", "memory-tier-v1"]
    performance_objectives: list[str] = Field(min_length=1)
    correctness_invariants: list[str] = Field(min_length=1)
    required_probes: list[str] = Field(min_length=1)
    project_fit_template_id: Literal["project-fit-system-path-v0.5.1"] = (
        "project-fit-system-path-v0.5.1"
    )
    comparison_scope: Literal["profile-local"] = "profile-local"
    generic_domain_score: Literal["forbidden"] = "forbidden"
    triton_portability_score: Literal["not-applicable"] = "not-applicable"

    @model_validator(mode="after")
    def domain_fields_and_sealability_are_coherent(self) -> SystemDraftProfile:
        if self.domain == "distributed-communication":
            if self.template_id != "communication-path-integration-v1":
                raise ValueError("communication profile requires communication template")
            if self.anchor_plugin != "communication-v1" or self.layer is None:
                raise ValueError("communication profile requires layer and communication anchors")
            if self.abstract_parent_profile is not None or self.object_kind is not None:
                raise ValueError("communication profile cannot declare offload object fields")
            if not self.sealable:
                raise ValueError("catalog communication profiles must be concrete")
            return self
        if self.template_id != "memory-tier-integration-v1":
            raise ValueError("memory-tier profile requires memory-tier template")
        if self.anchor_plugin != "memory-tier-v1" or self.layer is not None:
            raise ValueError("memory-tier profile requires memory-tier anchors and no comm layer")
        if self.profile_id == "memory-tiering-offload-runtime-v1":
            if self.sealable or self.object_kind is not None or self.abstract_parent_profile:
                raise ValueError("abstract memory-tier parent must remain unsealable")
            return self
        if not self.sealable or self.object_kind is None:
            raise ValueError("concrete memory-tier profiles require an object kind and sealability")
        if self.abstract_parent_profile != "memory-tiering-offload-runtime-v1":
            raise ValueError("concrete memory-tier profiles require the abstract parent")
        return self


class SystemDraftProfileCatalog(SystemPathModel):
    schema_version: Literal["0.5.2"] = "0.5.2"
    catalog_version: str
    status: Literal["proposed", "human-reviewed"] = "proposed"
    profile_order: list[str] = Field(min_length=1)
    profiles: dict[str, SystemDraftProfile]
    default_memory_profile: Literal["kv-cache-cpu-offload-v1"] = "kv-cache-cpu-offload-v1"
    cross_profile_ranking_allowed: Literal[False] = False

    @model_validator(mode="after")
    def order_keys_and_memory_profiles_are_complete(self) -> SystemDraftProfileCatalog:
        if len(self.profile_order) != len(set(self.profile_order)):
            raise ValueError("system profile order must be unique")
        if set(self.profile_order) != set(self.profiles):
            raise ValueError("system profile order must exactly cover catalog keys")
        if any(key != profile.profile_id for key, profile in self.profiles.items()):
            raise ValueError("system profile map keys must match profile ids")
        abstract = [
            profile
            for profile in self.profiles.values()
            if profile.profile_id == "memory-tiering-offload-runtime-v1"
        ]
        if len(abstract) != 1 or abstract[0].sealable:
            raise ValueError("catalog requires one unsealable memory-tier parent")
        concrete_objects = {
            profile.object_kind
            for profile in self.profiles.values()
            if profile.domain == "memory-tiering" and profile.sealable
        }
        if concrete_objects != {
            "kv-cache",
            "weight",
            "training-state",
            "activation",
            "checkpoint-staging",
        }:
            raise ValueError("catalog requires all five concrete memory object profiles")
        return self
