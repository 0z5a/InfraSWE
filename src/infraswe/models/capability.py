from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

ProofLevel = Literal[
    "CP0-declared",
    "CP1-inventory",
    "CP2-compile",
    "CP3-runtime",
    "CP4-behavior",
    "CP5-operational",
]
CapabilityStatus = Literal["supported", "unsupported", "unknown", "contradictory"]
RequirementMode = Literal[
    "required-present",
    "required-usable",
    "required-native",
    "required-absent",
    "forbidden-use",
    "preferred",
    "optional-observe",
    "explicitly-not-assumed",
]
TrialPhase = Literal["agent", "collect", "build", "verify", "measure", "profile", "judge"]

PROOF_LEVEL_ORDER = {
    "CP0-declared": 0,
    "CP1-inventory": 1,
    "CP2-compile": 2,
    "CP3-runtime": 3,
    "CP4-behavior": 4,
    "CP5-operational": 5,
}


class CapabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CapabilityParameterDefinition(CapabilityModel):
    type: Literal["enum-set", "integer-range", "number-range", "boolean", "string"]
    allowed_values: list[str] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def definition_is_coherent(self) -> CapabilityParameterDefinition:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("capability parameter minimum exceeds maximum")
        if self.type == "enum-set" and not self.allowed_values:
            raise ValueError("enum-set capability parameter requires allowed values")
        return self


class CapabilityProofPolicy(CapabilityModel):
    minimum_default: ProofLevel
    native_claim_minimum: Literal["CP4-behavior", "CP5-operational"] = "CP4-behavior"


class CapabilityRelationships(CapabilityModel):
    implies: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class CapabilityDefinition(CapabilityModel):
    capability_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    version: int = Field(ge=1)
    domain: str
    kind: Literal["semantic", "mechanism", "quantitative", "policy", "observability"]
    description: str = Field(min_length=12)
    parameters: dict[str, CapabilityParameterDefinition] = Field(default_factory=dict)
    proof_policy: CapabilityProofPolicy
    probes: dict[str, str] = Field(default_factory=dict)
    relationships: CapabilityRelationships = Field(default_factory=CapabilityRelationships)
    invalidation_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def relationships_do_not_self_reference(self) -> CapabilityDefinition:
        related = [
            *self.relationships.implies,
            *self.relationships.conflicts,
            *self.relationships.aliases,
        ]
        if self.capability_id in related:
            raise ValueError("capability relationship cannot reference itself")
        return self


class CapabilityRegistry(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    registry_id: str
    revision: int = Field(ge=1)
    definitions: list[CapabilityDefinition] = Field(min_length=1)
    registry_sha256: Digest

    @model_validator(mode="after")
    def definitions_are_unique(self) -> CapabilityRegistry:
        keys = [(item.capability_id, item.version) for item in self.definitions]
        if len(keys) != len(set(keys)):
            raise ValueError("capability registry definitions must be unique")
        aliases = [alias for item in self.definitions for alias in item.relationships.aliases]
        if len(aliases) != len(set(aliases)):
            raise ValueError("capability aliases must resolve uniquely")
        return self


class CandidateFallback(CapabilityModel):
    trigger: str
    behavior: Literal["explicit-unsupported", "explicit-error", "approved-fallback"]
    capability_id: str | None = None


class CandidateCapabilityDeclaration(CapabilityModel):
    requires: list[str] = Field(default_factory=list)
    uses: list[str] = Field(default_factory=list)
    does_not_use: list[str] = Field(default_factory=list)
    fallbacks: list[CandidateFallback] = Field(default_factory=list)
    declaration_sha256: Digest

    @model_validator(mode="after")
    def declaration_has_no_contradiction(self) -> CandidateCapabilityDeclaration:
        if set(self.uses) & set(self.does_not_use):
            raise ValueError("candidate cannot both use and disclaim a capability")
        return self


class CapabilityParameterConstraint(CapabilityModel):
    eq: Any | None = None
    gte: float | None = None
    lte: float | None = None
    set_contains: list[Any] = Field(default_factory=list)
    one_of: list[Any] = Field(default_factory=list)
    count_gte: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def constraint_is_nonempty_and_bounded(self) -> CapabilityParameterConstraint:
        values = (
            self.eq is not None,
            self.gte is not None,
            self.lte is not None,
            bool(self.set_contains),
            bool(self.one_of),
            self.count_gte is not None,
        )
        if not any(values):
            raise ValueError("capability parameter constraint cannot be empty")
        if self.gte is not None and self.lte is not None and self.gte > self.lte:
            raise ValueError("capability parameter constraint lower bound exceeds upper")
        return self


class CapabilityRequirement(CapabilityModel):
    capability_id: str
    mode: RequirementMode
    min_proof: ProofLevel = "CP1-inventory"
    definition_version: int | None = Field(default=None, ge=1)
    parameters: dict[str, CapabilityParameterConstraint] = Field(default_factory=dict)
    realization: str | None = None

    @model_validator(mode="after")
    def native_claim_requires_behavior_proof(self) -> CapabilityRequirement:
        if self.mode == "required-native" and PROOF_LEVEL_ORDER[self.min_proof] < 4:
            raise ValueError("required-native capability requires CP4 or CP5 proof")
        return self


class CapabilityExpression(CapabilityModel):
    operation: Literal["capability", "all_of", "any_of", "one_of", "not", "implies", "conditional"]
    requirement: CapabilityRequirement | None = None
    children: list[CapabilityExpression] = Field(default_factory=list)
    selected_variant_is: str | None = None

    @model_validator(mode="after")
    def expression_shape_is_valid(self) -> CapabilityExpression:
        if self.operation == "capability":
            if self.requirement is None or self.children or self.selected_variant_is is not None:
                raise ValueError("capability expression requires exactly one requirement")
        elif self.operation in {"all_of", "any_of", "one_of"}:
            if self.requirement is not None or len(self.children) < 1:
                raise ValueError("logical expression requires children only")
        elif self.operation == "not":
            if self.requirement is not None or len(self.children) != 1:
                raise ValueError("not expression requires exactly one child")
        elif self.operation == "implies":
            if self.requirement is not None or len(self.children) != 2:
                raise ValueError("implies expression requires antecedent and consequent")
        elif (
            self.requirement is not None or len(self.children) != 1 or not self.selected_variant_is
        ):
            raise ValueError("conditional expression requires a variant and one child")
        return self


class PhaseCapabilityContract(CapabilityModel):
    requirements: CapabilityExpression
    optional_observe: list[str] = Field(default_factory=list)
    forbidden_use: list[str] = Field(default_factory=list)
    network_policy_id: str
    toolchain_profile_id: str | None = None
    isolation_requirements: list[str] = Field(default_factory=list)


class CapabilityVariant(CapabilityModel):
    variant_id: str
    priority: int
    phases: dict[TrialPhase, PhaseCapabilityContract]
    resource_envelope_sha256: Digest
    topology_contract_sha256: Digest

    @model_validator(mode="after")
    def variant_has_execution_authority(self) -> CapabilityVariant:
        if not {"verify", "measure"} & set(self.phases):
            raise ValueError("capability variant requires verify or measure phase")
        return self


class CapabilityContract(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    policy_id: str
    registry_sha256: Digest
    variants: list[CapabilityVariant] = Field(min_length=1)
    allowed_candidate_capabilities: list[str] = Field(default_factory=list)
    closed_world_required: Literal[True] = True
    contract_sha256: Digest

    @model_validator(mode="after")
    def variants_are_unique(self) -> CapabilityContract:
        identifiers = [item.variant_id for item in self.variants]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("capability variants must be unique")
        return self


class CapabilityProbeIdentity(CapabilityModel):
    probe_id: str
    implementation_sha256: Digest
    image_sha256: Digest
    toolchain_sha256: Digest


class CapabilityAttestation(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    capability_id: str
    capability_definition_version: int = Field(ge=1)
    runner_snapshot_sha256: Digest
    status: CapabilityStatus
    proof_level: ProofLevel | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    probe: CapabilityProbeIdentity
    evidence_refs: list[str] = Field(default_factory=list)
    observed_at: datetime
    expires_at: datetime | None = None
    origin_trust: Literal["T4_INFRA_ATTESTED"] = "T4_INFRA_ATTESTED"
    attestation_sha256: Digest

    @model_validator(mode="after")
    def attestation_is_authoritative_and_time_bounded(self) -> CapabilityAttestation:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("capability attestation timestamp must be timezone-aware")
        if self.expires_at is not None:
            if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
                raise ValueError("capability attestation expiry must be timezone-aware")
            if self.expires_at <= self.observed_at:
                raise ValueError("capability attestation expiry must follow observation")
        if self.status == "unknown" and self.proof_level is not None:
            raise ValueError("unknown capability cannot claim a proof level")
        if self.status != "unknown" and self.proof_level is None:
            raise ValueError("decided capability attestation requires a proof level")
        if self.status in {"supported", "unsupported", "contradictory"} and not self.evidence_refs:
            raise ValueError("decided capability attestation requires evidence refs")
        return self


class RunnerHostIdentity(CapabilityModel):
    architecture: str
    sockets: int = Field(ge=1)
    numa_nodes: int = Field(ge=1)
    cpu_model: str


class RunnerAcceleratorIdentity(CapabilityModel):
    vendor: str
    model: str
    count: int = Field(ge=0)
    memory_bytes_each: int = Field(ge=0)


class RunnerManifest(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    runner_id: str
    revision: int = Field(ge=1)
    owner: str
    host: RunnerHostIdentity
    accelerators: list[RunnerAcceleratorIdentity] = Field(default_factory=list)
    network: dict[str, Any] = Field(default_factory=dict)
    software_profile: dict[str, Any]
    declared_capabilities: list[str] = Field(default_factory=list)
    attestation_policy_id: str
    manifest_sha256: Digest


class RunnerDeviceSnapshot(CapabilityModel):
    device_id: str
    kind: Literal["accelerator", "nic", "storage"]
    pci_bdf: str | None = None
    numa_node: int | None = Field(default=None, ge=0)
    free_memory_bytes: int | None = Field(default=None, ge=0)
    partition_mode: str = "none"
    persistence_mode: bool | None = None


class ResourceAvailability(CapabilityModel):
    total: float = Field(ge=0)
    available: float = Field(ge=0)
    allocatable: float = Field(ge=0)
    exclusive_available: bool = True

    @model_validator(mode="after")
    def amounts_are_coherent(self) -> ResourceAvailability:
        if self.available > self.total or self.allocatable > self.available:
            raise ValueError("resource allocatable <= available <= total is required")
        return self


class RunnerSnapshot(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    runner_manifest_sha256: Digest
    captured_at: datetime
    expires_at: datetime
    availability: Literal["available", "busy", "draining", "quarantined"]
    devices: list[RunnerDeviceSnapshot] = Field(default_factory=list)
    resources: dict[str, ResourceAvailability] = Field(default_factory=dict)
    dynamic: dict[str, Any] = Field(default_factory=dict)
    topology_sha256: Digest
    origin_trust: Literal["T4_INFRA_ATTESTED"] = "T4_INFRA_ATTESTED"
    snapshot_sha256: Digest

    @model_validator(mode="after")
    def snapshot_window_is_aware(self) -> RunnerSnapshot:
        for timestamp in (self.captured_at, self.expires_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("runner snapshot timestamps must be timezone-aware")
        if self.expires_at <= self.captured_at:
            raise ValueError("runner snapshot expiry must follow capture")
        identifiers = [item.device_id for item in self.devices]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("runner snapshot device ids must be unique")
        return self


class ResourceLimit(CapabilityModel):
    unit: Literal["count", "bytes", "seconds", "watts", "bps", "iops"]
    minimum_required: float = Field(default=0, ge=0)
    reserved: float = Field(default=0, ge=0)
    candidate_limit: float | None = Field(default=None, ge=0)
    measurement_reserve: float = Field(default=0, ge=0)
    exclusive: bool = False

    @model_validator(mode="after")
    def resource_partition_is_coherent(self) -> ResourceLimit:
        if self.reserved < self.minimum_required:
            raise ValueError("resource reservation cannot be below minimum required")
        if self.candidate_limit is not None and (
            self.candidate_limit + self.measurement_reserve > self.reserved
        ):
            raise ValueError("Candidate limit plus measurement reserve exceeds reservation")
        return self


class ResourcePhaseEnvelope(CapabilityModel):
    resources: dict[str, ResourceLimit] = Field(min_length=1)
    wall_time_s: int = Field(gt=0)
    official_repetitions: int = Field(default=1, ge=1)


class ResourceEnvelope(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    policy_id: str
    phases: dict[TrialPhase, ResourcePhaseEnvelope]
    overprovision_isolation_required: bool = True
    envelope_sha256: Digest


class ResourcePhaseResolution(CapabilityModel):
    phase: TrialPhase
    status: Literal["feasible", "capacity-unavailable", "unschedulable", "unresolved"]
    missing_resources: list[str] = Field(default_factory=list)
    busy_resources: list[str] = Field(default_factory=list)


class ResourceFeasibilityResult(CapabilityModel):
    status: Literal["feasible", "capacity-unavailable", "unschedulable", "unresolved"]
    phases: list[ResourcePhaseResolution]
    request_sha256: Digest


class TopologyVertex(CapabilityModel):
    vertex_id: str
    kind: Literal[
        "host",
        "socket",
        "numa-node",
        "cpu-set",
        "accelerator",
        "nic",
        "switch",
        "memory-tier",
        "storage",
        "process-role",
    ]
    role: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TopologyEdge(CapabilityModel):
    source: str
    target: str
    kind: Literal[
        "attached-to",
        "same-numa",
        "peer-accessible",
        "nvlink-connected",
        "pcie-path",
        "rdma-reachable",
        "shares-root-complex",
        "shares-copy-engine",
        "shares-memory-controller",
        "mounted-from",
    ]
    attributes: dict[str, Any] = Field(default_factory=dict)


class TopologyGraph(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    vertices: list[TopologyVertex] = Field(min_length=1)
    edges: list[TopologyEdge] = Field(default_factory=list)
    graph_sha256: Digest

    @model_validator(mode="after")
    def graph_is_well_formed(self) -> TopologyGraph:
        identifiers = [item.vertex_id for item in self.vertices]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("topology vertex ids must be unique")
        known = set(identifiers)
        if any(item.source not in known or item.target not in known for item in self.edges):
            raise ValueError("topology edges must reference known vertices")
        return self


class TopologyVertexRequirement(CapabilityModel):
    role: str
    kind: str
    count: int = Field(ge=1)
    attribute_equals: dict[str, Any] = Field(default_factory=dict)


class TopologyRelationRequirement(CapabilityModel):
    relation_id: str
    pattern: Literal[
        "same-node",
        "same-socket",
        "same-numa",
        "same-root-complex",
        "all-pairs",
        "ring",
        "mesh",
        "one-nic-per-k-gpu",
        "cross-node",
        "anti-affinity",
    ]
    source_role: str
    target_role: str | None = None
    via_edge_kinds: list[str] = Field(default_factory=list)
    peer_access_required: bool = False
    maximum_distance: float | None = Field(default=None, ge=0)
    k: int | None = Field(default=None, ge=1)


class TopologyContract(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    contract_id: str
    vertices: list[TopologyVertexRequirement] = Field(min_length=1)
    relations: list[TopologyRelationRequirement] = Field(default_factory=list)
    minimum_proof: Literal["CP3-runtime", "CP4-behavior", "CP5-operational"]
    contract_sha256: Digest


class TopologyMatchResult(CapabilityModel):
    status: Literal["satisfied", "unsatisfied", "unresolved", "probe-defect"]
    graph_sha256: Digest
    matched_relations: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)


class BenchmarkCellPolicy(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    policy_id: str
    comparison_included_fields: list[str] = Field(min_length=1)
    comparison_excluded_fields: list[str] = Field(default_factory=list)
    cross_cell_raw_performance_comparison: Literal["forbidden"] = "forbidden"
    post_result_field_exclusion: Literal["forbidden"] = "forbidden"
    policy_sha256: Digest


class BenchmarkCellManifest(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    cell_id: str
    policy_sha256: Digest
    task: dict[str, Any]
    runner: dict[str, Any]
    hardware: dict[str, Any]
    software: dict[str, Any]
    execution: dict[str, Any]
    benchmark: dict[str, Any]
    full_environment_digest: Digest
    comparison_cell_digest: Digest
    cell_sha256: Digest


class RunnerSelectionPolicy(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    policy_id: str
    variant_order: list[str] = Field(min_length=1)
    runner_order: list[str] = Field(min_length=1)
    probe_budget: int = Field(default=0, ge=0)
    post_result_reselection: Literal["forbidden"] = "forbidden"
    historical_candidate_performance_input: Literal["forbidden"] = "forbidden"
    policy_sha256: Digest


class CapabilityRequirementResolution(CapabilityModel):
    phase: TrialPhase
    capability_id: str
    mode: RequirementMode
    status: Literal["satisfied", "unsatisfied", "unresolved", "not-applicable"]
    proof_level: ProofLevel | None = None
    attestation_sha256: Digest | None = None
    failure_code: str | None = None


class ExcludedRunner(CapabilityModel):
    runner_id: str
    variant_id: str
    reason_code: str
    capability_id: str | None = None
    observed_status: str | None = None
    proof_level: ProofLevel | None = None


class CapabilityResolution(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    resolution_id: str
    task_seal_sha256: Digest
    candidate_sha256: Digest
    registry_sha256: Digest
    status: Literal[
        "eligible",
        "unschedulable",
        "capacity-unavailable",
        "unresolved",
        "candidate-declaration-ineligible",
        "runner-contradiction",
    ]
    selected_variant_id: str | None = None
    selected_runner_manifest_sha256: Digest | None = None
    selected_runner_snapshot_sha256: Digest | None = None
    requirements: list[CapabilityRequirementResolution] = Field(default_factory=list)
    topology: TopologyMatchResult | None = None
    resources: ResourceFeasibilityResult | None = None
    required_probes: list[str] = Field(default_factory=list)
    excluded_runners: list[ExcludedRunner] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    policy_id: str
    resolution_sha256: Digest

    @model_validator(mode="after")
    def eligible_resolution_is_complete(self) -> CapabilityResolution:
        if self.status == "eligible" and not all(
            (
                self.selected_variant_id,
                self.selected_runner_manifest_sha256,
                self.selected_runner_snapshot_sha256,
                self.topology is not None and self.topology.status == "satisfied",
                self.resources is not None and self.resources.status == "feasible",
            )
        ):
            raise ValueError(
                "eligible capability resolution must bind runner, cell inputs, and policy"
            )
        if self.status != "eligible" and self.selected_runner_snapshot_sha256 is not None:
            raise ValueError("non-eligible resolution cannot select a runner")
        return self


class ResourceLease(CapabilityModel):
    schema_version: Literal["0.1"] = "0.1"
    lease_id: str
    resolution_sha256: Digest
    status: Literal["active", "released", "broken", "expired"]
    allocations: dict[str, Any]
    isolation: dict[str, Any]
    acquired_at: datetime
    expires_at: datetime
    heartbeat_interval_s: int = Field(ge=1)
    pre_lease_snapshot_sha256: Digest
    post_lease_snapshot_sha256: Digest
    failure_codes: list[str] = Field(default_factory=list)
    lease_sha256: Digest

    @model_validator(mode="after")
    def lease_window_is_aware(self) -> ResourceLease:
        for timestamp in (self.acquired_at, self.expires_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("resource lease timestamps must be timezone-aware")
        if self.expires_at <= self.acquired_at:
            raise ValueError("resource lease expiry must follow acquisition")
        if self.status == "active" and self.failure_codes:
            raise ValueError("active resource lease cannot carry failure codes")
        if self.status == "broken" and not self.failure_codes:
            raise ValueError("broken resource lease requires failure codes")
        return self


class ResourceUsageObservation(CapabilityModel):
    phase: TrialPhase
    resource_id: str
    candidate_peak: float = Field(ge=0)
    external_interference: bool = False
    verifier_peak: float = Field(default=0, ge=0)


class ResourceUsageVerdict(CapabilityModel):
    status: Literal["PASS", "VALID_FAIL", "INFRA_INVALID", "BENCHMARK_DEFECT"]
    owner: Literal["candidate", "infrastructure", "benchmark", "none"]
    failure_codes: list[str] = Field(default_factory=list)


class EnvironmentSentinelResult(CapabilityModel):
    phase: Literal["pre-run", "during-run", "post-run"]
    status: Literal["PASS", "INFRA_INVALID", "BENCHMARK_DEFECT"]
    failure_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    result_sha256: Digest


class CandidateCapabilityUseObservation(CapabilityModel):
    capability_id: str
    declared: bool
    forbidden: bool
    native_required: bool
    native_proved: bool
    silent_fallback: bool
    evidence_refs: list[str] = Field(default_factory=list)


class CandidateCapabilityUseVerdict(CapabilityModel):
    status: Literal["PASS", "VALID_FAIL", "UNRESOLVED"]
    failure_codes: list[str] = Field(default_factory=list)
