from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.candidates import (
    CandidateBackend,
    CandidatePhase,
    DefaultCandidateResolution,
    OperatorFamily,
)
from infraswe.policy import (
    DEFAULT_EVALUATION_ENGINE,
    DEFAULT_EVALUATION_SCOPE,
    DEFAULT_SEAL_ENABLED,
)

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
DraftState = Literal[
    "D0-created",
    "D1-target-bound",
    "D2-precedents-retrieved",
    "D3-contract-proposed",
    "D4-human-reviewed",
    "D5-fast-loop",
    "D6-sealed",
    "D7-official-evaluation",
    "D8-decided",
    "D9-archived",
]
DefaultDraftProject = Literal[
    "vllm",
    "sglang",
    "flash-attention",
    "flashinfer",
    "cutlass-cute",
    "liger-kernel",
    "deepgemm",
    "megatron-core",
    "torchtitan",
    "verl",
]
DefaultContractKind = Literal[
    "api-abi",
    "lifecycle",
    "build-test-matrix",
    "dependency-policy",
    "fallback-policy",
    "deployment-workload-portfolio",
    "performance-acceptance-targets",
    "maintainability-probes",
]

DRAFT_STATE_ORDER = {
    state: index
    for index, state in enumerate(
        (
            "D0-created",
            "D1-target-bound",
            "D2-precedents-retrieved",
            "D3-contract-proposed",
            "D4-human-reviewed",
            "D5-fast-loop",
            "D6-sealed",
            "D7-official-evaluation",
            "D8-decided",
            "D9-archived",
        )
    )
}


class DraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionedArtifactRef(DraftModel):
    id: str
    sha256: Digest
    path: str | None = None


class ProjectOwnership(DraftModel):
    maintainers: list[str] = Field(min_length=1)
    review_required: int = Field(default=1, ge=1)
    last_reviewed_at: datetime | None = None
    update_policy: Literal["pull-request"] = "pull-request"


class EdgeProfileObjective(DraftModel):
    id: str
    status: Literal["active", "planned", "maintenance-only", "experimental", "out-of-scope"]
    target_release: str | None = None
    required_for_release: bool = False


class EdgeEcosystemObjective(DraftModel):
    owner: str
    policy: Literal["roadmap", "release-gate", "maintenance-only", "experimental", "out-of-scope"]
    profiles: list[EdgeProfileObjective] = Field(default_factory=list)


class ProjectObjectives(DraftModel):
    edge_ecosystem: EdgeEcosystemObjective


class TritonPortabilityPolicy(DraftModel):
    enabled: bool = False
    owner: str | None = None
    profile_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enabled_policy_has_owner_and_normalized_weights(self) -> TritonPortabilityPolicy:
        if not self.enabled:
            if self.profile_weights:
                raise ValueError("disabled Triton portability cannot carry profile weights")
            return self
        if not self.owner or not self.profile_weights:
            raise ValueError("enabled Triton portability requires an owner and profile weights")
        if any(weight <= 0 for weight in self.profile_weights.values()):
            raise ValueError("Triton portability weights must be positive")
        if abs(sum(self.profile_weights.values()) - 1.0) > 1e-9:
            raise ValueError("Triton portability profile weights must sum to one")
        return self


class TargetProjectProfile(DraftModel):
    schema_version: Literal["0.5"] = "0.5"
    id: str
    version: str
    status: Literal["proposed", "human-reviewed"]
    repository: str
    supported_revision_policy: str
    ownership: ProjectOwnership
    component_ownership: dict[str, list[str]]
    allowed_integration_points: list[str] = Field(min_length=1)
    api_abi_contract: VersionedArtifactRef
    lifecycle_contract: VersionedArtifactRef
    build_test_matrix: VersionedArtifactRef
    dependency_policy: VersionedArtifactRef
    fallback_policy: VersionedArtifactRef
    deployment_workload_portfolio: VersionedArtifactRef
    performance_acceptance_targets: VersionedArtifactRef
    maintainability_probes: VersionedArtifactRef
    project_objectives: ProjectObjectives
    scoring_template_id: Literal["project-fit-kernel-v0.5", "project-fit-triton-pure-v0.5"]
    triton_portability: TritonPortabilityPolicy = Field(default_factory=TritonPortabilityPolicy)

    @model_validator(mode="after")
    def triton_template_requires_enabled_policy(self) -> TargetProjectProfile:
        if self.status == "human-reviewed" and self.ownership.last_reviewed_at is None:
            raise ValueError("human-reviewed profiles require ownership.last_reviewed_at")
        if (
            self.scoring_template_id == "project-fit-triton-pure-v0.5"
            and not self.triton_portability.enabled
        ):
            raise ValueError("the pure Triton formula requires an enabled portability policy")
        return self


class DraftMetadata(DraftModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    revision: int = Field(ge=1)
    state: DraftState
    created_by: str


class DraftTarget(DraftModel):
    mode: Literal["catalog", "repository"]
    repository: str
    revision: Digest
    project_profile_sha256: Digest
    catalog_profile: str | None = None

    @model_validator(mode="after")
    def catalog_mode_has_profile(self) -> DraftTarget:
        if self.mode == "catalog" and not self.catalog_profile:
            raise ValueError("catalog targets require catalog_profile")
        if self.mode == "repository" and self.catalog_profile is not None:
            raise ValueError("repository targets cannot declare catalog_profile")
        return self


class DraftCandidate(DraftModel):
    kind: Literal["git-diff", "source-tree", "package", "generated"]
    revision: Digest
    intent: Literal["replace", "add-fastpath", "repair", "port", "integrate"]
    implementation_kind: Literal[
        "cuda-native", "hip-native", "cann-native", "triton-pure", "framework"
    ]
    entrypoints: list[str] = Field(min_length=1)
    operator_family: OperatorFamily = "generic"
    phase: CandidatePhase = "generic"
    backend: CandidateBackend = "generic"
    primary_host_candidate: str | None = None


class DraftBaseline(DraftModel):
    mode: Literal["target-head", "pinned-reference"]
    revision: Digest
    advisory_reference_profile: str | None = None


class DraftDeployment(DraftModel):
    workload_portfolio: VersionedArtifactRef
    required_cells: list[str] = Field(min_length=1)
    optional_cells: list[str] = Field(default_factory=list)
    request_or_step_protocol: VersionedArtifactRef

    @model_validator(mode="after")
    def required_and_optional_cells_are_disjoint(self) -> DraftDeployment:
        if set(self.required_cells) & set(self.optional_cells):
            raise ValueError("required and optional deployment cells must be disjoint")
        return self


class DraftRetrieval(DraftModel):
    enabled: bool = True
    corpus_cutoff: datetime
    sources: list[
        Literal[
            "target-code",
            "merged-prs",
            "rejected-prs",
            "reverted-prs",
            "review-comments",
            "issues",
            "ci-failures",
            "release-notes",
        ]
    ] = Field(min_length=1)
    precedent_set_sha256: Digest | None = None


class DraftAcceptanceContract(DraftModel):
    status: Literal["proposed", "human-reviewed", "sealed"]
    path: str
    sha256: Digest
    probe_set_sha256: Digest
    hidden_probe_policy_sha256: Digest
    human_review_sha256: Digest | None = None

    @model_validator(mode="after")
    def review_digest_matches_contract_status(self) -> DraftAcceptanceContract:
        if self.status == "proposed" and self.human_review_sha256 is not None:
            raise ValueError("proposed acceptance contracts cannot claim a human review")
        if self.status in {"human-reviewed", "sealed"} and self.human_review_sha256 is None:
            raise ValueError("reviewed acceptance contracts require human_review_sha256")
        return self


class DraftObjectiveBinding(DraftModel):
    edge_ecosystem_policy: Literal[
        "roadmap", "release-gate", "maintenance-only", "experimental", "out-of-scope"
    ]
    profile_set_sha256: Digest


class DraftPrecompilePolicy(DraftModel):
    """Draft-local switch for keeping unavoidable compilation out of timed cases."""

    mode: Literal["off", "auto"] = "auto"
    trigger: Literal["when-compilation-required"] = "when-compilation-required"
    cache_policy: Literal["content-addressed-evidence-identity"] = (
        "content-addressed-evidence-identity"
    )
    cache_miss_action: Literal["precompile-before-timed-cases"] = "precompile-before-timed-cases"
    timing_phases: list[Literal["precompile", "cold-start", "steady-state"]] = Field(
        default_factory=lambda: ["precompile", "cold-start", "steady-state"]
    )
    steady_state_compile_allowed: Literal[False] = False

    @model_validator(mode="after")
    def timing_phases_are_complete_and_unique(self) -> DraftPrecompilePolicy:
        expected = {"precompile", "cold-start", "steady-state"}
        if set(self.timing_phases) != expected or len(self.timing_phases) != len(expected):
            raise ValueError(
                "Draft precompile policy requires exactly precompile/cold-start/steady-state"
            )
        return self


class DraftBenchmarkLoop(DraftModel):
    fast_stage_max_official_fraction: float = Field(default=0.05, gt=0, le=1)
    affected_stage_max_official_fraction: float = Field(default=1.0, gt=0, le=1)
    official_replays: int = Field(default=7, ge=5, le=10)
    evaluation_scope: Literal["full", "staged"] = DEFAULT_EVALUATION_SCOPE
    early_exit_on_hard_gate: bool = False
    affected_case_selection: Literal["required"] = "required"
    benchmark_budget_policy_id: str
    evidence_policy_id: str
    precompile: DraftPrecompilePolicy = Field(default_factory=DraftPrecompilePolicy)

    @model_validator(mode="after")
    def fast_budget_does_not_exceed_affected_budget(self) -> DraftBenchmarkLoop:
        if self.fast_stage_max_official_fraction > self.affected_stage_max_official_fraction:
            raise ValueError("fast-stage budget cannot exceed affected-stage budget")
        if self.evaluation_scope == "full" and (
            self.affected_stage_max_official_fraction != 1.0 or self.early_exit_on_hard_gate
        ):
            raise ValueError("full evaluation requires all official cases and disables early exit")
        return self


class DraftScoringPolicy(DraftModel):
    formula_template_id: Literal["project-fit-kernel-v0.5", "project-fit-triton-pure-v0.5"]
    provisional_scoring_allowed: bool = True
    evaluation_engine: Literal["infraswe", "external"] = DEFAULT_EVALUATION_ENGINE
    seal_by_default: bool = DEFAULT_SEAL_ENABLED
    official_scoring_requires_seal: Literal[True] = True
    project_season: str


class DraftSpec(DraftModel):
    schema_version: Literal["0.5"] = "0.5"
    draft: DraftMetadata
    target: DraftTarget | None = None
    candidate: DraftCandidate
    default_candidates: DefaultCandidateResolution | None = None
    baseline: DraftBaseline | None = None
    deployment: DraftDeployment | None = None
    retrieval: DraftRetrieval | None = None
    acceptance_contract: DraftAcceptanceContract | None = None
    project_objectives: DraftObjectiveBinding | None = None
    benchmark_loop: DraftBenchmarkLoop | None = None
    scoring: DraftScoringPolicy | None = None

    @model_validator(mode="after")
    def fields_match_state(self) -> DraftSpec:
        state_index = DRAFT_STATE_ORDER[self.draft.state]
        required_by_state = {
            1: {"target": self.target, "baseline": self.baseline},
            2: {"retrieval": self.retrieval},
            3: {
                "deployment": self.deployment,
                "acceptance_contract": self.acceptance_contract,
                "project_objectives": self.project_objectives,
                "benchmark_loop": self.benchmark_loop,
                "scoring": self.scoring,
            },
        }
        missing = [
            name
            for minimum_state, values in required_by_state.items()
            if state_index >= minimum_state
            for name, value in values.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{self.draft.state} is missing required fields: " + ", ".join(missing)
            )
        if state_index >= 2 and self.retrieval and not self.retrieval.precedent_set_sha256:
            raise ValueError("retrieved Draft states require precedent_set_sha256")
        if (
            state_index >= 4
            and self.acceptance_contract
            and self.acceptance_contract.status not in {"human-reviewed", "sealed"}
        ):
            raise ValueError("D4+ requires a human-reviewed acceptance contract")
        if (
            state_index >= 6
            and self.acceptance_contract
            and self.acceptance_contract.status != "sealed"
        ):
            raise ValueError("D6+ requires a sealed acceptance contract")
        if (
            self.scoring
            and self.candidate.implementation_kind != "triton-pure"
            and self.scoring.formula_template_id == "project-fit-triton-pure-v0.5"
        ):
            raise ValueError("the pure Triton formula requires a triton-pure candidate")
        return self


class HumanReviewRecord(DraftModel):
    schema_version: Literal["0.5"] = "0.5"
    reviewer: str
    authority: Literal["project-maintainer"] = "project-maintainer"
    decision: Literal["approve", "request-changes", "reject"]
    reviewed_at: datetime
    target_profile_sha256: Digest
    acceptance_contract_sha256: Digest
    probe_set_sha256: Digest
    workload_portfolio_sha256: Digest
    formula_template_id: str
    notes_sha256: Digest


class SealMaterial(DraftModel):
    target_profile_sha256: Digest
    target_repository_sha256: Digest
    candidate_sha256: Digest
    precedent_set_sha256: Digest
    acceptance_contract_sha256: Digest
    probe_set_sha256: Digest
    workload_portfolio_sha256: Digest
    performance_target_sha256: Digest
    required_deployment_cell_set_sha256: Digest
    formula_template_id: str
    benchmark_budget_policy_id: str
    evidence_policy_id: str
    project_season: str


class SealedDraft(FrozenDraftModel):
    schema_version: Literal["0.5"] = "0.5"
    draft_id: str
    draft_revision: int = Field(ge=1)
    state: Literal["D6-sealed"] = "D6-sealed"
    sealed_at: datetime
    sealed_by: str
    human_review_sha256: Digest
    material: SealMaterial
    seal_sha256: Digest


class ProjectComparisonCell(FrozenDraftModel):
    schema_version: Literal["0.5"] = "0.5"
    target_project_profile_sha256: Digest
    target_repository_or_baseline_sha256: Digest
    change_intent: str
    semantic_contract_sha256: Digest
    acceptance_contract_sha256: Digest
    probe_set_sha256: Digest
    workload_portfolio_sha256: Digest
    performance_target_sha256: Digest
    required_deployment_cell_set_sha256: Digest
    formula_template_id: str
    evidence_policy_id: str
    project_season: str
    cross_project_ranking_allowed: Literal[False] = False


class AffectedCase(DraftModel):
    case_id: str
    symbols: list[str] = Field(default_factory=list)
    categories: list[
        Literal[
            "positive",
            "negative-control",
            "fallback-or-unsupported",
            "hidden-adjacent",
            "build-import-load",
        ]
    ] = Field(min_length=1)
    workload_ids: list[str] = Field(default_factory=list)
    required: bool = False


class AffectedCaseDecision(DraftModel):
    case_id: str
    selected: bool
    reasons: list[str] = Field(min_length=1)


class AffectedCasePlan(DraftModel):
    schema_version: Literal["0.5"] = "0.5"
    changed_symbols: list[str] = Field(min_length=1)
    decisions: list[AffectedCaseDecision] = Field(min_length=1)
    coverage_confidence: Literal["low", "medium", "high"]
    false_negative_audit_required: Literal[True] = True
    required_categories_present: bool
    failure_codes: list[str] = Field(default_factory=list)


class EvidenceCacheIdentity(DraftModel):
    target_repository_sha256: Digest
    project_profile_sha256: Digest
    candidate_sha256: Digest
    compiler: str
    runtime: str
    driver: str
    hardware_cell: str
    workload_case: str
    probe_version: str
    collector_version: str
    environment_digest: Digest


class DraftRevisionEvent(DraftModel):
    schema_version: Literal["0.5"] = "0.5"
    draft_id: str
    from_revision: int = Field(ge=1)
    to_revision: int = Field(ge=1)
    loop_kind: Literal["candidate", "contract"]
    old_candidate_sha256: Digest
    new_candidate_sha256: Digest
    old_contract_sha256: Digest
    new_contract_sha256: Digest
    actor_role: Literal["draft-owner", "project-maintainer"]
    reason: str

    @model_validator(mode="after")
    def revision_semantics_match_loop_kind(self) -> DraftRevisionEvent:
        if self.loop_kind == "candidate":
            if self.to_revision != self.from_revision:
                raise ValueError("candidate loops stay within one Draft revision")
            if self.old_contract_sha256 != self.new_contract_sha256:
                raise ValueError("candidate loops cannot change the acceptance contract")
            if self.old_candidate_sha256 == self.new_candidate_sha256:
                raise ValueError("candidate loops require a new candidate digest")
        else:
            if self.actor_role != "project-maintainer":
                raise ValueError("contract loops require project-maintainer authority")
            if self.to_revision != self.from_revision + 1:
                raise ValueError("contract loops create exactly one new Draft revision")
            if self.old_contract_sha256 == self.new_contract_sha256:
                raise ValueError("contract loops require a new contract digest")
        return self


class DefaultProjectContractArtifact(FrozenDraftModel):
    """A pinned, machine-proposed extraction from an upstream project snapshot."""

    schema_version: Literal["0.5"] = "0.5"
    project: DefaultDraftProject
    catalog_profile: str
    artifact_kind: DefaultContractKind
    source_repository: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    source_urls: list[str] = Field(min_length=1)
    requirements: list[str] = Field(min_length=1)
    probes: list[str] = Field(min_length=1)
    extraction_status: Literal["machine-proposed"] = "machine-proposed"


class DefaultDraftCatalogEntry(FrozenDraftModel):
    project: DefaultDraftProject
    aliases: list[str] = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    profile: TargetProjectProfile
    artifacts: dict[DefaultContractKind, DefaultProjectContractArtifact]

    @model_validator(mode="after")
    def artifact_set_is_complete(self) -> DefaultDraftCatalogEntry:
        expected = {
            "api-abi",
            "lifecycle",
            "build-test-matrix",
            "dependency-policy",
            "fallback-policy",
            "deployment-workload-portfolio",
            "performance-acceptance-targets",
            "maintainability-probes",
        }
        if set(self.artifacts) != expected:
            raise ValueError("default catalog entries require the complete contract artifact set")
        if any(artifact.project != self.project for artifact in self.artifacts.values()):
            raise ValueError("default catalog artifacts must belong to their catalog project")
        return self


class DefaultDraftCatalog(FrozenDraftModel):
    schema_version: Literal["0.5"] = "0.5"
    catalog_version: str
    status: Literal["proposed", "human-reviewed"]
    source_capture_date: str
    default_order: list[DefaultDraftProject]
    entries: dict[DefaultDraftProject, DefaultDraftCatalogEntry]

    @model_validator(mode="after")
    def exact_default_projects_are_present(self) -> DefaultDraftCatalog:
        expected = {
            "vllm",
            "sglang",
            "flash-attention",
            "flashinfer",
            "cutlass-cute",
            "liger-kernel",
            "deepgemm",
            "megatron-core",
            "torchtitan",
            "verl",
        }
        if set(self.entries) != expected or set(self.default_order) != expected:
            raise ValueError("the v0.5 default catalog requires the frozen ten-project set")
        if len(self.default_order) != len(expected):
            raise ValueError("default catalog order cannot contain duplicates")
        return self


class RemoteGitDraftLocation(DraftModel):
    repository: str
    revision: str = "HEAD"
    path: str

    @model_validator(mode="after")
    def path_is_repository_relative(self) -> RemoteGitDraftLocation:
        parts = self.path.replace("\\", "/").split("/")
        if self.path.startswith(("/", "\\")) or ".." in parts:
            raise ValueError("remote Draft path must be repository-relative")
        if not self.path or self.path.endswith("/"):
            raise ValueError("remote Draft path must name a file")
        return self


class DraftSourceResolution(FrozenDraftModel):
    schema_version: Literal["0.5"] = "0.5"
    source_kind: Literal["local", "remote-git", "default-catalog"]
    source: str
    draft: DraftSpec
    bundled_profile: TargetProjectProfile | None = None
    selected_default_project: DefaultDraftProject | None = None
    selection_reason: str
    audit_flags: list[str] = Field(default_factory=list)
