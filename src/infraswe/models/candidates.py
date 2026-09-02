from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
CandidateRole = Literal["oracle", "peer-impl", "host-project", "workload-source", "coverage-target"]
CandidateTier = Literal["P0", "P1", "P2"]
CandidatePhase = Literal["generic", "training", "inference", "communication"]
CandidateBackend = Literal["generic", "cuda", "rocm", "triton"]
OperatorFamily = Literal[
    "generic",
    "attention-training",
    "attention-inference-prefill",
    "attention-inference-decode",
    "paged-attention",
    "dense-gemm",
    "grouped-moe-gemm",
    "training-fused-ops",
    "quantization",
    "communication-collective",
    "moe-dispatch-combine",
    "gpu-initiated-communication",
]


class CandidateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateSource(CandidateModel):
    kind: Literal["pinned-git", "runtime-builtin", "draft-relative"]
    repository: str | None = None
    revision: GitRevision | None = None

    @model_validator(mode="after")
    def git_sources_are_pinned(self) -> CandidateSource:
        if self.kind == "pinned-git" and (not self.repository or not self.revision):
            raise ValueError("pinned-git candidate sources require repository and revision")
        if self.kind != "pinned-git" and self.revision is not None:
            raise ValueError("only pinned-git sources can carry a Git revision")
        return self


class CandidateBuildPolicy(CandidateModel):
    compilation_mode: Literal[
        "none", "environment-provided", "host-owned", "adapter-aot", "adapter-jit"
    ]
    adapter_id: str | None = None
    selection_action: Literal["metadata-only"] = "metadata-only"
    activation_scope: Literal["explicitly-activated-candidate-only"] = (
        "explicitly-activated-candidate-only"
    )
    registry_load_imports_candidate: Literal[False] = False
    registry_load_compiles_candidate: Literal[False] = False

    @model_validator(mode="after")
    def adapters_are_named_only_when_needed(self) -> CandidateBuildPolicy:
        needs_adapter = self.compilation_mode in {"adapter-aot", "adapter-jit"}
        if needs_adapter != bool(self.adapter_id):
            raise ValueError("adapter compilation modes require exactly one adapter_id")
        return self


class DefaultCandidateDefinition(CandidateModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{1,63}$")
    display_name: str
    roles: list[CandidateRole] = Field(min_length=1)
    tier: CandidateTier
    operator_families: list[OperatorFamily] = Field(min_length=1)
    phases: list[CandidatePhase] = Field(min_length=1)
    backends: list[CandidateBackend] = Field(min_length=1)
    source: CandidateSource
    build: CandidateBuildPolicy
    license_status: Literal["declared", "runtime-provided", "draft-owned"]
    default_eligible: bool = True

    @model_validator(mode="after")
    def role_and_scope_lists_are_unique(self) -> DefaultCandidateDefinition:
        for name, values in {
            "roles": self.roles,
            "operator_families": self.operator_families,
            "phases": self.phases,
            "backends": self.backends,
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(f"candidate {name} cannot contain duplicates")
        return self


class DefaultCandidateRule(CandidateModel):
    id: str
    order: int = Field(ge=1)
    operator_family: OperatorFamily
    phases: list[CandidatePhase] = Field(min_length=1)
    backends: list[CandidateBackend] = Field(min_length=1)
    oracles: list[str] = Field(min_length=1)
    primary_peer_impl: str
    secondary_peer_impls: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(min_length=1)
    workload_sources: list[str] = Field(min_length=1)
    coverage_targets: list[str] = Field(min_length=1)
    required_tests: list[str] = Field(min_length=1)


class DefaultCandidateRegistry(CandidateModel):
    schema_version: Literal["0.5"] = "0.5"
    registry_version: str
    status: Literal["proposed", "human-reviewed"] = "proposed"
    source_capture_date: str
    selection_policy_id: Literal["ordered-role-match-v0.5-r1"] = "ordered-role-match-v0.5-r1"
    learned_model_used: Literal[False] = False
    weighted_score_used: Literal[False] = False
    candidates: dict[str, DefaultCandidateDefinition]
    rules: list[DefaultCandidateRule] = Field(min_length=1)
    fallback_chains: dict[str, list[str]]

    @model_validator(mode="after")
    def registry_references_are_typed_and_complete(self) -> DefaultCandidateRegistry:
        if set(self.candidates) != {item.id for item in self.candidates.values()}:
            raise ValueError("candidate dictionary keys must match definition ids")
        orders = [rule.order for rule in self.rules]
        if len(orders) != len(set(orders)) or orders != sorted(orders):
            raise ValueError("candidate rules require unique ascending order values")
        fields = {
            "oracles": "oracle",
            "secondary_peer_impls": "peer-impl",
            "hosts": "host-project",
            "workload_sources": "workload-source",
            "coverage_targets": "coverage-target",
        }
        for rule in self.rules:
            role_refs = {**fields, "primary_peer_impl": "peer-impl"}
            for field_name, role in role_refs.items():
                value = getattr(rule, field_name)
                ids = [value] if isinstance(value, str) else value
                for candidate_id in ids:
                    candidate = self.candidates.get(candidate_id)
                    if candidate is None:
                        raise ValueError(f"rule {rule.id} references unknown {candidate_id}")
                    if role not in candidate.roles:
                        raise ValueError(
                            f"rule {rule.id} uses {candidate_id} outside its {role} role"
                        )
                    if not candidate.default_eligible:
                        raise ValueError(f"rule {rule.id} cannot select ineligible {candidate_id}")
        if self.rules[-1].operator_family != "generic":
            raise ValueError("the final candidate rule must be the generic fallback")
        return self


class CandidateSelectionRequest(CandidateModel):
    operator_family: OperatorFamily = "generic"
    phase: CandidatePhase = "generic"
    backend: CandidateBackend = "cuda"
    requested_primary_host: str | None = None


class CandidateSelectionTrace(CandidateModel):
    step: int = Field(ge=1)
    question: str
    result: Literal["matched", "not-matched", "selected", "fallback", "preserved"]
    explanation: str


class DefaultCandidateResolution(CandidateModel):
    schema_version: Literal["0.5"] = "0.5"
    registry_sha256: Digest
    request: CandidateSelectionRequest
    matched_rule_id: str
    oracles: list[str] = Field(min_length=1)
    primary_peer_impl: str
    secondary_peer_impls: list[str] = Field(default_factory=list)
    primary_host: str
    secondary_hosts: list[str] = Field(default_factory=list)
    workload_sources: list[str] = Field(min_length=1)
    coverage_targets: list[str] = Field(min_length=1)
    required_tests: list[str] = Field(min_length=1)
    trace: list[CandidateSelectionTrace] = Field(min_length=1)
    selection_side_effects: Literal["metadata-only-no-import-no-build"] = (
        "metadata-only-no-import-no-build"
    )
    compilation_started: Literal[False] = False
    learned_model_used: Literal[False] = False
    weighted_score_used: Literal[False] = False

    def selected_candidate_ids(self) -> set[str]:
        return {
            *self.oracles,
            self.primary_peer_impl,
            *self.secondary_peer_impls,
            self.primary_host,
            *self.secondary_hosts,
            *self.workload_sources,
            *self.coverage_targets,
        }


class CandidateActivationAction(CandidateModel):
    candidate_id: str
    compilation_mode: str
    compilation_required: bool
    cache_hit: bool
    action: Literal[
        "skip-no-compilation",
        "reuse-precompiled-artifact",
        "precompile-before-timed-cases",
        "compile-inline-with-warning",
    ]
    rationale_codes: list[str] = Field(min_length=1)


class CandidateActivationPlan(CandidateModel):
    schema_version: Literal["0.5"] = "0.5"
    registry_sha256: Digest
    resolution_sha256: Digest
    activation_policy: Literal["single-explicit-peer-v0.5"] = "single-explicit-peer-v0.5"
    activated_candidate_ids: list[str] = Field(min_length=1, max_length=1)
    actions: list[CandidateActivationAction] = Field(min_length=1, max_length=1)
    registry_candidate_count: int = Field(ge=1)
    inactive_candidate_count: int = Field(ge=0)
    selection_compile_seconds: float = Field(default=0.0, ge=0, le=0)
    timed_benchmark_started: Literal[False] = False

    @model_validator(mode="after")
    def actions_are_exactly_the_activated_subset(self) -> CandidateActivationPlan:
        action_ids = [item.candidate_id for item in self.actions]
        if action_ids != self.activated_candidate_ids:
            raise ValueError("activation actions must exactly follow activated_candidate_ids")
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("candidate activation cannot contain duplicates")
        if self.inactive_candidate_count != self.registry_candidate_count - len(action_ids):
            raise ValueError("inactive candidate count does not match activation scope")
        return self


class CandidateTimingGate(CandidateModel):
    """Auditable boundary between candidate preparation and benchmark timing."""

    schema_version: Literal["0.5"] = "0.5"
    activation_plan_sha256: Digest
    activated_candidate_ids: list[str] = Field(min_length=1, max_length=1)
    prepared_candidate_ids: list[str] = Field(default_factory=list, max_length=1)
    timed_benchmark_allowed: bool
    timing_eligibility: Literal["blocked", "diagnostic-only", "official"]
    steady_state_compile_allowed: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def readiness_matches_blockers(self) -> CandidateTimingGate:
        if not set(self.prepared_candidate_ids).issubset(self.activated_candidate_ids):
            raise ValueError("only the activated candidate can be marked prepared")
        if self.timed_benchmark_allowed == bool(self.blockers):
            raise ValueError("timed_benchmark_allowed must be true exactly when blockers are empty")
        expected = (
            "blocked" if self.blockers else "diagnostic-only" if self.warnings else "official"
        )
        if self.timing_eligibility != expected:
            raise ValueError("timing_eligibility does not match blockers and warnings")
        return self
