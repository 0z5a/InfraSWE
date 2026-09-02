from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

RetrievalChannel = Literal[
    "exact",
    "graph",
    "failure",
    "lifecycle",
    "lexical",
    "semantic",
    "negative",
]
PrecedentAuthority = Literal[
    "explicit-profile",
    "explicit-test-or-build-rule",
    "current-code-invariant",
    "repeated-accepted-precedent",
    "regression-or-revert-precedent",
    "single-accepted-precedent",
    "single-review-comment",
    "advisory-cross-project",
    "inferred",
    "conflicting",
]
PrecedentKind = Literal[
    "accepted-pattern",
    "rejected-pattern",
    "regression-precedent",
    "migration-precedent",
    "explicit-contract",
    "superseded-precedent",
    "advisory-precedent",
    "conflicting-precedent",
]
RuleTemplate = Literal[
    "required-integration-point",
    "required-interface",
    "forbidden-dependency",
    "required-build-target",
    "required-test-layer",
    "explicit-unsupported",
    "no-silent-fallback",
    "bounded-resource",
    "lifecycle-order",
    "stream-event-happens-before",
    "collective-order-consistency",
    "residency-consistency",
    "capacity-budget",
    "failure-code-required",
    "performance-regression-gate",
    "concurrency-probe-required",
    "maintenance-probe-required",
]


class RetrievalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RepositorySnapshot(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    repository: str
    revision: str
    repository_sha256: Digest
    source_manifest_sha256: Digest
    permission_snapshot_sha256: Digest
    captured_at: datetime
    corpus_cutoff: datetime
    parser_versions: dict[str, str] = Field(default_factory=dict)
    unparsed_files: list[str] = Field(default_factory=list)
    partial: bool = False

    @model_validator(mode="after")
    def timestamps_and_partial_state_are_coherent(self) -> RepositorySnapshot:
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (self.captured_at, self.corpus_cutoff)
        ):
            raise ValueError("snapshot timestamps must be timezone-aware")
        if self.unparsed_files and not self.partial:
            raise ValueError("unparsed files require partial=true")
        return self


class CommunicationFootprint(RetrievalModel):
    family: str
    collectives: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    transports: list[str] = Field(default_factory=list)
    topology_features: list[str] = Field(default_factory=list)
    communicator_lifecycle: list[str] = Field(default_factory=list)
    concurrency_surfaces: list[str] = Field(default_factory=list)
    failure_surfaces: list[str] = Field(default_factory=list)


class MemoryTieringFootprint(RetrievalModel):
    offload_object_kind: str
    mutability: str
    source_tier: str
    destination_tier: str
    residency_states: list[str] = Field(default_factory=list)
    transition_symbols: list[str] = Field(default_factory=list)
    allocator_symbols: list[str] = Field(default_factory=list)
    prefetch_symbols: list[str] = Field(default_factory=list)
    eviction_symbols: list[str] = Field(default_factory=list)
    budget_symbols: list[str] = Field(default_factory=list)
    copy_stream_symbols: list[str] = Field(default_factory=list)
    event_order_symbols: list[str] = Field(default_factory=list)
    version_key_symbols: list[str] = Field(default_factory=list)
    numa_policy_symbols: list[str] = Field(default_factory=list)


class FootprintExtractionRequest(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    draft_id: str
    draft_revision: int = Field(ge=1)
    candidate_sha256: Digest
    domain: Literal["auto", "distributed-communication", "memory-tiering"] = "auto"
    files: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def files_are_unique_relative_paths(self) -> FootprintExtractionRequest:
        if len(self.files) != len(set(self.files)):
            raise ValueError("footprint extraction files must be unique")
        invalid = [
            path
            for path in self.files
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
        ]
        if invalid:
            raise ValueError("footprint extraction files must stay under source root")
        return self


class CandidateFootprint(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    draft_id: str
    draft_revision: int = Field(ge=1)
    candidate_sha256: Digest
    files: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    callers: list[str] = Field(default_factory=list)
    dispatcher_points: list[str] = Field(default_factory=list)
    build_targets: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    config_keys: list[str] = Field(default_factory=list)
    failure_signatures: list[str] = Field(default_factory=list)
    resource_lifecycles: list[str] = Field(default_factory=list)
    workload_cases: list[str] = Field(default_factory=list)
    unresolved_surfaces: list[
        Literal[
            "unresolved-dynamic-dispatch",
            "unresolved-generated-source",
            "unresolved-plugin-boundary",
            "unresolved-runtime-symbol",
        ]
    ] = Field(default_factory=list)
    communication: CommunicationFootprint | None = None
    memory_tiering: MemoryTieringFootprint | None = None

    @model_validator(mode="after")
    def anchors_are_unique_and_domain_is_unambiguous(self) -> CandidateFootprint:
        anchor_fields = (
            "files",
            "symbols",
            "callers",
            "dispatcher_points",
            "build_targets",
            "tests",
            "config_keys",
            "failure_signatures",
            "resource_lifecycles",
            "workload_cases",
            "unresolved_surfaces",
        )
        duplicates = [
            name
            for name in anchor_fields
            if len(getattr(self, name)) != len(set(getattr(self, name)))
        ]
        if duplicates:
            raise ValueError("candidate footprint anchors must be unique: " + ", ".join(duplicates))
        if self.communication is not None and self.memory_tiering is not None:
            raise ValueError("single-domain footprint cannot mix communication and memory-tiering")
        return self


class QueryPass(RetrievalModel):
    id: RetrievalChannel
    required: bool
    budget: int = Field(ge=1)
    features: list[str] = Field(min_length=1)


class RRFPolicy(RetrievalModel):
    k: int = Field(default=60, ge=1)
    channel_weights: dict[RetrievalChannel, float]

    @model_validator(mode="after")
    def weights_are_positive(self) -> RRFPolicy:
        if not self.channel_weights or any(weight <= 0 for weight in self.channel_weights.values()):
            raise ValueError("RRF channel weights must be nonempty and positive")
        return self


class GraphExpansionBudget(RetrievalModel):
    max_hops: int = Field(default=2, ge=0, le=5)
    per_node_fanout: int = Field(default=20, ge=1)
    maximum_records: int = Field(default=200, ge=1)
    edge_allowlist: list[str] = Field(min_length=1)


class QueryPlan(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    policy_id: str
    target_snapshot_sha256: Digest
    candidate_footprint_sha256: Digest
    corpus_cutoff: datetime
    leakage_policy_id: str
    forbidden_source_ids: list[str] = Field(default_factory=list)
    passes: list[QueryPass] = Field(min_length=1)
    rrf: RRFPolicy
    graph: GraphExpansionBudget

    @model_validator(mode="after")
    def required_deterministic_passes_are_frozen(self) -> QueryPlan:
        if self.corpus_cutoff.tzinfo is None or self.corpus_cutoff.utcoffset() is None:
            raise ValueError("query-plan cutoff must be timezone-aware")
        identifiers = [item.id for item in self.passes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("query-plan passes must be unique")
        required = {item.id for item in self.passes if item.required}
        if not {"exact", "graph", "failure", "negative"}.issubset(required):
            raise ValueError("exact/graph/failure/negative passes must be required")
        semantic = next((item for item in self.passes if item.id == "semantic"), None)
        if semantic is not None and semantic.required:
            raise ValueError("semantic retrieval cannot be required for deterministic replay")
        if set(self.rrf.channel_weights) != set(identifiers):
            raise ValueError("RRF weights must exactly match query-plan passes")
        return self


class PrecedentValidity(RetrievalModel):
    repository: str
    first_revision: str
    last_revision: str | None = None
    corpus_cutoff_eligible: bool = True


class PrecedentScope(RetrievalModel):
    files: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    build_targets: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    configs: list[str] = Field(default_factory=list)
    failure_signatures: list[str] = Field(default_factory=list)
    lifecycle_tags: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)


class PrecedentRecord(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    precedent_id: str
    source_kind: Literal[
        "code",
        "test",
        "build-rule",
        "pull-request",
        "review",
        "issue",
        "ci-failure",
        "regression",
        "revert",
        "release-note",
        "archived-trial",
    ]
    source_locator: str
    source_digest: Digest
    source_event_id: str
    observed_at: datetime
    validity: PrecedentValidity
    kind: PrecedentKind
    authority: PrecedentAuthority
    target_authority: bool
    confidence: float = Field(ge=0, le=1)
    scope: PrecedentScope
    relations: dict[str, list[str]] = Field(default_factory=dict)
    text: str = ""
    change_fingerprint: str | None = None
    proposed_rule_templates: list[RuleTemplate] = Field(default_factory=list)

    @model_validator(mode="after")
    def authority_and_timestamp_are_coherent(self) -> PrecedentRecord:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("precedent timestamps must be timezone-aware")
        if not self.target_authority and self.authority != "advisory-cross-project":
            raise ValueError("cross-project precedent must remain advisory")
        return self


class PrecedentGraphEdge(RetrievalModel):
    source_id: str
    target_id: str
    kind: Literal[
        "MODIFIES",
        "CALLS",
        "REGISTERS",
        "TESTED_BY",
        "BUILT_BY",
        "CONFIGURED_BY",
        "REVIEWED_IN",
        "FAILED_BY",
        "FIXES",
        "REVERTS",
        "SUPERSEDES",
        "REGRESSES",
        "DEPENDS_ON",
        "OWNED_BY",
        "TOUCHES_LIFECYCLE",
        "TOUCHES_CAPABILITY",
        "OBSERVED_IN_WORKLOAD",
    ]

    @model_validator(mode="after")
    def edge_is_not_self_referential(self) -> PrecedentGraphEdge:
        if self.source_id == self.target_id:
            raise ValueError("precedent graph edges cannot be self-referential")
        return self


class ChannelHit(RetrievalModel):
    precedent_id: str
    channel: RetrievalChannel
    rank: int = Field(ge=1)
    matched_features: list[str] = Field(default_factory=list)


class FusedHit(RetrievalModel):
    precedent_id: str
    score: float = Field(gt=0)
    channel_ranks: dict[RetrievalChannel, int]


class LeakageExclusion(RetrievalModel):
    precedent_id: str
    reason: Literal[
        "forbidden-source-id",
        "after-corpus-cutoff",
        "future-follow-up",
        "known-solution-fingerprint",
        "suspected-near-duplicate",
    ]


class LeakageAudit(RetrievalModel):
    status: Literal["pass", "fail", "unresolved"]
    allowed_precedent_ids: list[str]
    exclusions: list[LeakageExclusion] = Field(default_factory=list)
    known_solution_leaked: bool = False
    suspected_near_duplicate: bool = False
    failure_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_findings(self) -> LeakageAudit:
        if self.known_solution_leaked and self.status != "fail":
            raise ValueError("known solution leakage requires status=fail")
        if self.suspected_near_duplicate and self.status == "pass":
            raise ValueError("suspected near-duplicate cannot pass leakage audit")
        return self


class ConflictSet(RetrievalModel):
    conflict_id: str
    precedent_ids: list[str] = Field(min_length=2)
    disposition: Literal[
        "human-review-required",
        "superseded-by-current-contract",
        "unresolved",
    ]
    reason: str


class RuleCandidate(RetrievalModel):
    rule_id: str
    modality: Literal["MUST", "SHOULD", "MAY", "FORBID"]
    template: RuleTemplate
    arguments: dict[str, Any]
    source_precedents: list[str] = Field(min_length=1)
    authority: PrecedentAuthority
    confidence: float = Field(ge=0, le=1)
    status: Literal[
        "proposed",
        "accepted",
        "edited",
        "advisory-only",
        "rejected",
        "conflicted",
    ] = "proposed"


class HumanRuleDecision(RetrievalModel):
    rule_id: str
    action: Literal[
        "accept",
        "accept-with-edits",
        "advisory-only",
        "reject",
        "conflict-unresolved",
    ]
    before_sha256: Digest
    after_sha256: Digest
    reviewer: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    reviewed_at: datetime

    @model_validator(mode="after")
    def review_timestamp_is_timezone_aware(self) -> HumanRuleDecision:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("human rule review timestamps must be timezone-aware")
        return self


class RetrievalCoverage(RetrievalModel):
    status: Literal["complete", "partial", "blocked", "stale", "conflicting"]
    required_passes_completed: list[RetrievalChannel]
    missing_sources: list[str] = Field(default_factory=list)
    unresolved_surfaces: list[str] = Field(default_factory=list)
    known_blind_spots: list[str] = Field(default_factory=list)


class RetrievalTrustCard(RetrievalModel):
    snapshot_integrity: Literal["pass", "fail", "unresolved"]
    deterministic_replay: Literal["pass", "fail", "unresolved"]
    parser_coverage: float = Field(ge=0, le=1)
    anchor_coverage: float = Field(ge=0, le=1)
    provenance_completeness: float = Field(ge=0, le=1)
    conflict_detection_status: Literal["complete", "partial", "unresolved"]
    leakage_audit: Literal["pass", "fail", "unresolved"]
    embedding_required: Literal[False] = False
    human_rule_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    unresolved_sources: list[str] = Field(default_factory=list)
    candidate_score_effect: Literal["none"] = "none"


class PrecedentSet(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    draft_id: str
    draft_revision: int = Field(ge=1)
    target_snapshot_sha256: Digest
    corpus_cutoff: datetime
    query_policy_id: str
    leakage_policy_id: str
    records: list[PrecedentRecord]
    graph_edges: list[PrecedentGraphEdge] = Field(default_factory=list)
    conflict_sets: list[ConflictSet] = Field(default_factory=list)
    omitted_records_path: str
    digest: Digest

    @model_validator(mode="after")
    def identities_are_unique_and_edges_resolve(self) -> PrecedentSet:
        identifiers = [item.precedent_id for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("PrecedentSet records must be unique")
        known = set(identifiers)
        dangling = [
            edge
            for edge in self.graph_edges
            if edge.source_id not in known or edge.target_id not in known
        ]
        if dangling:
            raise ValueError("PrecedentSet graph edges must resolve to included records")
        return self


class RetrievalBundle(RetrievalModel):
    schema_version: Literal["0.5.1"] = "0.5.1"
    snapshot: RepositorySnapshot
    footprint: CandidateFootprint
    query_plan: QueryPlan
    channel_hits: list[ChannelHit]
    fused_ranking: list[FusedHit]
    leakage_audit: LeakageAudit
    coverage: RetrievalCoverage
    rules: list[RuleCandidate]
    human_decisions: list[HumanRuleDecision] = Field(default_factory=list)
    trust: RetrievalTrustCard
    precedent_set: PrecedentSet
    bundle_sha256: Digest
