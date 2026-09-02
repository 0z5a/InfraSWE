from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

JudgeMode = Literal[
    "off",
    "advisory",
    "verifier-audit",
    "bounded-semantic",
    "human-assisted",
    "exploratory-proxy",
    "trajectory-audit",
]
JudgeAuthority = Literal[
    "none",
    "diagnostic",
    "audit",
    "bounded-score",
    "human-decision-support",
]
JudgeComponent = Literal["U", "M", "P"]
JudgeOwner = Literal["deterministic", "semantic-judge"]
JudgeDomain = Literal[
    "kernel",
    "distributed-communication",
    "memory-tiering",
    "training-integration",
    "inference-runtime",
]

GLOBAL_JUDGE_WEIGHT_CAPS: dict[JudgeComponent, float] = {
    "U": 0.10,
    "M": 0.20,
    "P": 0.25,
}
FORBIDDEN_JUDGE_BINDINGS = frozenset({"InfraCert", "C", "R", "O", "X", "CellArtifactPerformance"})


class JudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class JudgeModelIdentity(JudgeModel):
    provider: str
    family: str
    snapshot: str | None = None
    weights_sha256: Digest | None = None
    api_revision: str | None = None
    runtime_image_sha256: Digest | None = None
    quantization_sha256: Digest | None = None
    inference_engine_sha256: Digest | None = None
    chat_template_sha256: Digest | None = None

    @property
    def pinned(self) -> bool:
        if self.weights_sha256 is not None:
            return bool(
                self.runtime_image_sha256
                and self.inference_engine_sha256
                and self.chat_template_sha256
            )
        return bool(self.snapshot and self.api_revision)


class JudgePanelMember(JudgeModel):
    member_id: str
    model: JudgeModelIdentity
    repetitions: int = Field(default=2, ge=1, le=16)
    calibration_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def calibration_weights_are_bounded(self) -> JudgePanelMember:
        if any(value < 0 or value > 1 for value in self.calibration_weights.values()):
            raise ValueError("Judge calibration weights must stay in [0, 1]")
        return self


class JudgeEligibilityFloors(JudgeModel):
    weighted_kappa_lower_95ci: float = Field(default=0.60, ge=-1, le=1)
    hard_violation_false_pass_rate: float = Field(default=0.02, ge=0, le=1)
    test_retest_agreement: float = Field(default=0.90, ge=0, le=1)
    position_consistency: float = Field(default=0.90, ge=0, le=1)
    semantic_equivalence_invariance: float = Field(default=0.90, ge=0, le=1)
    evidence_grounding_precision: float = Field(default=0.95, ge=0, le=1)
    schema_valid_rate: float = Field(default=0.995, ge=0, le=1)
    prompt_injection_asr: float = Field(default=0.05, ge=0, le=1)
    in_domain_non_abstain_coverage: float = Field(default=0.75, ge=0, le=1)


class JudgeAggregationPolicy(JudgeModel):
    policy_id: Literal["weighted-median-abstain-v1"] = "weighted-median-abstain-v1"
    minimum_model_families: int = Field(default=2, ge=2)
    minimum_valid_members: int = Field(default=2, ge=2)
    repetitions_per_member: int = Field(default=2, ge=2)
    maximum_within_member_range: float = Field(default=0.25, ge=0, le=1)
    maximum_cross_family_range: float = Field(default=0.25, ge=0, le=1)
    require_evidence_refs: Literal[True] = True
    require_cross_family_agreement: Literal[True] = True
    required_criterion_abstain: Literal["unresolved"] = "unresolved"


class JudgeSecurityPolicy(JudgeModel):
    policy_id: Literal["untrusted-code-boundary-v1"] = "untrusted-code-boundary-v1"
    policy_sha256: Digest
    candidate_identity_blinding: Literal[True] = True
    score_blinding: Literal[True] = True
    prompt_injection_audit: Literal[True] = True
    secret_scan_required: Literal[True] = True
    network_access: Literal["forbidden"] = "forbidden"
    data_egress: Literal["local-only", "approved-hosted", "redacted-hosted"] = "local-only"


class JudgeBudget(JudgeModel):
    maximum_calls: int = Field(default=8, ge=0)
    maximum_input_tokens: int = Field(default=240_000, ge=0)
    maximum_output_tokens: int = Field(default=32_000, ge=0)
    cache_required: Literal[True] = True


class JudgeProfile(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    profile_id: str
    benchmark_season: str
    mode: JudgeMode
    authority: JudgeAuthority
    execution_mode: Literal[
        "direct-llm",
        "read-only-agent",
        "panel-mixed",
        "local-open-weight",
        "hosted-snapshot",
    ]
    supported_domains: list[JudgeDomain] = Field(min_length=1)
    system_prompt_sha256: Digest
    prompt_template_sha256: Digest
    context_compiler_sha256: Digest
    adapter_sha256: Digest
    security: JudgeSecurityPolicy
    panel: list[JudgePanelMember] = Field(default_factory=list)
    aggregation: JudgeAggregationPolicy = Field(default_factory=JudgeAggregationPolicy)
    component_judge_weight_caps: dict[JudgeComponent, float] = Field(default_factory=dict)
    forbidden_component_bindings: list[
        Literal["InfraCert", "C", "R", "O", "X", "CellArtifactPerformance"]
    ] = Field(default_factory=lambda: sorted(FORBIDDEN_JUDGE_BINDINGS))
    calibration_set_sha256: Digest | None = None
    calibration_report_sha256: Digest | None = None
    drift_sentinel_sha256: Digest | None = None
    floors: JudgeEligibilityFloors = Field(default_factory=JudgeEligibilityFloors)
    budget: JudgeBudget = Field(default_factory=JudgeBudget)
    same_family_as_candidate_policy: Literal["leave-family-out-or-zero-weight"] = (
        "leave-family-out-or-zero-weight"
    )

    @model_validator(mode="after")
    def official_profile_is_pinned_and_bounded(self) -> JudgeProfile:
        member_ids = [member.member_id for member in self.panel]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Judge panel member ids must be unique")
        if set(self.forbidden_component_bindings) != FORBIDDEN_JUDGE_BINDINGS:
            raise ValueError("Judge profile must forbid InfraCert/C/R/O/X/performance bindings")
        for component, cap in self.component_judge_weight_caps.items():
            if cap < 0 or cap > GLOBAL_JUDGE_WEIGHT_CAPS[component]:
                raise ValueError(f"Judge weight cap for {component} exceeds the global cap")
        if self.authority == "bounded-score":
            if self.mode != "bounded-semantic":
                raise ValueError("bounded-score authority requires bounded-semantic mode")
            if not self.component_judge_weight_caps:
                raise ValueError("bounded-score profile requires component Judge weight caps")
            if len(self.panel) < self.aggregation.minimum_valid_members:
                raise ValueError("bounded-score profile has too few panel members")
            families = {member.model.family for member in self.panel}
            if len(families) < self.aggregation.minimum_model_families:
                raise ValueError("bounded-score profile requires multiple model families")
            if any(not member.model.pinned for member in self.panel):
                raise ValueError("bounded-score profile requires exact pinned model identities")
            if any(
                member.repetitions < self.aggregation.repetitions_per_member
                for member in self.panel
            ):
                raise ValueError("bounded-score panel members require sealed repetitions")
            if not all(
                (
                    self.calibration_set_sha256,
                    self.calibration_report_sha256,
                    self.drift_sentinel_sha256,
                )
            ):
                raise ValueError("bounded-score profile requires calibration and drift digests")
        elif self.mode == "bounded-semantic":
            raise ValueError("bounded-semantic mode requires bounded-score authority")
        return self


class JudgeCriterionScale(JudgeModel):
    type: Literal["anchored-ordinal-5"] = "anchored-ordinal-5"
    anchors: dict[int, str] = Field(
        default_factory=lambda: {
            0: "clear-violation",
            1: "major-misalignment",
            2: "mixed-or-insufficient",
            3: "mostly-aligned",
            4: "clearly-aligned",
        }
    )

    @model_validator(mode="after")
    def anchors_cover_the_frozen_scale(self) -> JudgeCriterionScale:
        if set(self.anchors) != set(range(5)) or any(not value for value in self.anchors.values()):
            raise ValueError("Judge criterion anchors must define nonempty grades 0 through 4")
        return self


class JudgeCriterion(JudgeModel):
    criterion_id: str
    owner_component: JudgeComponent
    owner_type: JudgeOwner
    weight_within_component: float = Field(gt=0, le=1)
    question: str
    scale: JudgeCriterionScale = Field(default_factory=JudgeCriterionScale)
    required_evidence_types: list[str] = Field(default_factory=list)
    forbidden_inferences: list[str] = Field(default_factory=list)
    abstain_when: list[str] = Field(default_factory=list)
    required: bool = True

    @model_validator(mode="after")
    def semantic_criterion_has_grounding_and_abstention(self) -> JudgeCriterion:
        if self.owner_type == "semantic-judge" and (
            not self.required_evidence_types or not self.abstain_when
        ):
            raise ValueError(
                "semantic Judge criteria require evidence types and explicit abstention conditions"
            )
        return self


class JudgeRubric(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    rubric_id: str
    domain: JudgeDomain
    review_status: Literal["human-reviewed"] = "human-reviewed"
    human_review_sha256: Digest
    criteria: list[JudgeCriterion] = Field(min_length=1)

    @model_validator(mode="after")
    def ownership_is_unique_complete_and_capped(self) -> JudgeRubric:
        identifiers = [criterion.criterion_id for criterion in self.criteria]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Judge rubric criterion ids must be unique")
        for component in {criterion.owner_component for criterion in self.criteria}:
            component_criteria = [
                criterion for criterion in self.criteria if criterion.owner_component == component
            ]
            total = sum(item.weight_within_component for item in component_criteria)
            if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"criterion weights for {component} must sum to 1")
            judge_weight = sum(
                item.weight_within_component
                for item in component_criteria
                if item.owner_type == "semantic-judge"
            )
            if judge_weight > GLOBAL_JUDGE_WEIGHT_CAPS[component] + 1e-12:
                raise ValueError(f"semantic Judge weight for {component} exceeds the global cap")
        return self


class JudgeCalibrationMetrics(JudgeModel):
    weighted_kappa_lower_95ci: float = Field(ge=-1, le=1)
    hard_violation_false_pass_rate: float = Field(ge=0, le=1)
    test_retest_agreement: float = Field(ge=0, le=1)
    position_consistency: float = Field(ge=0, le=1)
    semantic_equivalence_invariance: float = Field(ge=0, le=1)
    evidence_grounding_precision: float = Field(ge=0, le=1)
    schema_valid_rate: float = Field(ge=0, le=1)
    prompt_injection_asr: float = Field(ge=0, le=1)
    in_domain_non_abstain_coverage: float = Field(ge=0, le=1)


class JudgeCalibrationReport(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    profile_id: str
    calibration_set_sha256: Digest
    domain: JudgeDomain
    sample_count: int = Field(ge=1)
    confidence_interval_policy_id: str
    metrics: JudgeCalibrationMetrics
    status: Literal["pass", "fail"]


class JudgeDriftSentinel(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    profile_id: str
    sentinel_set_sha256: Digest
    model_identity_sha256: Digest
    checks: dict[str, bool]
    status: Literal["pass", "fail", "drifted"]

    @model_validator(mode="after")
    def passing_sentinel_has_no_failed_checks(self) -> JudgeDriftSentinel:
        if self.status == "pass" and (not self.checks or not all(self.checks.values())):
            raise ValueError("passing drift sentinel requires every sealed check to pass")
        return self


class JudgeCell(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    profile_id: str
    benchmark_season: str
    profile_sha256: Digest
    rubric_sha256: Digest
    criterion_ownership_sha256: Digest
    panel_sha256: Digest
    aggregation_policy_sha256: Digest
    calibration_report_sha256: Digest
    drift_sentinel_sha256: Digest
    security_policy_sha256: Digest
    judge_cell_sha256: Digest


class JudgeBlindnessManifest(JudgeModel):
    candidate_author_hidden: Literal[True] = True
    candidate_agent_hidden: Literal[True] = True
    organization_hidden: Literal[True] = True
    review_popularity_hidden: Literal[True] = True
    aggregate_score_hidden: Literal[True] = True
    other_judges_hidden: Literal[True] = True
    human_final_decision_hidden: Literal[True] = True
    leaderboard_rank_hidden: Literal[True] = True


class JudgePackArtifactSpec(JudgeModel):
    ref_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*:[a-zA-Z0-9_.-]+$")
    path: str
    evidence_type: str
    authority: Literal[
        "target-authority",
        "deterministic-evidence",
        "human-reviewed-precedent",
        "candidate-controlled",
        "rubric",
        "advisory",
    ]
    candidate_controlled: bool = False

    @model_validator(mode="after")
    def candidate_authority_is_explicit(self) -> JudgePackArtifactSpec:
        if self.candidate_controlled != (self.authority == "candidate-controlled"):
            raise ValueError("candidate-controlled authority and flag must agree")
        return self


class JudgeInputPackSpec(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    draft_id: str
    draft_revision: int = Field(ge=1)
    candidate_sha256: Digest
    target_revision_sha256: Digest
    rubric_sha256: Digest
    artifacts: list[JudgePackArtifactSpec] = Field(min_length=1)
    blindness: JudgeBlindnessManifest = Field(default_factory=JudgeBlindnessManifest)
    data_egress: Literal["local-only", "approved-hosted", "redacted-hosted"] = "local-only"

    @model_validator(mode="after")
    def refs_are_unique_and_pack_is_grounded(self) -> JudgeInputPackSpec:
        refs = [artifact.ref_id for artifact in self.artifacts]
        if len(refs) != len(set(refs)):
            raise ValueError("Judge input pack artifact refs must be unique")
        authorities = {artifact.authority for artifact in self.artifacts}
        if not authorities & {"target-authority", "deterministic-evidence"}:
            raise ValueError("Judge input pack requires authoritative target or evidence material")
        if "rubric" not in authorities:
            raise ValueError("Judge input pack requires its sealed rubric")
        return self


class JudgeInputArtifact(JudgeModel):
    ref_id: str
    pack_path: str
    evidence_type: str
    authority: Literal[
        "target-authority",
        "deterministic-evidence",
        "human-reviewed-precedent",
        "candidate-controlled",
        "rubric",
        "advisory",
    ]
    source_sha256: Digest
    content_sha256: Digest
    candidate_controlled: bool
    boundary_encoding: Literal["none", "html-escaped-untrusted-v1"]


class JudgeInputPackManifest(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    draft_id: str
    draft_revision: int = Field(ge=1)
    candidate_sha256: Digest
    target_revision_sha256: Digest
    rubric_sha256: Digest
    blindness: JudgeBlindnessManifest
    data_egress: Literal["local-only", "approved-hosted", "redacted-hosted"]
    secret_scan_status: Literal["pass"] = "pass"
    artifacts: list[JudgeInputArtifact] = Field(min_length=1)
    pack_sha256: Digest


JudgeVerdict = Literal[
    "clear-violation",
    "major-misalignment",
    "mixed-or-insufficient",
    "mostly-aligned",
    "clearly-aligned",
    "abstain",
    "insufficient-evidence",
    "out-of-scope",
    "possible-prompt-injection",
]


class JudgeCriterionOutput(JudgeModel):
    criterion_id: str
    verdict: JudgeVerdict
    ordinal_grade: int | None = Field(default=None, ge=0, le=4)
    normalized_value: float | None = Field(default=None, ge=0, le=1)
    rationale_summary: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    counterevidence_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    requested_probes: list[str] = Field(default_factory=list)
    security_flags: list[str] = Field(default_factory=list)
    abstain_reason: str | None = None

    @model_validator(mode="after")
    def score_and_abstention_are_coherent(self) -> JudgeCriterionOutput:
        scored = self.verdict in {
            "clear-violation",
            "major-misalignment",
            "mixed-or-insufficient",
            "mostly-aligned",
            "clearly-aligned",
        }
        if scored:
            if self.ordinal_grade is None or self.normalized_value is None:
                raise ValueError("scored Judge verdict requires ordinal and normalized values")
            if not math.isclose(
                self.normalized_value,
                self.ordinal_grade / 4,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("Judge normalized value must equal ordinal_grade / 4")
            if self.abstain_reason is not None:
                raise ValueError("scored Judge verdict cannot carry an abstain reason")
        elif (
            self.ordinal_grade is not None
            or self.normalized_value is not None
            or not self.abstain_reason
        ):
            raise ValueError("abstaining Judge verdict requires only an abstain reason")
        return self


class JudgeOutputGlobal(JudgeModel):
    schema_valid: Literal[True] = True
    prompt_injection_suspected: bool = False
    out_of_scope: bool = False


class JudgeOutput(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    judge_run_id: str
    judge_cell_sha256: Digest
    input_pack_sha256: Digest
    rubric_sha256: Digest
    mode: JudgeMode
    criteria: list[JudgeCriterionOutput] = Field(min_length=1)
    global_status: JudgeOutputGlobal

    @model_validator(mode="after")
    def criterion_outputs_are_unique(self) -> JudgeOutput:
        identifiers = [criterion.criterion_id for criterion in self.criteria]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Judge output criterion ids must be unique")
        return self


class JudgeRunRecord(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    member_id: str
    model_family: str
    repetition: int = Field(ge=1)
    decoding_seed: int
    calibration_weight: float = Field(ge=0, le=1)
    criterion_calibration_weights: dict[str, float] = Field(default_factory=dict)
    candidate_family_excluded: bool = False
    validation_status: Literal["valid", "invalid"]
    failure_codes: list[str] = Field(default_factory=list)
    output: JudgeOutput

    @model_validator(mode="after")
    def validation_state_is_coherent(self) -> JudgeRunRecord:
        if any(value < 0 or value > 1 for value in self.criterion_calibration_weights.values()):
            raise ValueError("criterion calibration weights must stay in [0, 1]")
        if self.validation_status == "valid" and self.failure_codes:
            raise ValueError("valid Judge run cannot carry failure codes")
        if self.validation_status == "invalid" and not self.failure_codes:
            raise ValueError("invalid Judge run requires failure codes")
        if self.candidate_family_excluded and (
            self.calibration_weight != 0 or any(self.criterion_calibration_weights.values())
        ):
            raise ValueError("same-family excluded Judge run must have zero weight")
        return self


class JudgeCriterionAggregation(JudgeModel):
    criterion_id: str
    status: Literal[
        "valid",
        "unresolved",
        "judge-disagreement",
        "security-review-required",
    ]
    normalized_value: float | None = Field(default=None, ge=0, le=1)
    weighted_mad: float | None = Field(default=None, ge=0, le=1)
    valid_vote_count: int = Field(ge=0)
    valid_member_count: int = Field(ge=0)
    valid_family_count: int = Field(ge=0)
    abstention_rate: float = Field(ge=0, le=1)
    repeat_agreement: float = Field(ge=0, le=1)
    cross_family_range: float | None = Field(default=None, ge=0, le=1)
    failure_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_aggregate_has_value_and_no_failures(self) -> JudgeCriterionAggregation:
        if self.status == "valid" and (
            self.normalized_value is None or self.weighted_mad is None or self.failure_codes
        ):
            raise ValueError("valid Judge aggregate requires a value and no failures")
        if self.status != "valid" and (self.normalized_value is not None or not self.failure_codes):
            raise ValueError("unresolved Judge aggregate requires failure codes and no value")
        return self


class JudgeAggregation(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    judge_cell_sha256: Digest
    input_pack_sha256: Digest
    rubric_sha256: Digest
    policy_id: Literal["weighted-median-abstain-v1"]
    status: Literal[
        "official",
        "unresolved-judge",
        "judge-disagreement",
        "security-review-required",
    ]
    top_level_score_status: Literal["not-a-score"] = "not-a-score"
    criteria: list[JudgeCriterionAggregation] = Field(min_length=1)
    aggregation_sha256: Digest


class JudgeComponentProjection(JudgeModel):
    component: JudgeComponent
    status: Literal["official", "unresolved-judge", "deterministic-only"]
    deterministic_core_projection: float = Field(ge=0, le=1)
    judge_assisted_projection: float | None = Field(default=None, ge=0, le=1)
    judge_weight_within_component: float = Field(ge=0, le=1)
    criterion_values: dict[str, float]

    @model_validator(mode="after")
    def assisted_projection_matches_status(self) -> JudgeComponentProjection:
        if self.status == "official" and self.judge_assisted_projection is None:
            raise ValueError("official Judge component requires assisted projection")
        if self.status != "official" and self.judge_assisted_projection is not None:
            raise ValueError("non-official Judge component cannot publish assisted projection")
        return self


class JudgeScoreProjection(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    infra_cert_status: Literal["pass", "fail", "unresolved"]
    status: Literal[
        "official",
        "unresolved-judge",
        "hard-gate-failed",
        "hard-gate-unresolved",
    ]
    components: dict[JudgeComponent, JudgeComponentProjection]
    cross_judge_cell_ranking_allowed: Literal[False] = False
    projection_sha256: Digest

    @model_validator(mode="after")
    def hard_gate_cannot_be_overridden(self) -> JudgeScoreProjection:
        if self.infra_cert_status == "fail":
            if self.status != "hard-gate-failed":
                raise ValueError("failed InfraCert must remain a hard-gate failure")
            if any(
                component.judge_assisted_projection is not None
                for component in self.components.values()
            ):
                raise ValueError("Judge cannot publish an assisted score after InfraCert failure")
        if self.infra_cert_status == "unresolved" and self.status != "hard-gate-unresolved":
            raise ValueError("unresolved InfraCert must keep the score projection unresolved")
        return self


class JudgeTrustCard(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    status: Literal["pass", "fail", "drifted", "unpinned"]
    judge_cell_sha256: Digest | None = None
    profile_id: str
    domain: JudgeDomain
    calibration_report_sha256: Digest | None = None
    drift_sentinel_sha256: Digest | None = None
    metrics: JudgeCalibrationMetrics | None = None
    failure_codes: list[str] = Field(default_factory=list)
    candidate_score_effect: Literal[False] = False

    @model_validator(mode="after")
    def trust_state_is_coherent(self) -> JudgeTrustCard:
        if self.status == "pass" and (
            self.judge_cell_sha256 is None
            or self.calibration_report_sha256 is None
            or self.drift_sentinel_sha256 is None
            or self.metrics is None
            or self.failure_codes
        ):
            raise ValueError("passing JudgeTrust requires a Cell, calibration, drift, and metrics")
        if self.status != "pass" and not self.failure_codes:
            raise ValueError("non-passing JudgeTrust requires failure codes")
        return self


class JudgeCostCard(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    judge_cell_sha256: Digest
    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    wall_p50_seconds: float = Field(ge=0)
    wall_p95_seconds: float = Field(ge=0)
    provider_errors: int = Field(ge=0)
    retries: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    candidate_score_effect: Literal[False] = False

    @model_validator(mode="after")
    def counts_and_latency_are_coherent(self) -> JudgeCostCard:
        if self.cache_hits > self.calls:
            raise ValueError("Judge cache hits cannot exceed calls")
        if self.wall_p95_seconds < self.wall_p50_seconds:
            raise ValueError("Judge wall p95 cannot be below p50")
        return self


class JudgeVerifierAuditResult(JudgeModel):
    schema_version: Literal["0.5.3"] = "0.5.3"
    status: Literal[
        "agree",
        "possible-false-positive",
        "possible-false-negative",
        "verifier-scope-gap",
        "insufficient-evidence",
    ]
    evidence_refs: list[str] = Field(default_factory=list)
    recommended_actions: list[
        Literal["human-review", "add-deterministic-probe", "version-and-rerun"]
    ] = Field(default_factory=list)
    candidate_score_affected: Literal[False] = False
