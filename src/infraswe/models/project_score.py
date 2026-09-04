from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from infraswe.models.draft import Digest, DraftState, ProjectComparisonCell
from infraswe.policy import MERGE_ACCEPT_SCORE_FLOOR_100, overall_score_decision_band

ProjectComponentStatus = Literal[
    "scored", "not_applicable", "unresolved", "diagnostic", "not_run_due_to_gate"
]


class ProjectScoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PullRequestReviewContext(ProjectScoreModel):
    """Outcome-free live review context used only to decide CHECK eligibility."""

    created_at: datetime
    observed_at: datetime
    last_activity_at: datetime
    last_human_review_at: datetime | None = None
    current_head_human_non_author_review_count: int = Field(default=0, ge=0)
    total_human_non_author_review_count: int = Field(default=0, ge=0)
    pending_human_review_request: bool = False

    @model_validator(mode="after")
    def timestamps_and_review_counts_are_coherent(self) -> PullRequestReviewContext:
        timestamps = [self.created_at, self.observed_at, self.last_activity_at]
        if self.last_human_review_at is not None:
            timestamps.append(self.last_human_review_at)
        if any(value.tzinfo is None or value.utcoffset() is None for value in timestamps):
            raise ValueError("pull-request review timestamps must be timezone-aware")
        if self.observed_at < self.created_at:
            raise ValueError("review context cannot be observed before PR creation")
        if not self.created_at <= self.last_activity_at <= self.observed_at:
            raise ValueError("last PR activity must fall between creation and observation")
        if self.last_human_review_at is not None and not (
            self.created_at <= self.last_human_review_at <= self.observed_at
        ):
            raise ValueError("last human review must fall between creation and observation")
        if (
            self.last_human_review_at is not None
            and self.last_activity_at < self.last_human_review_at
        ):
            raise ValueError("last activity cannot predate the last human review")
        if (
            self.current_head_human_non_author_review_count
            > self.total_human_non_author_review_count
        ):
            raise ValueError("current-head review count cannot exceed total review count")
        if (
            self.current_head_human_non_author_review_count > 0
            and self.last_human_review_at is None
        ):
            raise ValueError("current-head human reviews require a review timestamp")
        return self


class ProjectScoreComponent(ProjectScoreModel):
    status: ProjectComponentStatus
    value: float | None = Field(default=None, ge=0, le=1)
    formula_version: str
    input_evidence_digests: list[Digest] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high", "not_applicable"]
    failure_codes: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def value_matches_status(self) -> ProjectScoreComponent:
        if self.status == "scored" and self.value is None:
            raise ValueError("scored project components require a value")
        if (
            self.status in {"not_applicable", "unresolved", "not_run_due_to_gate"}
            and self.value is not None
        ):
            raise ValueError(f"{self.status} project components cannot carry a value")
        if self.status == "not_applicable" and self.confidence != "not_applicable":
            raise ValueError("not_applicable requires not_applicable confidence")
        return self


class ProjectFitScore(ProjectScoreModel):
    status: Literal["provisional", "official", "not_acceptable", "unresolved", "not_issued"]
    formula_template_id: Literal[
        "project-fit-kernel-v0.5",
        "project-fit-triton-pure-v0.5",
        "project-fit-system-path-v0.5.1",
    ]
    score_100: float | None = Field(default=None, ge=0, le=100)
    components: dict[str, ProjectScoreComponent]
    component_floors: dict[str, float]
    confidence: Literal["low", "medium", "high", "not_applicable"]
    comparison_cell: ProjectComparisonCell
    cross_project_ranking_allowed: Literal[False] = False
    failure_codes: list[str] = Field(default_factory=list)
    audit_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def formula_and_components_are_coherent(self) -> ProjectFitScore:
        expected = {
            "evolutionary_maintainability",
            "project_contract_fit",
            "performance_reuse_utilization",
            "operational_fit",
            "pure_triton_portability",
        }
        if set(self.components) != expected:
            raise ValueError("ProjectFit requires the frozen M/P/R/O/X component envelope")
        portability = self.components["pure_triton_portability"]
        if self.formula_template_id in {
            "project-fit-kernel-v0.5",
            "project-fit-system-path-v0.5.1",
        }:
            if portability.status != "not_applicable":
                raise ValueError("ordinary ProjectFit requires X=not_applicable")
        elif self.score_100 is not None and portability.status != "scored":
            raise ValueError("numeric pure Triton ProjectFit requires a scored X component")
        if self.status in {"official", "not_acceptable"} and self.score_100 is None:
            raise ValueError(f"{self.status} ProjectFit requires score_100")
        if self.status in {"unresolved", "not_issued"} and self.score_100 is not None:
            raise ValueError(f"{self.status} ProjectFit cannot carry score_100")
        return self


class PureTritonEligibilityEvidence(ProjectScoreModel):
    implementation_kind: Literal[
        "cuda-native", "hip-native", "cann-native", "triton-pure", "framework"
    ]
    core_kernel_language: str
    backend_native_kernel_calls: list[str] = Field(default_factory=list)
    backend_specific_code_locations: list[str] = Field(default_factory=list)
    backend_specific_code_is_capability_or_launch_only: bool
    shared_semantic_implementation_family: bool
    required_profile_evidence: dict[str, Literal["verified", "missing", "unsupported"]]
    local_baseline_normalization: bool
    absolute_cross_hardware_latency_ranking: bool
    explicit_unsupported_and_fallback: bool
    evidence_digests: list[Digest] = Field(default_factory=list)


class TritonPurityAudit(ProjectScoreModel):
    status: Literal["pass", "fail", "not_applicable", "unresolved"]
    failure_codes: list[str] = Field(default_factory=list)
    evidence_digests: list[Digest] = Field(default_factory=list)
    reason: str | None = None


class BenchmarkTrustCard(ProjectScoreModel):
    status: Literal["scored", "unresolved"]
    score_100: float | None = Field(default=None, ge=0, le=100)
    components: dict[str, float | None]
    formula_version: Literal["benchmark-trust-v0.5"] = "benchmark-trust-v0.5"
    evidence_digests: list[Digest] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def trust_value_matches_status(self) -> BenchmarkTrustCard:
        expected = {"reproducibility", "evidence", "statistics", "environment"}
        if set(self.components) != expected:
            raise ValueError("BenchmarkTrust requires repro/evidence/stats/environment")
        if self.status == "scored" and self.score_100 is None:
            raise ValueError("scored BenchmarkTrust requires score_100")
        if self.status == "unresolved" and self.score_100 is not None:
            raise ValueError("unresolved BenchmarkTrust cannot carry score_100")
        invalid = [
            name
            for name, value in self.components.items()
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
        ]
        if invalid:
            raise ValueError("BenchmarkTrust components must be finite and in [0, 1]")
        return self


class BenchmarkCostCard(ProjectScoreModel):
    status: Literal["complete", "partial", "unresolved"] = "complete"
    wall_time_seconds: float | None = Field(default=None, ge=0)
    accelerator_seconds: float | None = Field(default=None, ge=0)
    compile_seconds: float | None = Field(default=None, ge=0)
    precompile_seconds: float | None = Field(default=None, ge=0)
    cold_start_seconds: float | None = Field(default=None, ge=0)
    steady_state_seconds: float | None = Field(default=None, ge=0)
    steady_state_compile_seconds: float | None = Field(default=None, ge=0)
    compilation_path: Literal["not-required", "precompile", "cache-reuse", "inline"] | None = None
    profiler_seconds: float | None = Field(default=None, ge=0)
    executed_cases: int | None = Field(default=None, ge=0)
    skipped_cases: int | None = Field(default=None, ge=0)
    cache_hit_ratio: float | None = Field(default=None, ge=0, le=1)
    time_to_first_diagnostic_seconds: float | None = Field(default=None, ge=0)
    time_to_actionable_decision_seconds: float | None = Field(default=None, ge=0)
    fast_stage_resolution_rate: float | None = Field(default=None, ge=0, le=1)
    serialization_config_compatibility: Literal["pass", "fail", "unresolved"]
    failure_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_cost_has_all_required_values(self) -> BenchmarkCostCard:
        required = {
            "wall_time_seconds": self.wall_time_seconds,
            "accelerator_seconds": self.accelerator_seconds,
            "compile_seconds": self.compile_seconds,
            "precompile_seconds": self.precompile_seconds,
            "cold_start_seconds": self.cold_start_seconds,
            "steady_state_seconds": self.steady_state_seconds,
            "steady_state_compile_seconds": self.steady_state_compile_seconds,
            "compilation_path": self.compilation_path,
            "profiler_seconds": self.profiler_seconds,
            "executed_cases": self.executed_cases,
            "skipped_cases": self.skipped_cases,
            "cache_hit_ratio": self.cache_hit_ratio,
            "fast_stage_resolution_rate": self.fast_stage_resolution_rate,
        }
        if self.status == "complete" and any(value is None for value in required.values()):
            raise ValueError("complete BenchmarkCost requires all frozen cost fields")
        if self.status == "unresolved" and any(value is not None for value in required.values()):
            raise ValueError("partially observed BenchmarkCost must use status=partial")
        if (
            self.compile_seconds is not None
            and self.precompile_seconds is not None
            and self.precompile_seconds > self.compile_seconds
        ):
            raise ValueError("precompile_seconds cannot exceed total compile_seconds")
        if self.compilation_path in {"precompile", "cache-reuse", "not-required"} and (
            self.steady_state_compile_seconds not in {None, 0}
        ):
            raise ValueError("precompiled benchmark paths cannot compile during steady state")
        if self.compilation_path == "not-required" and self.compile_seconds not in {None, 0}:
            raise ValueError("not-required compilation paths must report zero compile_seconds")
        return self


class ProjectObjectiveResult(ProjectScoreModel):
    policy: Literal["roadmap", "release-gate", "maintenance-only", "experimental", "out-of-scope"]
    status: Literal[
        "verified",
        "degraded",
        "explicit-unsupported",
        "not-tested",
        "blocked-by-infra",
        "out-of-scope",
    ]
    weighted_score: None = None
    release_gate_passed: bool | None = None
    evidence_digests: list[Digest] = Field(default_factory=list)


class CellEfficiencyReference(ProjectScoreModel):
    status: Literal["available", "unresolved", "not_applicable"]
    details_path: str | None = None
    cross_cell_ranking_allowed: Literal[False] = False


class InfraSWEMicroscores(ProjectScoreModel):
    """Explanatory child scores nested under the sole overall score."""

    project_fit: ProjectFitScore
    benchmark_trust: BenchmarkTrustCard


class OrderedEvaluationGate(ProjectScoreModel):
    name: Literal["maintainability", "deployability", "performance", "overall-score"]
    status: Literal["pass", "fail", "unresolved", "not-run"]
    rationale_codes: list[str] = Field(min_length=1)


class MergeabilityDecision(ProjectScoreModel):
    verdict: Literal[
        "accept",
        "accept_with_scope",
        "check",
        "reject",
        "unresolved",
    ]
    supported_scope: list[str] = Field(default_factory=list)
    excluded_scope: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    rationale_codes: list[str] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_legacy_revise(cls, value: object) -> object:
        """Read the v0.1 legacy wire label while emitting only ``check``."""

        return "check" if value == "revise" else value


class InfraSWEDecision(ProjectScoreModel):
    """Current three-class disposition with scope represented as a qualifier."""

    classification: Literal["accept", "check", "reject"]
    acceptance_scope: Literal["full", "limited", "not-applicable"]
    supported_scope: list[str] = Field(default_factory=list)
    excluded_scope: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    rationale_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def scope_qualifier_matches_classification(self) -> InfraSWEDecision:
        if self.classification != "accept" and self.acceptance_scope != "not-applicable":
            raise ValueError("only accept classifications can carry an acceptance scope")
        if self.classification == "accept":
            expected_scope = "limited" if self.excluded_scope else "full"
            if self.acceptance_scope != expected_scope:
                raise ValueError("acceptance scope must reflect the declared excluded scope")
        return self


class V05ScoreResult(ProjectScoreModel):
    schema_version: Literal["0.5"] = "0.5"
    draft_id: str
    draft_revision: int = Field(ge=1)
    draft_state: DraftState
    sealed_draft_sha256: Digest | None = None
    target_project_profile_sha256: Digest
    target_repository_sha256: Digest
    candidate_sha256: Digest
    acceptance_contract_sha256: Digest
    infra_cert: Literal["pass", "fail", "unresolved"]
    project_fit: ProjectFitScore
    leaderboard_effective_project_fit_100: float | None = Field(default=None, ge=0, le=100)
    benchmark_trust: BenchmarkTrustCard
    benchmark_cost: BenchmarkCostCard
    evidence_grade: Literal[
        "E0-runtime", "E1-framework", "E2-system-trace", "E3-kernel-counter", "E4-sealed"
    ]
    project_objectives: dict[str, ProjectObjectiveResult]
    cell_efficiency: CellEfficiencyReference
    decision: MergeabilityDecision
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    failure_codes: list[str] = Field(default_factory=list)
    audit_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def result_layers_are_coherent(self) -> V05ScoreResult:
        if self.decision.verdict in {"accept", "accept_with_scope"} and (
            self.project_fit.score_100 is None
            or self.project_fit.score_100 < MERGE_ACCEPT_SCORE_FLOOR_100
        ):
            raise ValueError("accepted mergeability decisions require ProjectFit >= 85")
        if self.decision.verdict == "check" and not (
            {
                "ACTIVE_NEW_PR_REVIEW_CHECK_ELIGIBLE",
                "ACTIVE_NEW_PR_REVIEW_REVISE_ELIGIBLE",
            }
            & set(self.decision.rationale_codes)
        ):
            raise ValueError("CHECK requires explicit active-new-PR review eligibility")
        if self.project_fit.status == "official":
            if self.draft_state != "D8-decided" or self.sealed_draft_sha256 is None:
                raise ValueError("official ProjectFit requires a decided sealed Draft")
            if self.infra_cert != "pass":
                raise ValueError("official ProjectFit requires InfraCert pass")
            if self.leaderboard_effective_project_fit_100 != self.project_fit.score_100:
                raise ValueError("official acceptable ProjectFit must publish its effective score")
        if (
            self.project_fit.status == "provisional"
            and self.leaderboard_effective_project_fit_100 is not None
        ):
            raise ValueError("provisional ProjectFit cannot enter a leaderboard")
        if self.project_fit.status == "provisional" and self.draft_state != "D5-fast-loop":
            raise ValueError("provisional ProjectFit is only valid in the D5 fast loop")
        if self.infra_cert == "fail":
            if self.project_fit.score_100 is not None:
                raise ValueError("failed InfraCert cannot publish ProjectFit-100")
            if self.leaderboard_effective_project_fit_100 != 0:
                raise ValueError("failed InfraCert requires effective ProjectFit 0")
            if self.decision.verdict not in {"reject", "check"}:
                raise ValueError("failed InfraCert must produce REJECT or CHECK")
        if self.infra_cert == "unresolved":
            if self.leaderboard_effective_project_fit_100 is not None:
                raise ValueError("unresolved InfraCert cannot publish an effective score")
            if self.decision.verdict != "unresolved":
                raise ValueError("unresolved InfraCert requires an unresolved decision")
        if (
            self.project_fit.status == "not_acceptable"
            and self.leaderboard_effective_project_fit_100 != 0
        ):
            raise ValueError("not acceptable ProjectFit requires effective score 0")
        if self.project_fit.status == "not_acceptable" and (
            self.draft_state != "D8-decided" or self.sealed_draft_sha256 is None
        ):
            raise ValueError("not acceptable ProjectFit requires a decided sealed Draft")
        return self


class InfraSWEOverallResult(ProjectScoreModel):
    """Current result envelope with one composite and nested explanatory microscores."""

    schema_version: Literal["0.1"] = "0.1"
    draft_id: str
    draft_revision: int = Field(ge=1)
    draft_state: DraftState
    sealed_draft_sha256: Digest | None = None
    target_project_profile_sha256: Digest
    target_repository_sha256: Digest
    candidate_sha256: Digest
    acceptance_contract_sha256: Digest
    infra_cert: Literal["pass", "fail", "unresolved"]
    overall_score_100: float | None = Field(default=None, ge=0, le=100)
    overall_score_formula_id: Literal["infraswe-overall-v0.1"] = "infraswe-overall-v0.1"
    microscores: InfraSWEMicroscores
    ordered_gates: list[OrderedEvaluationGate] = Field(min_length=4, max_length=4)
    benchmark_cost: BenchmarkCostCard
    evidence_grade: Literal[
        "E0-runtime", "E1-framework", "E2-system-trace", "E3-kernel-counter", "E4-sealed"
    ]
    project_objectives: dict[str, ProjectObjectiveResult]
    cell_efficiency: CellEfficiencyReference
    decision: InfraSWEDecision
    evaluation_engine: Literal["infraswe", "external"] = "infraswe"
    evaluation_scope: Literal["full", "staged"] = "full"
    seal_enabled: bool = True
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    failure_codes: list[str] = Field(default_factory=list)
    audit_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def hierarchy_and_decision_are_coherent(self) -> InfraSWEOverallResult:
        expected_order = ["maintainability", "deployability", "performance", "overall-score"]
        if [gate.name for gate in self.ordered_gates] != expected_order:
            raise ValueError(
                "evaluation gates must use maintainability/deployability/performance/overall order"
            )
        first_non_pass = next(
            (index for index, gate in enumerate(self.ordered_gates[:3]) if gate.status != "pass"),
            None,
        )
        if first_non_pass is not None and any(
            gate.status != "not-run" for gate in self.ordered_gates[first_non_pass + 1 :]
        ):
            raise ValueError("a blocking hard gate requires every later gate to be not-run")
        hard_gates_passed = first_non_pass is None
        score_gate = self.ordered_gates[3]
        if hard_gates_passed:
            if self.overall_score_100 is None and score_gate.status != "unresolved":
                raise ValueError("missing overall score requires an unresolved overall-score gate")
            if self.overall_score_100 is not None and score_gate.status != "pass":
                raise ValueError("issued overall score requires a passed overall-score gate")
        elif self.overall_score_100 is not None:
            raise ValueError("overall score cannot be issued before every hard gate passes")
        if self.overall_score_100 is not None:
            project_fit_score = self.microscores.project_fit.score_100
            benchmark_trust_score = self.microscores.benchmark_trust.score_100
            if project_fit_score is None or benchmark_trust_score is None:
                raise ValueError("overall score requires both explanatory microscores")
            expected_score = 100 * (
                (project_fit_score / 100) ** 0.85 * (benchmark_trust_score / 100) ** 0.15
            )
            if not math.isclose(self.overall_score_100, expected_score, rel_tol=1e-9):
                raise ValueError("overall score disagrees with its frozen microscore formula")
        hard_gates = self.ordered_gates[:3]
        if any(gate.status == "fail" for gate in hard_gates):
            expected_decision = "reject"
        elif (
            any(gate.status in {"unresolved", "not-run"} for gate in hard_gates)
            or self.overall_score_100 is None
        ):
            expected_decision = "check"
        else:
            expected_decision = overall_score_decision_band(self.overall_score_100)
        if self.decision.classification != expected_decision:
            raise ValueError("decision disagrees with ordered hard gates and overall score")
        if self.decision.classification == "accept" and (
            not self.seal_enabled
            or self.sealed_draft_sha256 is None
            or self.draft_state != "D8-decided"
        ):
            raise ValueError("accept requires the enabled Draft seal path and D8 state")
        return self
