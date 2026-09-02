from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComponentStatus = Literal["scored", "not_applicable", "unresolved", "diagnostic"]
EvidenceGrade = Literal[
    "legacy-framework-trace",
    "E0-runtime",
    "E1-framework",
    "E2-system-trace",
    "E3-kernel-counter",
    "E4-sealed",
]


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    reasons: list[str] = Field(default_factory=list)


class CoreComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: float = Field(ge=0, le=1)
    regression: float = Field(ge=0, le=1)
    fresh_replay: float = Field(ge=0, le=1)
    efficiency: float = Field(ge=0, le=1)
    protocol: float = Field(ge=0, le=1)


class InfraComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slo_goodput: float = Field(ge=0, le=1)
    fault_recovery: float = Field(ge=0, le=1)
    safety_rollback: float = Field(ge=0, le=1)
    resource_efficiency: float = Field(ge=0, le=1)
    topology_robustness: float = Field(ge=0, le=1)
    observability: float = Field(ge=0, le=1)


class KernelSearchScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auc_primary_100: float | None = Field(default=None, ge=0, le=100)
    primary_budget_axis: str | None = None
    auc_leased_device_100: float | None = Field(default=None, ge=0, le=100)
    auc_wall_100: float | None = Field(default=None, ge=0, le=100)
    auc_token_100: float | None = Field(default=None, ge=0, le=100)
    sealed_checkpoint_count: int = Field(default=0, ge=0)
    sealed_feedback_during_episode: Literal[False] = False
    time_to_first_correct_sec: float | None = Field(default=None, ge=0)
    time_to_baseline_sec: float | None = Field(default=None, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    compile_attempts: int = Field(default=0, ge=0)
    profile_invocations: int = Field(default=0, ge=0)


class KernelScoreEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicable: bool
    certified: bool | None = None
    verdict: Literal["pass", "fail", "unresolved", "not_applicable"] = "not_applicable"
    disposition: Literal["valid", "invalid", "quarantined", "not_applicable"] = "not_applicable"
    artifact_status: Literal["scored", "unscored_invalid", "quarantined", "not_applicable"] = (
        "not_applicable"
    )
    artifact_100: float | None = Field(default=None, ge=0, le=100)
    leaderboard_effective_artifact_100: float | None = Field(default=None, ge=0, le=100)
    benchmark_cell_id: str | None = None
    leaderboard_season: str | None = None
    formula_version: str | None = None
    formula_origin: str | None = None
    component_formula_origins: dict[str, str] = Field(default_factory=dict)
    formula_parameters_sha256: str | None = None
    profile: str | None = None
    components: dict[str, float] = Field(default_factory=dict)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    search: KernelSearchScore | None = None
    roles: dict[str, str] = Field(default_factory=dict)
    audit_flags: list[str] = Field(default_factory=list)


class ScoreComponent(BaseModel):
    """A score value bound to its formula and input evidence.

    Missing, unsupported, and diagnostic-only evidence are represented explicitly;
    callers must never turn those states into a numeric zero or renormalize weights.
    """

    model_config = ConfigDict(extra="forbid")

    status: ComponentStatus
    value: float | None = Field(default=None, ge=0, le=1)
    formula_version: str
    input_evidence_digests: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high", "not_applicable"] = "low"
    reason: str | None = None

    @model_validator(mode="after")
    def value_matches_status(self) -> ScoreComponent:
        if self.status == "scored" and self.value is None:
            raise ValueError("scored components require a value")
        if self.status in {"not_applicable", "unresolved"} and self.value is not None:
            raise ValueError(f"{self.status} components cannot carry a numeric value")
        if self.status == "not_applicable" and self.confidence != "not_applicable":
            raise ValueError("not_applicable components require not_applicable confidence")
        return self


class DeployabilityScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["scored", "not_deployable", "unresolved", "not_applicable"]
    formula_template_id: Literal["deployability-v0.4"] = "deployability-v0.4"
    score_100: float | None = Field(default=None, ge=0, le=100)
    components: dict[str, ScoreComponent] = Field(default_factory=dict)
    component_floors: dict[str, float] = Field(default_factory=dict)
    cross_cell_ranking_allowed: bool = True

    @model_validator(mode="after")
    def deployability_is_coherent(self) -> DeployabilityScore:
        expected = {"concurrent_stability", "kernel_reuse", "maintainability"}
        if self.status != "not_applicable" and set(self.components) != expected:
            raise ValueError(
                "deployability requires exactly concurrent_stability/kernel_reuse/maintainability"
            )
        if self.status in {"scored", "not_deployable"} and self.score_100 is None:
            raise ValueError(f"{self.status} deployability requires score_100")
        if self.status in {"unresolved", "not_applicable"} and self.score_100 is not None:
            raise ValueError(f"{self.status} deployability cannot carry score_100")
        return self


class CellEfficiencyScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["scored", "unresolved", "diagnostic", "not_applicable"]
    work_model_id: str | None = None
    sol_efficiency: ScoreComponent
    useful_memory_band_efficiency: ScoreComponent
    physical_memory_band_efficiency: ScoreComponent
    traffic_amplification: ScoreComponent
    raw: dict[str, Any] = Field(default_factory=dict)
    counter_confidence: Literal["low", "medium", "high", "not_applicable"]
    cross_cell_ranking_allowed: Literal[False] = False


class CellArtifactScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["scored", "not_deployable", "unresolved", "not_applicable"]
    formula_template_id: Literal[
        "cell-artifact-mixed-v0.4",
        "cell-artifact-memory-v0.4",
        "cell-artifact-compute-v0.4",
        "cell-artifact-distributed-v0.4",
        "cell-artifact-memory-tiering-v0.5.2",
    ]
    score_100: float | None = Field(default=None, ge=0, le=100)
    components: dict[str, ScoreComponent] = Field(default_factory=dict)
    cross_cell_ranking_allowed: Literal[False] = False

    @model_validator(mode="after")
    def value_matches_status(self) -> CellArtifactScore:
        if self.status in {"scored", "not_deployable"} and self.score_100 is None:
            raise ValueError(f"{self.status} cell artifact requires score_100")
        if self.status in {"unresolved", "not_applicable"} and self.score_100 is not None:
            raise ValueError(f"{self.status} cell artifact cannot carry score_100")
        return self


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "0.3", "0.4"] = "0.1"
    resolved_at_1: bool | None = None
    stable_resolved_at_1: bool | None = None
    coverage: float | None = Field(default=None, ge=0, le=1)
    gate: GateResult | None = None
    core_components: CoreComponents | None = None
    infra_components: InfraComponents | None = None
    core_100: float | None = Field(default=None, ge=0, le=100)
    infra_ext_100: float | None = Field(default=None, ge=0, le=100)
    infra_total: float | None = Field(default=None, ge=0, le=100)
    raw: dict[str, Any] = Field(default_factory=dict)
    kernel: KernelScoreEnvelope | None = None
    infra_cert: Literal["pass", "fail", "unresolved", "not_applicable"] | None = None
    disposition: Literal["valid", "invalid", "quarantined", "partial", "not_applicable"] | None = (
        None
    )
    benchmark_cell_id: str | None = None
    evidence_grade: EvidenceGrade | None = None
    deployability: DeployabilityScore | None = None
    leaderboard_effective_deployability_100: float | None = Field(default=None, ge=0, le=100)
    cell_efficiency: CellEfficiencyScore | None = None
    cell_artifact: CellArtifactScore | None = None
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    search: KernelSearchScore | None = None
    failure_codes: list[str] = Field(default_factory=list)
    audit_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def versioned_envelope_is_coherent(self) -> ScoreResult:
        legacy_values = {
            "resolved_at_1": self.resolved_at_1,
            "stable_resolved_at_1": self.stable_resolved_at_1,
            "coverage": self.coverage,
            "gate": self.gate,
            "core_components": self.core_components,
            "infra_components": self.infra_components,
            "core_100": self.core_100,
            "infra_ext_100": self.infra_ext_100,
            "infra_total": self.infra_total,
        }
        if self.schema_version != "0.4":
            missing = [name for name, value in legacy_values.items() if value is None]
            if missing:
                raise ValueError("legacy score result missing fields: " + ", ".join(missing))
            return self
        required = {
            "infra_cert": self.infra_cert,
            "disposition": self.disposition,
            "benchmark_cell_id": self.benchmark_cell_id,
            "evidence_grade": self.evidence_grade,
            "deployability": self.deployability,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("v0.4 score result missing fields: " + ", ".join(missing))
        if self.infra_cert == "fail":
            if self.deployability and self.deployability.score_100 is not None:
                raise ValueError("failed InfraCert cannot publish Deployability-100")
            if self.leaderboard_effective_deployability_100 != 0:
                raise ValueError("failed InfraCert requires effective deployability 0")
        if (
            self.infra_cert == "unresolved"
            and self.leaderboard_effective_deployability_100 is not None
        ):
            raise ValueError("unresolved InfraCert cannot publish an effective score")
        return self
