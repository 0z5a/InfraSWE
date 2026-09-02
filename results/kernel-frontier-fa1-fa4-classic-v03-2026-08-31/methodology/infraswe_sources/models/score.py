from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    disposition: Literal["valid", "invalid", "quarantined", "not_applicable"] = (
        "not_applicable"
    )
    artifact_status: Literal[
        "scored", "unscored_invalid", "quarantined", "not_applicable"
    ] = "not_applicable"
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


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    resolved_at_1: bool
    stable_resolved_at_1: bool
    coverage: float = Field(ge=0, le=1)
    gate: GateResult
    core_components: CoreComponents
    infra_components: InfraComponents
    core_100: float = Field(ge=0, le=100)
    infra_ext_100: float = Field(ge=0, le=100)
    infra_total: float = Field(ge=0, le=100)
    raw: dict[str, Any] = Field(default_factory=dict)
    kernel: KernelScoreEnvelope | None = None
