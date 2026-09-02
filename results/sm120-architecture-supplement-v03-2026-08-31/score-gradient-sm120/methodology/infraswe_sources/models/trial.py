from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from infraswe.io import utc_now


class TrialState(StrEnum):
    PENDING = "PENDING"
    LEASING = "LEASING"
    SETUP = "SETUP"
    AGENT_RUNNING = "AGENT_RUNNING"
    COLLECTING = "COLLECTING"
    AGENT_DESTROYED = "AGENT_DESTROYED"
    VERIFYING = "VERIFYING"
    SCORING = "SCORING"
    ARCHIVING = "ARCHIVING"
    COMPLETED = "COMPLETED"
    FAILED_INFRA = "FAILED_INFRA"
    FAILED_AGENT = "FAILED_AGENT"
    INVALID_TASK = "INVALID_TASK"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class FailureKind(StrEnum):
    TASK_INVALID = "TASK_INVALID"
    ENV_BUILD_FAILED = "ENV_BUILD_FAILED"
    LEASE_FAILED = "LEASE_FAILED"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_BUDGET_EXCEEDED = "AGENT_BUDGET_EXCEEDED"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    PATCH_APPLY_FAILED = "PATCH_APPLY_FAILED"
    FUNCTIONAL_FAILED = "FUNCTIONAL_FAILED"
    REGRESSION_FAILED = "REGRESSION_FAILED"
    SLO_FAILED = "SLO_FAILED"
    FAULT_RECOVERY_FAILED = "FAULT_RECOVERY_FAILED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    SILENT_FALLBACK = "SILENT_FALLBACK"
    DEADLOCK = "DEADLOCK"
    DATA_CORRUPTION = "DATA_CORRUPTION"
    RESOURCE_LEAK = "RESOURCE_LEAK"
    FLAKY_REPLAY = "FLAKY_REPLAY"
    VERIFIER_INFRA_FAILED = "VERIFIER_INFRA_FAILED"


class TrialEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: datetime = Field(default_factory=utc_now)
    state: TrialState
    detail: str = ""


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_time_sec: float = Field(default=0, ge=0)
    agent_time_sec: float = Field(default=0, ge=0)
    verifier_time_sec: float = Field(default=0, ge=0)
    gpu_minutes: float = Field(default=0, ge=0)
    model_cost_usd: float = Field(default=0, ge=0)
    infra_cost_usd: float = Field(default=0, ge=0)


class ReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    passed: bool
    exit_code: int
    duration_sec: float = Field(ge=0)
    assertions: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    faults: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    failure: FailureKind | None = None
    message: str = ""


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: str = "0.1"
    trial_id: str
    task_id: str
    state: TrialState = TrialState.PENDING
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    events: list[TrialEvent] = Field(default_factory=list)
    replays: list[ReplayResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    failure: FailureKind | None = None
    failure_detail: str = ""

    def transition(self, state: TrialState, detail: str = "") -> None:
        self.state = state
        self.events.append(TrialEvent(state=state, detail=detail))
