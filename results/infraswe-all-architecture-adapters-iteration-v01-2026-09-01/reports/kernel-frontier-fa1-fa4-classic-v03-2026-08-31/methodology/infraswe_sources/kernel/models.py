from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Authority(StrEnum):
    GATE = "gate"
    METRIC = "metric"
    SCORE = "score"
    AUDIT = "audit"
    ADVISORY = "advisory"


class Scope(StrEnum):
    TRIAL = "trial"
    REPLAY = "replay"
    CASE = "case"
    REPLAY_CASE = "replay-case"
    SEARCH = "search"


class RoleStatus(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class Disposition(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    QUARANTINED = "quarantined"
    NOT_APPLICABLE = "not_applicable"


class KernelModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetricValue(KernelModel):
    value: float | int | str | bool | None
    unit: str = Field(min_length=1)
    statistic: str = Field(min_length=1)
    population: str = Field(min_length=1)


class FailureCode(KernelModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*(?::.*)?$")
    severity: str
    owner: str
    retryable: bool
    evidence_sha256: str | None = None


class EvidenceRef(KernelModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str

    @model_validator(mode="after")
    def path_is_trial_relative(self) -> EvidenceRef:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence path must be trial-root relative")
        return self


class RoleIdentity(KernelModel):
    task_package_sha256: str
    candidate_source_sha256: str
    build_artifact_sha256: str
    role_graph_sha256: str
    evaluator_sha256: str
    hardware_class_sha256: str
    environment_contract_sha256: str
    execution_environment_sha256: str


class RoleInput(KernelModel):
    role_instance_id: str
    result_sha256: str


class RoleResult(KernelModel):
    schema_version: str = Field(default="0.3", pattern=r"^0\.3$")
    role_id: str
    role_instance_id: str
    authority: Authority
    scope: Scope
    status: RoleStatus
    verdict: Verdict
    disposition: Disposition
    profile: str
    replay_index: int | None = Field(default=None, ge=1)
    case_id: str | None = None
    attempt_index: int = Field(default=0, ge=0)
    identity: RoleIdentity
    inputs: list[RoleInput] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0, le=1)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    assertions: dict[str, bool] = Field(default_factory=dict)
    failure_codes: list[FailureCode] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    result_sha256: str = ""
    signature: str = ""
    message: str = ""

    @model_validator(mode="after")
    def enforce_single_authority_contract(self) -> RoleResult:
        if self.authority == Authority.SCORE and self.score is None:
            raise ValueError("score authority requires score")
        if self.authority != Authority.SCORE and self.score is not None:
            raise ValueError("only score authority may set score")
        if self.status != RoleStatus.COMPLETED and self.disposition == Disposition.VALID:
            raise ValueError("incomplete role result cannot have valid disposition")
        if self.disposition == Disposition.NOT_APPLICABLE:
            if self.verdict != Verdict.NOT_APPLICABLE:
                raise ValueError("not-applicable disposition requires not-applicable verdict")
        elif self.verdict == Verdict.NOT_APPLICABLE:
            raise ValueError("not-applicable verdict requires not-applicable disposition")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        return self


@dataclass(frozen=True, order=True)
class RoleKey:
    role_id: str
    scope: str
    replay_index: int | None
    case_id: str | None
    attempt_index: int

    @classmethod
    def from_result(cls, result: RoleResult) -> RoleKey:
        return cls(
            role_id=result.role_id,
            scope=result.scope.value,
            replay_index=result.replay_index,
            case_id=result.case_id,
            attempt_index=result.attempt_index,
        )


class KernelAggregate(KernelModel):
    certified: bool
    verdict: str
    disposition: str
    artifact_status: str
    artifact_100: float | None = Field(default=None, ge=0, le=100)
    leaderboard_effective_artifact_100: float | None = Field(default=None, ge=0, le=100)
    components: dict[str, float] = Field(default_factory=dict)
    failure_codes: list[str] = Field(default_factory=list)


class AnchorCaseResult(KernelModel):
    status: str
    scoring_baseline_latency_us: float = Field(gt=0)
    candidate_latency_us: float = Field(gt=0)
    anchor_latency_us: float = Field(gt=0)
    speedup_vs_scoring_baseline_raw: float = Field(gt=0)
    anchor_efficiency_raw: float = Field(gt=0)
    anchor_score_raw: float | None = None
    failure_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
