from __future__ import annotations

import tomllib
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    model_validator,
)


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TaskMetadata(ProtocolModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    title: str
    track: str
    repository: str
    base_commit: str
    level: Literal["L1", "L2", "L3", "L4", "K0", "K1", "K2", "K3", "K4", "K5"] = "L1"
    kind: (
        Literal[
            "infra-regression",
            "kernel-micro",
            "kernel-library",
            "kernel-integrated",
            "kernel-distributed",
        ]
        | None
    ) = None


class EnvironmentConfig(ProtocolModel):
    profile: str = "cpu-small"
    agent_mode: Literal["local", "docker", "vm", "kubernetes"] = "docker"
    verifier_mode: Literal["separate"] = "separate"
    network: Literal["deny", "allowlist"] = "deny"
    agent_image: str = "python:3.12-slim"
    verifier_image: str = "python:3.12-slim"
    gpu_count: int = Field(default=0, ge=0, le=16)
    shm_size: str = Field(default="256m", pattern=r"^[1-9][0-9]*(?:[kKmMgG])$")
    exclusive_gpu_lease: bool = False
    mps: Literal["disabled", "enabled", "not-applicable"] = "not-applicable"


class BudgetConfig(ProtocolModel):
    agent_timeout_sec: PositiveInt = 900
    verifier_timeout_sec: PositiveInt = 300
    gpu_minutes: float = Field(default=0, ge=0)
    max_model_cost_usd: float = Field(default=0, ge=0)
    max_infra_cost_usd: float = Field(default=0, ge=0)


class ReplayConfig(ProtocolModel):
    count: int = Field(default=3, ge=1, le=10)
    require_all: bool = True


class ArtifactConfig(ProtocolModel):
    config_paths: list[str] = Field(default_factory=list)
    require_clean_patch: bool = True


class GateConfig(ProtocolModel):
    forbid_test_modification: bool = True
    forbid_credential_access: bool = True
    forbid_silent_fallback: bool = True
    forbid_data_corruption: bool = True
    forbid_resource_leak: bool = True


class ExecutionConfig(ProtocolModel):
    repo: str = "fixture/repo"
    instruction: str = "instruction.md"
    solution_command: list[str] = Field(default_factory=lambda: ["python", "solution/solve.py"])
    verifier_command: list[str] = Field(default_factory=lambda: ["python", "tests/verify.py"])
    allowed_patch_paths: list[str] = Field(default_factory=lambda: ["**"])

    @model_validator(mode="after")
    def commands_are_not_empty(self) -> ExecutionConfig:
        if not self.solution_command or not self.verifier_command:
            raise ValueError("solution_command and verifier_command must be non-empty")
        return self


class KernelRoleRequirements(ProtocolModel):
    certification_roles: list[str] = Field(default_factory=list)
    artifact_roles: list[str] = Field(default_factory=list)
    search_roles: list[str] = Field(default_factory=list)
    fresh_replays: int = Field(default=3, ge=1, le=10)
    required_passes: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def required_passes_fit_replays(self) -> KernelRoleRequirements:
        if self.required_passes > self.fresh_replays:
            raise ValueError("required_passes cannot exceed fresh_replays")
        return self


class KernelRoleGraphRef(ProtocolModel):
    path: str
    sha256: str


class KernelPerformanceConfig(ProtocolModel):
    primary: str = "anchor_score"
    report: list[str] = Field(default_factory=list)
    scoring_baseline_sha256: str
    anchor_manifest_sha256: str
    min_headroom: float = Field(default=1.10, gt=1)
    beyond_anchor_tolerance: float = Field(default=0.03, ge=0)
    case_aggregation: str = "weighted-certified-only"
    sampling_plan_sha256: str


class KernelSearchConfig(ProtocolModel):
    enabled: bool = False
    primary_budget_axis: Literal["leased_device_seconds", "wall_seconds", "tokens", "usd"] = (
        "leased_device_seconds"
    )
    max_candidates: PositiveInt = 128
    sealed_checkpoint_fractions: list[float] = Field(default_factory=list)
    dev_private_query_limit: int = Field(default=0, ge=0)
    sealed_feedback_during_episode: Literal[False] = False
    publish_axes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def checkpoints_are_strictly_increasing(self) -> KernelSearchConfig:
        points = self.sealed_checkpoint_fractions
        if points and (points[-1] != 1.0 or any(not 0 < point <= 1 for point in points)):
            raise ValueError("sealed checkpoints must be in (0, 1] and end at 1.0")
        if any(left >= right for left, right in pairwise(points)):
            raise ValueError("sealed checkpoints must be strictly increasing")
        return self


class KernelScoringConfig(ProtocolModel):
    benchmark_cell_id: str
    leaderboard_season: str
    formula_version: str
    formula_origin: str = "infraswe-profile-template"
    component_formula_origins: dict[str, str] = Field(default_factory=dict)
    formula_parameters_sha256: str
    profile: Literal["kernel-micro", "kernel-library", "kernel-integrated", "kernel-distributed"]
    role_graph: KernelRoleGraphRef
    role_requirements: KernelRoleRequirements
    performance: KernelPerformanceConfig
    search: KernelSearchConfig = Field(default_factory=KernelSearchConfig)


class ScoringConfig(ProtocolModel):
    reference_wall_time_sec: PositiveFloat = 60.0
    reference_cost_usd: float = Field(default=0.0, ge=0)
    slo_metric: str = "slo_goodput_ratio"
    resource_metric: str = "resource_efficiency_ratio"
    kernel: KernelScoringConfig | None = None


class KernelCaseSet(ProtocolModel):
    path: str | None = None
    count: int = Field(ge=0)
    digest: str | None = None
    provenance: str | None = None
    feedback_policy: str | None = None


class KernelContract(ProtocolModel):
    entrypoint: str
    reference_entrypoint: str
    target_arch: list[str]
    allowed_backends: list[str]
    artifact_surface: Literal[
        "device-kernel", "dispatcher", "framework-patch", "communication-program", "runtime-utility"
    ]
    execution_scope: Literal["single-device", "multi-device-intranode", "multi-node"]
    workload_semantics: Literal[
        "operator", "fusion", "collective", "expert-parallel", "setup-utility"
    ]
    mechanism_policy: Literal[
        "strict-native", "approved-compositional", "framework-integrated", "direct-communication"
    ]
    measurement_domain: Literal["device-time", "host-e2e", "request-e2e", "step-e2e"]
    capabilities: list[
        Literal[
            "dispatch",
            "integration",
            "communication",
            "overlap",
            "topology",
            "graph-capture",
        ]
    ] = Field(default_factory=list)
    public_cases: KernelCaseSet | None = None
    dev_private_cases: KernelCaseSet | None = None
    hidden_cases: KernelCaseSet | None = None
    numerics: dict[str, Any] = Field(default_factory=dict)
    resource_contract: dict[str, Any] = Field(default_factory=dict)
    anti_hack_policy: dict[str, Any] = Field(default_factory=dict)


class TaskPackage(ProtocolModel):
    schema_version: Literal["0.1", "0.3"] = "0.1"
    task: TaskMetadata
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    gates: GateConfig = Field(default_factory=GateConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    kernel_contract: KernelContract | None = None
    _package_dir: Path = PrivateAttr()

    @model_validator(mode="after")
    def kernel_envelope_is_coherent(self) -> TaskPackage:
        is_kernel = bool(self.task.kind and self.task.kind.startswith("kernel-"))
        if is_kernel and (self.schema_version != "0.3" or self.kernel_contract is None):
            raise ValueError("kernel tasks require schema_version 0.3 and kernel_contract")
        if is_kernel and self.scoring.kernel is None:
            raise ValueError("kernel tasks require scoring.kernel")
        if not is_kernel and (self.kernel_contract is not None or self.scoring.kernel is not None):
            raise ValueError("kernel scoring fields require task.kind=kernel-*")
        if is_kernel and self.scoring.kernel is not None:
            if self.scoring.kernel.profile != self.task.kind:
                raise ValueError("kernel scoring profile must match task.kind")
            requirements = self.scoring.kernel.role_requirements
            if self.replay.count != requirements.fresh_replays:
                raise ValueError("replay.count must match kernel fresh_replays")
            if self.replay.require_all and requirements.required_passes != self.replay.count:
                raise ValueError("require_all tasks require one pass per fresh replay")
            if self.environment.gpu_count and not self.environment.exclusive_gpu_lease:
                raise ValueError("GPU kernel tasks require an exclusive GPU lease")
            if self.environment.gpu_count and self.environment.mps != "disabled":
                raise ValueError("GPU kernel tasks require MPS disabled")
        return self

    @classmethod
    def load(cls, path: str | Path) -> TaskPackage:
        source = Path(path).expanduser().resolve()
        config_path = source / "task.toml" if source.is_dir() else source
        if not config_path.is_file():
            raise FileNotFoundError(f"task config not found: {config_path}")
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        package = cls.model_validate(data)
        package._package_dir = config_path.parent
        return package

    @property
    def package_dir(self) -> Path:
        return self._package_dir

    def resolve(self, relative: str) -> Path:
        candidate = (self.package_dir / relative).resolve()
        try:
            candidate.relative_to(self.package_dir)
        except ValueError as error:
            raise ValueError(f"task path escapes package: {relative}") from error
        return candidate

    def validate_layout(self) -> list[str]:
        errors: list[str] = []
        required = {
            "repo": self.resolve(self.execution.repo),
            "instruction": self.resolve(self.execution.instruction),
            "verifier": self.resolve(self.execution.verifier_command[-1]),
            "solution": self.resolve(self.execution.solution_command[-1]),
        }
        for label, path in required.items():
            if not path.exists():
                errors.append(f"missing {label}: {path.relative_to(self.package_dir)}")
        if required["repo"].exists() and not required["repo"].is_dir():
            errors.append("execution.repo must be a directory")
        return errors
