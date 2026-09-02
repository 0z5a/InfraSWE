from __future__ import annotations

import tomllib
from hashlib import sha256
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

from infraswe.models.training import (
    TrainingImplementationConfig,
    TrainingProfilingConfig,
    TrainingTrainerConfig,
    TrainingWorkloadConfig,
)

LegacyKernelTaskKind = Literal[
    "kernel-micro",
    "kernel-library",
    "kernel-integrated",
    "kernel-distributed",
]
V04TaskKind = Literal[
    "benchmark-replacement",
    "backend-conformance",
    "backend-repair",
    "cross-backend-port",
    "distributed-integration",
    "kernel-frontier",
    "training-workflow",
]
TaskKind = Literal[
    "infra-regression",
    "kernel-micro",
    "kernel-library",
    "kernel-integrated",
    "kernel-distributed",
    "benchmark-replacement",
    "backend-conformance",
    "backend-repair",
    "cross-backend-port",
    "distributed-integration",
    "kernel-frontier",
    "training-workflow",
]
ImplementationLevel = Literal["micro", "library", "integrated", "distributed", "utility"]

LEGACY_KERNEL_TASK_KINDS = frozenset(
    {"kernel-micro", "kernel-library", "kernel-integrated", "kernel-distributed"}
)
V04_TASK_KINDS = frozenset(
    {
        "benchmark-replacement",
        "backend-conformance",
        "backend-repair",
        "cross-backend-port",
        "distributed-integration",
        "kernel-frontier",
        "training-workflow",
    }
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
    kind: TaskKind | None = None
    implementation_level: ImplementationLevel | None = None


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
    capability_probe: bool = False


class BudgetConfig(ProtocolModel):
    agent_timeout_sec: PositiveInt = 900
    verifier_timeout_sec: PositiveInt = 300
    gpu_minutes: float = Field(default=0, ge=0)
    max_model_cost_usd: float = Field(default=0, ge=0)
    max_infra_cost_usd: float = Field(default=0, ge=0)


class ReplayConfig(ProtocolModel):
    count: int = Field(default=3, ge=1, le=10)
    require_all: bool = True
    fresh_process: bool = True


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
    profile: LegacyKernelTaskKind | V04TaskKind
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
    deployability_template: Literal["deployability-v0.4"] | None = None
    cell_artifact_template: (
        Literal[
            "cell-artifact-mixed-v0.4",
            "cell-artifact-memory-v0.4",
            "cell-artifact-compute-v0.4",
            "cell-artifact-distributed-v0.4",
        ]
        | None
    ) = None
    absolute_latency_global_ranking: Literal["forbidden"] | None = None
    raw_peak_performance_in_cross_cell_score: Literal[False] = False


class SemanticContractRef(ProtocolModel):
    path: str
    sha256: str


class BackendProfileRef(ProtocolModel):
    id: str
    adapter: str
    benchmark_cell_id: str


class CertificationConfig(ProtocolModel):
    hidden_correctness_required: float = Field(default=1.0, ge=0, le=1)
    silent_fallback_rate_max: float = Field(default=0.0, ge=0, le=1)
    fresh_replays: int = Field(default=7, ge=1, le=10)
    require_all: bool = True


class ConcurrencyConfig(ProtocolModel):
    protocol_id: str
    reference_saturation_anchor: str
    load_ratios: list[float] = Field(min_length=1)
    minimum_completed_requests_per_cell: PositiveInt = 1000
    burst_or_soak: Literal["required", "optional", "not_applicable"] = "required"
    request_mix_sha256: str

    @model_validator(mode="after")
    def load_ratios_are_frozen_and_increasing(self) -> ConcurrencyConfig:
        if any(ratio <= 0 for ratio in self.load_ratios):
            raise ValueError("concurrency load ratios must be positive")
        if any(left >= right for left, right in pairwise(self.load_ratios)):
            raise ValueError("concurrency load ratios must be strictly increasing")
        return self


class ReuseContractConfig(ProtocolModel):
    sha256: str
    expected_variant_budget: PositiveInt
    max_variant_budget: PositiveInt
    specialization_dimensions: list[str] = Field(default_factory=list)
    require_case_to_implementation_map: bool = True
    compile_cache_observability: Literal[
        "required", "required-if-applicable", "optional", "not_applicable"
    ] = "required-if-applicable"

    @model_validator(mode="after")
    def variant_budget_is_coherent(self) -> ReuseContractConfig:
        if self.expected_variant_budget > self.max_variant_budget:
            raise ValueError("expected_variant_budget cannot exceed max_variant_budget")
        return self


class MaintainabilityConfig(ProtocolModel):
    probe_set_sha256: str
    require_capability_contract: bool = True
    require_structured_failure_codes: bool = True
    build_profiles: list[str] = Field(min_length=1)


class EfficiencyConfig(ProtocolModel):
    work_model_id: str
    regime: Literal[
        "launch-bound",
        "memory-bound",
        "compute-bound",
        "mixed",
        "communication-bound",
        "utility/no-efficiency-score",
    ]
    work_model_confidence_min: Literal["low", "medium", "high"] = "high"
    calibration_manifest_sha256: str | None = None
    traffic_amplification_budget: float = Field(default=1.0, ge=1)


EvidenceGrade = Literal[
    "E0-runtime",
    "E1-framework",
    "E2-system-trace",
    "E3-kernel-counter",
    "E4-sealed",
]


class EvidenceConfig(ProtocolModel):
    minimum_grade_for_deployability: EvidenceGrade = "E2-system-trace"
    minimum_grade_for_cell_efficiency: EvidenceGrade = "E3-kernel-counter"
    collectors: list[str] = Field(min_length=1)


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
        "operator",
        "fusion",
        "collective",
        "expert-parallel",
        "setup-utility",
        "system-pipeline",
        "build-portability",
        "memory-ordering",
    ]
    mechanism_policy: Literal[
        "strict-native",
        "approved-compositional",
        "framework-integrated",
        "direct-communication",
        "capability-gated",
    ]
    measurement_domain: Literal["device-time", "host-e2e", "request-e2e", "step-e2e", "system-e2e"]
    capabilities: list[str] = Field(default_factory=list)
    public_cases: KernelCaseSet | None = None
    dev_private_cases: KernelCaseSet | None = None
    hidden_cases: KernelCaseSet | None = None
    numerics: dict[str, Any] = Field(default_factory=dict)
    resource_contract: dict[str, Any] = Field(default_factory=dict)
    anti_hack_policy: dict[str, Any] = Field(default_factory=dict)


class TaskPackage(ProtocolModel):
    schema_version: Literal["0.1", "0.3", "0.4"] = "0.1"
    task: TaskMetadata
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    gates: GateConfig = Field(default_factory=GateConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    kernel_contract: KernelContract | None = None
    semantic_contract: SemanticContractRef | None = None
    backend_profile: BackendProfileRef | None = None
    certification: CertificationConfig | None = None
    concurrency: ConcurrencyConfig | None = None
    reuse_contract: ReuseContractConfig | None = None
    maintainability: MaintainabilityConfig | None = None
    efficiency: EfficiencyConfig | None = None
    evidence: EvidenceConfig | None = None
    workload: TrainingWorkloadConfig | None = None
    trainer: TrainingTrainerConfig | None = None
    implementation: TrainingImplementationConfig | None = None
    profiling: TrainingProfilingConfig | None = None
    _package_dir: Path = PrivateAttr()

    @model_validator(mode="after")
    def kernel_envelope_is_coherent(self) -> TaskPackage:
        kind = self.task.kind
        is_legacy_kernel = kind in LEGACY_KERNEL_TASK_KINDS
        is_v04 = kind in V04_TASK_KINDS
        is_training = kind == "training-workflow"
        has_kernel_contract = self.kernel_contract is not None
        has_kernel_scoring = self.scoring.kernel is not None
        if is_legacy_kernel and (self.schema_version != "0.3" or not has_kernel_contract):
            raise ValueError("kernel tasks require schema_version 0.3 and kernel_contract")
        if is_legacy_kernel and not has_kernel_scoring:
            raise ValueError("kernel tasks require scoring.kernel")
        if has_kernel_contract != has_kernel_scoring:
            raise ValueError("kernel_contract and scoring.kernel must be declared together")
        if not (is_legacy_kernel or is_v04) and has_kernel_contract:
            raise ValueError("kernel scoring fields require an eligible task kind")
        if has_kernel_scoring:
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
        if self.schema_version == "0.4":
            if not is_v04:
                raise ValueError("schema_version 0.4 requires a v0.4 task kind")
            if self.task.implementation_level is None:
                raise ValueError("schema_version 0.4 requires task.implementation_level")
            required_contracts = {
                "semantic_contract": self.semantic_contract,
                "backend_profile": self.backend_profile,
                "certification": self.certification,
                "concurrency": self.concurrency,
                "reuse_contract": self.reuse_contract,
                "maintainability": self.maintainability,
                "evidence": self.evidence,
            }
            missing = [name for name, value in required_contracts.items() if value is None]
            if missing:
                raise ValueError("schema_version 0.4 missing contracts: " + ", ".join(missing))
            if self.scoring.deployability_template != "deployability-v0.4":
                raise ValueError("schema_version 0.4 requires deployability-v0.4")
            if self.scoring.absolute_latency_global_ranking != "forbidden":
                raise ValueError("schema_version 0.4 forbids absolute latency global ranking")
            if self.certification and self.replay.count != self.certification.fresh_replays:
                raise ValueError("replay.count must match certification fresh_replays")
            if self.efficiency is None and self.scoring.cell_artifact_template is not None:
                raise ValueError("cell artifact scoring requires an efficiency contract")
        elif is_v04:
            raise ValueError("v0.4 task kinds require schema_version 0.4")

        training_fields = {
            "workload": self.workload,
            "trainer": self.trainer,
            "implementation": self.implementation,
            "profiling": self.profiling,
        }
        if is_training:
            missing_training = [name for name, value in training_fields.items() if value is None]
            if missing_training:
                raise ValueError("training-workflow missing fields: " + ", ".join(missing_training))
            if self.task.track != "training":
                raise ValueError("training-workflow requires task.track=training")
            if has_kernel_contract or has_kernel_scoring:
                raise ValueError("training-workflow cannot use the legacy kernel score envelope")
            if self.replay.count < 5:
                raise ValueError("official training tasks require at least five fresh replays")
            if not self.replay.fresh_process:
                raise ValueError("training replays must use fresh processes")
            if not self.environment.capability_probe:
                raise ValueError("training-workflow requires an environment capability probe")
            if self.environment.gpu_count and not self.environment.exclusive_gpu_lease:
                raise ValueError("GPU training tasks require an exclusive GPU lease")
            if self.environment.gpu_count and self.environment.mps != "disabled":
                raise ValueError("GPU training tasks require MPS disabled")
            if self.profiling and self.profiling.minimum_grade_for_deployability not in {
                "G3",
                "G4",
            }:
                raise ValueError(
                    "v0.4 requires a G3 system trace or stronger for training deployability"
                )
            if self.workload and self.semantic_contract:
                workload_contract = self.workload.semantic_contract
                if (
                    workload_contract.path != self.semantic_contract.path
                    or workload_contract.sha256 != self.semantic_contract.sha256
                ):
                    raise ValueError(
                        "training workload and v0.4 semantic contract references must match"
                    )
        elif any(value is not None for value in training_fields.values()):
            raise ValueError("training fields require task.kind=training-workflow")
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
        if self.workload:
            training_refs = {
                "training semantic contract": self.workload.semantic_contract,
                "training work model": self.workload.work_model,
            }
            for label, reference in training_refs.items():
                path = self.resolve(reference.path)
                if not path.is_file():
                    errors.append(f"missing {label}: {path.relative_to(self.package_dir)}")
                    continue
                observed = "sha256:" + sha256(path.read_bytes()).hexdigest()
                if observed != reference.sha256.lower():
                    errors.append(
                        f"{label} digest mismatch: expected {reference.sha256}, observed {observed}"
                    )
        return errors
