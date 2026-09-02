from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from infraswe.models.score import ScoreResult

_SIMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_NAMESPACED_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,63}$")
_BUILTIN_IDS = frozenset(
    {
        "supervised",
        "online-rl",
        "optimizer",
        "training-kernel",
        "sft",
        "grpo",
        "dapo",
        "adamw",
        "muon",
        "muon-plus-adamw",
        "sft-contract",
        "grpo-contract",
        "grpo-online",
        "dapo-loss-contract",
        "dapo-recipe-contract",
        "dapo-online",
        "muon-contract",
        "matrix",
        "native-pytorch",
        "hf-transformers",
        "trl",
        "verl",
        "torchtune",
        "axolotl",
        "megatron-lm",
        "megatron-core",
        "torchrun",
        "accelerate",
        "deepspeed",
        "ray",
        "single",
        "ddp",
        "fsdp2",
        "zero2",
        "zero3",
        "tp",
        "pp",
        "cp",
        "ep",
        "none",
        "transformers",
        "vllm",
        "sglang",
        "eager",
        "torch-compile",
        "framework-compiled",
        "inductor",
        "pytorch",
        "triton",
        "cuda-cpp",
        "hip-cpp",
        "backend-native",
        "cuda",
        "rocm",
        "cann",
        "cpu",
        "generated",
        "pytorch-eager",
        "pytorch-compile",
        "triton-custom",
        "cuda-extension",
        "shared-object",
        "ptx",
        "cubin",
        "hsaco",
        "packed-variable",
        "valid-target-token-mean",
        "global-l2",
        "cosine",
        "custom",
    }
)


def validate_extensible_id(value: str) -> str:
    """Accept stable built-in ids and namespaced extension ids.

    Core schemas intentionally do not enumerate frameworks. Third-party ids use
    ``org.example/adapter-name`` so adding one never requires a scorer change.
    """

    if (value in _BUILTIN_IDS and _SIMPLE_ID.fullmatch(value)) or _NAMESPACED_ID.fullmatch(value):
        return value
    raise ValueError("id must be a lowercase built-in id or a namespaced custom id")


class TrainingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrainingArtifactRef(TrainingModel):
    path: str
    sha256: str = Field(pattern=r"^sha256:[0-9a-fA-F]{8,64}$")


class TrainingWorkloadConfig(TrainingModel):
    family: str
    algorithm: str
    optimizer: str
    semantic_contract: TrainingArtifactRef
    work_model: TrainingArtifactRef
    certification_scope: str

    @field_validator("family", "algorithm", "optimizer", "certification_scope")
    @classmethod
    def ids_are_extensible(cls, value: str) -> str:
        return validate_extensible_id(value)


class TrainingTrainerConfig(TrainingModel):
    adapter: str = "matrix"
    required: list[str] = Field(min_length=1)
    optional: list[str] = Field(default_factory=list)
    launcher: str = "custom"
    distributed: str = "single"
    rollout_engine: str = "none"

    @field_validator("adapter", "launcher", "distributed", "rollout_engine")
    @classmethod
    def scalar_ids_are_extensible(cls, value: str) -> str:
        return validate_extensible_id(value)

    @field_validator("required", "optional")
    @classmethod
    def list_ids_are_extensible(cls, values: list[str]) -> list[str]:
        checked = [validate_extensible_id(value) for value in values]
        if len(set(checked)) != len(checked):
            raise ValueError("adapter lists cannot contain duplicates")
        return checked

    @model_validator(mode="after")
    def required_and_optional_are_disjoint(self) -> TrainingTrainerConfig:
        overlap = sorted(set(self.required) & set(self.optional))
        if overlap:
            raise ValueError("required and optional adapters overlap: " + ", ".join(overlap))
        return self


class TrainingImplementationConfig(TrainingModel):
    candidates: list[str] = Field(min_length=1)
    graph_mode: str = "eager"
    graph_compiler: str = "none"
    kernel_language: str = "pytorch"
    device_runtime: str = "cuda"
    artifact_formats: list[str] = Field(default_factory=list)
    forbid_silent_fallback: Literal[True] = True

    @field_validator("graph_mode", "graph_compiler", "kernel_language", "device_runtime")
    @classmethod
    def implementation_ids_are_extensible(cls, value: str) -> str:
        return validate_extensible_id(value)

    @field_validator("candidates", "artifact_formats")
    @classmethod
    def candidate_ids_are_extensible(cls, values: list[str]) -> list[str]:
        checked = [validate_extensible_id(value) for value in values]
        if len(set(checked)) != len(checked):
            raise ValueError("implementation lists cannot contain duplicates")
        return checked


TrainingProfilerGrade = Literal["G0", "G1", "G2", "G3", "G4"]


class TrainingProfilingConfig(TrainingModel):
    minimum_grade_for_deployability: TrainingProfilerGrade = "G3"
    concurrency_grade: TrainingProfilerGrade = "G3"
    cell_efficiency_grade: Literal["G4"] = "G4"
    authoritative_timing: Literal["separate-unprofiled-run"] = "separate-unprofiled-run"
    profiled_timing_authoritative: Literal[False] = False


class SeedBundle(TrainingModel):
    model: int = Field(ge=0)
    data: int = Field(ge=0)
    sampling: int = Field(ge=0)
    dropout: int = Field(ge=0)


class NormalizedTrainingConfig(TrainingModel):
    global_batch_tokens: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    sequence_length_policy: str
    precision: Literal["fp32", "bf16", "fp16", "tf32", "fp8"]
    loss_reduction: str
    gradient_clipping: str
    optimizer: str
    learning_rate_schedule: str
    activation_checkpointing: bool
    seed_bundle: SeedBundle

    @field_validator(
        "sequence_length_policy",
        "loss_reduction",
        "gradient_clipping",
        "optimizer",
        "learning_rate_schedule",
    )
    @classmethod
    def config_ids_are_extensible(cls, value: str) -> str:
        return validate_extensible_id(value)


class TensorComparisonEvidence(TrainingModel):
    reference: list[float] = Field(min_length=1)
    candidate: list[float] = Field(min_length=1)
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    quantity: str
    dtype: str
    sequence_bucket: int = Field(gt=0)
    distributed_mode: str = "single"


class SFTSemanticEvidence(TrainingModel):
    token_losses: list[float] = Field(min_length=1)
    target_mask: list[bool] = Field(min_length=1)
    observed_loss: float
    observed_denominator: int = Field(ge=0)
    packed_sample_ids: list[str] = Field(min_length=1)
    observed_attention_edges: list[tuple[int, int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def token_vectors_have_equal_length(self) -> SFTSemanticEvidence:
        lengths = {
            len(self.token_losses),
            len(self.target_mask),
            len(self.packed_sample_ids),
        }
        if len(lengths) != 1:
            raise ValueError("SFT token losses, target mask, and sample ids must align")
        return self


class GRPORolloutSample(TrainingModel):
    prompt_id: str
    group_id: str
    sample_id: str
    policy_version: int = Field(ge=0)
    train_policy_version: int = Field(ge=0)
    sampling_seed: int = Field(ge=0)
    token_ids: list[int] = Field(min_length=1)
    old_log_probs: list[float] = Field(min_length=1)
    reward: float
    observed_advantage: float
    valid_token_mask: list[bool] = Field(min_length=1)

    @model_validator(mode="after")
    def token_vectors_have_equal_length(self) -> GRPORolloutSample:
        if not (len(self.token_ids) == len(self.old_log_probs) == len(self.valid_token_mask)):
            raise ValueError("rollout token ids, log-probs, and valid-token mask must align")
        return self


class GRPOSemanticEvidence(TrainingModel):
    samples: list[GRPORolloutSample] = Field(min_length=2)
    expected_group_size: int = Field(gt=1)
    advantage_epsilon: float = Field(default=1e-8, gt=0)
    advantage_tolerance: float = Field(default=1e-6, ge=0)
    max_policy_staleness: int = Field(default=1, ge=0)
    kl_definition: str
    kl_sign: Literal["penalty-positive"] = "penalty-positive"


class DAPOSemanticEvidence(TrainingModel):
    token_level_policy_gradient: bool
    asymmetric_clip_higher: bool
    dynamic_sampling: bool
    overlong_policy_exact: bool
    soft_overlong_punishment_exact: bool
    reward_aggregation_exact: bool


class ParameterGroupRecord(TrainingModel):
    name: str
    shape: list[int] = Field(min_length=1)
    semantic_role: Literal["hidden-matrix", "embedding", "output-head", "norm", "bias", "other"]
    group_id: str
    optimizer: str
    state_shape: list[int] = Field(default_factory=list)
    update_count: int = Field(default=1, ge=0)

    @field_validator("optimizer")
    @classmethod
    def optimizer_id_is_extensible(cls, value: str) -> str:
        return validate_extensible_id(value)


class MuonSemanticEvidence(TrainingModel):
    trainable_parameters: list[str] = Field(min_length=1)
    parameter_groups: list[ParameterGroupRecord] = Field(min_length=1)
    newton_schulz_iterations: int = Field(gt=0)
    coefficients_id: str
    normalization_id: str
    epsilon: float = Field(gt=0)
    update_comparison: TensorComparisonEvidence


class CheckpointEvidence(TrainingModel):
    saved_components: list[str] = Field(min_length=1)
    restored_components: list[str] = Field(min_length=1)
    next_step_comparison: TensorComparisonEvidence
    rng_streams_restored: list[str] = Field(default_factory=list)
    fresh_process: bool


class RuntimeSafetyEvidence(TrainingModel):
    loss_values: list[float] = Field(min_length=1)
    silent_fallback_count: int = Field(ge=0)
    declared_fallbacks: list[str] = Field(default_factory=list)
    deadlock: bool = False
    watchdog_passed: bool = True
    resource_leaks: list[str] = Field(default_factory=list)
    half_batch_updates: int = Field(default=0, ge=0)


class IntegrityEvidence(TrainingModel):
    manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-fA-F]{8,64}$")
    raw_evidence_digests: list[str] = Field(min_length=1)
    timeline_consistent: bool
    versions_exact: bool

    @field_validator("raw_evidence_digests")
    @classmethod
    def evidence_digests_are_sha256(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"sha256:[0-9a-fA-F]{8,64}", value):
                raise ValueError("raw evidence digests must use sha256:<hex>")
        return values


class TrainingEvidenceBundle(TrainingModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    algorithm: str
    optimizer: str
    certification_scope: str
    adapter_id: str
    framework_version: str
    framework_stack_id: str
    hardware_cell_id: str
    implementation_bundle_id: str
    evidence_manifest_sha256: str = Field(pattern=r"^sha256:[0-9a-fA-F]{8,64}$")
    normalized_config: NormalizedTrainingConfig
    forward: TensorComparisonEvidence | None = None
    backward: TensorComparisonEvidence | None = None
    optimizer_update: TensorComparisonEvidence | None = None
    checkpoint: CheckpointEvidence | None = None
    runtime: RuntimeSafetyEvidence | None = None
    integrity: IntegrityEvidence | None = None
    sft: SFTSemanticEvidence | None = None
    grpo: GRPOSemanticEvidence | None = None
    dapo: DAPOSemanticEvidence | None = None
    muon: MuonSemanticEvidence | None = None

    @field_validator("algorithm", "optimizer", "certification_scope", "adapter_id")
    @classmethod
    def bundle_ids_are_extensible(cls, value: str) -> str:
        return validate_extensible_id(value)


class TrainingGate(TrainingModel):
    status: Literal["pass", "fail", "unresolved", "not_applicable"]
    failure_codes: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def failure_codes_match_status(self) -> TrainingGate:
        if self.status == "fail" and not self.failure_codes:
            raise ValueError("failed training gates require a failure code")
        return self


class TrainingCertification(TrainingModel):
    schema_version: Literal["0.1"] = "0.1"
    status: Literal["pass", "fail", "unresolved"]
    algorithm: str
    certification_scope: str
    gates: dict[str, TrainingGate]
    failure_codes: list[str] = Field(default_factory=list)
    evidence_manifest_sha256: str

    @model_validator(mode="after")
    def status_matches_gates(self) -> TrainingCertification:
        statuses = {gate.status for gate in self.gates.values()}
        expected = (
            "fail" if "fail" in statuses else ("unresolved" if "unresolved" in statuses else "pass")
        )
        if self.status != expected:
            raise ValueError("training certification status does not match its gates")
        if self.status == "fail" and not self.failure_codes:
            raise ValueError("failed training certification requires failure codes")
        return self


class TrainingReuseEvidence(TrainingModel):
    shape_coverage: float = Field(ge=0, le=1)
    dtype_coverage: float = Field(ge=0, le=1)
    layout_coverage: float = Field(ge=0, le=1)
    observed_variants: int = Field(ge=0)
    expected_variant_budget: int = Field(gt=0)
    max_variant_budget: int = Field(gt=0)
    dispatcher_reuse: float = Field(ge=0, le=1)
    compile_cache_reuse: float = Field(ge=0, le=1)
    portability_reuse: float = Field(ge=0, le=1)
    silent_fallback_rate: float = Field(default=0, ge=0, le=1)


class TrainingMaintainabilityEvidence(TrainingModel):
    contract: float = Field(ge=0, le=1)
    locality: float = Field(ge=0, le=1)
    tests: float = Field(ge=0, le=1)
    build_reproducibility: float = Field(ge=0, le=1)


class TrainingCellEfficiencyEvidence(TrainingModel):
    work_model_id: str
    regime: Literal[
        "launch-bound",
        "memory-bound",
        "compute-bound",
        "mixed",
        "communication-bound",
        "utility/no-efficiency-score",
    ]
    work_model: dict[str, Any]
    calibration: dict[str, Any]
    candidate_time_seconds: float = Field(gt=0)
    actual_memory_bytes: float | None = Field(default=None, ge=0)
    traffic_amplification_budget: float = Field(default=1.0, ge=1)
    counter_evidence_available: bool
    counter_confidence: Literal["low", "medium", "high"] = "high"
    evidence_digests: list[str] = Field(default_factory=list)


class TrainingScoreInput(TrainingModel):
    schema_version: Literal["0.1"] = "0.1"
    benchmark_cell_id: str
    profiler_grade: TrainingProfilerGrade
    sealed: bool = False
    fresh_process_replays: int = Field(ge=1, le=10)
    load_cells: list[dict[str, Any]] = Field(default_factory=list)
    reuse: TrainingReuseEvidence | None = None
    maintainability: TrainingMaintainabilityEvidence | None = None
    cell_artifact_template: (
        Literal[
            "cell-artifact-mixed-v0.4",
            "cell-artifact-memory-v0.4",
            "cell-artifact-compute-v0.4",
            "cell-artifact-distributed-v0.4",
        ]
        | None
    ) = None
    cell_efficiency: TrainingCellEfficiencyEvidence | None = None
    evidence_digests: list[str] = Field(default_factory=list)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)


class TrainingRawMetric(TrainingModel):
    value: float | int | str | bool | None = None
    unit: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def missing_values_have_a_reason(self) -> TrainingRawMetric:
        if self.value is None and not self.reason:
            raise ValueError("a missing raw metric requires an explicit reason")
        return self


class TrainingComparability(TrainingModel):
    semantic_contract_id: str
    hardware_cell_id: str
    normalized_execution_contract_id: str
    work_model_id: str
    concurrency_protocol_id: str
    leaderboard_season: str
    cross_hardware_absolute_performance: Literal[False] = False
    cell_local_efficiency_only: Literal[True] = True


class TrainingResult(TrainingModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    algorithm: str
    optimizer: str
    framework_stack_id: str
    hardware_cell_id: str
    implementation_bundle_id: str
    training_cert: TrainingCertification
    scoring_authority: Literal["infraswe-scoring-v0.4"] = "infraswe-scoring-v0.4"
    v04_score: ScoreResult
    profiler_grade: TrainingProfilerGrade
    raw_metrics: dict[str, TrainingRawMetric] = Field(default_factory=dict)
    comparability: TrainingComparability

    @model_validator(mode="after")
    def certification_controls_score_issuance(self) -> TrainingResult:
        if self.v04_score.schema_version != "0.4":
            raise ValueError("training results require the v0.4 score envelope")
        deployability = self.v04_score.deployability
        if self.training_cert.status == "fail":
            if self.v04_score.infra_cert != "fail":
                raise ValueError("failed TrainingCert must fail the v0.4 hard-gate envelope")
            if deployability and deployability.score_100 is not None:
                raise ValueError("failed TrainingCert cannot publish Deployability-100")
        if (
            self.training_cert.status == "unresolved"
            and self.v04_score.leaderboard_effective_deployability_100 is not None
        ):
            raise ValueError("unresolved TrainingCert cannot publish an effective score")
        return self


CapabilityLevel = Literal[
    "unsupported", "protocol-supported", "adapter-implemented", "cell-certified"
]


class TrainingAdapterCapability(TrainingModel):
    adapter_id: str
    capability_level: CapabilityLevel
    runtime_available: bool
    framework_version: str | None = None
    algorithms: dict[str, CapabilityLevel] = Field(default_factory=dict)
    missing_dependencies: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class TrainingCapabilityManifest(TrainingModel):
    schema_version: Literal["0.1"] = "0.1"
    generated_at: str
    probe_version: str
    status: Literal["ready", "partial", "protocol_only", "not_ready"]
    adapters: dict[str, TrainingAdapterCapability]
    platform: dict[str, Any]
    scoring_authority: Literal["infraswe-scoring-v0.4"] = "infraswe-scoring-v0.4"
    missing_evidence_policy: Literal["unresolved-not-zero"] = "unresolved-not-zero"
    failure_codes: list[str] = Field(default_factory=list)


class TrainingEvidenceArtifact(TrainingModel):
    path: str
    sha256: str = Field(pattern=r"^sha256:[0-9a-fA-F]{8,64}$")
    size_bytes: int = Field(ge=0)
    category: Literal[
        "contract",
        "framework",
        "run",
        "algorithm",
        "checkpoint",
        "kernel",
        "profiler",
        "fault",
        "score",
    ]


class TrainingEvidencePackManifest(TrainingModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    sealed: bool
    artifacts: list[TrainingEvidenceArtifact] = Field(min_length=1)
    required_categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_categories_are_present(self) -> TrainingEvidencePackManifest:
        observed = {artifact.category for artifact in self.artifacts}
        missing = sorted(set(self.required_categories) - observed)
        if missing:
            raise ValueError("evidence pack missing categories: " + ", ".join(missing))
        return self
