from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from infraswe.models.draft import Digest
from infraswe.models.system_paths import SystemPathLoadCell

PositiveMilliseconds = Annotated[float, Field(gt=0)]


class CommunicationPhaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class CommunicationGpuTimingProvenance(CommunicationPhaseModel):
    """Content-addressed source of the declared GPU timestamp semantics."""

    capture_kind: Literal["profiler-kernel", "cuda-event-bracket"]
    adapter: str = Field(min_length=1)
    artifact_sha256: Digest
    observed_kernel_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def kernel_capture_names_observed_kernels(self) -> CommunicationGpuTimingProvenance:
        if self.capture_kind == "profiler-kernel" and not self.observed_kernel_names:
            raise ValueError("profiler-kernel timing requires observed kernel names")
        if any(not name for name in self.observed_kernel_names):
            raise ValueError("observed kernel names cannot be empty")
        return self


class CommunicationExecutionIdentity(CommunicationPhaseModel):
    """Controlled model/policy/checkpoint/topology identity for an A/B comparison."""

    model_revision_sha256: Digest
    checkpoint_sha256: Digest
    policy_state_sha256: Digest
    topology_sha256: Digest


class CommunicationArtifactCoverage(CommunicationPhaseModel):
    """Coverage claim for the evidence artifact supplied to the scorer."""

    claim_scope: Literal["full-run", "partial-shard"]
    manifest_sha256: Digest
    expected_units: int = Field(ge=1)
    verified_units: int = Field(ge=0)
    reconstructed_units: int = Field(ge=0)
    exact_order_verified: bool


class CommunicationExperimentProvenance(CommunicationPhaseModel):
    """Separates candidate selection from independent confirmation evidence."""

    phase: Literal["candidate-selection", "confirmation"]
    independent_process_run_ids: tuple[str, ...] = Field(min_length=1)
    independent_process_artifact_sha256: tuple[Digest, ...] = Field(min_length=1)

    @field_validator("independent_process_run_ids")
    @classmethod
    def process_run_ids_are_unique_and_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not run_id for run_id in value):
            raise ValueError("independent process run IDs cannot be empty")
        if len(value) != len(set(value)):
            raise ValueError("independent process run IDs must be unique")
        return value

    @model_validator(mode="after")
    def every_process_run_has_one_unique_artifact(self) -> CommunicationExperimentProvenance:
        if len(self.independent_process_run_ids) != len(self.independent_process_artifact_sha256):
            raise ValueError("every independent process run requires one artifact digest")
        if len(self.independent_process_artifact_sha256) != len(
            set(self.independent_process_artifact_sha256)
        ):
            raise ValueError("independent process artifact digests must be unique")
        return self


class CommunicationResourceLifecycleEvent(CommunicationPhaseModel):
    """One acquisition or final-release observation for a rank-local resource."""

    process_group_id: str = Field(min_length=1)
    logical_operation_id: str = Field(min_length=1)
    rank: int = Field(ge=0)
    event: Literal["acquire", "release"]
    timestamp_ns: int = Field(ge=0)
    message_bytes: int = Field(gt=0)


class CommunicationPhaseTraceRecord(CommunicationPhaseModel):
    """One framework-neutral collective observation from one rank."""

    schema_version: Literal["0.1"] = "0.1"
    framework: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    rank: int = Field(ge=0)
    world_size: int = Field(ge=2)
    local_rank: int | None = Field(default=None, ge=0)
    node: int | None = Field(default=None, ge=0)
    step: int = Field(ge=0)
    microbatch: int | None = Field(default=None, ge=0)
    layer: int | None = Field(default=None, ge=0)
    direction: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    logical_operation_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    pair_role: Literal["a", "b"]
    process_group_id: str = Field(min_length=1)
    process_group_ranks: tuple[int, ...] = Field(min_length=2)
    communicator_sequence_id: int = Field(ge=0)
    stream_id: str = Field(min_length=1)
    message_bytes: int = Field(gt=0)
    requested_offset_us: float = 0.0
    api_launch_timestamp_ns: int = Field(ge=0)
    api_return_timestamp_ns: int | None = Field(default=None, ge=0)
    gpu_start_timestamp_ns: int = Field(ge=0)
    gpu_end_timestamp_ns: int = Field(ge=0)
    completion_timestamp_ns: int | None = Field(default=None, ge=0)
    consumer_timestamp_ns: int | None = Field(default=None, ge=0)
    transport: str = Field(min_length=1)
    topology_class: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("process_group_ranks")
    @classmethod
    def process_group_ranks_are_canonical(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(rank < 0 for rank in value):
            raise ValueError("process_group_ranks cannot contain negative ranks")
        if tuple(sorted(set(value))) != value:
            raise ValueError("process_group_ranks must be sorted and unique")
        return value

    @model_validator(mode="after")
    def rank_and_timestamps_are_coherent(self) -> CommunicationPhaseTraceRecord:
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.local_rank is not None and self.local_rank >= self.world_size:
            raise ValueError("local_rank must be smaller than world_size")
        if self.rank not in self.process_group_ranks:
            raise ValueError("record rank must belong to process_group_ranks")
        if any(rank >= self.world_size for rank in self.process_group_ranks):
            raise ValueError("process-group ranks must be smaller than world_size")
        if self.gpu_end_timestamp_ns < self.gpu_start_timestamp_ns:
            raise ValueError("GPU end timestamp cannot precede GPU start")
        if (
            self.api_return_timestamp_ns is not None
            and self.api_return_timestamp_ns < self.api_launch_timestamp_ns
        ):
            raise ValueError("API return timestamp cannot precede API launch")
        if (
            self.completion_timestamp_ns is not None
            and self.completion_timestamp_ns < self.api_launch_timestamp_ns
        ):
            raise ValueError("completion timestamp cannot precede API launch")
        if (
            self.completion_timestamp_ns is not None
            and self.api_return_timestamp_ns is not None
            and self.completion_timestamp_ns < self.api_return_timestamp_ns
        ):
            raise ValueError("completion timestamp cannot precede API return")
        return self


class CommunicationPhaseTraceSet(CommunicationPhaseModel):
    """A complete trace for one policy run inside one sealed comparison cell."""

    schema_version: Literal["0.1"] = "0.1"
    framework: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    world_size: int = Field(ge=2)
    cell_identity_sha256: Digest
    workload_sha256: Digest
    execution_identity: CommunicationExecutionIdentity
    artifact_coverage: CommunicationArtifactCoverage
    experiment_provenance: CommunicationExperimentProvenance
    timestamp_domain: str = Field(min_length=1)
    gpu_timestamp_semantics: Literal["kernel-observed", "event-bracket"]
    gpu_timing_provenance: CommunicationGpuTimingProvenance
    clock_sync_error_bound_us: float = Field(ge=0)
    records: list[CommunicationPhaseTraceRecord] = Field(min_length=1)
    resource_lifecycle_events: list[CommunicationResourceLifecycleEvent] = Field(min_length=1)
    step_time_ms: list[PositiveMilliseconds] = Field(default_factory=list)
    isolated_latency_ms_by_operation: dict[str, PositiveMilliseconds] = Field(default_factory=dict)
    evidence_digests: list[Digest] = Field(min_length=1)

    @model_validator(mode="after")
    def record_identity_matches_trace_set(self) -> CommunicationPhaseTraceSet:
        expected_timing = (
            "kernel-observed"
            if self.gpu_timing_provenance.capture_kind == "profiler-kernel"
            else "event-bracket"
        )
        if self.gpu_timestamp_semantics != expected_timing:
            raise ValueError("GPU timestamp semantics do not match timing provenance")
        if not set(self.experiment_provenance.independent_process_artifact_sha256).issubset(
            self.evidence_digests
        ):
            raise ValueError("independent process artifacts must be bound as evidence digests")
        for record in self.records:
            if record.framework != self.framework:
                raise ValueError("record framework does not match trace-set framework")
            if record.run_id != self.run_id:
                raise ValueError("record run_id does not match trace-set run_id")
            if record.world_size != self.world_size:
                raise ValueError("record world_size does not match trace-set world_size")
        return self


class CommunicationPhaseRegressionPolicy(CommunicationPhaseModel):
    """Frozen within-cell regression tolerances; none depend on a particular rank count."""

    schema_version: Literal["0.1"] = "0.1"
    formula_version: Literal["communication-phase-regression-v0.1"] = (
        "communication-phase-regression-v0.1"
    )
    max_step_time_p95_regression_fraction: float = Field(default=0.02, ge=0)
    max_pair_completion_p95_regression_fraction: float = Field(default=0.02, ge=0)
    max_contention_stretch_p95_regression_fraction: float = Field(default=0.05, ge=0)
    max_realized_offset_error_p95_us: float = Field(default=250.0, gt=0)
    max_rank_skew_regression_fraction: float = Field(default=0.10, ge=0)
    rank_skew_absolute_allowance_us: float = Field(default=50.0, ge=0)
    consumer_wait_p95_allowance_us: float = Field(default=0.0, ge=0)
    max_clock_sync_error_us: float = Field(default=50.0, ge=0)
    max_outstanding_bytes: int | None = Field(default=None, gt=0)
    max_outstanding_collectives: int | None = Field(default=None, ge=1)
    min_confirmation_process_runs: int = Field(default=1, ge=1)


class CommunicationPhaseRunMetrics(CommunicationPhaseModel):
    timestamp_domain: str
    gpu_timestamp_semantics: Literal["kernel-observed", "event-bracket"]
    clock_sync_error_bound_us: float = Field(ge=0)
    transport: str | None = None
    topology_class: str | None = None
    pair_count: int = Field(ge=0)
    completed_pair_count: int = Field(ge=0)
    step_time_p50_ms: float | None = Field(default=None, gt=0)
    step_time_p95_ms: float | None = Field(default=None, gt=0)
    pair_completion_p50_ms: float | None = Field(default=None, ge=0)
    pair_completion_p95_ms: float | None = Field(default=None, ge=0)
    pair_completion_p99_ms: float | None = Field(default=None, ge=0)
    api_launch_offset_p50_us: float | None = None
    realized_offset_p50_us: float | None = None
    realized_offset_p95_us: float | None = None
    realized_offset_error_p95_us: float | None = Field(default=None, ge=0)
    contention_stretch_p95: float | None = Field(default=None, ge=0)
    actual_overlap_p50_ms: float | None = Field(default=None, ge=0)
    rank_start_skew_p95_us: float | None = Field(default=None, ge=0)
    rank_finish_skew_p95_us: float | None = Field(default=None, ge=0)
    consumer_slack_p50_us: float | None = None
    consumer_wait_p95_us: float | None = Field(default=None, ge=0)
    consumer_deadline_miss_count: int = Field(default=0, ge=0)
    max_inflight_bytes: int = Field(ge=0)
    max_inflight_collectives: int = Field(ge=0)
    collective_order_safe: bool
    order_violations: list[str] = Field(default_factory=list)
    resource_lifecycle_safe: bool
    resource_lifecycle_violations: list[str] = Field(default_factory=list)
    artifact_coverage_complete: bool


class CommunicationPhaseRegressionMetrics(CommunicationPhaseModel):
    pair_completion_gain_fraction: float | None = None
    step_time_gain_fraction: float | None = None
    contention_stretch_change_fraction: float | None = None
    consumer_slack_utilization_p50: float | None = None
    rank_start_skew_change_fraction: float | None = None
    rank_finish_skew_change_fraction: float | None = None


class CommunicationPhaseRegressionResult(CommunicationPhaseModel):
    """A within-cell verdict and inputs for the existing concurrent-stability score."""

    schema_version: Literal["0.1"] = "0.1"
    formula_version: Literal["communication-phase-regression-v0.1"] = (
        "communication-phase-regression-v0.1"
    )
    status: Literal["pass", "fail", "unresolved"]
    score_status: Literal["within-cell-regression-only"] = "within-cell-regression-only"
    cell_identity_sha256: Digest
    workload_sha256: Digest
    baseline_framework: str
    candidate_framework: str
    baseline_run_id: str
    candidate_run_id: str
    world_size: int = Field(ge=2)
    components: dict[str, float | None]
    baseline: CommunicationPhaseRunMetrics
    candidate: CommunicationPhaseRunMetrics
    comparison: CommunicationPhaseRegressionMetrics
    load_cell: SystemPathLoadCell | None = None
    failure_codes: list[str] = Field(default_factory=list)
    unresolved_reasons: list[str] = Field(default_factory=list)
    cross_cell_ranking_allowed: Literal[False] = False

    @model_validator(mode="after")
    def verdict_is_coherent(self) -> CommunicationPhaseRegressionResult:
        expected = {
            "comm_phase_sweep",
            "comm_contention_stretch",
            "realized_offset_stability",
            "collective_order_safety",
            "windowed_scheduler_gain",
            "consumer_slack_utilization",
        }
        if set(self.components) != expected:
            raise ValueError("communication phase result requires the frozen six components")
        if any(value is not None and not 0 <= value <= 1 for value in self.components.values()):
            raise ValueError("communication phase components must be in [0, 1]")
        if self.status == "pass" and (self.failure_codes or self.unresolved_reasons):
            raise ValueError("passing regression cannot have failures or unresolved evidence")
        if self.status == "fail" and not self.failure_codes:
            raise ValueError("failed regression requires failure codes")
        if self.status == "unresolved" and not self.unresolved_reasons:
            raise ValueError("unresolved regression requires unresolved reasons")
        if self.status != "pass" and self.load_cell is not None and self.unresolved_reasons:
            raise ValueError("unresolved evidence cannot publish a load cell")
        return self
