from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RequestSample(EvidenceModel):
    schema_version: Literal["0.4"] = "0.4"
    protocol_id: str
    replay_index: int = Field(ge=1, le=10)
    regime: Literal["light", "normal", "knee", "saturation", "overload", "burst_or_soak"]
    request_id: str
    tenant_id: str | None = None
    offered_at_seconds: float = Field(ge=0)
    completed_at_seconds: float | None = Field(default=None, ge=0)
    latency_seconds: float | None = Field(default=None, ge=0)
    completed: bool
    output_valid: bool
    slo_met: bool
    error_code: str | None = None

    @model_validator(mode="after")
    def completion_fields_are_coherent(self) -> RequestSample:
        if self.completed and (self.completed_at_seconds is None or self.latency_seconds is None):
            raise ValueError("completed requests require completion time and latency")
        if not self.completed and self.slo_met:
            raise ValueError("incomplete requests cannot meet the SLO")
        if (
            self.completed_at_seconds is not None
            and self.completed_at_seconds < self.offered_at_seconds
        ):
            raise ValueError("request completion cannot precede offer time")
        return self


class LoadCellEvidence(EvidenceModel):
    schema_version: Literal["0.4"] = "0.4"
    protocol_id: str
    regime: Literal["light", "normal", "knee", "saturation", "overload", "burst_or_soak"]
    load_ratio: float = Field(gt=0)
    offered_requests: int = Field(ge=1)
    completed_requests: int = Field(ge=0)
    slo_goodput_ratio: float = Field(ge=0, le=1)
    error_drop_rate: float = Field(ge=0, le=1)
    tail_score: float = Field(ge=0, le=1)
    replay_jitter_score: float = Field(ge=0, le=1)
    resource_stability_score: float = Field(ge=0, le=1)
    fairness_score: float = Field(ge=0, le=1)
    p99_status: Literal["official", "exploratory"]
    request_samples_sha256: str
    deadlock: bool = False
    livelock: bool = False
    queue_unbounded: bool = False
    memory_growth_limit_exceeded: bool = False
    silent_fallback: bool = False
    error_drop_rate_above_max: bool = False

    @model_validator(mode="after")
    def sample_count_controls_p99_status(self) -> LoadCellEvidence:
        if self.completed_requests > self.offered_requests:
            raise ValueError("completed_requests cannot exceed offered_requests")
        if self.completed_requests < 1000 and self.p99_status != "exploratory":
            raise ValueError("p99 must be exploratory below 1000 completed requests")
        return self


class NormalizedProfilerMetrics(EvidenceModel):
    compute_throughput_pct: float | None = Field(default=None, ge=0)
    memory_throughput_pct: float | None = Field(default=None, ge=0)
    dram_bytes_read: float | None = Field(default=None, ge=0)
    dram_bytes_write: float | None = Field(default=None, ge=0)
    l2_bytes: float | None = Field(default=None, ge=0)
    achieved_occupancy: float | None = Field(default=None, ge=0, le=1)
    registers_per_thread: int | None = Field(default=None, ge=0)
    local_memory_bytes: float | None = Field(default=None, ge=0)
    graph_break_count: int | None = Field(default=None, ge=0)
    recompile_count: int | None = Field(default=None, ge=0)
    guard_count: int | None = Field(default=None, ge=0)
    compiled_graph_count: int | None = Field(default=None, ge=0)
    generated_kernel_count: int | None = Field(default=None, ge=0)
    compile_cache_hit_count: int | None = Field(default=None, ge=0)
    compile_cache_miss_count: int | None = Field(default=None, ge=0)
    cpu_launch_gap_seconds: float | None = Field(default=None, ge=0)
    serialized_stream_fraction: float | None = Field(default=None, ge=0, le=1)
    memcpy_critical_path_fraction: float | None = Field(default=None, ge=0, le=1)


class ProfilerEvidence(EvidenceModel):
    schema_version: Literal["0.4"] = "0.4"
    collector_kind: Literal["framework", "system-trace", "kernel-counter"]
    collector_backend: str
    collector_version: str | None = None
    metric_map_version: str
    status: Literal["captured", "not_applicable", "unresolved"]
    normalized: NormalizedProfilerMetrics = Field(default_factory=NormalizedProfilerMetrics)
    unavailable_reasons: dict[str, str] = Field(default_factory=dict)
    raw_evidence: list[str] = Field(default_factory=list)
    raw_evidence_digests: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high", "not_applicable"]
    official_timing_source: Literal["separate-unprofiled-run"] = "separate-unprofiled-run"
    profiled_timing_authoritative: Literal[False] = False

    @model_validator(mode="after")
    def capture_status_is_coherent(self) -> ProfilerEvidence:
        if self.status == "captured" and not self.raw_evidence:
            raise ValueError("captured profiler evidence requires raw evidence paths")
        if self.status == "not_applicable" and self.confidence != "not_applicable":
            raise ValueError("not_applicable profiler evidence requires matching confidence")
        if len(self.raw_evidence) != len(self.raw_evidence_digests):
            raise ValueError("every raw evidence path must have a digest")
        return self
