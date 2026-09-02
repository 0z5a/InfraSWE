from __future__ import annotations

import pytest

from infraswe.models.evidence import LoadCellEvidence, ProfilerEvidence, RequestSample
from infraswe.telemetry.profiler_v04 import (
    framework_compile_evidence,
    kernel_counter_evidence,
    system_trace_evidence,
)


def test_native_framework_collector_is_na_not_zero() -> None:
    evidence = framework_compile_evidence(None, applicable=False)
    assert evidence.status == "not_applicable"
    assert evidence.confidence == "not_applicable"
    assert evidence.normalized.graph_break_count is None


def test_missing_counter_collector_is_unresolved_not_zero() -> None:
    evidence = kernel_counter_evidence(None, applicable=True)
    assert evidence.status == "unresolved"
    assert evidence.normalized.dram_bytes_read is None
    assert evidence.unavailable_reasons


def test_failed_profiler_capture_preserves_raw_failure_evidence() -> None:
    evidence = kernel_counter_evidence(
        None,
        applicable=True,
        raw_evidence=["ncu-error.log"],
        raw_evidence_digests=["sha256:error"],
    )

    assert evidence.status == "unresolved"
    assert evidence.raw_evidence == ["ncu-error.log"]
    assert evidence.raw_evidence_digests == ["sha256:error"]


def test_captured_collectors_require_raw_evidence_and_digests() -> None:
    with pytest.raises(ValueError, match="raw evidence"):
        system_trace_evidence({"serialized_stream_fraction": 0.2})

    evidence = system_trace_evidence(
        {"serialized_stream_fraction": 0.2, "cpu_launch_gap_seconds": 0.001},
        raw_evidence=["evidence/profiles/system-trace/run.nsys-rep"],
        raw_evidence_digests=["sha256:trace"],
    )
    assert evidence.status == "captured"
    assert evidence.profiled_timing_authoritative is False
    assert evidence.official_timing_source == "separate-unprofiled-run"


def test_profiler_schema_forbids_authoritative_profiled_latency() -> None:
    with pytest.raises(ValueError):
        ProfilerEvidence.model_validate(
            {
                "collector_kind": "kernel-counter",
                "collector_backend": "ncu",
                "metric_map_version": "nvidia-ncu-v3",
                "status": "unresolved",
                "confidence": "low",
                "profiled_timing_authoritative": True,
            }
        )


def test_request_samples_and_p99_minimum_are_machine_validated() -> None:
    request = RequestSample(
        protocol_id="serving-load-normalized-v1",
        replay_index=1,
        regime="normal",
        request_id="req-1",
        offered_at_seconds=1.0,
        completed_at_seconds=1.1,
        latency_seconds=0.1,
        completed=True,
        output_valid=True,
        slo_met=True,
    )
    assert request.completed

    with pytest.raises(ValueError, match="exploratory"):
        LoadCellEvidence(
            protocol_id="serving-load-normalized-v1",
            regime="normal",
            load_ratio=0.5,
            offered_requests=100,
            completed_requests=100,
            slo_goodput_ratio=1,
            error_drop_rate=0,
            tail_score=1,
            replay_jitter_score=1,
            resource_stability_score=1,
            fairness_score=1,
            p99_status="official",
            request_samples_sha256="sha256:samples",
        )
