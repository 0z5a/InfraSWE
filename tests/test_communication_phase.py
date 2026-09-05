from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from infraswe.cli import app
from infraswe.models.communication_phase import (
    CommunicationArtifactCoverage,
    CommunicationExecutionIdentity,
    CommunicationExperimentProvenance,
    CommunicationGpuTimingProvenance,
    CommunicationPhaseRegressionPolicy,
    CommunicationPhaseRegressionResult,
    CommunicationPhaseTraceRecord,
    CommunicationPhaseTraceSet,
    CommunicationResourceLifecycleEvent,
)
from infraswe.scoring.communication_phase import (
    evaluate_communication_phase_regression,
    summarize_communication_phase,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _content_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _groups(world_size: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if world_size == 2:
        return (((0, 1),), ((0, 1),))
    if world_size == 4:
        return (((0, 1), (2, 3)), ((0, 2), (1, 3)))
    raise AssertionError("test fixture only reserves CPU coverage for 2-rank and 4-rank layouts")


def _trace(
    *,
    world_size: int,
    run_id: str,
    framework: str,
    policy: str,
    requested_offset_us: float,
    duration_scale: float,
    step_scale: float,
    realized_offset_us: float | None = None,
    divergent_order: bool = False,
    include_consumers: bool = True,
) -> CommunicationPhaseTraceSet:
    groups_a, groups_b = _groups(world_size)
    records: list[CommunicationPhaseTraceRecord] = []
    for pair_index in range(3):
        pair_id = f"step-{pair_index}"
        pair_base_ns = 1_000_000_000 + pair_index * 10_000_000
        for role, operation, groups, role_offset_us, nominal_duration_ns in (
            ("a", "ep_dispatch", groups_a, 0.0, 1_000_000),
            ("b", "dp_reduce_scatter", groups_b, requested_offset_us, 800_000),
        ):
            for group_index, group in enumerate(groups):
                group_id = f"{role}-group-{group_index}"
                logical_id = f"{pair_id}/{group_id}"
                for rank in group:
                    sequence_id = pair_index
                    if (
                        divergent_order
                        and group_id == "a-group-0"
                        and rank == group[-1]
                        and pair_index < 2
                    ):
                        sequence_id = 1 - pair_index
                    gpu_role_offset_us = (
                        role_offset_us
                        if realized_offset_us is None or role == "a"
                        else realized_offset_us
                    )
                    start_ns = (
                        pair_base_ns
                        + int(gpu_role_offset_us * 1_000)
                        + group_index * 2_000
                        + rank * 1_000
                    )
                    duration_ns = int(nominal_duration_ns * duration_scale)
                    end_ns = start_ns + duration_ns
                    records.append(
                        CommunicationPhaseTraceRecord(
                            framework=framework,
                            run_id=run_id,
                            rank=rank,
                            world_size=world_size,
                            local_rank=rank,
                            node=0,
                            step=pair_index,
                            microbatch=pair_index,
                            layer=3,
                            direction="backward",
                            operation=operation,
                            logical_operation_id=logical_id,
                            pair_id=pair_id,
                            pair_role=role,
                            process_group_id=group_id,
                            process_group_ranks=group,
                            communicator_sequence_id=sequence_id,
                            stream_id=f"stream-{role}",
                            message_bytes=1_024,
                            requested_offset_us=role_offset_us,
                            api_launch_timestamp_ns=start_ns - 50_000,
                            api_return_timestamp_ns=start_ns - 40_000,
                            gpu_start_timestamp_ns=start_ns,
                            gpu_end_timestamp_ns=end_ns,
                            completion_timestamp_ns=end_ns + 10_000,
                            consumer_timestamp_ns=(
                                end_ns + 500_000 if role == "b" and include_consumers else None
                            ),
                            transport="nccl",
                            topology_class="single-node-test",
                        )
                    )
    resource_lifecycle_events = []
    for record in records:
        assert record.completion_timestamp_ns is not None
        resource_lifecycle_events.extend(
            (
                CommunicationResourceLifecycleEvent(
                    process_group_id=record.process_group_id,
                    logical_operation_id=record.logical_operation_id,
                    rank=record.rank,
                    event="acquire",
                    timestamp_ns=record.gpu_start_timestamp_ns,
                    message_bytes=record.message_bytes,
                ),
                CommunicationResourceLifecycleEvent(
                    process_group_id=record.process_group_id,
                    logical_operation_id=record.logical_operation_id,
                    rank=record.rank,
                    event="release",
                    timestamp_ns=record.completion_timestamp_ns,
                    message_bytes=record.message_bytes,
                ),
            )
        )
    process_run_ids = tuple(f"{run_id}-process-{index}" for index in range(5))
    process_artifact_digests = tuple(
        _content_digest(process_run_id) for process_run_id in process_run_ids
    )
    return CommunicationPhaseTraceSet(
        framework=framework,
        run_id=run_id,
        policy=policy,
        world_size=world_size,
        cell_identity_sha256=_digest("a"),
        workload_sha256=_digest("b"),
        execution_identity=CommunicationExecutionIdentity(
            model_revision_sha256=_digest("c"),
            checkpoint_sha256=_digest("d"),
            policy_state_sha256=_digest("e"),
            topology_sha256=_digest("f"),
        ),
        artifact_coverage=CommunicationArtifactCoverage(
            claim_scope="full-run",
            manifest_sha256=_digest("3"),
            expected_units=len(records),
            verified_units=len(records),
            reconstructed_units=len(records),
            exact_order_verified=True,
        ),
        experiment_provenance=CommunicationExperimentProvenance(
            phase="confirmation",
            independent_process_run_ids=process_run_ids,
            independent_process_artifact_sha256=process_artifact_digests,
        ),
        timestamp_domain="shared-monotonic-ns",
        gpu_timestamp_semantics="kernel-observed",
        gpu_timing_provenance=CommunicationGpuTimingProvenance(
            capture_kind="profiler-kernel",
            adapter="nsys-nccl-kernel-v1",
            artifact_sha256=_digest("4"),
            observed_kernel_names=("ncclKernel_AllReduce_RING_LL",),
        ),
        clock_sync_error_bound_us=5,
        records=records,
        resource_lifecycle_events=resource_lifecycle_events,
        step_time_ms=[10.0 * step_scale, 10.2 * step_scale, 9.8 * step_scale],
        isolated_latency_ms_by_operation={
            "ep_dispatch": 0.8,
            "dp_reduce_scatter": 0.6,
        },
        evidence_digests=[
            _digest("1" if run_id == "baseline" else "2"),
            *process_artifact_digests,
        ],
    )


def test_four_rank_overlap_grid_scores_within_one_cell_across_frameworks() -> None:
    baseline = _trace(
        world_size=4,
        run_id="baseline",
        framework="verl",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    candidate = _trace(
        world_size=4,
        run_id="candidate",
        framework="slime",
        policy="windowed",
        requested_offset_us=100,
        duration_scale=0.9,
        step_scale=0.95,
    )

    summary = summarize_communication_phase(candidate)
    assert summary.pair_count == summary.completed_pair_count == 3
    assert summary.collective_order_safe
    assert summary.realized_offset_p50_us == pytest.approx(100)

    result = evaluate_communication_phase_regression(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(
            max_outstanding_bytes=4_096,
            max_outstanding_collectives=4,
        ),
        regime="normal",
        load_ratio=0.5,
    )
    assert result.status == "pass"
    assert result.baseline_framework == "verl"
    assert result.candidate_framework == "slime"
    assert result.world_size == 4
    assert result.load_cell is not None
    assert result.load_cell.p99_status == "exploratory"
    assert result.load_cell.domain == "distributed-communication"
    assert all(value == pytest.approx(1.0) for value in result.components.values())
    assert not result.cross_cell_ranking_allowed


def test_two_rank_sequence_divergence_is_a_hard_regression() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="megatron-core",
        policy="eager",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="megatron-core",
        policy="ordered-overlap",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
        divergent_order=True,
    )

    result = evaluate_communication_phase_regression(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(),
        regime="knee",
        load_ratio=0.8,
    )
    assert result.status == "fail"
    assert "COLLECTIVE_ORDER_VIOLATION" in result.failure_codes
    assert result.components["collective_order_safety"] == 0
    assert result.load_cell is not None
    assert "COLLECTIVE_ORDER_VIOLATION" in result.load_cell.hard_gate_failure_codes


def test_missing_consumer_timestamp_is_unresolved_not_zero() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="verl",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="verl",
        policy="windowed",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
        include_consumers=False,
    )

    result = evaluate_communication_phase_regression(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(),
        regime="light",
        load_ratio=0.25,
    )
    assert result.status == "unresolved"
    assert "CONSUMER_TIMESTAMP_EVIDENCE_INCOMPLETE" in result.unresolved_reasons
    assert result.components["consumer_slack_utilization"] is None
    assert result.load_cell is None


def test_cross_cell_or_workload_comparison_is_rejected() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="verl",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="verl",
        policy="windowed",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    different_cell = candidate.model_copy(update={"cell_identity_sha256": _digest("c")})
    with pytest.raises(ValueError, match="different cells"):
        evaluate_communication_phase_regression(
            baseline,
            different_cell,
            CommunicationPhaseRegressionPolicy(),
            regime="normal",
            load_ratio=0.5,
        )


def test_offset_timing_and_inflight_window_regressions_are_fail_closed() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="megatron-core",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="megatron-core",
        policy="windowed",
        requested_offset_us=100,
        realized_offset_us=1_000,
        duration_scale=1.2,
        step_scale=1.1,
    ).model_copy(update={"clock_sync_error_bound_us": 75})

    result = evaluate_communication_phase_regression(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(
            max_outstanding_bytes=1_000,
            max_outstanding_collectives=1,
        ),
        regime="overload",
        load_ratio=1.2,
    )
    assert result.status == "fail"
    assert {
        "REALIZED_OFFSET_UNSTABLE",
        "OUTSTANDING_BYTE_BUDGET_EXCEEDED",
        "OUTSTANDING_COLLECTIVE_BUDGET_EXCEEDED",
        "STEP_TIME_P95_REGRESSION",
        "CLOCK_SYNC_ERROR_BOUND_EXCEEDED",
    }.issubset(result.failure_codes)
    assert result.components["realized_offset_stability"] == pytest.approx(250 / 900)


def test_trace_record_rejects_noncanonical_process_group_membership() -> None:
    record = _trace(
        world_size=2,
        run_id="baseline",
        framework="custom-runtime",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    ).records[0]
    payload = record.model_dump(mode="json")
    payload["process_group_ranks"] = [1, 0]
    with pytest.raises(ValidationError, match="sorted and unique"):
        CommunicationPhaseTraceRecord.model_validate(payload)


def test_event_brackets_remain_diagnostic_only() -> None:
    event_provenance = CommunicationGpuTimingProvenance(
        capture_kind="cuda-event-bracket",
        adapter="cuda-event-v1",
        artifact_sha256=_digest("5"),
    )
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="megatron-core",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    ).model_copy(
        update={
            "gpu_timestamp_semantics": "event-bracket",
            "gpu_timing_provenance": event_provenance,
        }
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="megatron-core",
        policy="ordered-overlap",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    ).model_copy(
        update={
            "gpu_timestamp_semantics": "event-bracket",
            "gpu_timing_provenance": event_provenance,
        }
    )

    official = evaluate_communication_phase_regression(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(),
        regime="normal",
        load_ratio=0.5,
    )
    assert official.status == "unresolved"
    assert "KERNEL_OBSERVED_TIMESTAMPS_REQUIRED" in official.unresolved_reasons
    assert official.load_cell is None


def test_phase_regression_cli_writes_machine_readable_result(tmp_path) -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="verl",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="verl",
        policy="windowed",
        requested_offset_us=100,
        duration_scale=0.9,
        step_scale=0.95,
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "regression.json"
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")

    invocation = CliRunner().invoke(
        app,
        [
            "communication",
            "phase-regression",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--regime",
            "normal",
            "--load-ratio",
            "0.5",
            "--output",
            str(output_path),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    result = CommunicationPhaseRegressionResult.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert result.status == "pass"
    assert "world_size=2" in invocation.output


def _evaluate(
    baseline: CommunicationPhaseTraceSet,
    candidate: CommunicationPhaseTraceSet,
    policy: CommunicationPhaseRegressionPolicy | None = None,
) -> CommunicationPhaseRegressionResult:
    return evaluate_communication_phase_regression(
        baseline,
        candidate,
        policy or CommunicationPhaseRegressionPolicy(),
        regime="normal",
        load_ratio=0.5,
    )


def test_hard_negative_catalog_is_source_annotated_and_complete() -> None:
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "communication_phase"
        / "hard-negative-mutations-v0.1.json"
    )
    catalog = json.loads(path.read_text(encoding="utf-8"))
    assert catalog["source"]["section"].startswith("10. InfraSWE")
    assert catalog["allowed_logical_world_sizes"] == [2, 4]
    assert "do not establish agent precision" in catalog["claim_limit"]
    assert {case["id"] for case in catalog["cases"]} == {
        "requested-offset-as-realized-offset",
        "event-bracket-as-kernel-observed",
        "a-faster-pair-and-consumer-worse",
        "missing-rank",
        "missing-or-divergent-sequence",
        "cross-topology-identity",
        "cross-policy-state-identity",
        "cross-checkpoint-identity",
        "count-compliant-byte-budget-exceeded",
        "duplicate-resource-release",
        "missing-resource-release",
        "partial-shard-as-full-reconstruction",
        "selection-best-as-generalized-confirmation",
    }
    assert all(case["invariant"] and case["expected"] for case in catalog["cases"])


def test_requested_offset_cannot_masquerade_as_gpu_realized_offset() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="verl",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="verl",
        policy="requested-800us",
        requested_offset_us=800,
        realized_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )

    summary = summarize_communication_phase(candidate)
    assert summary.realized_offset_p50_us == pytest.approx(0)
    assert summary.realized_offset_error_p95_us == pytest.approx(800)
    result = _evaluate(candidate=candidate, baseline=baseline)
    assert result.status == "fail"
    assert "REALIZED_OFFSET_UNSTABLE" in result.failure_codes


def test_kernel_observed_label_requires_matching_capture_provenance() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="megatron-core",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="megatron-core",
        policy="candidate",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    event_provenance = CommunicationGpuTimingProvenance(
        capture_kind="cuda-event-bracket",
        adapter="cuda-event-v1",
        artifact_sha256=_digest("5"),
    )
    mislabeled = candidate.model_copy(update={"gpu_timing_provenance": event_provenance})

    result = _evaluate(baseline, mislabeled)
    assert result.status == "fail"
    assert "GPU_TIMING_PROVENANCE_MISMATCH" in result.failure_codes
    payload = mislabeled.model_dump(mode="json")
    with pytest.raises(ValidationError, match="timing provenance"):
        CommunicationPhaseTraceSet.model_validate(payload)


def test_local_a_improvement_cannot_hide_pair_or_consumer_regression() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="slime",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="slime",
        policy="locally-faster-a",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    records = []
    release_timestamps: dict[tuple[str, str, int], int] = {}
    for record in candidate.records:
        duration_ns = 400_000 if record.pair_role == "a" else 1_800_000
        end_ns = record.gpu_start_timestamp_ns + duration_ns
        completion_ns = end_ns + 10_000
        records.append(
            record.model_copy(
                update={
                    "gpu_end_timestamp_ns": end_ns,
                    "completion_timestamp_ns": completion_ns,
                    "consumer_timestamp_ns": (
                        end_ns - 100_000 if record.pair_role == "b" else None
                    ),
                }
            )
        )
        release_timestamps[(record.process_group_id, record.logical_operation_id, record.rank)] = (
            completion_ns
        )
    lifecycle = [
        event.model_copy(
            update={
                "timestamp_ns": release_timestamps[
                    (event.process_group_id, event.logical_operation_id, event.rank)
                ]
            }
        )
        if event.event == "release"
        else event
        for event in candidate.resource_lifecycle_events
    ]
    candidate = candidate.model_copy(
        update={"records": records, "resource_lifecycle_events": lifecycle}
    )

    baseline_a_ns = max(
        record.gpu_end_timestamp_ns - record.gpu_start_timestamp_ns
        for record in baseline.records
        if record.pair_role == "a"
    )
    candidate_a_ns = max(
        record.gpu_end_timestamp_ns - record.gpu_start_timestamp_ns
        for record in candidate.records
        if record.pair_role == "a"
    )
    assert candidate_a_ns < baseline_a_ns
    result = _evaluate(baseline, candidate)
    assert result.status == "fail"
    assert {
        "PAIR_COMPLETION_P95_REGRESSION",
        "CONSUMER_DEADLINE_REGRESSION",
    }.issubset(result.failure_codes)


def test_missing_rank_is_rejected_even_when_artifact_count_claim_is_resealed() -> None:
    baseline = _trace(
        world_size=4,
        run_id="baseline",
        framework="verl",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=4,
        run_id="candidate",
        framework="verl",
        policy="missing-rank",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    records = [record for record in candidate.records if record.rank != 3]
    lifecycle = [event for event in candidate.resource_lifecycle_events if event.rank != 3]
    count = len(records)
    coverage = candidate.artifact_coverage.model_copy(
        update={
            "expected_units": count,
            "verified_units": count,
            "reconstructed_units": count,
        }
    )
    candidate = candidate.model_copy(
        update={
            "records": records,
            "resource_lifecycle_events": lifecycle,
            "artifact_coverage": coverage,
        }
    )

    result = _evaluate(baseline, candidate)
    assert result.status == "fail"
    assert "TRACE_INTEGRITY_VIOLATION" in result.failure_codes
    assert any(item.startswith("incomplete-world:3") for item in result.candidate.order_violations)


@pytest.mark.parametrize(
    ("field", "digest_character"),
    (
        ("topology_sha256", "6"),
        ("policy_state_sha256", "7"),
        ("checkpoint_sha256", "8"),
    ),
)
def test_cross_execution_identity_is_rejected_before_scoring(
    field: str,
    digest_character: str,
) -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="verl",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="verl",
        policy="candidate",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    identity = candidate.execution_identity.model_copy(update={field: _digest(digest_character)})

    with pytest.raises(ValueError, match="one execution identity"):
        _evaluate(baseline, candidate.model_copy(update={"execution_identity": identity}))


def test_collective_count_compliance_does_not_override_byte_budget() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="slime",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="slime",
        policy="two-credit-window",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )

    result = _evaluate(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(
            max_outstanding_collectives=2,
            max_outstanding_bytes=1_500,
        ),
    )
    assert result.candidate.max_inflight_collectives == 2
    assert result.candidate.max_inflight_bytes == 2_048
    assert "OUTSTANDING_BYTE_BUDGET_EXCEEDED" in result.failure_codes
    assert "OUTSTANDING_COLLECTIVE_BUDGET_EXCEEDED" not in result.failure_codes


@pytest.mark.parametrize("mutation", ("duplicate", "missing"))
def test_duplicate_or_missing_resource_release_fails_closed(mutation: str) -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="slime",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="slime",
        policy="windowed",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    target = next(
        event for event in candidate.resource_lifecycle_events if event.event == "release"
    )
    lifecycle = list(candidate.resource_lifecycle_events)
    if mutation == "duplicate":
        lifecycle.append(target.model_copy())
    else:
        lifecycle.remove(target)

    result = _evaluate(
        baseline,
        candidate.model_copy(update={"resource_lifecycle_events": lifecycle}),
    )
    assert result.status == "fail"
    assert "RESOURCE_LIFECYCLE_VIOLATION" in result.failure_codes
    assert any(
        violation.startswith("resource-release-count:")
        for violation in result.candidate.resource_lifecycle_violations
    )


def test_partial_shard_verification_cannot_claim_full_reconstruction() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="custom-runtime",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="custom-runtime",
        policy="partial-evidence",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    partial = candidate.artifact_coverage.model_copy(
        update={
            "claim_scope": "partial-shard",
            "verified_units": len(candidate.records) // 2,
            "reconstructed_units": len(candidate.records) // 2,
            "exact_order_verified": False,
        }
    )

    result = _evaluate(
        baseline,
        candidate.model_copy(update={"artifact_coverage": partial}),
    )
    assert result.status == "fail"
    assert "ARTIFACT_COVERAGE_INCOMPLETE" in result.failure_codes
    assert result.candidate.artifact_coverage_complete is False


def test_selection_best_sample_is_not_generalized_confirmation() -> None:
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="megatron-core",
        policy="baseline",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="megatron-core",
        policy="sweep-best",
        requested_offset_us=0,
        duration_scale=0.8,
        step_scale=0.8,
    )
    selection_only = CommunicationExperimentProvenance(
        phase="candidate-selection",
        independent_process_run_ids=("candidate-selection-process-0",),
        independent_process_artifact_sha256=(_content_digest("candidate-selection-process-0"),),
    )
    candidate = candidate.model_copy(
        update={
            "experiment_provenance": selection_only,
            "evidence_digests": [
                *candidate.evidence_digests,
                *selection_only.independent_process_artifact_sha256,
            ],
        }
    )

    result = _evaluate(
        baseline,
        candidate,
        CommunicationPhaseRegressionPolicy(min_confirmation_process_runs=5),
    )
    assert result.status == "unresolved"
    assert "CANDIDATE_CONFIRMATION_EVIDENCE_INSUFFICIENT" in result.unresolved_reasons
    assert result.load_cell is None
