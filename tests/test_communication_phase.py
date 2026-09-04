from __future__ import annotations

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from infraswe.cli import app
from infraswe.models.communication_phase import (
    CommunicationPhaseRegressionPolicy,
    CommunicationPhaseRegressionResult,
    CommunicationPhaseTraceRecord,
    CommunicationPhaseTraceSet,
)
from infraswe.scoring.communication_phase import (
    evaluate_communication_phase_regression,
    summarize_communication_phase,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


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
    return CommunicationPhaseTraceSet(
        framework=framework,
        run_id=run_id,
        policy=policy,
        world_size=world_size,
        cell_identity_sha256=_digest("a"),
        workload_sha256=_digest("b"),
        timestamp_domain="shared-monotonic-ns",
        gpu_timestamp_semantics="kernel-observed",
        clock_sync_error_bound_us=5,
        records=records,
        step_time_ms=[10.0 * step_scale, 10.2 * step_scale, 9.8 * step_scale],
        isolated_latency_ms_by_operation={
            "ep_dispatch": 0.8,
            "dp_reduce_scatter": 0.6,
        },
        evidence_digests=[_digest("1" if run_id == "baseline" else "2")],
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
    baseline = _trace(
        world_size=2,
        run_id="baseline",
        framework="megatron-core",
        policy="concurrent",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    ).model_copy(update={"gpu_timestamp_semantics": "event-bracket"})
    candidate = _trace(
        world_size=2,
        run_id="candidate",
        framework="megatron-core",
        policy="ordered-overlap",
        requested_offset_us=0,
        duration_scale=1.0,
        step_scale=1.0,
    ).model_copy(update={"gpu_timestamp_semantics": "event-bracket"})

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
