from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from infraswe.models.communication_phase import (
    CommunicationPhaseRegressionMetrics,
    CommunicationPhaseRegressionPolicy,
    CommunicationPhaseRegressionResult,
    CommunicationPhaseRunMetrics,
    CommunicationPhaseTraceRecord,
    CommunicationPhaseTraceSet,
    CommunicationResourceLifecycleEvent,
)
from infraswe.models.system_paths import SystemPathLoadCell


@dataclass(frozen=True)
class _RunSummary:
    metrics: CommunicationPhaseRunMetrics
    pair_ids: frozenset[str]
    completed_pair_ids: frozenset[str]
    missing_isolation_operations: frozenset[str]
    consumer_timestamps_complete: bool
    artifact_coverage_complete: bool


_ORDER_VIOLATION_PREFIXES = (
    "duplicate-sequence:",
    "non-contiguous-sequence:",
    "sequence-divergence:",
)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _fractional_change(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return 0.0 if candidate == 0 else None
    return candidate / baseline - 1.0


def _lower_is_better_retention(baseline: float, candidate: float) -> float:
    if candidate <= baseline:
        return 1.0
    if baseline <= 0:
        return 0.0
    return baseline / candidate


def _allowed_retention(candidate: float, allowed: float) -> float:
    if candidate <= allowed:
        return 1.0
    if allowed <= 0:
        return 0.0
    return allowed / candidate


def _resource_key(
    value: CommunicationPhaseTraceRecord | CommunicationResourceLifecycleEvent,
) -> tuple[str, str, int]:
    return value.process_group_id, value.logical_operation_id, value.rank


def _resource_lifecycle_violations(trace: CommunicationPhaseTraceSet) -> list[str]:
    violations: set[str] = set()
    records_by_key: dict[tuple[str, str, int], list[CommunicationPhaseTraceRecord]] = defaultdict(
        list
    )
    events_by_key: dict[tuple[str, str, int], list[CommunicationResourceLifecycleEvent]] = (
        defaultdict(list)
    )
    for record in trace.records:
        records_by_key[_resource_key(record)].append(record)
    for event in trace.resource_lifecycle_events:
        events_by_key[_resource_key(event)].append(event)

    for key in sorted(set(events_by_key) - set(records_by_key)):
        violations.add("orphan-resource-event:" + ":".join(map(str, key)))
    for key, records in sorted(records_by_key.items()):
        label = ":".join(map(str, key))
        if len(records) != 1:
            violations.add(f"resource-record-count:{label}:{len(records)}")
            continue
        record = records[0]
        events = events_by_key.get(key, [])
        acquires = [event for event in events if event.event == "acquire"]
        releases = [event for event in events if event.event == "release"]
        if len(acquires) != 1:
            violations.add(f"resource-acquire-count:{label}:{len(acquires)}")
        if len(releases) != 1:
            violations.add(f"resource-release-count:{label}:{len(releases)}")
        for event in events:
            if event.message_bytes != record.message_bytes:
                violations.add(f"resource-byte-mismatch:{label}")
        if len(acquires) == 1 and acquires[0].timestamp_ns > record.gpu_start_timestamp_ns:
            violations.add(f"resource-acquired-after-gpu-start:{label}")
        if len(acquires) == 1 and len(releases) == 1:
            terminal_timestamp = max(
                record.gpu_end_timestamp_ns,
                record.completion_timestamp_ns or record.gpu_end_timestamp_ns,
            )
            if releases[0].timestamp_ns < acquires[0].timestamp_ns:
                violations.add(f"resource-release-before-acquire:{label}")
            if releases[0].timestamp_ns < terminal_timestamp:
                violations.add(f"resource-release-before-completion:{label}")
    return sorted(violations)


def _max_inflight(events: Iterable[CommunicationResourceLifecycleEvent]) -> tuple[int, int]:
    by_rank: dict[int, list[CommunicationResourceLifecycleEvent]] = defaultdict(list)
    for event in events:
        by_rank[event.rank].append(event)
    maximum_bytes = 0
    maximum_collectives = 0
    for rank_events in by_rank.values():
        acquires: dict[int, list[CommunicationResourceLifecycleEvent]] = defaultdict(list)
        releases: dict[int, list[CommunicationResourceLifecycleEvent]] = defaultdict(list)
        for event in rank_events:
            target = acquires if event.event == "acquire" else releases
            target[event.timestamp_ns].append(event)
        inflight_bytes = 0
        inflight_collectives = 0
        for timestamp in sorted(set(acquires) | set(releases)):
            for event in releases[timestamp]:
                inflight_bytes -= event.message_bytes
                inflight_collectives -= 1
            for event in acquires[timestamp]:
                inflight_bytes += event.message_bytes
                inflight_collectives += 1
            maximum_bytes = max(maximum_bytes, inflight_bytes)
            maximum_collectives = max(maximum_collectives, inflight_collectives)
    return maximum_bytes, maximum_collectives


def _artifact_coverage_complete(trace: CommunicationPhaseTraceSet) -> bool:
    coverage = trace.artifact_coverage
    return (
        coverage.claim_scope == "full-run"
        and coverage.expected_units == len(trace.records)
        and coverage.verified_units == coverage.expected_units
        and coverage.reconstructed_units == coverage.expected_units
        and coverage.exact_order_verified
    )


def _timing_provenance_matches(trace: CommunicationPhaseTraceSet) -> bool:
    expected = (
        "kernel-observed"
        if trace.gpu_timing_provenance.capture_kind == "profiler-kernel"
        else "event-bracket"
    )
    return trace.gpu_timestamp_semantics == expected


def _collective_violations(trace: CommunicationPhaseTraceSet) -> list[str]:
    violations: set[str] = set()
    observed_world = {record.rank for record in trace.records}
    expected_world = set(range(trace.world_size))
    if observed_world != expected_world:
        missing_ranks = ",".join(str(rank) for rank in sorted(expected_world - observed_world))
        violations.add("incomplete-world:" + missing_ranks)

    by_group: dict[str, list[CommunicationPhaseTraceRecord]] = defaultdict(list)
    for record in trace.records:
        by_group[record.process_group_id].append(record)
    for group_id, records in sorted(by_group.items()):
        memberships = {record.process_group_ranks for record in records}
        if len(memberships) != 1:
            violations.add(f"group-membership-divergence:{group_id}")
            continue
        membership = next(iter(memberships))
        signatures: dict[int, list[tuple[int, str]]] = {}
        for rank in membership:
            rank_records = [record for record in records if record.rank == rank]
            sequence_ids = [record.communicator_sequence_id for record in rank_records]
            if len(sequence_ids) != len(set(sequence_ids)):
                violations.add(f"duplicate-sequence:{group_id}:rank-{rank}")
            ordered = sorted(
                (
                    record.communicator_sequence_id,
                    record.logical_operation_id,
                )
                for record in rank_records
            )
            if ordered:
                observed_sequences = [sequence for sequence, _ in ordered]
                expected_sequences = list(
                    range(observed_sequences[0], observed_sequences[0] + len(observed_sequences))
                )
                if observed_sequences != expected_sequences:
                    violations.add(f"non-contiguous-sequence:{group_id}:rank-{rank}")
            signatures[rank] = ordered
        reference_rank = membership[0]
        reference = signatures[reference_rank]
        for rank in membership[1:]:
            if signatures[rank] != reference:
                violations.add(f"sequence-divergence:{group_id}:rank-{reference_rank}-vs-{rank}")

    by_collective: dict[tuple[str, str], list[CommunicationPhaseTraceRecord]] = defaultdict(list)
    for record in trace.records:
        by_collective[(record.process_group_id, record.logical_operation_id)].append(record)
    for (group_id, logical_id), records in sorted(by_collective.items()):
        memberships = {record.process_group_ranks for record in records}
        if len(memberships) != 1:
            violations.add(f"collective-membership-divergence:{group_id}:{logical_id}")
            continue
        expected_ranks = set(next(iter(memberships)))
        observed_ranks = [record.rank for record in records]
        if len(observed_ranks) != len(set(observed_ranks)):
            violations.add(f"duplicate-rank-record:{group_id}:{logical_id}")
        if set(observed_ranks) != expected_ranks:
            violations.add(f"incomplete-collective:{group_id}:{logical_id}")
        if len({record.operation for record in records}) != 1:
            violations.add(f"operation-divergence:{group_id}:{logical_id}")
        if len({record.pair_id for record in records}) != 1:
            violations.add(f"pair-divergence:{group_id}:{logical_id}")
        if len({record.pair_role for record in records}) != 1:
            violations.add(f"pair-role-divergence:{group_id}:{logical_id}")

    if len({record.transport for record in trace.records}) != 1:
        violations.add("transport-divergence")
    if len({record.topology_class for record in trace.records}) != 1:
        violations.add("topology-class-divergence")
    return sorted(violations)


def _collective_is_complete(records: Sequence[CommunicationPhaseTraceRecord]) -> bool:
    memberships = {record.process_group_ranks for record in records}
    if len(memberships) != 1:
        return False
    expected = set(next(iter(memberships)))
    observed = [record.rank for record in records]
    return len(observed) == len(set(observed)) and set(observed) == expected


def _summarize(
    trace: CommunicationPhaseTraceSet,
    isolation_reference_ms: dict[str, float],
) -> _RunSummary:
    violations = _collective_violations(trace)
    by_pair: dict[str, dict[str, list[CommunicationPhaseTraceRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in trace.records:
        by_pair[record.pair_id][record.pair_role].append(record)

    pair_completion_ms: list[float] = []
    api_offsets_us: list[float] = []
    realized_offsets_us: list[float] = []
    realized_offset_errors_us: list[float] = []
    contention_stretches: list[float] = []
    actual_overlaps_ms: list[float] = []
    rank_start_skews_us: list[float] = []
    rank_finish_skews_us: list[float] = []
    consumer_slacks_us: list[float] = []
    consumer_waits_us: list[float] = []
    missing_isolation_operations: set[str] = set()
    completed_pair_ids: set[str] = set()
    consumer_timestamps_complete = True

    for pair_id, roles in sorted(by_pair.items()):
        if set(roles) != {"a", "b"}:
            violations.append(f"incomplete-pair:{pair_id}")
            continue
        role_complete = True
        for role in ("a", "b"):
            collectives: dict[tuple[str, str], list[CommunicationPhaseTraceRecord]] = defaultdict(
                list
            )
            for record in roles[role]:
                collectives[(record.process_group_id, record.logical_operation_id)].append(record)
            if not collectives or not all(
                _collective_is_complete(records) for records in collectives.values()
            ):
                role_complete = False
        if not role_complete:
            violations.append(f"incomplete-pair-collective:{pair_id}")
            continue

        completed_pair_ids.add(pair_id)
        records_a = roles["a"]
        records_b = roles["b"]
        if not ({record.rank for record in records_a} & {record.rank for record in records_b}):
            violations.append(f"disjoint-process-groups:{pair_id}")
        starts_a = [record.gpu_start_timestamp_ns for record in records_a]
        starts_b = [record.gpu_start_timestamp_ns for record in records_b]
        ends_a = [record.gpu_end_timestamp_ns for record in records_a]
        ends_b = [record.gpu_end_timestamp_ns for record in records_b]
        api_a = [record.api_launch_timestamp_ns for record in records_a]
        api_b = [record.api_launch_timestamp_ns for record in records_b]
        requested_a = [record.requested_offset_us for record in records_a]
        requested_b = [record.requested_offset_us for record in records_b]
        if len(set(requested_a)) != 1 or len(set(requested_b)) != 1:
            violations.append(f"requested-offset-divergence:{pair_id}")

        gpu_start_a = float(_percentile([float(value) for value in starts_a], 0.5))
        gpu_start_b = float(_percentile([float(value) for value in starts_b], 0.5))
        api_start_a = float(_percentile([float(value) for value in api_a], 0.5))
        api_start_b = float(_percentile([float(value) for value in api_b], 0.5))
        requested_offset_us = float(_percentile(requested_b, 0.5)) - float(
            _percentile(requested_a, 0.5)
        )
        realized_offset_us = (gpu_start_b - gpu_start_a) / 1_000
        api_offsets_us.append((api_start_b - api_start_a) / 1_000)
        realized_offsets_us.append(realized_offset_us)
        realized_offset_errors_us.append(abs(realized_offset_us - requested_offset_us))
        pair_completion_ms.append(
            (max([*ends_a, *ends_b]) - min([*starts_a, *starts_b])) / 1_000_000
        )
        actual_overlaps_ms.append(
            max(0, min(max(ends_a), max(ends_b)) - max(min(starts_a), min(starts_b))) / 1_000_000
        )

        for role_records in (records_a, records_b):
            collectives = defaultdict(list)
            for record in role_records:
                collectives[(record.process_group_id, record.logical_operation_id)].append(record)
            for collective_records in collectives.values():
                starts = [record.gpu_start_timestamp_ns for record in collective_records]
                ends = [record.gpu_end_timestamp_ns for record in collective_records]
                rank_start_skews_us.append((max(starts) - min(starts)) / 1_000)
                rank_finish_skews_us.append((max(ends) - min(ends)) / 1_000)
                operation = collective_records[0].operation
                isolated_ms = isolation_reference_ms.get(operation)
                if isolated_ms is None:
                    missing_isolation_operations.add(operation)
                else:
                    duration_ms = (max(ends) - min(starts)) / 1_000_000
                    contention_stretches.append(duration_ms / isolated_ms)

        for record in records_b:
            if record.consumer_timestamp_ns is None:
                consumer_timestamps_complete = False
                continue
            slack_us = (record.consumer_timestamp_ns - record.gpu_end_timestamp_ns) / 1_000
            consumer_slacks_us.append(slack_us)
            consumer_waits_us.append(max(0.0, -slack_us))

    resource_violations = _resource_lifecycle_violations(trace)
    max_inflight_bytes, max_inflight_collectives = _max_inflight(trace.resource_lifecycle_events)
    artifact_coverage_complete = _artifact_coverage_complete(trace)
    metrics = CommunicationPhaseRunMetrics(
        timestamp_domain=trace.timestamp_domain,
        gpu_timestamp_semantics=trace.gpu_timestamp_semantics,
        clock_sync_error_bound_us=trace.clock_sync_error_bound_us,
        transport=(
            next(iter({record.transport for record in trace.records}))
            if len({record.transport for record in trace.records}) == 1
            else None
        ),
        topology_class=(
            next(iter({record.topology_class for record in trace.records}))
            if len({record.topology_class for record in trace.records}) == 1
            else None
        ),
        pair_count=len(by_pair),
        completed_pair_count=len(completed_pair_ids),
        step_time_p50_ms=_percentile(trace.step_time_ms, 0.50),
        step_time_p95_ms=_percentile(trace.step_time_ms, 0.95),
        pair_completion_p50_ms=_percentile(pair_completion_ms, 0.50),
        pair_completion_p95_ms=_percentile(pair_completion_ms, 0.95),
        pair_completion_p99_ms=_percentile(pair_completion_ms, 0.99),
        api_launch_offset_p50_us=_percentile(api_offsets_us, 0.50),
        realized_offset_p50_us=_percentile(realized_offsets_us, 0.50),
        realized_offset_p95_us=_percentile(realized_offsets_us, 0.95),
        realized_offset_error_p95_us=_percentile(realized_offset_errors_us, 0.95),
        contention_stretch_p95=_percentile(contention_stretches, 0.95),
        actual_overlap_p50_ms=_percentile(actual_overlaps_ms, 0.50),
        rank_start_skew_p95_us=_percentile(rank_start_skews_us, 0.95),
        rank_finish_skew_p95_us=_percentile(rank_finish_skews_us, 0.95),
        consumer_slack_p50_us=_percentile(consumer_slacks_us, 0.50),
        consumer_wait_p95_us=_percentile(consumer_waits_us, 0.95),
        consumer_deadline_miss_count=sum(slack < 0 for slack in consumer_slacks_us),
        max_inflight_bytes=max_inflight_bytes,
        max_inflight_collectives=max_inflight_collectives,
        collective_order_safe=not violations,
        order_violations=sorted(set(violations)),
        resource_lifecycle_safe=not resource_violations,
        resource_lifecycle_violations=resource_violations,
        artifact_coverage_complete=artifact_coverage_complete,
    )
    return _RunSummary(
        metrics=metrics,
        pair_ids=frozenset(by_pair),
        completed_pair_ids=frozenset(completed_pair_ids),
        missing_isolation_operations=frozenset(missing_isolation_operations),
        consumer_timestamps_complete=consumer_timestamps_complete,
        artifact_coverage_complete=artifact_coverage_complete,
    )


def summarize_communication_phase(
    trace: CommunicationPhaseTraceSet,
) -> CommunicationPhaseRunMetrics:
    """Summarize one trace without creating a cross-cell or global score."""

    return _summarize(trace, trace.isolated_latency_ms_by_operation).metrics


def _required_metric(
    baseline: float | None,
    candidate: float | None,
    missing_name: str,
    unresolved: set[str],
) -> tuple[float, float] | None:
    if baseline is None or candidate is None:
        unresolved.add(missing_name)
        return None
    return baseline, candidate


def evaluate_communication_phase_regression(
    baseline: CommunicationPhaseTraceSet,
    candidate: CommunicationPhaseTraceSet,
    policy: CommunicationPhaseRegressionPolicy,
    *,
    regime: str,
    load_ratio: float,
) -> CommunicationPhaseRegressionResult:
    """Compare traces only inside one exact cell/workload and emit one load-ladder cell."""

    if baseline.cell_identity_sha256 != candidate.cell_identity_sha256:
        raise ValueError("communication phase regression cannot compare different cells")
    if baseline.workload_sha256 != candidate.workload_sha256:
        raise ValueError("communication phase regression cannot compare different workloads")
    if baseline.world_size != candidate.world_size:
        raise ValueError("communication phase regression requires identical world_size")
    if baseline.timestamp_domain != candidate.timestamp_domain:
        raise ValueError("communication phase regression requires one timestamp domain")
    if baseline.gpu_timestamp_semantics != candidate.gpu_timestamp_semantics:
        raise ValueError("communication phase regression requires identical GPU timing semantics")
    if baseline.run_id == candidate.run_id:
        raise ValueError("baseline and candidate must be distinct runs")
    if baseline.execution_identity != candidate.execution_identity:
        raise ValueError("communication phase regression requires one execution identity")
    if set(baseline.experiment_provenance.independent_process_run_ids) & set(
        candidate.experiment_provenance.independent_process_run_ids
    ) or set(baseline.experiment_provenance.independent_process_artifact_sha256) & set(
        candidate.experiment_provenance.independent_process_artifact_sha256
    ):
        raise ValueError("baseline and candidate require independent process runs")

    baseline_summary = _summarize(baseline, baseline.isolated_latency_ms_by_operation)
    candidate_summary = _summarize(candidate, candidate.isolated_latency_ms_by_operation)
    baseline_metrics = baseline_summary.metrics
    candidate_metrics = candidate_summary.metrics
    failures: set[str] = set()
    unresolved: set[str] = set()

    if baseline_metrics.order_violations:
        unresolved.add("BASELINE_TRACE_INTEGRITY_INVALID")
    if candidate_metrics.order_violations:
        failures.add("TRACE_INTEGRITY_VIOLATION")
        if any(
            violation.startswith(_ORDER_VIOLATION_PREFIXES)
            for violation in candidate_metrics.order_violations
        ):
            failures.add("COLLECTIVE_ORDER_VIOLATION")
        if any(
            violation.startswith("requested-offset-divergence:")
            for violation in candidate_metrics.order_violations
        ):
            failures.add("REQUESTED_OFFSET_DIVERGENCE")
        if any(
            violation.startswith("disjoint-process-groups:")
            for violation in candidate_metrics.order_violations
        ):
            failures.add("PROCESS_GROUPS_DO_NOT_OVERLAP")
    if baseline_metrics.resource_lifecycle_violations:
        unresolved.add("BASELINE_RESOURCE_LIFECYCLE_INVALID")
    if candidate_metrics.resource_lifecycle_violations:
        failures.add("RESOURCE_LIFECYCLE_VIOLATION")
    if not baseline_summary.artifact_coverage_complete:
        unresolved.add("BASELINE_ARTIFACT_COVERAGE_INCOMPLETE")
    if not candidate_summary.artifact_coverage_complete:
        failures.add("ARTIFACT_COVERAGE_INCOMPLETE")
    if not _timing_provenance_matches(baseline):
        unresolved.add("BASELINE_GPU_TIMING_PROVENANCE_MISMATCH")
    if not _timing_provenance_matches(candidate):
        failures.add("GPU_TIMING_PROVENANCE_MISMATCH")
    if (
        baseline.experiment_provenance.phase != "confirmation"
        or len(baseline.experiment_provenance.independent_process_run_ids)
        < policy.min_confirmation_process_runs
    ):
        unresolved.add("BASELINE_CONFIRMATION_EVIDENCE_INSUFFICIENT")
    if (
        candidate.experiment_provenance.phase != "confirmation"
        or len(candidate.experiment_provenance.independent_process_run_ids)
        < policy.min_confirmation_process_runs
    ):
        unresolved.add("CANDIDATE_CONFIRMATION_EVIDENCE_INSUFFICIENT")
    if baseline_summary.pair_ids != candidate_summary.pair_ids:
        failures.add("PAIR_COVERAGE_MISMATCH")
    if baseline_summary.completed_pair_ids != baseline_summary.pair_ids:
        unresolved.add("BASELINE_PAIR_TRACE_INCOMPLETE")
    if candidate_summary.completed_pair_ids != candidate_summary.pair_ids:
        failures.add("CANDIDATE_PAIR_TRACE_INCOMPLETE")
    if (
        baseline_summary.missing_isolation_operations
        or candidate_summary.missing_isolation_operations
    ):
        unresolved.add("ISOLATED_LATENCY_REFERENCE_MISSING")
    if baseline.gpu_timestamp_semantics != "kernel-observed":
        unresolved.add("KERNEL_OBSERVED_TIMESTAMPS_REQUIRED")
    if baseline.clock_sync_error_bound_us > policy.max_clock_sync_error_us:
        unresolved.add("BASELINE_CLOCK_SYNC_ERROR_BOUND_EXCEEDED")
    if candidate.clock_sync_error_bound_us > policy.max_clock_sync_error_us:
        failures.add("CLOCK_SYNC_ERROR_BOUND_EXCEEDED")
    if (
        baseline_metrics.transport != candidate_metrics.transport
        or baseline_metrics.topology_class != candidate_metrics.topology_class
    ):
        failures.add("CELL_METADATA_MISMATCH")

    components: dict[str, float | None] = {
        "comm_phase_sweep": None,
        "comm_contention_stretch": None,
        "realized_offset_stability": None,
        "collective_order_safety": 1.0 if candidate_metrics.collective_order_safe else 0.0,
        "windowed_scheduler_gain": None,
        "consumer_slack_utilization": None,
    }

    pair_times = _required_metric(
        baseline_metrics.pair_completion_p95_ms,
        candidate_metrics.pair_completion_p95_ms,
        "PAIR_COMPLETION_EVIDENCE_MISSING",
        unresolved,
    )
    pair_retention: float | None = None
    if pair_times is not None:
        baseline_pair, candidate_pair = pair_times
        pair_retention = _lower_is_better_retention(baseline_pair, candidate_pair)
        components["comm_phase_sweep"] = pair_retention
        if candidate_pair > baseline_pair * (
            1 + policy.max_pair_completion_p95_regression_fraction
        ):
            failures.add("PAIR_COMPLETION_P95_REGRESSION")

    step_times = _required_metric(
        baseline_metrics.step_time_p95_ms,
        candidate_metrics.step_time_p95_ms,
        "STEP_TIME_EVIDENCE_MISSING",
        unresolved,
    )
    step_retention: float | None = None
    if step_times is not None:
        baseline_step, candidate_step = step_times
        step_retention = _lower_is_better_retention(baseline_step, candidate_step)
        if candidate_step > baseline_step * (1 + policy.max_step_time_p95_regression_fraction):
            failures.add("STEP_TIME_P95_REGRESSION")

    stretches = _required_metric(
        baseline_metrics.contention_stretch_p95,
        candidate_metrics.contention_stretch_p95,
        "CONTENTION_STRETCH_EVIDENCE_MISSING",
        unresolved,
    )
    stretch_retention: float | None = None
    if stretches is not None:
        baseline_stretch, candidate_stretch = stretches
        stretch_retention = _lower_is_better_retention(baseline_stretch, candidate_stretch)
        components["comm_contention_stretch"] = stretch_retention
        if candidate_stretch > baseline_stretch * (
            1 + policy.max_contention_stretch_p95_regression_fraction
        ):
            failures.add("CONTENTION_STRETCH_P95_REGRESSION")

    offset_error = candidate_metrics.realized_offset_error_p95_us
    if offset_error is None:
        unresolved.add("REALIZED_OFFSET_EVIDENCE_MISSING")
        offset_score = None
    else:
        offset_score = _allowed_retention(offset_error, policy.max_realized_offset_error_p95_us)
        components["realized_offset_stability"] = offset_score
        if offset_error > policy.max_realized_offset_error_p95_us:
            failures.add("REALIZED_OFFSET_UNSTABLE")

    rank_start = _required_metric(
        baseline_metrics.rank_start_skew_p95_us,
        candidate_metrics.rank_start_skew_p95_us,
        "RANK_START_SKEW_EVIDENCE_MISSING",
        unresolved,
    )
    rank_finish = _required_metric(
        baseline_metrics.rank_finish_skew_p95_us,
        candidate_metrics.rank_finish_skew_p95_us,
        "RANK_FINISH_SKEW_EVIDENCE_MISSING",
        unresolved,
    )
    rank_skew_score: float | None = None
    if rank_start is not None and rank_finish is not None:
        baseline_start, candidate_start = rank_start
        baseline_finish, candidate_finish = rank_finish
        allowed_start = (
            baseline_start * (1 + policy.max_rank_skew_regression_fraction)
            + policy.rank_skew_absolute_allowance_us
        )
        allowed_finish = (
            baseline_finish * (1 + policy.max_rank_skew_regression_fraction)
            + policy.rank_skew_absolute_allowance_us
        )
        rank_skew_score = min(
            _allowed_retention(candidate_start, allowed_start),
            _allowed_retention(candidate_finish, allowed_finish),
        )
        if candidate_start > allowed_start:
            failures.add("RANK_START_SKEW_REGRESSION")
        if candidate_finish > allowed_finish:
            failures.add("RANK_FINISH_SKEW_REGRESSION")

    consumer_score: float | None
    consumers_complete = (
        baseline_summary.consumer_timestamps_complete
        and candidate_summary.consumer_timestamps_complete
    )
    consumer_wait = _required_metric(
        baseline_metrics.consumer_wait_p95_us,
        candidate_metrics.consumer_wait_p95_us,
        "CONSUMER_WAIT_EVIDENCE_MISSING",
        unresolved,
    )
    if not consumers_complete or consumer_wait is None:
        unresolved.add("CONSUMER_TIMESTAMP_EVIDENCE_INCOMPLETE")
        consumer_score = None
    else:
        baseline_wait, candidate_wait = consumer_wait
        allowed_wait = baseline_wait + policy.consumer_wait_p95_allowance_us
        consumer_score = _allowed_retention(candidate_wait, allowed_wait)
        components["consumer_slack_utilization"] = consumer_score
        if candidate_wait > allowed_wait:
            failures.add("CONSUMER_DEADLINE_REGRESSION")

    if policy.max_outstanding_bytes is None:
        bytes_score = _lower_is_better_retention(
            float(baseline_metrics.max_inflight_bytes),
            float(candidate_metrics.max_inflight_bytes),
        )
    else:
        bytes_score = _allowed_retention(
            float(candidate_metrics.max_inflight_bytes),
            float(policy.max_outstanding_bytes),
        )
        if candidate_metrics.max_inflight_bytes > policy.max_outstanding_bytes:
            failures.add("OUTSTANDING_BYTE_BUDGET_EXCEEDED")
    if policy.max_outstanding_collectives is None:
        collectives_score = _lower_is_better_retention(
            float(baseline_metrics.max_inflight_collectives),
            float(candidate_metrics.max_inflight_collectives),
        )
    else:
        collectives_score = _allowed_retention(
            float(candidate_metrics.max_inflight_collectives),
            float(policy.max_outstanding_collectives),
        )
        if candidate_metrics.max_inflight_collectives > policy.max_outstanding_collectives:
            failures.add("OUTSTANDING_COLLECTIVE_BUDGET_EXCEEDED")
    resource_score = min(bytes_score, collectives_score)
    if step_retention is not None:
        components["windowed_scheduler_gain"] = math.sqrt(step_retention * resource_score)

    order_score = 1.0 if candidate_metrics.collective_order_safe else 0.0
    if rank_skew_score is not None:
        components["collective_order_safety"] = min(order_score, rank_skew_score)

    comparison = CommunicationPhaseRegressionMetrics(
        pair_completion_gain_fraction=(
            None
            if baseline_metrics.pair_completion_p50_ms is None
            or candidate_metrics.pair_completion_p50_ms is None
            or baseline_metrics.pair_completion_p50_ms <= 0
            else 1
            - candidate_metrics.pair_completion_p50_ms / baseline_metrics.pair_completion_p50_ms
        ),
        step_time_gain_fraction=(
            None
            if baseline_metrics.step_time_p50_ms is None
            or candidate_metrics.step_time_p50_ms is None
            else 1 - candidate_metrics.step_time_p50_ms / baseline_metrics.step_time_p50_ms
        ),
        contention_stretch_change_fraction=_fractional_change(
            baseline_metrics.contention_stretch_p95,
            candidate_metrics.contention_stretch_p95,
        ),
        consumer_slack_utilization_p50=(
            None
            if baseline_metrics.consumer_slack_p50_us is None
            or candidate_metrics.consumer_slack_p50_us is None
            or baseline_metrics.consumer_slack_p50_us <= 0
            else (baseline_metrics.consumer_slack_p50_us - candidate_metrics.consumer_slack_p50_us)
            / baseline_metrics.consumer_slack_p50_us
        ),
        rank_start_skew_change_fraction=_fractional_change(
            baseline_metrics.rank_start_skew_p95_us,
            candidate_metrics.rank_start_skew_p95_us,
        ),
        rank_finish_skew_change_fraction=_fractional_change(
            baseline_metrics.rank_finish_skew_p95_us,
            candidate_metrics.rank_finish_skew_p95_us,
        ),
    )

    load_cell: SystemPathLoadCell | None = None
    if not unresolved and all(value is not None for value in components.values()):
        fairness_score = min(order_score, float(rank_skew_score), float(consumer_score))
        load_cell = SystemPathLoadCell(
            domain="distributed-communication",
            protocol_id="communication-concurrency-core-v1",
            regime=regime,  # type: ignore[arg-type]
            load_ratio=load_ratio,
            offered_work=len(baseline_summary.pair_ids),
            completed_work=len(
                candidate_summary.completed_pair_ids & baseline_summary.completed_pair_ids
            ),
            goodput_score=float(step_retention),
            tail_score=float(pair_retention),
            jitter_score=float(offset_score),
            overlap_progress_score=float(stretch_retention),
            resource_stability_score=resource_score,
            fairness_score=fairness_score,
            p99_status=(
                "official" if len(candidate_summary.completed_pair_ids) >= 1_000 else "exploratory"
            ),
            hard_gate_failure_codes=sorted(failures),
            evidence_digests=sorted(set(baseline.evidence_digests + candidate.evidence_digests)),
        )

    status = "fail" if failures else "unresolved" if unresolved else "pass"
    return CommunicationPhaseRegressionResult(
        status=status,
        cell_identity_sha256=baseline.cell_identity_sha256,
        workload_sha256=baseline.workload_sha256,
        baseline_framework=baseline.framework,
        candidate_framework=candidate.framework,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        world_size=baseline.world_size,
        components=components,
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        comparison=comparison,
        load_cell=load_cell,
        failure_codes=sorted(failures),
        unresolved_reasons=sorted(unresolved),
    )
