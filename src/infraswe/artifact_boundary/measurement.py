from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.artifact_boundary import (
    MeasurementIntegrityReport,
    OfficialTimingSample,
)


def build_timing_integrity_report(
    samples: Sequence[OfficialTimingSample],
    *,
    expected_repetitions: int,
    timer_policy_id: str = "trusted-host-device-dual-clock-v1",
) -> MeasurementIntegrityReport:
    """Validate paired official timing completeness before it becomes authoritative."""

    if expected_repetitions < 2:
        raise ValueError("official paired timing requires at least two repetitions")
    failures: list[str] = []
    by_pair: defaultdict[str, list[OfficialTimingSample]] = defaultdict(list)
    sample_ids = [item.sample_id for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        failures.append("TIMING_SAMPLE_ID_DUPLICATE")
    for sample in samples:
        by_pair[sample.pair_id].append(sample)
        if sample.completion_counter_after <= sample.completion_counter_before:
            failures.append("TIMING_SAMPLE_EARLY_RETURN:" + sample.sample_id)
        if sample.device_elapsed_seconds is None and sample.synchronization not in {
            "blocking-contract",
            "distributed-barrier-and-device-sync",
        }:
            failures.append("TIMING_DEVICE_CLOCK_MISSING:" + sample.sample_id)
    expected_pair_ids = {f"pair-{index}" for index in range(1, expected_repetitions + 1)}
    if set(by_pair) != expected_pair_ids:
        failures.append("TIMING_PAIR_SET_INCOMPLETE")
    positions: Counter[str] = Counter()
    for pair_id, pair_samples in by_pair.items():
        roles = {item.role for item in pair_samples}
        pair_positions = {item.pair_position for item in pair_samples}
        repetitions = {item.repetition for item in pair_samples}
        if len(pair_samples) != 2 or roles != {"baseline", "candidate"}:
            failures.append("TIMING_PAIR_ROLE_INCOMPLETE:" + pair_id)
        if len(pair_positions) != 1 or len(repetitions) != 1:
            failures.append("TIMING_PAIR_METADATA_CONFLICT:" + pair_id)
        elif pair_positions:
            positions.update(pair_positions)
    if not {"baseline-first", "candidate-first"} <= set(positions):
        failures.append("TIMING_ORDER_NOT_COUNTERBALANCED")
    preliminary = MeasurementIntegrityReport(
        timer_policy_id=timer_policy_id,
        paired_order_policy="counterbalanced",
        samples=list(samples),
        all_samples_retained=not any("INCOMPLETE" in item for item in failures),
        status="PASS" if not failures else "BENCHMARK_DEFECT",
        failure_codes=sorted(set(failures)),
        report_sha256="sha256:" + "0" * 64,
    )
    material = preliminary.model_dump(mode="json", exclude={"report_sha256"})
    return preliminary.model_copy(update={"report_sha256": canonical_sha256(material)})


def audit_timing_integrity(report: MeasurementIntegrityReport) -> list[str]:
    material = report.model_dump(mode="json", exclude={"report_sha256"})
    return (
        []
        if report.report_sha256 == canonical_sha256(material)
        else ["TIMING_INTEGRITY_REPORT_DIGEST_MISMATCH"]
    )
