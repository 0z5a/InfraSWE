from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from infraswe.models.evidence import NormalizedProfilerMetrics, ProfilerEvidence


def _record(
    *,
    kind: Literal["framework", "system-trace", "kernel-counter"],
    backend: str,
    version: str | None,
    metric_map_version: str,
    applicable: bool,
    payload: Mapping[str, Any] | None,
    raw_evidence: Sequence[str],
    raw_evidence_digests: Sequence[str],
    unavailable_reason: str,
) -> ProfilerEvidence:
    if not applicable:
        return ProfilerEvidence(
            collector_kind=kind,
            collector_backend=backend,
            collector_version=version,
            metric_map_version=metric_map_version,
            status="not_applicable",
            confidence="not_applicable",
            unavailable_reasons={"collector": unavailable_reason},
        )
    if payload is None:
        return ProfilerEvidence(
            collector_kind=kind,
            collector_backend=backend,
            collector_version=version,
            metric_map_version=metric_map_version,
            status="unresolved",
            confidence="low",
            unavailable_reasons={"collector": unavailable_reason},
            raw_evidence=list(raw_evidence),
            raw_evidence_digests=list(raw_evidence_digests),
        )
    normalized = NormalizedProfilerMetrics.model_validate(dict(payload))
    return ProfilerEvidence(
        collector_kind=kind,
        collector_backend=backend,
        collector_version=version,
        metric_map_version=metric_map_version,
        status="captured",
        normalized=normalized,
        raw_evidence=list(raw_evidence),
        raw_evidence_digests=list(raw_evidence_digests),
        confidence="high",
    )


def framework_compile_evidence(
    payload: Mapping[str, Any] | None,
    *,
    applicable: bool,
    backend: str = "torch-compile",
    version: str | None = None,
    raw_evidence: Sequence[str] = (),
    raw_evidence_digests: Sequence[str] = (),
) -> ProfilerEvidence:
    return _record(
        kind="framework",
        backend=backend,
        version=version,
        metric_map_version="framework-compile-v0.4",
        applicable=applicable,
        payload=payload,
        raw_evidence=raw_evidence,
        raw_evidence_digests=raw_evidence_digests,
        unavailable_reason=(
            "framework compilation is not applicable to this native path"
            if not applicable
            else "framework compile evidence was not captured"
        ),
    )


def system_trace_evidence(
    payload: Mapping[str, Any] | None,
    *,
    backend: str = "nsys",
    version: str | None = None,
    raw_evidence: Sequence[str] = (),
    raw_evidence_digests: Sequence[str] = (),
) -> ProfilerEvidence:
    return _record(
        kind="system-trace",
        backend=backend,
        version=version,
        metric_map_version="system-trace-v0.4",
        applicable=True,
        payload=payload,
        raw_evidence=raw_evidence,
        raw_evidence_digests=raw_evidence_digests,
        unavailable_reason="system trace collector failed or was unavailable",
    )


def kernel_counter_evidence(
    payload: Mapping[str, Any] | None,
    *,
    applicable: bool,
    backend: str = "ncu",
    version: str | None = None,
    metric_map_version: str = "nvidia-ncu-v3",
    raw_evidence: Sequence[str] = (),
    raw_evidence_digests: Sequence[str] = (),
) -> ProfilerEvidence:
    return _record(
        kind="kernel-counter",
        backend=backend,
        version=version,
        metric_map_version=metric_map_version,
        applicable=applicable,
        payload=payload,
        raw_evidence=raw_evidence,
        raw_evidence_digests=raw_evidence_digests,
        unavailable_reason=(
            "kernel counters are unsupported for this backend or workload"
            if not applicable
            else "kernel counter evidence was not captured"
        ),
    )
