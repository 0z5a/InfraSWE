from __future__ import annotations

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    ResourceEnvelope,
    ResourceFeasibilityResult,
    ResourcePhaseResolution,
    ResourceUsageObservation,
    ResourceUsageVerdict,
    RunnerSnapshot,
)


def evaluate_resource_feasibility(
    envelope: ResourceEnvelope,
    snapshot: RunnerSnapshot,
) -> ResourceFeasibilityResult:
    phases: list[ResourcePhaseResolution] = []
    for phase, phase_envelope in envelope.phases.items():
        missing: list[str] = []
        busy: list[str] = []
        unresolved: list[str] = []
        for resource_id, limit in phase_envelope.resources.items():
            available = snapshot.resources.get(resource_id)
            if available is None:
                unresolved.append(resource_id)
                continue
            required = max(limit.minimum_required, limit.reserved)
            if available.total < required:
                missing.append(resource_id)
            elif available.allocatable < required or (
                limit.exclusive and not available.exclusive_available
            ):
                busy.append(resource_id)
        status = (
            "unschedulable"
            if missing
            else "capacity-unavailable"
            if busy
            else "unresolved"
            if unresolved
            else "feasible"
        )
        phases.append(
            ResourcePhaseResolution(
                phase=phase,
                status=status,
                missing_resources=sorted([*missing, *unresolved]),
                busy_resources=sorted(busy),
            )
        )
    statuses = {item.status for item in phases}
    overall = (
        "unschedulable"
        if "unschedulable" in statuses
        else "capacity-unavailable"
        if "capacity-unavailable" in statuses
        else "unresolved"
        if "unresolved" in statuses
        else "feasible"
    )
    material = {
        "envelope_sha256": envelope.envelope_sha256,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "phases": [item.model_dump(mode="json") for item in phases],
    }
    return ResourceFeasibilityResult(
        status=overall,
        phases=phases,
        request_sha256=canonical_sha256(material),
    )


def evaluate_resource_usage(
    envelope: ResourceEnvelope,
    observations: list[ResourceUsageObservation],
) -> ResourceUsageVerdict:
    failures: list[str] = []
    owner = "none"
    status = "PASS"
    phase_limits = {
        (phase, resource_id): limit
        for phase, phase_envelope in envelope.phases.items()
        for resource_id, limit in phase_envelope.resources.items()
    }
    for observation in observations:
        limit = phase_limits.get((observation.phase, observation.resource_id))
        if limit is None:
            failures.append("RESOURCE_OBSERVATION_UNDECLARED:" + observation.resource_id)
            status = "BENCHMARK_DEFECT"
            owner = "benchmark"
            continue
        if observation.external_interference:
            failures.append("EXTERNAL_INTERFERENCE:" + observation.resource_id)
            if status != "BENCHMARK_DEFECT":
                status = "INFRA_INVALID"
                owner = "infrastructure"
        if observation.verifier_peak > limit.measurement_reserve:
            failures.append("VERIFIER_RESERVE_EXCEEDED:" + observation.resource_id)
            status = "BENCHMARK_DEFECT"
            owner = "benchmark"
        if (
            limit.candidate_limit is not None
            and observation.candidate_peak > limit.candidate_limit
            and status not in {"BENCHMARK_DEFECT", "INFRA_INVALID"}
        ):
            failures.append("RESOURCE_CONTRACT_VIOLATION:" + observation.resource_id)
            status = "VALID_FAIL"
            owner = "candidate"
    return ResourceUsageVerdict(status=status, owner=owner, failure_codes=failures)
