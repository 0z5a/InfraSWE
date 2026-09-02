from __future__ import annotations

from infraswe.models.score import InfraComponents
from infraswe.models.task import TaskPackage
from infraswe.models.trial import ReplayResult


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _metric(replays: list[ReplayResult], name: str, default: float = 0.0) -> float:
    return _clamp(
        _mean([replay.metrics[name] for replay in replays if name in replay.metrics], default)
    )


def infra_components(task: TaskPackage, replays: list[ReplayResult]) -> InfraComponents:
    safety_values: list[float] = []
    for replay in replays:
        selected = [
            float(passed)
            for name, passed in replay.assertions.items()
            if name.startswith("safety.")
        ]
        if selected:
            safety_values.append(_mean(selected))
    fault_recovery = _mean(
        [
            _clamp(float(replay.faults.get("score", replay.faults.get("passed", False))))
            for replay in replays
        ]
    )
    return InfraComponents(
        slo_goodput=_metric(replays, task.scoring.slo_metric),
        fault_recovery=fault_recovery,
        safety_rollback=_clamp(_mean(safety_values)),
        resource_efficiency=_metric(replays, task.scoring.resource_metric),
        topology_robustness=_metric(replays, "topology_robustness_ratio"),
        observability=_metric(replays, "observability_ratio"),
    )


def infra_score(components: InfraComponents) -> float:
    value = 100 * (
        0.25 * components.slo_goodput
        + 0.20 * components.fault_recovery
        + 0.20 * components.safety_rollback
        + 0.15 * components.resource_efficiency
        + 0.10 * components.topology_robustness
        + 0.10 * components.observability
    )
    return round(value, 6)
