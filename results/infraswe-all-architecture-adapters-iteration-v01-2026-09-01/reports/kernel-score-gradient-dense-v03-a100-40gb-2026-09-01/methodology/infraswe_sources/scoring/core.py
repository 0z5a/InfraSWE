from __future__ import annotations

from infraswe.models.score import CoreComponents
from infraswe.models.task import TaskPackage
from infraswe.models.trial import ReplayResult, Usage


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _assertion_ratio(replays: list[ReplayResult], prefix: str, inverse: bool = False) -> float:
    values: list[float] = []
    for replay in replays:
        selected = [
            float(passed)
            for name, passed in replay.assertions.items()
            if (name.startswith(prefix)) ^ inverse
        ]
        if selected:
            values.append(_mean(selected))
    return _mean(values)


def core_components(
    task: TaskPackage,
    replays: list[ReplayResult],
    usage: Usage,
    protocol_complete: bool,
) -> CoreComponents:
    regression = _assertion_ratio(replays, "regression.")
    correctness_values: list[float] = []
    for replay in replays:
        selected = [
            float(passed)
            for name, passed in replay.assertions.items()
            if not name.startswith("regression.")
        ]
        if selected:
            correctness_values.append(_mean(selected))
    correctness = _mean(correctness_values)
    fresh_replay = _mean([float(replay.passed) for replay in replays])
    resolved = bool(replays) and replays[0].passed
    if resolved:
        reference = task.scoring.reference_wall_time_sec
        efficiency = min(1.0, reference / max(usage.wall_time_sec, 0.001))
    else:
        efficiency = 0.0
    return CoreComponents(
        correctness=correctness,
        regression=regression,
        fresh_replay=fresh_replay,
        efficiency=efficiency,
        protocol=float(protocol_complete),
    )


def core_score(components: CoreComponents, protocol_gate: bool) -> float:
    if not protocol_gate:
        return 0.0
    value = 100 * (
        0.55 * components.correctness
        + 0.20 * components.regression
        + 0.10 * components.fresh_replay
        + 0.10 * components.efficiency
        + 0.05 * components.protocol
    )
    return round(value, 6)
