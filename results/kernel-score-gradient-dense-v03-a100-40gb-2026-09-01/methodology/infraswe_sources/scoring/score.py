from __future__ import annotations

from infraswe.models.artifact import ArtifactManifest
from infraswe.models.score import KernelScoreEnvelope, ScoreResult
from infraswe.models.task import TaskPackage
from infraswe.models.trial import TrialRecord
from infraswe.scoring.core import core_components, core_score
from infraswe.scoring.gates import evaluate_gates
from infraswe.scoring.infra import infra_components, infra_score


def _coverage(task: TaskPackage, record: TrialRecord, manifest_valid: bool) -> float:
    checks = [manifest_valid, bool(record.replays)]
    for replay in record.replays:
        checks.extend(
            [
                bool(replay.assertions),
                task.scoring.slo_metric in replay.metrics,
                task.scoring.resource_metric in replay.metrics,
                "topology_robustness_ratio" in replay.metrics,
                "observability_ratio" in replay.metrics,
                bool(replay.faults),
                bool(replay.policy),
            ]
        )
    return sum(checks) / len(checks) if checks else 0.0


def score_trial(
    task: TaskPackage,
    record: TrialRecord,
    manifest: ArtifactManifest,
    run_dir,
) -> ScoreResult:
    manifest_valid = not manifest.verify(run_dir)
    gate, protocol_gate, catastrophic_gate = evaluate_gates(record.replays, manifest_valid)
    protocol_complete = manifest_valid and all(
        replay.policy.get("passed", False) for replay in record.replays
    )
    core_parts = core_components(task, record.replays, record.usage, protocol_complete)
    infra_parts = infra_components(task, record.replays)
    core_100 = core_score(core_parts, protocol_gate)
    infra_ext_100 = infra_score(infra_parts)
    if catastrophic_gate and core_100 > 0 and infra_ext_100 > 0:
        infra_total = 100 * (core_100 / 100) ** 0.40 * (infra_ext_100 / 100) ** 0.60
    else:
        infra_total = 0.0
    resolved = bool(record.replays) and record.replays[0].passed
    stable = len(record.replays) == task.replay.count and all(
        replay.passed for replay in record.replays
    )
    return ScoreResult(
        resolved_at_1=resolved,
        stable_resolved_at_1=stable,
        coverage=_coverage(task, record, manifest_valid),
        gate=gate,
        core_components=core_parts,
        infra_components=infra_parts,
        core_100=core_100,
        infra_ext_100=infra_ext_100,
        infra_total=round(infra_total, 6),
        raw={
            "manifest_valid": manifest_valid,
            "protocol_gate": protocol_gate,
            "catastrophic_gate": catastrophic_gate,
            "usage": record.usage.model_dump(mode="json"),
            "replays": [replay.model_dump(mode="json") for replay in record.replays],
        },
        kernel=KernelScoreEnvelope(applicable=False),
    )
