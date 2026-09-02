from __future__ import annotations

from infraswe.models.score import GateResult
from infraswe.models.trial import ReplayResult

CATASTROPHIC_FAILURES = {
    "DATA_CORRUPTION",
    "SILENT_FALLBACK",
    "RESOURCE_LEAK",
    "POLICY_VIOLATION",
    "DEADLOCK",
}


def evaluate_gates(
    replays: list[ReplayResult], manifest_valid: bool
) -> tuple[GateResult, bool, bool]:
    reasons: list[str] = []
    protocol_gate = manifest_valid
    if not manifest_valid:
        reasons.append("ARTIFACT_INVALID")
    if not replays or any(not replay.assertions for replay in replays):
        protocol_gate = False
        reasons.append("EMPTY_ORACLE")
    hard_failures = {
        failure for replay in replays for failure in replay.policy.get("hard_failures", [])
    }
    if "POLICY_VIOLATION" in hard_failures:
        protocol_gate = False
    catastrophic_gate = not bool(hard_failures & CATASTROPHIC_FAILURES)
    reasons.extend(sorted(hard_failures))
    unique = sorted(set(reasons))
    return (
        GateResult(passed=protocol_gate and catastrophic_gate, reasons=unique),
        protocol_gate,
        catastrophic_gate,
    )
