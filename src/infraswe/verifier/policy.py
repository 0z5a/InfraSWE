from __future__ import annotations

from typing import Any

from infraswe.models.task import TaskPackage


def merge_policy_results(
    task: TaskPackage,
    agent_policy: dict[str, Any],
    verifier_policy: dict[str, Any],
) -> dict[str, Any]:
    hard_failures = list(agent_policy.get("hard_failures", []))
    hard_failures.extend(verifier_policy.get("hard_failures", []))
    if task.gates.forbid_silent_fallback and verifier_policy.get("silent_fallback", False):
        hard_failures.append("SILENT_FALLBACK")
    if task.gates.forbid_data_corruption and verifier_policy.get("data_corruption", False):
        hard_failures.append("DATA_CORRUPTION")
    if task.gates.forbid_resource_leak and verifier_policy.get("resource_leak", False):
        hard_failures.append("RESOURCE_LEAK")
    unique = sorted(set(hard_failures))
    return {
        "passed": bool(agent_policy.get("passed", True))
        and bool(verifier_policy.get("passed", True))
        and not unique,
        "agent": agent_policy,
        "verifier": verifier_policy,
        "hard_failures": unique,
    }
