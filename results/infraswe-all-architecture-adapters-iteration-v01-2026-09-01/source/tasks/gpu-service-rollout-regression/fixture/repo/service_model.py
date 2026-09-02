from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutOutcome:
    total_requests: int
    successful_requests: int
    corrupted_responses: int
    dropped_requests: int
    rollback_completed: bool
    orphan_replicas: int
    recovery_time_seconds: float
    events: list[dict[str, Any]]

    @property
    def goodput_ratio(self) -> float:
        return self.successful_requests / self.total_requests


def load_deployment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("spec"), dict):
        raise ValueError("deployment must contain an object-valued spec")
    return value


def simulate_rollout(config: dict[str, Any], *, request_count: int = 400) -> RolloutOutcome:
    spec = config["spec"]
    probe = spec.get("readinessProbe", {})
    strategy = spec.get("strategy", {})
    events: list[dict[str, Any]] = [{"at": 0.0, "event": "rollout_started"}]
    dropped = 0

    if probe.get("path") != "/readyz":
        dropped += 42
        events.append({"at": 0.2, "event": "readiness_endpoint_mismatch"})
    if int(probe.get("initialDelaySeconds", 0)) < 2:
        dropped += 18
        events.append({"at": 0.5, "event": "replica_routed_before_ready"})
    if int(strategy.get("maxUnavailable", 1)) != 0:
        dropped += 24
        events.append({"at": 1.0, "event": "capacity_gap"})
    if int(strategy.get("maxSurge", 0)) < 1:
        dropped += 12
        events.append({"at": 1.2, "event": "replacement_not_prestarted"})

    grace = int(spec.get("terminationGracePeriodSeconds", 0))
    if grace < 5:
        dropped += 28
        events.append({"at": 2.0, "event": "inflight_requests_terminated"})
        recovery_time = 8.0
    else:
        recovery_time = 2.0
        events.append({"at": 2.0, "event": "inflight_requests_drained"})

    rollback_completed = bool(spec.get("rollbackEnabled", False))
    orphan_replicas = 0 if rollback_completed else 1
    if rollback_completed:
        events.append({"at": recovery_time, "event": "rollback_completed"})
    successful = max(0, request_count - dropped)
    return RolloutOutcome(
        total_requests=request_count,
        successful_requests=successful,
        corrupted_responses=0,
        dropped_requests=dropped,
        rollback_completed=rollback_completed,
        orphan_replicas=orphan_replicas,
        recovery_time_seconds=recovery_time,
        events=events,
    )
