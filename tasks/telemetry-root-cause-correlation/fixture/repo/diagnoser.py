from __future__ import annotations

from typing import Any


def diagnose(signals: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    del policy
    logs = signals.get("logs", [])
    if logs:
        first = logs[0]
        return {
            "schema_version": "1",
            "status": "diagnosed",
            "root_cause": first.get("cause"),
            "correlation_id": first.get("correlation_id"),
            "confidence": 1.0,
            "evidence_ids": [first.get("id")],
            "reason": "first_error_log",
            "fallback_reported": False,
        }
    return {
        "schema_version": "1",
        "status": "inconclusive",
        "root_cause": None,
        "correlation_id": None,
        "confidence": 0.0,
        "evidence_ids": [],
        "reason": "no_logs",
        "fallback_reported": False,
    }
