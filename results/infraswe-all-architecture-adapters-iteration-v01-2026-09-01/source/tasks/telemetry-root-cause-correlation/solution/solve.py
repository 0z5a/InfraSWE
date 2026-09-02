from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from collections import defaultdict
from typing import Any

MODALITIES = ("logs", "metrics", "profiles", "traces")


def _inconclusive(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "inconclusive",
        "root_cause": None,
        "correlation_id": None,
        "confidence": 0.0,
        "evidence_ids": [],
        "reason": reason,
        "fallback_reported": True,
    }


def diagnose(signals: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(signals, dict) or not isinstance(policy, dict):
        raise ValueError("signals and policy must be objects")
    minimum = policy.get("minimum_modalities")
    window = policy.get("maximum_window_ms")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or not 3 <= minimum <= len(MODALITIES)
        or isinstance(window, bool)
        or not isinstance(window, int)
        or window <= 0
        or policy.get("require_correlation_id") is not True
    ):
        raise ValueError("signal policy must require correlated multi-modal evidence")
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for modality in MODALITIES:
        items = signals.get(modality)
        if not isinstance(items, list):
            raise ValueError(f"signals.{modality} must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"signals.{modality} entries must be objects")
            identifier = item.get("id")
            correlation_id = item.get("correlation_id")
            cause = item.get("cause")
            timestamp = item.get("at_ms")
            if (
                not isinstance(identifier, str)
                or not identifier
                or not isinstance(correlation_id, str)
                or not correlation_id
                or not isinstance(cause, str)
                or not cause
                or isinstance(timestamp, bool)
                or not isinstance(timestamp, int | float)
            ):
                continue
            groups[(correlation_id, cause)][modality].append(item)

    candidates: list[tuple[int, str, str, list[str]]] = []
    for (correlation_id, cause), by_modality in groups.items():
        timestamps = [
            float(item["at_ms"])
            for items in by_modality.values()
            for item in items
        ]
        if len(by_modality) < minimum or max(timestamps) - min(timestamps) > window:
            continue
        evidence_ids = sorted(
            str(item["id"]) for items in by_modality.values() for item in items
        )
        candidates.append((len(by_modality), cause, correlation_id, evidence_ids))
    if not candidates:
        return _inconclusive("insufficient_correlated_evidence")
    modalities, cause, correlation_id, evidence_ids = min(
        candidates, key=lambda item: (-item[0], item[1], item[2], item[3])
    )
    return {
        "schema_version": "1",
        "status": "diagnosed",
        "root_cause": cause,
        "correlation_id": correlation_id,
        "confidence": modalities / len(MODALITIES),
        "evidence_ids": evidence_ids,
        "reason": "correlated_multi_modal_evidence",
        "fallback_reported": False,
    }
"""

policy = {
    "maximum_window_ms": 100,
    "minimum_modalities": 3,
    "require_correlation_id": True,
}

Path("diagnoser.py").write_text(source, encoding="utf-8")
Path("signal_policy.json").write_text(
    json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
