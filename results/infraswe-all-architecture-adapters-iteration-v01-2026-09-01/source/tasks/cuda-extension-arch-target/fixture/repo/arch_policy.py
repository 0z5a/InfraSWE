from __future__ import annotations

from typing import Any


def select_targets(
    device_sms: list[int], toolkit: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    del device_sms, toolkit
    targets = config.get("requested_sms", [90])
    return {
        "schema_version": "1",
        "status": "ready",
        "sass_targets": targets,
        "ptx_target": max(targets) if targets else 90,
        "reason": "configured_targets",
        "fallback_reported": False,
    }
