from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "blocked",
        "sass_targets": [],
        "ptx_target": None,
        "reason": reason,
        "fallback_reported": True,
    }


def select_targets(
    device_sms: list[int], toolkit: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(device_sms, list)
        or not device_sms
        or any(isinstance(sm, bool) or not isinstance(sm, int) or sm < 10 for sm in device_sms)
        or not isinstance(toolkit, dict)
        or not isinstance(config, dict)
    ):
        raise ValueError("device_sms must be a non-empty list of SM integers")
    required = {
        "allow_unsupported_fallback": False,
        "emit_ptx": False,
        "require_native_sass": True,
        "target_mode": "visible_devices",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("build configuration must require visible native SASS targets")
    supported = toolkit.get("supported_sms")
    if not isinstance(supported, list) or any(
        isinstance(sm, bool) or not isinstance(sm, int) for sm in supported
    ):
        raise ValueError("toolkit.supported_sms must be a list of integers")
    targets = sorted(set(device_sms))
    if any(target not in supported for target in targets):
        return _blocked("visible_sm_not_supported_by_toolkit")
    return {
        "schema_version": "1",
        "status": "ready",
        "sass_targets": targets,
        "ptx_target": None,
        "reason": "all_visible_sms_have_native_sass",
        "fallback_reported": False,
    }
"""

config = {
    "allow_unsupported_fallback": False,
    "emit_ptx": False,
    "require_native_sass": True,
    "target_mode": "visible_devices",
}

Path("arch_policy.py").write_text(source, encoding="utf-8")
Path("build_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
