from __future__ import annotations

import json
from pathlib import Path

selector_source = """from __future__ import annotations

from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _request_fields(request: dict[str, Any]) -> tuple[int, int, int, int]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    device_sm = _integer(request.get("device_sm"), "device_sm", minimum=10)
    driver_cuda = _integer(request.get("driver_cuda"), "driver_cuda", minimum=10)
    runtime_cuda = _integer(request.get("runtime_cuda"), "runtime_cuda", minimum=10)
    cxx11_abi = _integer(request.get("cxx11_abi"), "cxx11_abi")
    if cxx11_abi not in {0, 1}:
        raise ValueError("cxx11_abi must be 0 or 1")
    return device_sm, driver_cuda, runtime_cuda, cxx11_abi


def _artifact_id(artifact: dict[str, Any]) -> str | None:
    value = artifact.get("id")
    return value if isinstance(value, str) and value else None


def _native_compatible(
    artifact: dict[str, Any],
    *,
    device_sm: int,
    driver_cuda: int,
    runtime_cuda: int,
    cxx11_abi: int,
) -> bool:
    sms = artifact.get("sms")
    built_cuda = artifact.get("built_cuda")
    artifact_abi = artifact.get("cxx11_abi")
    return bool(
        artifact.get("kind") == "sass"
        and _artifact_id(artifact)
        and isinstance(sms, list)
        and all(isinstance(sm, int) and not isinstance(sm, bool) for sm in sms)
        and device_sm in sms
        and isinstance(built_cuda, int)
        and not isinstance(built_cuda, bool)
        and built_cuda <= runtime_cuda <= driver_cuda
        and artifact_abi == cxx11_abi
    )


def _ptx_compatible(
    artifact: dict[str, Any],
    *,
    device_sm: int,
    driver_cuda: int,
    cxx11_abi: int,
) -> bool:
    compute = artifact.get("compute")
    minimum_driver = artifact.get("requires_driver_cuda")
    artifact_abi = artifact.get("cxx11_abi")
    return bool(
        artifact.get("kind") == "ptx"
        and _artifact_id(artifact)
        and isinstance(compute, int)
        and not isinstance(compute, bool)
        and compute <= device_sm
        and isinstance(minimum_driver, int)
        and not isinstance(minimum_driver, bool)
        and minimum_driver <= driver_cuda
        and artifact_abi == cxx11_abi
    )


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "blocked",
        "artifact_id": None,
        "mechanism": "none",
        "reason": reason,
        "fallback_reported": True,
    }


def select_artifact(
    request: dict[str, Any],
    artifacts: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(artifacts, list) or any(
        not isinstance(artifact, dict) for artifact in artifacts
    ):
        raise ValueError("artifacts must be a list of objects")
    if not isinstance(policy, dict):
        raise ValueError("policy must be an object")
    if policy.get("allow_cpu_fallback") is not False:
        raise ValueError("CPU fallback must be disabled")
    if policy.get("require_explicit_fallback") is not True:
        raise ValueError("explicit fallback reporting must be required")
    if policy.get("selection_order") != ["native_sass", "ptx_jit"]:
        raise ValueError("selection_order must prefer native_sass then ptx_jit")

    device_sm, driver_cuda, runtime_cuda, cxx11_abi = _request_fields(request)
    if runtime_cuda > driver_cuda:
        return _blocked("driver_runtime_incompatible")

    native = sorted(
        (
            artifact
            for artifact in artifacts
            if _native_compatible(
                artifact,
                device_sm=device_sm,
                driver_cuda=driver_cuda,
                runtime_cuda=runtime_cuda,
                cxx11_abi=cxx11_abi,
            )
        ),
        key=lambda artifact: str(artifact["id"]),
    )
    if native:
        return {
            "schema_version": "1",
            "status": "ready",
            "artifact_id": native[0]["id"],
            "mechanism": "native_sass",
            "reason": "exact_sm_cuda_and_abi",
            "fallback_reported": False,
        }

    ptx = sorted(
        (
            artifact
            for artifact in artifacts
            if _ptx_compatible(
                artifact,
                device_sm=device_sm,
                driver_cuda=driver_cuda,
                cxx11_abi=cxx11_abi,
            )
        ),
        key=lambda artifact: str(artifact["id"]),
    )
    if ptx:
        return {
            "schema_version": "1",
            "status": "ready",
            "artifact_id": ptx[0]["id"],
            "mechanism": "ptx_jit",
            "reason": "native_unavailable_explicit_ptx_jit",
            "fallback_reported": True,
        }
    return _blocked("no_compatible_cuda_artifact")
"""

policy = {
    "allow_cpu_fallback": False,
    "require_explicit_fallback": True,
    "selection_order": ["native_sass", "ptx_jit"],
}

Path("selector.py").write_text(selector_source, encoding="utf-8")
Path("build_policy.json").write_text(
    json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
