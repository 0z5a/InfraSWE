from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

import re
from typing import Any

DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "status": "blocked",
        "candidate_id": None,
        "reason": reason,
        "fallback_reported": True,
        "lock": [],
    }


def _locked_candidate(
    request: dict[str, str], candidate: dict[str, Any]
) -> list[dict[str, str]] | None:
    if (
        candidate.get("name") != request["name"]
        or candidate.get("version") != request["version"]
        or candidate.get("architecture") != request["architecture"]
        or not isinstance(candidate.get("id"), str)
        or not candidate["id"]
        or not isinstance(candidate.get("digest"), str)
        or not DIGEST.fullmatch(candidate["digest"])
    ):
        return None
    dependencies = candidate.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(dependency, dict) for dependency in dependencies
    ):
        return None
    entries = [
        {
            "name": candidate["name"],
            "version": candidate["version"],
            "architecture": candidate["architecture"],
            "digest": candidate["digest"],
        }
    ]
    seen: dict[str, tuple[str, str, str]] = {}
    for dependency in dependencies:
        name = dependency.get("name")
        version = dependency.get("version")
        architecture = dependency.get("architecture")
        digest = dependency.get("digest")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or architecture != request["architecture"]
            or not isinstance(digest, str)
            or not DIGEST.fullmatch(digest)
        ):
            return None
        identity = (version, architecture, digest)
        if name in seen and seen[name] != identity:
            return None
        seen[name] = identity
    entries.extend(
        {
            "name": name,
            "version": identity[0],
            "architecture": identity[1],
            "digest": identity[2],
        }
        for name, identity in seen.items()
    )
    return sorted(entries, key=lambda entry: (entry["name"], entry["version"], entry["digest"]))


def resolve_image(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(request, dict) or not isinstance(policy, dict):
        raise ValueError("request and policy must be objects")
    if not isinstance(candidates, list) or any(
        not isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError("candidates must be a list of objects")
    if (
        policy.get("require_digest") is not True
        or policy.get("forbid_mutable_tags") is not True
        or policy.get("reject_dependency_conflicts") is not True
    ):
        raise ValueError("lock policy must require immutable conflict-free artifacts")
    architectures = policy.get("allowed_architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("allowed_architectures must be non-empty")
    normalized = {
        "name": _text(request.get("name"), "request.name"),
        "version": _text(request.get("version"), "request.version"),
        "architecture": _text(request.get("architecture"), "request.architecture"),
    }
    if normalized["version"] == "latest" or normalized["architecture"] not in architectures:
        return _blocked("mutable_or_disallowed_request")

    compatible: list[tuple[str, list[dict[str, str]]]] = []
    for candidate in candidates:
        lock = _locked_candidate(normalized, candidate)
        if lock is not None:
            compatible.append((candidate["id"], lock))
    if not compatible:
        return _blocked("no_immutable_architecture_compatible_candidate")
    candidate_id, lock = min(compatible, key=lambda item: item[0])
    return {
        "schema_version": "1",
        "status": "ready",
        "candidate_id": candidate_id,
        "reason": "immutable_dependency_lock",
        "fallback_reported": False,
        "lock": lock,
    }
"""

policy = {
    "allowed_architectures": ["amd64", "arm64"],
    "forbid_mutable_tags": True,
    "reject_dependency_conflicts": True,
    "require_digest": True,
}

Path("resolver.py").write_text(source, encoding="utf-8")
Path("lock_policy.json").write_text(
    json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
