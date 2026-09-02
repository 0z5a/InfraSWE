from __future__ import annotations

import ast
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

Decision = Literal["accept_with_scope", "reject", "unresolved"]

_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".hip",
    ".py",
    ".pyi",
    ".rs",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
}
_TEST_PARTS = {"test", "tests", "testing", "unit_tests"}
_SILENT_EXCEPTION = re.compile(
    r"(?ms)^\+\s*except(?:\s+Exception)?\s*:\s*\n"
    r"(?:^\+[^\n]*\n){0,2}^\+\s*(?:pass|return\s+None)\s*$"
)
_REMOVED_GUARD = re.compile(
    r"^-\s*(?:assert\b|(?:if\b.*:\s*)?(?:raise\b|.*\b(?:check|validate)\s*\())",
    re.MULTILINE,
)
_ADDED_GUARD = re.compile(
    r"^\+\s*(?:assert\b|(?:if\b.*:\s*)?(?:raise\b|.*\b(?:check|validate)\s*\())",
    re.MULTILINE,
)
_REMOVED_SYNC = re.compile(
    r"^-.*\b(?:barrier|synchronize|wait|lock|fence)\b", re.IGNORECASE | re.MULTILINE
)
_ADDED_SYNC = re.compile(
    r"^\+.*\b(?:barrier|synchronize|wait|lock|fence)\b", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class StaticCohortAssessment:
    decision: Decision
    score_100: float | None
    checks: tuple[dict[str, Any], ...]
    rationale_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "score_100": self.score_100,
            "checks": list(self.checks),
            "rationale_codes": list(self.rationale_codes),
        }


def is_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    name = pure.name.lower()
    return (
        bool(_TEST_PARTS.intersection(part.lower() for part in pure.parts))
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.cc", "_test.cpp", "_test.cu"))
    )


def is_source_path(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in _SOURCE_SUFFIXES


def _syntax_check(path: str, content: bytes) -> tuple[bool | None, str]:
    suffix = PurePosixPath(path).suffix.lower()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None, "non-UTF-8 source is outside the portable parser set"
    try:
        if suffix in {".py", ".pyi"}:
            ast.parse(text, filename=path)
        elif suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        else:
            return None, "no language-complete parser is registered for this suffix"
    except (SyntaxError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, "head source parses with the registered standard-library parser"


def assess_static_change(
    *,
    expected_paths: Sequence[str],
    changed_files: Sequence[Mapping[str, Any]],
    head_sources: Mapping[str, bytes],
) -> StaticCohortAssessment:
    """Apply the preregistered, outcome-free R8 static scoring policy.

    This deliberately does not infer runtime correctness. It only rejects a small set of
    reproducible hard defects and otherwise grants a scoped static pass above the 85 floor.
    """

    observed_paths = sorted(str(item["filename"]) for item in changed_files)
    expected = sorted(expected_paths)
    checks: list[dict[str, Any]] = []
    if observed_paths != expected:
        checks.append(
            {
                "id": "exact-path-parity",
                "status": "unresolved",
                "failure_code": "R8_PATH_PARITY_MISMATCH",
                "details": {"expected": expected, "observed": observed_paths},
            }
        )
        return StaticCohortAssessment(
            decision="unresolved",
            score_100=None,
            checks=tuple(checks),
            rationale_codes=("R8_PATH_PARITY_MISMATCH",),
        )
    checks.append(
        {
            "id": "exact-path-parity",
            "status": "pass",
            "details": f"all {len(expected)} locked paths match the API diff",
        }
    )

    source_paths = [path for path in expected if is_source_path(path) and not is_test_path(path)]
    test_paths = [path for path in expected if is_test_path(path)]
    checks.append(
        {
            "id": "implementation-scope",
            "status": "pass" if source_paths else "fail",
            "failure_code": None if source_paths else "R8_NO_IMPLEMENTATION_SOURCE",
            "details": {"source_paths": source_paths, "test_paths": test_paths},
        }
    )

    missing_sources = [
        path
        for path in source_paths
        if path not in head_sources
        and next(
            (item.get("status") for item in changed_files if item["filename"] == path),
            None,
        )
        != "removed"
    ]
    if missing_sources:
        checks.append(
            {
                "id": "head-source-acquisition",
                "status": "unresolved",
                "failure_code": "R8_HEAD_SOURCE_UNAVAILABLE",
                "details": missing_sources,
            }
        )
        return StaticCohortAssessment(
            decision="unresolved",
            score_100=None,
            checks=tuple(checks),
            rationale_codes=("R8_HEAD_SOURCE_UNAVAILABLE",),
        )
    checks.append(
        {
            "id": "head-source-acquisition",
            "status": "pass",
            "details": f"retrieved {len(head_sources)} non-deleted head files by locked SHA",
        }
    )

    syntax_failures: list[dict[str, str]] = []
    syntax_passes = 0
    syntax_skips = 0
    for path, content in sorted(head_sources.items()):
        status, detail = _syntax_check(path, content)
        if status is False:
            syntax_failures.append({"path": path, "details": detail})
        elif status is True:
            syntax_passes += 1
        else:
            syntax_skips += 1
    checks.append(
        {
            "id": "registered-head-syntax",
            "status": "fail" if syntax_failures else "pass",
            "failure_code": "R8_HEAD_SYNTAX_INVALID" if syntax_failures else None,
            "details": {
                "parsed": syntax_passes,
                "not_applicable": syntax_skips,
                "failures": syntax_failures,
            },
        }
    )

    patches = "\n".join(str(item.get("patch") or "") for item in changed_files)
    conflict_marker = any(
        marker in content
        for content in head_sources.values()
        for marker in (b"<<<<<<< ", b">>>>>>> ")
    )
    checks.append(
        {
            "id": "merge-conflict-markers",
            "status": "fail" if conflict_marker else "pass",
            "failure_code": "R8_MERGE_CONFLICT_MARKER" if conflict_marker else None,
            "details": "scanned retrieved head sources",
        }
    )

    silent_exception = bool(_SILENT_EXCEPTION.search(patches)) and not test_paths
    checks.append(
        {
            "id": "silent-exception-fallback",
            "status": "fail" if silent_exception else "pass",
            "failure_code": "R8_SILENT_EXCEPTION_WITHOUT_TEST" if silent_exception else None,
            "details": "hard only for an added silent broad/bare handler without a changed test",
        }
    )

    removed_guard = bool(_REMOVED_GUARD.search(patches))
    replacement_guard = bool(_ADDED_GUARD.search(patches))
    unguarded = removed_guard and not replacement_guard and not test_paths
    checks.append(
        {
            "id": "validation-boundary-retention",
            "status": "fail" if unguarded else "pass",
            "failure_code": "R8_REMOVED_GUARD_WITHOUT_REPLACEMENT" if unguarded else None,
            "details": {
                "removed_guard": removed_guard,
                "replacement_guard": replacement_guard,
                "changed_test_present": bool(test_paths),
            },
        }
    )

    removed_sync = bool(_REMOVED_SYNC.search(patches))
    replacement_sync = bool(_ADDED_SYNC.search(patches))
    unsynchronized = removed_sync and not replacement_sync and not test_paths
    checks.append(
        {
            "id": "synchronization-retention",
            "status": "fail" if unsynchronized else "pass",
            "failure_code": "R8_REMOVED_SYNC_WITHOUT_REPLACEMENT" if unsynchronized else None,
            "details": {
                "removed_sync": removed_sync,
                "replacement_sync": replacement_sync,
                "changed_test_present": bool(test_paths),
            },
        }
    )

    hard_codes = tuple(
        str(item["failure_code"])
        for item in checks
        if item["status"] == "fail" and item.get("failure_code")
    )
    if hard_codes:
        return StaticCohortAssessment(
            decision="reject",
            score_100=35.0,
            checks=tuple(checks),
            rationale_codes=hard_codes,
        )
    score = 94.0 if test_paths else 88.0
    return StaticCohortAssessment(
        decision="accept_with_scope",
        score_100=score,
        checks=tuple(checks),
        rationale_codes=(
            "R8_STATIC_CONTRACT_PASS_WITH_CHANGED_TEST"
            if test_paths
            else "R8_STATIC_CONTRACT_PASS_NO_CHANGED_TEST",
        ),
    )
