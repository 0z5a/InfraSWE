#!/usr/bin/env python3
"""Produce outcome-free static evidence for every frozen R14 case."""

from __future__ import annotations

import argparse
import ast
import base64
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
CONFLICT_PATTERNS = {
    "<<<<<<<": re.compile(r"(?m)^\s*<<<<<<<\s+.+$"),
    "=======": re.compile(r"(?m)^\s*=======\s*$"),
    ">>>>>>>": re.compile(r"(?m)^\s*>>>>>>>\s+.+$"),
}
SIGNATURE_PATTERNS = {
    "collective": re.compile(
        r"all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|broadcast|collective",
        re.IGNORECASE,
    ),
    "p2p_or_transport": re.compile(
        r"\bp2p\b|\bsend\b|\brecv\b|nixl|mooncake|ipc|weight.?transfer|kv.?transfer",
        re.IGNORECASE,
    ),
    "rank_or_group": re.compile(
        r"get_rank|world_size|local_rank|process_group|new_group|parallel_state|\bgroup\b",
        re.IGNORECASE,
    ),
    "stream_or_progress": re.compile(
        r"stream|event|wait|synchroniz|barrier|timeout|deadlock|progress|async",
        re.IGNORECASE,
    ),
    "shape_or_shard": re.compile(
        r"\.shape|\.size\(|numel|reshape|view\(|split|chunk|shard|partition",
        re.IGNORECASE,
    ),
    "lifecycle": re.compile(
        r"close|destroy|unlink|cleanup|release|evict|flock|finaliz|__del__",
        re.IGNORECASE,
    ),
    "guard_or_error": re.compile(
        r"\bassert\b|\braise\b|valueerror|runtimeerror|warning|fail.?fast",
        re.IGNORECASE,
    ),
}
TEST_FUNCTION = re.compile(r"^\+\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)", re.MULTILINE)
DEFINITION = re.compile(
    r"^\+\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _decode(record: dict[str, Any]) -> bytes | None:
    if not record["available"]:
        return None
    return base64.b64decode(record["content_base64"])


def _text(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith(("_test.py", "_test.cpp", "_test.cu", ".test.ts"))
    )


def _python_parse(path: str, text: str | None) -> dict[str, Any] | None:
    if not path.endswith(".py") or text is None:
        return None
    try:
        ast.parse(text, filename=path)
    except SyntaxError as error:
        return {
            "ok": False,
            "line": error.lineno,
            "offset": error.offset,
            "message": error.msg,
        }
    return {"ok": True}


def _patch_lines(patch: str | None) -> tuple[list[str], list[str]]:
    if patch is None:
        return [], []
    added = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    return added, removed


def _conflict_counts(text: str | None) -> dict[str, int | None]:
    if text is None:
        return {marker: None for marker in CONFLICT_MARKERS}
    return {marker: len(CONFLICT_PATTERNS[marker].findall(text)) for marker in CONFLICT_MARKERS}


def _signature_counts(lines: list[str]) -> dict[str, int]:
    text = "\n".join(lines)
    return {name: len(pattern.findall(text)) for name, pattern in SIGNATURE_PATTERNS.items()}


def _file_evidence(item: dict[str, Any]) -> dict[str, Any]:
    base_bytes = _decode(item["base"])
    head_bytes = _decode(item["head"])
    base_text = _text(base_bytes)
    head_text = _text(head_bytes)
    patch = item.get("patch")
    added, removed = _patch_lines(patch)
    path = item["head_path"]
    return {
        "path": path,
        "base_path": item["base_path"],
        "change_type": item["change_type"],
        "is_test": _is_test_path(path),
        "patch_available": patch is not None,
        "patch_sha256": item["patch_sha256"],
        "base_available": item["base"]["available"],
        "head_available": item["head"]["available"],
        "base_sha256": item["base"].get("sha256"),
        "head_sha256": item["head"].get("sha256"),
        "base_is_utf8": base_text is not None,
        "head_is_utf8": head_text is not None,
        "base_python_parse": _python_parse(item["base_path"], base_text),
        "head_python_parse": _python_parse(path, head_text),
        "base_conflict_markers": _conflict_counts(base_text),
        "head_conflict_markers": _conflict_counts(head_text),
        "added_line_count_in_patch": len(added),
        "removed_line_count_in_patch": len(removed),
        "added_signatures": _signature_counts(added),
        "removed_signatures": _signature_counts(removed),
        "added_test_functions": TEST_FUNCTION.findall(patch or "") if _is_test_path(path) else [],
        "added_definitions": DEFINITION.findall(patch or ""),
        "added_todo_count": sum("todo" in line.lower() for line in added),
        "added_skip_or_xfail_count": sum(
            "skip" in line.lower() or "xfail" in line.lower() for line in added
        ),
    }


def _body_evidence(body: str) -> dict[str, Any]:
    lowered = body.lower()
    hardware_terms = [
        term
        for term in ("a100", "h100", "h200", "b200", "gb200", "gb300", "sm100", "sm12")
        if term in lowered
    ]
    return {
        "byte_count": len(body.encode("utf-8")),
        "sha256": canonical_sha256(body),
        "mentions_test": any(term in lowered for term in ("test", "pytest", "unittest")),
        "mentions_benchmark": any(
            term in lowered for term in ("benchmark", "latency", "throughput", "speedup")
        ),
        "mentions_multi_gpu": any(
            term in lowered for term in ("multi-gpu", "multi gpu", "2 gpu", "8 gpu", "tp2", "ep2")
        ),
        "mentions_failure_or_limit": any(
            term in lowered for term in ("fail", "error", "limitation", "not support", "todo")
        ),
        "hardware_terms": hardware_terms,
        "code_fence_count": body.count("```"),
        "checkbox_count": body.count("- ["),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = _read(args.bundle)
    bundle_material = {key: value for key, value in bundle.items() if key != "source_bundle_sha256"}
    if bundle["source_bundle_sha256"] != canonical_sha256(bundle_material):
        raise SystemExit("R14 source bundle digest mismatch")
    hidden = (
        bundle["state_or_merge_fields_requested"],
        bundle["review_or_comment_fields_requested"],
        bundle["ci_or_label_fields_requested"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R14 source bundle exposes forbidden evidence")

    cases = []
    for case in bundle["cases"]:
        files = [_file_evidence(item) for item in case["files"]]
        source_files = [item for item in files if not item["is_test"]]
        test_files = [item for item in files if item["is_test"]]
        head_syntax_failures = [
            item["path"]
            for item in files
            if item["head_python_parse"] is not None and not item["head_python_parse"]["ok"]
        ]
        head_conflict_files = []
        for item in files:
            marker_counts = item["head_conflict_markers"]
            if all((marker_counts.get(marker) or 0) > 0 for marker in CONFLICT_MARKERS):
                head_conflict_files.append(item["path"])
        aggregate_added = {
            name: sum(item["added_signatures"][name] for item in files)
            for name in SIGNATURE_PATTERNS
        }
        aggregate_removed = {
            name: sum(item["removed_signatures"][name] for item in files)
            for name in SIGNATURE_PATTERNS
        }
        body = case["body_projection"]["body"] or ""
        cases.append(
            {
                "case_id": case["case_id"],
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "base_sha": case["evaluation_base_sha"],
                "head_sha": case["head_sha"],
                "head_matches_selection_at_acquisition": case[
                    "head_matches_selection_at_acquisition"
                ],
                "changed_file_count": len(files),
                "source_file_count": len(source_files),
                "test_file_count": len(test_files),
                "candidate_test_path_present": bool(test_files),
                "candidate_test_functions_added": sorted(
                    {name for item in test_files for name in item["added_test_functions"]}
                ),
                "patch_missing_count": sum(not item["patch_available"] for item in files),
                "head_python_syntax_failures": head_syntax_failures,
                "head_conflict_marker_files": head_conflict_files,
                "aggregate_added_signatures": aggregate_added,
                "aggregate_removed_signatures": aggregate_removed,
                "body_evidence": _body_evidence(body),
                "files": files,
            }
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": "r14-outcome-free-static-evidence-v0.1",
        "selection_lock_sha256": bundle["selection_lock_sha256"],
        "test_plan_sha256": bundle["test_plan_sha256"],
        "source_bundle_sha256": bundle["source_bundle_sha256"],
        "analyzed_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_used": False,
        "cases": cases,
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(cases)}")
    syntax_failure_count = sum(bool(case["head_python_syntax_failures"]) for case in cases)
    print(f"syntax_failure_count={syntax_failure_count}")
    print(f"conflict_case_count={sum(bool(case['head_conflict_marker_files']) for case in cases)}")
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
