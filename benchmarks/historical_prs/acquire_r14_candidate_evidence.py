#!/usr/bin/env python3
"""Acquire exact R14 candidate body/diff/source after the test-plan lock."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

BODY_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      title
      body
      baseRefOid
      headRefOid
    }
  }
}
"""

OUTCOME_BEARING_BODY_BLOCKS = (
    (
        "sglang-pr-states",
        re.compile(
            r"<!--\s*pr-states:start\s*-->.*?<!--\s*pr-states:end\s*-->",
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ),
)


def _sanitize_body(body: str | None) -> tuple[str, list[dict[str, Any]]]:
    """Remove machine-maintained outcome/CI blocks from otherwise allowed PR prose."""
    sanitized = body or ""
    redactions: list[dict[str, Any]] = []
    for block_name, pattern in OUTCOME_BEARING_BODY_BLOCKS:
        matches = list(pattern.finditer(sanitized))
        if not matches:
            continue
        redactions.append(
            {
                "block_name": block_name,
                "count": len(matches),
                "raw_block_sha256": [canonical_sha256(match.group(0)) for match in matches],
            }
        )
        sanitized = pattern.sub(f"[outcome-bearing body block redacted: {block_name}]", sanitized)
    return sanitized, redactions


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _run_gh(arguments: list[str], *, binary: bool = False) -> tuple[int, bytes, bytes]:
    process: subprocess.CompletedProcess[bytes] | None = None
    for attempt in range(4):
        process = subprocess.run(["gh", "api", *arguments], check=False, capture_output=True)
        if process.returncode == 0:
            break
        if attempt < 3 and process.returncode not in {1, 4}:
            time.sleep(2**attempt)
    assert process is not None
    stdout = process.stdout
    stderr = process.stderr
    if not binary:
        stdout.decode("utf-8")
        stderr.decode("utf-8", errors="replace")
    return process.returncode, stdout, stderr


def _json_gh(arguments: list[str]) -> dict[str, Any]:
    returncode, stdout, stderr = _run_gh(arguments)
    if returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())
    payload = json.loads(stdout)
    if isinstance(payload, dict) and payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub response was not an object")
    return payload


def _body_projection(case: dict[str, Any]) -> dict[str, Any]:
    owner, name = case["repository"].split("/", 1)
    payload = _json_gh(
        [
            "graphql",
            "-f",
            f"query={BODY_QUERY}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={case['pull_number']}",
        ]
    )
    projected = payload["data"]["repository"]["pullRequest"]
    if projected is None:
        raise RuntimeError(f"missing PR body projection for {case['case_id']}")
    raw_body = projected["body"] or ""
    sanitized_body, redactions = _sanitize_body(raw_body)
    return {
        "number": int(projected["number"]),
        "title": projected["title"],
        "body": sanitized_body,
        "body_sanitization": {
            "raw_body_sha256": canonical_sha256(raw_body),
            "sanitized_body_sha256": canonical_sha256(sanitized_body),
            "redactions": redactions,
        },
        "current_base_ref_oid": projected["baseRefOid"],
        "current_head_ref_oid": projected["headRefOid"],
    }


def _compare_projection(case: dict[str, Any]) -> dict[str, Any]:
    endpoint = f"repos/{case['repository']}/compare/{case['base_tip_sha']}...{case['head_sha']}"
    payload = _json_gh([endpoint])
    files = []
    for item in payload.get("files", []):
        files.append(
            {
                "filename": item["filename"],
                "previous_filename": item.get("previous_filename"),
                "change_type": item["status"],
                "additions": int(item["additions"]),
                "deletions": int(item["deletions"]),
                "changes": int(item["changes"]),
                "head_blob_sha": item.get("sha"),
                "patch": item.get("patch"),
            }
        )
    return {
        "git_relationship": payload.get("status"),
        "ahead_by": payload.get("ahead_by"),
        "behind_by": payload.get("behind_by"),
        "total_commits": payload.get("total_commits"),
        "merge_base_sha": payload["merge_base_commit"]["sha"],
        "files": files,
    }


def _fetch_file(repository: str, ref: str, path: str) -> dict[str, Any]:
    encoded = quote(path, safe="/")
    endpoint = f"repos/{repository}/contents/{encoded}?ref={quote(ref, safe='')}"
    returncode, stdout, stderr = _run_gh(
        ["-H", "Accept: application/vnd.github.raw+json", endpoint], binary=True
    )
    if returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip()
        if "HTTP 404" in error or "Not Found" in error:
            return {"available": False, "error": "not-found"}
        return {"available": False, "error": error or f"gh-exit-{returncode}"}
    return {
        "available": True,
        "byte_count": len(stdout),
        "sha256": "sha256:" + hashlib.sha256(stdout).hexdigest(),
        "content_base64": base64.b64encode(stdout).decode("ascii"),
    }


def _without_content(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "content_base64"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R14 selection digest mismatch")
    plan = _read(args.test_plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit("R14 test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R14 plan/selection binding mismatch")
    if not plan["frozen_before_candidate_body_access"]:
        raise SystemExit("R14 body acquisition was not authorized by a prior plan lock")
    if not plan["frozen_before_source_diff_content_access"]:
        raise SystemExit("R14 source acquisition was not authorized by a prior plan lock")

    cases: list[dict[str, Any]] = selection_material["cases"]
    acquired_at = datetime.now(UTC).isoformat()
    bundle_cases: list[dict[str, Any]] = []
    fetch_jobs: dict[Any, tuple[int, int, str]] = {}
    for case in cases:
        body = _body_projection(case)
        compare = _compare_projection(case)
        selected_paths = set(case["paths"])
        compared_paths = {item["filename"] for item in compare["files"]}
        if selected_paths != compared_paths:
            raise SystemExit(
                f"{case['case_id']}: exact compare path mismatch: "
                f"selected_only={sorted(selected_paths - compared_paths)}, "
                f"compare_only={sorted(compared_paths - selected_paths)}"
            )
        file_records = []
        for item in compare["files"]:
            base_path = item.get("previous_filename") or item["filename"]
            record = {
                **item,
                "base_path": base_path,
                "head_path": item["filename"],
                "patch_sha256": (
                    canonical_sha256(item["patch"]) if item.get("patch") is not None else None
                ),
                "base": None,
                "head": None,
            }
            file_records.append(record)
        bundle_cases.append(
            {
                "case_id": case["case_id"],
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "selected_title": case["title"],
                "base_sha": case["base_sha"],
                "base_tip_sha": case["base_tip_sha"],
                "evaluation_base_sha": compare["merge_base_sha"],
                "head_sha": case["head_sha"],
                "body_projection": body,
                "head_matches_selection_at_acquisition": (
                    body["current_head_ref_oid"] == case["head_sha"]
                ),
                "compare": {key: value for key, value in compare.items() if key != "files"},
                "files": file_records,
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        for case_index, case in enumerate(bundle_cases):
            for file_index, item in enumerate(case["files"]):
                base_future = executor.submit(
                    _fetch_file,
                    case["repository"],
                    case["evaluation_base_sha"],
                    item["base_path"],
                )
                head_future = executor.submit(
                    _fetch_file,
                    case["repository"],
                    case["head_sha"],
                    item["head_path"],
                )
                fetch_jobs[base_future] = (case_index, file_index, "base")
                fetch_jobs[head_future] = (case_index, file_index, "head")
        for future in as_completed(fetch_jobs):
            case_index, file_index, side = fetch_jobs[future]
            bundle_cases[case_index]["files"][file_index][side] = future.result()

    bundle_material = {
        "schema_version": "0.1",
        "protocol_id": "r14-exact-candidate-evidence-v0.2-sanitized-body",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "acquired_at": acquired_at,
        "body_fields_requested": ["number", "title", "body", "baseRefOid", "headRefOid"],
        "diff_endpoint": "exact base...head Git compare",
        "state_or_merge_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "outcome_bearing_body_blocks_removed_before_storage": True,
        "cases": bundle_cases,
    }
    bundle_sha256 = canonical_sha256(bundle_material)
    bundle = {**bundle_material, "source_bundle_sha256": bundle_sha256}
    atomic_write_json(args.bundle_output, bundle)

    manifest_cases = []
    for case in bundle_cases:
        body_text = case["body_projection"]["body"] or ""
        manifest_cases.append(
            {
                "case_id": case["case_id"],
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "base_sha": case["base_sha"],
                "base_tip_sha": case["base_tip_sha"],
                "evaluation_base_sha": case["evaluation_base_sha"],
                "head_sha": case["head_sha"],
                "head_matches_selection_at_acquisition": case[
                    "head_matches_selection_at_acquisition"
                ],
                "body_byte_count": len(body_text.encode("utf-8")),
                "body_sha256": canonical_sha256(body_text),
                "body_sanitization": case["body_projection"]["body_sanitization"],
                "compare": case["compare"],
                "files": [
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if key not in {"base", "head", "patch"}
                        },
                        "base": _without_content(item["base"]),
                        "head": _without_content(item["head"]),
                    }
                    for item in case["files"]
                ],
            }
        )
    manifest_material = {
        "schema_version": "0.1",
        "protocol_id": bundle_material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": bundle_sha256,
        "acquired_at": acquired_at,
        "outcome_review_ci_fields_requested": False,
        "cases": manifest_cases,
    }
    manifest = {
        **manifest_material,
        "evidence_manifest_sha256": canonical_sha256(manifest_material),
    }
    atomic_write_json(args.manifest_output, manifest)
    print(f"case_count={len(bundle_cases)}")
    print(f"source_bundle_sha256={bundle_sha256}")
    print(f"evidence_manifest_sha256={manifest['evidence_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
