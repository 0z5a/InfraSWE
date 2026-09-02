#!/usr/bin/env python3
"""Acquire exact, outcome-free R16 candidate evidence after the plan lock."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acquire_r14_candidate_evidence import (
    _body_projection,
    _compare_projection,
    _fetch_file,
    _read,
    _without_content,
)

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def main(round_label: str = "R16") -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--bundle-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit(f"{round_label} selection digest mismatch")
    plan = _read(args.test_plan)
    plan_material = {
        key: value for key, value in plan.items() if key != "test_plan_sha256"
    }
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit(f"{round_label} test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit(f"{round_label} plan/selection binding mismatch")
    if not plan["frozen_before_candidate_body_access"]:
        raise SystemExit(f"{round_label} body acquisition lacks a prior plan lock")
    if not plan["frozen_before_source_diff_content_access"]:
        raise SystemExit(f"{round_label} source acquisition lacks a prior plan lock")
    if any(
        plan[field]
        for field in (
            "merge_outcome_or_state_requested",
            "review_or_comment_requested",
            "ci_or_label_requested",
        )
    ):
        raise SystemExit(f"{round_label} plan requests forbidden outcome evidence")

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
            file_records.append(
                {
                    **item,
                    "base_path": base_path,
                    "head_path": item["filename"],
                    "patch_sha256": (
                        canonical_sha256(item["patch"])
                        if item.get("patch") is not None
                        else None
                    ),
                    "base": None,
                    "head": None,
                }
            )
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
                "compare": {
                    key: value for key, value in compare.items() if key != "files"
                },
                "files": file_records,
            }
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        for case_index, case in enumerate(bundle_cases):
            for file_index, item in enumerate(case["files"]):
                for side, ref, path in (
                    ("base", case["evaluation_base_sha"], item["base_path"]),
                    ("head", case["head_sha"], item["head_path"]),
                ):
                    future = executor.submit(_fetch_file, case["repository"], ref, path)
                    fetch_jobs[future] = (case_index, file_index, side)
        for future in as_completed(fetch_jobs):
            case_index, file_index, side = fetch_jobs[future]
            bundle_cases[case_index]["files"][file_index][side] = future.result()

    bundle_material = {
        "schema_version": "0.1",
        "protocol_id": (
            f"{round_label.lower()}-exact-candidate-evidence-v0.1-"
            "sanitized-before-storage"
        ),
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "acquired_at": acquired_at,
        "body_fields_requested": [
            "number",
            "title",
            "body",
            "baseRefOid",
            "headRefOid",
        ],
        "diff_endpoint": "exact base...head Git compare",
        "state_or_merge_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "outcome_bearing_body_blocks_removed_before_storage": True,
        "cases": bundle_cases,
    }
    bundle_sha256 = canonical_sha256(bundle_material)
    atomic_write_json(
        args.bundle_output,
        {**bundle_material, "source_bundle_sha256": bundle_sha256},
    )

    manifest_cases = []
    for case in bundle_cases:
        body = case["body_projection"]
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
                "body_byte_count": len((body["body"] or "").encode("utf-8")),
                "body_sha256": canonical_sha256(body["body"] or ""),
                "body_sanitization": body["body_sanitization"],
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
    manifest_sha256 = canonical_sha256(manifest_material)
    atomic_write_json(
        args.manifest_output,
        {**manifest_material, "evidence_manifest_sha256": manifest_sha256},
    )
    redactions = sum(
        bool(case["body_projection"]["body_sanitization"]["redactions"])
        for case in bundle_cases
    )
    print(f"case_count={len(bundle_cases)}")
    print(f"sanitized_body_case_count={redactions}")
    print(f"source_bundle_sha256={bundle_sha256}")
    print(f"evidence_manifest_sha256={manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("R16"))
