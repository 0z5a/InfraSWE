#!/usr/bin/env python3
"""Produce outcome-free static evidence for every frozen R16 case."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from acquire_r14_candidate_evidence import _read
from analyze_r14_candidate_evidence import (
    CONFLICT_MARKERS,
    SIGNATURE_PATTERNS,
    _body_evidence,
    _file_evidence,
)

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = _read(args.bundle)
    material = {
        key: value for key, value in bundle.items() if key != "source_bundle_sha256"
    }
    if bundle["source_bundle_sha256"] != canonical_sha256(material):
        raise SystemExit("R16 source bundle digest mismatch")
    if any(
        bundle[field]
        for field in (
            "state_or_merge_fields_requested",
            "review_or_comment_fields_requested",
            "ci_or_label_fields_requested",
        )
    ):
        raise SystemExit("R16 source bundle exposes forbidden evidence")
    if not bundle["outcome_bearing_body_blocks_removed_before_storage"]:
        raise SystemExit("R16 body was not sanitized before storage")

    cases = []
    for case in bundle["cases"]:
        files = [_file_evidence(item) for item in case["files"]]
        source_files = [item for item in files if not item["is_test"]]
        test_files = [item for item in files if item["is_test"]]
        head_syntax_failures = [
            item["path"]
            for item in files
            if item["head_python_parse"] is not None
            and not item["head_python_parse"]["ok"]
        ]
        head_conflict_files = []
        for item in files:
            counts = item["head_conflict_markers"]
            if all((counts.get(marker) or 0) > 0 for marker in CONFLICT_MARKERS):
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
                    {
                        name
                        for item in test_files
                        for name in item["added_test_functions"]
                    }
                ),
                "patch_missing_count": sum(
                    not item["patch_available"] for item in files
                ),
                "head_python_syntax_failures": head_syntax_failures,
                "head_conflict_marker_files": head_conflict_files,
                "aggregate_added_signatures": aggregate_added,
                "aggregate_removed_signatures": aggregate_removed,
                "body_evidence": _body_evidence(body),
                "files": files,
            }
        )

    evidence_material = {
        "schema_version": "0.1",
        "protocol_id": "r16-outcome-free-static-evidence-v0.1",
        "selection_lock_sha256": bundle["selection_lock_sha256"],
        "test_plan_sha256": bundle["test_plan_sha256"],
        "source_bundle_sha256": bundle["source_bundle_sha256"],
        "analyzed_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_used": False,
        "cases": cases,
    }
    payload = {
        **evidence_material,
        "evidence_sha256": canonical_sha256(evidence_material),
    }
    atomic_write_json(args.output, payload)
    print(f"case_count={len(cases)}")
    print(
        "syntax_failure_count="
        f"{sum(bool(case['head_python_syntax_failures']) for case in cases)}"
    )
    print(
        "conflict_case_count="
        f"{sum(bool(case['head_conflict_marker_files']) for case in cases)}"
    )
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
