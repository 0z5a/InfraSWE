#!/usr/bin/env python3
"""Freeze R7 selection after outcome-free metadata and path-parity checks."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.draft.defaults import build_default_catalog
from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

PROJECT_TO_REPO_DIR = {
    "cutlass-cute": "cutlass",
    "liger-kernel": "liger",
    "deepgemm": "deepgemm",
    "megatron-core": "megatron",
    "torchtitan": "torchtitan",
    "verl": "verl",
}


def _selected(discovery: dict[str, Any], project: str) -> dict[str, Any]:
    item = discovery["discoveries"][project]
    number = item["selected_pull_number"]
    return next(candidate for candidate in item["candidates"] if candidate["number"] == number)


def _path_status(repo: Path, base: str, head: str) -> list[dict[str, str]]:
    process = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-status", base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, str]] = []
    for line in process.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append({"status": parts[0], "path": parts[-1]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--path-parity-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit("R7 discovery digest mismatch")
    if discovery["review_text_visible_during_discovery"] is not False:
        raise SystemExit("R7 discovery did not keep review text hidden")
    if discovery["merge_outcomes_visible_during_discovery"] is not False:
        raise SystemExit("R7 discovery did not keep outcomes hidden")

    parity_cases: list[dict[str, Any]] = []
    selected_cases: list[dict[str, Any]] = []
    catalog = build_default_catalog()
    profile_material: list[dict[str, Any]] = []
    for project in policy["projects_in_order"]:
        candidate = _selected(discovery, project)
        base = candidate["first_pr_commit_first_parent_sha"]
        head = candidate["head_sha"]
        rows = _path_status(args.repo_root / PROJECT_TO_REPO_DIR[project], base, head)
        actual_paths = [item["path"] for item in rows]
        expected_paths = candidate["paths"]
        path_parity = actual_paths == expected_paths
        if not path_parity:
            raise SystemExit(f"R7 path parity failed for {project}")
        case_id = f"{PROJECT_TO_REPO_DIR[project]}-pr-{candidate['number']}"
        entry = catalog.entries[project]
        profile = entry.profile.model_dump(mode="json")
        profile_sha = canonical_sha256(profile)
        profile_material.append(
            {"project": project, "profile_id": entry.profile.id, "sha256": profile_sha}
        )
        parity_cases.append(
            {
                "case_id": case_id,
                "project": project,
                "repository": discovery["discoveries"][project]["repository"],
                "pull_number": candidate["number"],
                "metadata_paths": expected_paths,
                "first_parent_to_head_path_status": rows,
                "path_parity": path_parity,
                "diff_content_inspected": False,
            }
        )
        selected_cases.append(
            {
                "case_id": case_id,
                "project": project,
                "default_profile_id": entry.profile.id,
                "default_profile_sha256": profile_sha,
                "repository": discovery["discoveries"][project]["repository"],
                "pull_number": candidate["number"],
                "title": candidate["title"],
                "created_at": candidate["created_at"],
                "base_ref": candidate["base_ref"],
                "api_base_ref_oid": candidate["base_ref_oid"],
                "base_sha": base,
                "base_derivation": "first-pr-commit-first-parent-path-parity",
                "first_pr_commit_sha": candidate["first_pr_commit_sha"],
                "head_sha": head,
                "changed_files": candidate["changed_files"],
                "additions": candidate["additions"],
                "deletions": candidate["deletions"],
                "paths": expected_paths,
            }
        )

    frozen_at = datetime.now(UTC)
    parity_material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "checked_at": frozen_at.isoformat(),
        "inspection_scope": "git diff --name-status only",
        "review_text_visible": False,
        "merge_outcomes_visible": False,
        "cases": parity_cases,
    }
    parity_payload = {
        **parity_material,
        "path_parity_sha256": canonical_sha256(parity_material),
    }
    atomic_write_json(args.path_parity_output, parity_payload)

    catalog_path = Path("catalog/default-drafts-v0.5/catalog.json")
    catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    selection_material = {
        "frozen_at": frozen_at.isoformat(),
        "selection_rule": policy["ranking"],
        "preselection_policy_sha256": canonical_sha256(policy),
        "discovery_sha256": discovery["discovery_sha256"],
        "path_parity_sha256": parity_payload["path_parity_sha256"],
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "learned_model_used": False,
        "weighted_score_used": False,
        "repository_identity_amendments": policy.get("pre_discovery_amendments", []),
        "profile_catalog": {
            "path": str(catalog_path),
            "catalog_version": catalog.catalog_version,
            "catalog_sha256": canonical_sha256(catalog_payload),
            "selected_profile_set_sha256": canonical_sha256(profile_material),
        },
        "acquisition": {
            "provider": "GitHub GraphQL search API",
            "requested_fields": policy["metadata_only_discovery_fields"],
            "outcome_fields_requested": False,
            "review_fields_requested": False,
            "issue_comment_fields_requested": False,
            "ci_fields_requested": False,
            "diff_content_requested": False,
        },
        "cases": selected_cases,
    }
    selection_payload = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "selection_material": selection_material,
        "selection_lock_sha256": canonical_sha256(selection_material),
    }
    atomic_write_json(args.selection_output, selection_payload)
    print(
        json.dumps(
            {
                "selection_lock_sha256": selection_payload["selection_lock_sha256"],
                "path_parity_sha256": parity_payload["path_parity_sha256"],
                "cases": [item["case_id"] for item in selected_cases],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
