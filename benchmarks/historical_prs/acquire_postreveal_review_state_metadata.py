#!/usr/bin/env python3
"""Acquire text-free review-state metadata for a sealed learning cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _checked(path: Path, digest_field: str, *, material_field: str | None = None) -> dict[str, Any]:
    payload = _read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def _graphql(query: str) -> tuple[dict[str, Any], str]:
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(3):
        process = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    return json.loads(process.stdout)["data"], process.stdout


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _count_states(reviews: list[dict[str, Any]]) -> dict[str, int]:
    states = ("APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING")
    return {state: sum(item["state"] == state for item in reviews) for state in states}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _checked(
        args.selection_lock,
        "selection_lock_sha256",
        material_field="selection_material",
    )
    locks = _checked(args.judgment_locks, "lock_set_sha256")
    reveal = _checked(args.reveal, "reveal_sha256")
    if reveal["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("reveal/judgment binding mismatch")
    if locks["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("judgment/selection binding mismatch")

    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    locked = {item["material"]["case_id"]: item["material"] for item in locks["locks"]}
    if not locked.keys() <= selected.keys():
        raise SystemExit("judgment contains an unknown selected case")

    repositories: dict[str, list[dict[str, Any]]] = {}
    for case_id in locked:
        case = selected[case_id]
        repositories.setdefault(case["repository"], []).append(case)
    fields: list[str] = []
    for repo_index, (repository, cases) in enumerate(repositories.items()):
        owner, name = repository.split("/", 1)
        pulls = " ".join(
            f"p{case['pull_number']}:pullRequest(number:{case['pull_number']})"
            "{author{login __typename} reviews(first:100)"
            "{nodes{author{login __typename} authorAssociation state submittedAt "
            "commit{oid}}}}"
            for case in cases
        )
        fields.append(f'r{repo_index}:repository(owner:"{owner}",name:"{name}"){{{pulls}}}')
    query = "query {" + " ".join(fields) + "}"
    data, raw = _graphql(query)

    records: dict[str, dict[str, Any]] = {}
    for repo_index, (_repository, cases) in enumerate(repositories.items()):
        for case in cases:
            pull = data[f"r{repo_index}"][f"p{case['pull_number']}"]
            if pull is None:
                raise SystemExit(f"{case['case_id']}: pull request metadata unavailable")
            author = str((pull.get("author") or {}).get("login") or "unknown")
            cutoff = _time(locked[case["case_id"]]["frozen_at"])
            reviews: list[dict[str, Any]] = []
            for item in pull["reviews"]["nodes"]:
                reviewer = item.get("author") or {}
                submitted_at = item.get("submittedAt")
                if (
                    reviewer.get("__typename") == "Bot"
                    or str(reviewer.get("login") or "unknown") == author
                    or submitted_at is None
                    or _time(submitted_at) > cutoff
                ):
                    continue
                reviews.append(
                    {
                        "reviewer": str(reviewer.get("login") or "unknown"),
                        "reviewer_association": str(item.get("authorAssociation") or "UNKNOWN"),
                        "state": str(item.get("state") or "UNKNOWN"),
                        "submitted_at": submitted_at,
                        "commit_id": (item.get("commit") or {}).get("oid"),
                    }
                )
            latest_by_reviewer: dict[str, dict[str, Any]] = {}
            for item in sorted(reviews, key=lambda value: value["submitted_at"]):
                latest_by_reviewer[item["reviewer"]] = item
            latest = list(latest_by_reviewer.values())
            final_head = [item for item in reviews if item["commit_id"] == case["head_sha"]]
            records[case["case_id"]] = {
                "case_id": case["case_id"],
                "locked_head_sha": case["head_sha"],
                "review_record_count_at_lock": len(reviews),
                "distinct_human_non_author_reviewer_count_at_lock": len(latest),
                "review_state_counts_at_lock": _count_states(reviews),
                "latest_reviewer_state_counts_at_lock": _count_states(latest),
                "final_head_review_state_counts_at_lock": _count_states(final_head),
                "review_records": reviews,
            }

    material = {
        "schema_version": "0.1",
        "protocol_id": "postreveal-outcome-free-review-state-metadata-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "judgment_lock_set_sha256": locks["lock_set_sha256"],
        "reveal_sha256": reveal["reveal_sha256"],
        "acquired_after_reveal": True,
        "learning_only": True,
        "state_or_merge_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "review_text_requested": False,
        "review_state_metadata_requested": True,
        "raw_response_sha256": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(),
        "cases": [records[case_id] for case_id in locked],
    }
    payload = {**material, "metadata_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "cases": len(records),
                "latest_approved_cases": sum(
                    item["latest_reviewer_state_counts_at_lock"]["APPROVED"] > 0
                    for item in records.values()
                ),
                "final_head_approved_cases": sum(
                    item["final_head_review_state_counts_at_lock"]["APPROVED"] > 0
                    for item in records.values()
                ),
                "metadata_sha256": payload["metadata_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
