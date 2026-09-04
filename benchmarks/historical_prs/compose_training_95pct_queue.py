#!/usr/bin/env python3
"""Compose a 95% training queue while preserving an executed prefix exactly."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read_checked(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    material = {key: value for key, value in payload.items() if key != "queue_lock_sha256"}
    if payload.get("queue_lock_sha256") != canonical_sha256(material):
        raise SystemExit(f"{path}: queue digest mismatch")
    return payload


def _identity(case: dict[str, Any]) -> tuple[str, int]:
    return str(case["repository"]).lower(), int(case["pull_number"])


def compose_queue(
    *,
    coverage: dict[str, Any],
    original: dict[str, Any],
    prefix_overrides: list[dict[str, Any]],
    processed_count: int,
    group_index_base: int,
    group_size: int,
    created_at: str,
) -> dict[str, Any]:
    if processed_count <= 0 or group_index_base <= 0 or group_size <= 0:
        raise ValueError("processed count, group index base, and group size must be positive")
    if coverage.get("profile") != "training":
        raise ValueError("coverage queue must use the training profile")
    if coverage.get("target_fraction") != 0.95:
        raise ValueError("coverage queue must be frozen at target_fraction=0.95")

    original_cases = original["cases"]
    if processed_count >= len(coverage["cases"]) or processed_count > len(original_cases):
        raise ValueError("processed prefix must leave an unprocessed coverage suffix")

    prefix_by_index = {
        int(case["queue_index"]): dict(case) for case in original_cases[:processed_count]
    }
    for override in prefix_overrides:
        for case in override["cases"]:
            source_index = int(case.get("source_queue_index", case["queue_index"]))
            if source_index < processed_count:
                replacement = dict(case)
                replacement["queue_index"] = source_index
                replacement.pop("source_queue_index", None)
                prefix_by_index[source_index] = replacement

    if sorted(prefix_by_index) != list(range(processed_count)):
        raise ValueError("processed prefix queue indexes are not contiguous")
    prefix = [prefix_by_index[index] for index in range(processed_count)]
    prefix_identities = {_identity(case) for case in prefix}
    if len(prefix_identities) != processed_count:
        raise ValueError("processed prefix contains duplicate identities")

    coverage_by_identity = {_identity(case): case for case in coverage["cases"]}
    if len(coverage_by_identity) != len(coverage["cases"]):
        raise ValueError("coverage queue contains duplicate identities")
    missing = sorted(prefix_identities - coverage_by_identity.keys())
    if missing:
        preview = ", ".join(f"{repository}#{number}" for repository, number in missing[:5])
        raise ValueError(f"processed prefix is not a subset of the 95% sample: {preview}")

    suffix = [dict(case) for case in coverage["cases"] if _identity(case) not in prefix_identities]
    combined = prefix + suffix
    if len(combined) != len(coverage["cases"]):
        raise ValueError("composed queue changed the frozen 95% target count")

    for queue_index, case in enumerate(combined):
        case["queue_index"] = queue_index
        case.pop("source_queue_index", None)
        if queue_index < processed_count:
            continue
        suffix_index = queue_index - processed_count
        case["coverage_queue_index"] = int(coverage_by_identity[_identity(case)]["queue_index"])
        case["group_index"] = group_index_base + suffix_index // group_size
        case["group_offset"] = suffix_index % group_size

    suffix_group_count = (len(suffix) + group_size - 1) // group_size
    project_quotas = {
        project: sum(case["project"] == project for case in combined)
        for project in sorted(coverage["repositories"])
    }
    if project_quotas != coverage["project_quotas"]:
        raise ValueError("composed project quotas differ from the frozen 95% sample")
    if not all(project_quotas.values()):
        raise ValueError("every training draft repository must have positive coverage")

    material = {
        "schema_version": "0.1",
        "protocol_id": "training-bulk-outcome-blind-95pct-composed-queue-v0.1",
        "profile": "training",
        "created_at": created_at,
        "seed": coverage["seed"],
        "identity_source": coverage["identity_source"],
        "target_count": len(combined),
        "target_fraction": 0.95,
        "available_count_after_prior_exclusions": coverage[
            "available_count_after_prior_exclusions"
        ],
        "processed_count": processed_count,
        "processed_prefix_sha256": canonical_sha256(prefix),
        "source_coverage_queue_lock_sha256": coverage["queue_lock_sha256"],
        "source_original_queue_lock_sha256": original["queue_lock_sha256"],
        "source_prefix_override_lock_sha256s": [
            override["queue_lock_sha256"] for override in prefix_overrides
        ],
        "group_index_base": group_index_base,
        "group_size": group_size,
        "group_count": group_index_base + suffix_group_count,
        "suffix_group_count": suffix_group_count,
        "last_group_size": len(suffix) - (suffix_group_count - 1) * group_size,
        "allowed_graphql_fields": [],
        "allowed_git_ref_fields": ["pull_number", "head_sha"],
        "outcome_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "diff_or_body_fields_requested": False,
        "repositories": coverage["repositories"],
        "acquisitions": coverage["acquisitions"],
        "project_quotas": project_quotas,
        "cases": combined,
    }
    return {**material, "queue_lock_sha256": canonical_sha256(material)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-queue", type=Path, required=True)
    parser.add_argument("--original-queue", type=Path, required=True)
    parser.add_argument("--prefix-override", type=Path, action="append", default=[])
    parser.add_argument("--processed-count", type=int, required=True)
    parser.add_argument("--group-index-base", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = compose_queue(
        coverage=_read_checked(args.coverage_queue),
        original=_read_checked(args.original_queue),
        prefix_overrides=[_read_checked(path) for path in args.prefix_override],
        processed_count=args.processed_count,
        group_index_base=args.group_index_base,
        group_size=args.group_size,
        created_at=datetime.now(UTC).isoformat(),
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "target_count": payload["target_count"],
                "processed_count": payload["processed_count"],
                "remaining_count": payload["target_count"] - payload["processed_count"],
                "group_index_base": payload["group_index_base"],
                "suffix_group_count": payload["suffix_group_count"],
                "group_size": payload["group_size"],
                "last_group_size": payload["last_group_size"],
                "project_quotas": payload["project_quotas"],
                "queue_lock_sha256": payload["queue_lock_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
