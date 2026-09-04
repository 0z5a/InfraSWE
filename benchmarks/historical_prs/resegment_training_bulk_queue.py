#!/usr/bin/env python3
"""Re-segment the untouched suffix of a frozen outcome-blind PR queue."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--processed-count", type=int, required=True)
    parser.add_argument("--group-index-base", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.processed_count < 0 or args.group_index_base < 0 or args.group_size <= 0:
        raise SystemExit("processed count/index must be non-negative and size positive")

    source = _read_checked(args.source_queue)
    source_cases = source["cases"]
    if args.processed_count >= len(source_cases):
        raise SystemExit("processed count must leave at least one case")
    remaining = [dict(case) for case in source_cases[args.processed_count :]]
    for offset, case in enumerate(remaining):
        case["source_queue_index"] = int(case["queue_index"])
        case["group_index"] = args.group_index_base + offset // args.group_size
        case["group_offset"] = offset % args.group_size

    group_count = (len(remaining) + args.group_size - 1) // args.group_size
    material = {
        "schema_version": "0.1",
        "protocol_id": "training-bulk-outcome-blind-resegmented-queue-v0.1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_queue_lock_sha256": source["queue_lock_sha256"],
        "source_processed_prefix_sha256": canonical_sha256(source_cases[: args.processed_count]),
        "source_target_count": len(source_cases),
        "processed_count": args.processed_count,
        "target_count": len(remaining),
        "group_index_base": args.group_index_base,
        "group_size": args.group_size,
        "group_count": group_count,
        "last_group_size": len(remaining) - (group_count - 1) * args.group_size,
        "outcome_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "diff_or_body_fields_requested": False,
        "repositories": source["repositories"],
        "project_quotas": {
            project: sum(case["project"] == project for case in remaining)
            for project in sorted(source["repositories"])
        },
        "cases": remaining,
    }
    payload = {**material, "queue_lock_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "processed_count": args.processed_count,
                "remaining_count": len(remaining),
                "group_index_base": args.group_index_base,
                "group_count": group_count,
                "group_size": args.group_size,
                "last_group_size": material["last_group_size"],
                "queue_lock_sha256": payload["queue_lock_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
