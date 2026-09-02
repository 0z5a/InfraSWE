#!/usr/bin/env python3
"""Run outcome-free, human-readable static heuristics over an exact PR patch."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import (
    R2_POLICY_ID,
    R3_POLICY_ID,
    R4_POLICY_ID,
    R5_POLICY_ID,
    analyze_python_changes,
)
from infraswe.io import atomic_write_json


def _pair(value: str) -> tuple[str, str]:
    parts = value.split("::", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("pairs must use BEFORE::AFTER")
    return parts[0], parts[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--pair", action="append", type=_pair, required=True)
    parser.add_argument(
        "--policy-id",
        choices=(R2_POLICY_ID, R3_POLICY_ID, R4_POLICY_ID, R5_POLICY_ID),
        default=R4_POLICY_ID,
    )
    parser.add_argument("--evidence-code", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    before_sources = {
        old: (args.base_root / old).read_text(encoding="utf-8") for old, _ in args.pair
    }
    after_sources = {
        new: (args.head_root / new).read_text(encoding="utf-8") for _, new in args.pair
    }
    completed = subprocess.run(
        ["git", "-C", str(args.base_root), "diff", args.base_sha, args.head_sha, "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    observations = analyze_python_changes(
        before_sources,
        after_sources,
        unified_diff=completed.stdout,
        policy_id=args.policy_id,
        evidence_codes=frozenset(args.evidence_code),
    )
    source_identity = {
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
        "pairs": [{"before": old, "after": new} for old, new in args.pair],
        "before_source_sha256": {
            path: canonical_sha256(source) for path, source in before_sources.items()
        },
        "after_source_sha256": {
            path: canonical_sha256(source) for path, source in after_sources.items()
        },
        "unified_diff_sha256": canonical_sha256(completed.stdout),
    }
    material = {
        "schema_version": "0.5",
        "protocol_id": f"historical-explainable-static-v0.5-{args.policy_id.rsplit('-', 1)[-1]}",
        "policy_id": args.policy_id,
        "declared_evidence_codes": sorted(set(args.evidence_code)),
        "case_id": args.case_id,
        "source_identity": source_identity,
        "observations": [item.model_dump(mode="json") for item in observations],
        "duration_seconds": time.perf_counter() - started,
        "compilation_path": "not-required",
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
