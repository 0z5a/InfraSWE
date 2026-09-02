#!/usr/bin/env python3
"""Deterministically remove embedded outcome-bearing blocks from an acquired R14 bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

BLOCKS = (
    (
        "sglang-pr-states",
        re.compile(
            r"<!--\s*pr-states:start\s*-->.*?<!--\s*pr-states:end\s*-->",
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ),
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _sanitize(body: str) -> tuple[str, list[dict[str, Any]]]:
    sanitized = body
    redactions: list[dict[str, Any]] = []
    for name, pattern in BLOCKS:
        matches = list(pattern.finditer(sanitized))
        if not matches:
            continue
        redactions.append(
            {
                "block_name": name,
                "count": len(matches),
                "raw_block_sha256": [canonical_sha256(match.group(0)) for match in matches],
            }
        )
        sanitized = pattern.sub(f"[outcome-bearing body block redacted: {name}]", sanitized)
    return sanitized, redactions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    raw = _read(args.input)
    raw_material = {key: value for key, value in raw.items() if key != "source_bundle_sha256"}
    if raw.get("source_bundle_sha256") != canonical_sha256(raw_material):
        raise SystemExit("raw R14 source bundle digest mismatch")

    cases = json.loads(json.dumps(raw["cases"]))
    manifest_cases: list[dict[str, Any]] = []
    for case in cases:
        projection = case["body_projection"]
        raw_body = projection.get("body") or ""
        sanitized_body, redactions = _sanitize(raw_body)
        projection["body"] = sanitized_body
        projection["body_sanitization"] = {
            "raw_body_sha256": canonical_sha256(raw_body),
            "sanitized_body_sha256": canonical_sha256(sanitized_body),
            "redactions": redactions,
        }
        manifest_cases.append(
            {
                "case_id": case["case_id"],
                **projection["body_sanitization"],
            }
        )

    material = {
        **{
            key: value
            for key, value in raw_material.items()
            if key not in {"protocol_id", "cases"}
        },
        "protocol_id": "r14-exact-candidate-evidence-v0.2-sanitized-body",
        "raw_source_bundle_sha256": raw["source_bundle_sha256"],
        "outcome_bearing_body_blocks_removed_before_judgment": True,
        "cases": cases,
    }
    sanitized_sha256 = canonical_sha256(material)
    atomic_write_json(
        args.output,
        {**material, "source_bundle_sha256": sanitized_sha256},
    )

    manifest_material = {
        "schema_version": "0.1",
        "protocol_id": "r14-candidate-body-sanitization-v0.1",
        "selection_lock_sha256": raw["selection_lock_sha256"],
        "test_plan_sha256": raw["test_plan_sha256"],
        "raw_source_bundle_sha256": raw["source_bundle_sha256"],
        "sanitized_source_bundle_sha256": sanitized_sha256,
        "cases": manifest_cases,
        "redacted_case_count": sum(bool(case["redactions"]) for case in manifest_cases),
        "redacted_block_count": sum(
            sum(item["count"] for item in case["redactions"])
            for case in manifest_cases
        ),
        "integrity_note": (
            "The evaluator noticed one embedded SGLang CI block before this sanitizer was "
            "introduced. That observation is quarantined and must not influence judgments; "
            "all judgment inputs use the sanitized bundle."
        ),
    }
    atomic_write_json(
        args.manifest_output,
        {**manifest_material, "manifest_sha256": canonical_sha256(manifest_material)},
    )
    print(f"sanitized_source_bundle_sha256={sanitized_sha256}")
    print(f"redacted_case_count={manifest_material['redacted_case_count']}")
    print(f"redacted_block_count={manifest_material['redacted_block_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
