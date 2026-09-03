#!/usr/bin/env python3
"""Seal prior revealed PR outcomes as a deterministic retrieval set."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from discover_r14_candidates import _is_test_path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

STOP_TOKENS = {
    "a",
    "add",
    "an",
    "and",
    "bug",
    "bugfix",
    "core",
    "feat",
    "feature",
    "fix",
    "for",
    "in",
    "of",
    "on",
    "pr",
    "refactor",
    "support",
    "test",
    "tests",
    "the",
    "to",
    "update",
    "with",
}


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def checked(path: Path, digest_field: str, *, material_field: str | None = None) -> dict[str, Any]:
    payload = read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def title_tokens(title: str) -> list[str]:
    text = re.sub(r"#\d+", " ", title.lower())
    return sorted(
        {
            token
            for token in re.findall(r"[a-z0-9]+", text)
            if len(token) > 1 and token not in STOP_TOKENS
        }
    )


def source_paths(paths: list[str]) -> list[str]:
    return sorted(path for path in paths if not _is_test_path(path) and not path.endswith(".md"))


def directory_prefixes(paths: list[str]) -> list[str]:
    prefixes: set[str] = set()
    for path in paths:
        parts = path.split("/")
        for depth in range(1, min(4, len(parts))):
            prefixes.add("/".join(parts[:depth]))
    return sorted(prefixes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-round", required=True)
    parser.add_argument("--source-selection", type=Path, action="append", required=True)
    parser.add_argument("--source-audit", type=Path, action="append", required=True)
    parser.add_argument("--negative-consensus-minimum", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.negative_consensus_minimum < 2:
        raise SystemExit("--negative-consensus-minimum must be at least 2")
    if len(args.source_selection) != len(args.source_audit):
        raise SystemExit("selection and audit source counts differ")

    bindings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selection_path, audit_path in zip(args.source_selection, args.source_audit, strict=True):
        selection = checked(
            selection_path, "selection_lock_sha256", material_field="selection_material"
        )
        audit = checked(audit_path, "audit_sha256")
        if audit["source_digests"]["selection_lock"] != canonical_sha256(selection):
            raise SystemExit(f"{audit_path.name} is not bound to {selection_path.name}")
        selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
        audited = {item["case_id"]: item for item in audit["cases"]}
        if selected.keys() != audited.keys():
            lock_path = audit_path.parent / "machine-judgment-locks.json"
            locks = checked(lock_path, "lock_set_sha256")
            if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
                raise SystemExit(f"{audit_path.name} is not bound to {lock_path.name}")
            locked_ids = {item["material"]["case_id"] for item in locks["locks"]}
            if audited.keys() != locked_ids or not locked_ids <= selected.keys():
                raise SystemExit(f"case mismatch between {selection_path} and {audit_path}")
        for case_id in audited:
            case = selected[case_id]
            if case_id in seen:
                raise SystemExit(f"duplicate precedent identity: {case_id}")
            seen.add(case_id)
            paths = source_paths(list(case["paths"]))
            records.append(
                {
                    "precedent_id": case_id,
                    "project": case["project"],
                    "repository": case["repository"],
                    "pull_number": case["pull_number"],
                    "risk_family": case["risk_family"],
                    "title": case["title"],
                    "title_tokens": title_tokens(case["title"]),
                    "source_paths": paths,
                    "source_directory_prefixes": directory_prefixes(paths),
                    "oracle_decision": audited[case_id]["oracle_decision"],
                    "oracle_reason": audited[case_id]["oracle_reason"],
                    "source_selection_lock_sha256": selection["selection_lock_sha256"],
                    "source_audit_sha256": audit["audit_sha256"],
                }
            )
        bindings.append(
            {
                "selection_path": str(selection_path),
                "selection_lock_sha256": selection["selection_lock_sha256"],
                "audit_path": str(audit_path),
                "audit_sha256": audit["audit_sha256"],
                "case_count": len(selected),
            }
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-disposition-precedents-v0.1",
        "target_round": args.target_round.upper(),
        "built_only_from_prior_revealed_rounds": True,
        "current_target_outcomes_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "retrieval_policy": {
            "same_project_required": True,
            "same_risk_family_required": True,
            "title_token_jaccard_min": 0.35,
            "source_path_jaccard_min": 0.4,
            "source_directory_prefix_jaccard_min": 0.75,
            "negative_consensus_minimum": args.negative_consensus_minimum,
            "accepted_neighbor_vetoes_negative_consensus": True,
            "weighted_disposition_score_used": False,
        },
        "source_bindings": bindings,
        "records": sorted(records, key=lambda item: item["precedent_id"]),
    }
    payload = {**material, "precedent_set_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "target_round": payload["target_round"],
                "precedent_count": len(records),
                "precedent_set_sha256": payload["precedent_set_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
