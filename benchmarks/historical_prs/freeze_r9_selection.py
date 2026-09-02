#!/usr/bin/env python3
"""Freeze five R9 cases from an existing outcome-free discovery pool."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

SELECTORS = {
    "cutlass-cute": 3332,
    "flashinfer": 3950,
    "sglang": 31349,
    "sglang-2": 31346,
    "vllm": 48695,
}

CASE_IDS = {
    ("cutlass-cute", 3332): "cutlass-pr-3332",
    ("flashinfer", 3950): "flashinfer-pr-3950",
    ("sglang", 31349): "sglang-pr-31349",
    ("sglang", 31346): "sglang-pr-31346",
    ("vllm", 48695): "vllm-pr-48695",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-lock", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _read(args.discovery_lock)
    source_material = {
        key: value for key, value in source.items() if key != "selection_lock_sha256"
    }
    if source["selection_lock_sha256"] != canonical_sha256(source_material):
        raise SystemExit("source discovery lock digest mismatch")
    if source["outcome_fields_requested"] is not False:
        raise SystemExit("source discovery requested outcome fields")
    if source["review_fields_requested"] is not False:
        raise SystemExit("source discovery requested review fields")

    policy = _read(args.policy)
    if policy["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R9 policy exposes outcomes")
    if policy["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R9 policy exposes review text")
    if policy["selected_pull_numbers"] != [3332, 3950, 31346, 31349, 48695]:
        raise SystemExit("R9 policy selectors changed")

    already_selected = {(item["project"], item["pull_number"]) for item in source["selected_cases"]}
    cases: list[HistoricalPRCandidate] = []
    seen: set[tuple[str, int]] = set()
    for selector, pull_number in SELECTORS.items():
        project = selector.removesuffix("-2")
        discovery = source["discoveries"][project]
        item = next(
            candidate for candidate in discovery["candidates"] if candidate["number"] == pull_number
        )
        identity = (project, pull_number)
        if identity in already_selected:
            raise SystemExit(f"R9 case was already scored in R8: {identity}")
        if identity in seen or not item["eligible"]:
            raise SystemExit(f"R9 selector is duplicate or ineligible: {identity}")
        seen.add(identity)
        cases.append(
            HistoricalPRCandidate(
                case_id=CASE_IDS[identity],
                project=project,
                repository=discovery["repository"],
                pull_number=pull_number,
                title=item["title"],
                created_at=item["created_at"],
                base_ref=item["base_ref"],
                base_tip_sha=item["base_ref_oid"],
                base_sha=item["base_sha"],
                base_derivation="first-pr-commit-first-parent-path-parity",
                head_sha=item["head_sha"],
                pr_commit_shas=[item["head_sha"]],
                changed_files=item["changed_files"],
                additions=item["additions"],
                deletions=item["deletions"],
                paths=item["paths"],
                acquisition_query=discovery["query"],
                selection_policy_id=policy["protocol_id"],
                outcome_fields_requested=False,
            )
        )

    cases.sort(key=lambda item: item.case_id)
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "source_outcome_free_discovery_sha256": source["selection_lock_sha256"],
        "preselection_policy_sha256": canonical_sha256(policy),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "diff_content_visible_during_selection": False,
        "selection_basis": "title, changed paths, size, and exact SHA metadata only",
        "frozen_at": datetime.now(UTC).isoformat(),
        "cases": [item.model_dump(mode="json") for item in cases],
    }
    payload = {"selection_material": material, "selection_lock_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps([item.case_id for item in cases], indent=2))
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
