#!/usr/bin/env python3
"""Freeze twenty R11 cases from the outcome-free R8 discovery lock."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

SELECTORS = [
    ("cutlass-cute", 3352),
    ("cutlass-cute", 3380),
    ("deepgemm", 327),
    ("deepgemm", 337),
    ("flash-attention", 2662),
    ("flash-attention", 2678),
    ("flashinfer", 3930),
    ("flashinfer", 3990),
    ("liger-kernel", 1251),
    ("liger-kernel", 1283),
    ("megatron-core", 5726),
    ("megatron-core", 5759),
    ("sglang", 31339),
    ("sglang", 31351),
    ("torchtitan", 3861),
    ("torchtitan", 3869),
    ("verl", 7010),
    ("verl", 7046),
    ("vllm", 48754),
    ("vllm", 48755),
]


def _case_id(project: str, pull_number: int) -> str:
    prefixes = {
        "cutlass-cute": "cutlass",
        "deepgemm": "deepgemm",
        "flash-attention": "flashattention",
        "flashinfer": "flashinfer",
        "liger-kernel": "liger",
        "megatron-core": "megatron",
        "sglang": "sglang",
        "torchtitan": "torchtitan",
        "verl": "verl",
        "vllm": "vllm",
    }
    return f"{prefixes[project]}-pr-{pull_number}"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _selected_identities(payload: dict[str, Any]) -> set[tuple[str, int]]:
    identities: set[tuple[str, int]] = set()
    collections = [payload.get("selected_cases", [])]
    material = payload.get("selection_material")
    if isinstance(material, dict):
        collections.append(material.get("cases", []))
    for collection in collections:
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            project = item.get("project")
            pull_number = item.get("pull_number")
            if isinstance(project, str) and isinstance(pull_number, int):
                identities.add((project, pull_number))
    return identities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-lock", type=Path, required=True)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
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
    if policy["source_discovery_lock_sha256"] != source["selection_lock_sha256"]:
        raise SystemExit("R11 policy is not bound to the discovery lock")
    if policy["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R11 policy exposes outcomes")
    if policy["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R11 policy exposes review text")
    expected_case_ids = [_case_id(*identity) for identity in SELECTORS]
    if policy["selected_case_ids"] != expected_case_ids:
        raise SystemExit("R11 policy selectors changed")

    already_selected = _selected_identities(source)
    prior_digests = []
    for path in args.prior_lock:
        prior = _read(path)
        already_selected.update(_selected_identities(prior))
        prior_digests.append(prior["selection_lock_sha256"])
    if prior_digests != policy["prior_selection_lock_sha256"]:
        raise SystemExit("R11 prior selection lock order or digest changed")

    cases: list[HistoricalPRCandidate] = []
    seen: set[tuple[str, int]] = set()
    project_counts: dict[str, int] = {}
    for project, pull_number in SELECTORS:
        discovery = source["discoveries"][project]
        item = next(
            candidate for candidate in discovery["candidates"] if candidate["number"] == pull_number
        )
        identity = (project, pull_number)
        if identity in already_selected:
            raise SystemExit(f"R11 case was already scored in R1-R10: {identity}")
        if identity in seen or not item["eligible"]:
            raise SystemExit(f"R11 selector is duplicate or ineligible: {identity}")
        seen.add(identity)
        project_counts[project] = project_counts.get(project, 0) + 1
        cases.append(
            HistoricalPRCandidate(
                case_id=_case_id(project, pull_number),
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

    if len(cases) != policy["case_count"] or set(project_counts.values()) != {2}:
        raise SystemExit("R11 allocation must be exactly two cases per project")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "source_outcome_free_discovery_sha256": source["selection_lock_sha256"],
        "prior_selection_lock_sha256": prior_digests,
        "preselection_policy_sha256": canonical_sha256(policy),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "diff_content_visible_during_selection": False,
        "selection_basis": "title, changed paths, size, and exact SHA metadata only",
        "machine_policy_id": policy["machine_policy"]["policy_id"],
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
