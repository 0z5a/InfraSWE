#!/usr/bin/env python3
"""Freeze fourteen training-path R13 cases from the outcome-free R8 discovery lock."""

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
    ("flash-attention", 2654),
    ("liger-kernel", 1274),
    ("liger-kernel", 1268),
    ("liger-kernel", 1230),
    ("megatron-core", 5808),
    ("megatron-core", 5798),
    ("megatron-core", 5743),
    ("megatron-core", 5742),
    ("torchtitan", 3841),
    ("torchtitan", 3897),
    ("torchtitan", 3867),
    ("verl", 7014),
    ("verl", 7013),
    ("verl", 6984),
]

TRAINING_SIGNALS = (
    "bwd",
    "backward",
    "grad",
    "gradient",
    "loss",
    "trainer",
    "training",
    "optimizer",
    "checkpoint",
    "fsdp",
    "hsdp",
    "activation",
    "rematerialization",
    "weight sync",
)


def _case_id(project: str, pull_number: int) -> str:
    prefixes = {
        "flash-attention": "flashattention",
        "liger-kernel": "liger",
        "megatron-core": "megatron",
        "torchtitan": "torchtitan",
        "verl": "verl",
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
        raise SystemExit("R13 policy is not bound to the discovery lock")
    hidden = (
        policy["merge_outcomes_visible_to_machine_judge"],
        policy["review_text_visible_to_machine_judge"],
        policy["ci_fields_visible_to_machine_judge"],
        policy["candidate_body_visible_during_policy_freeze"],
        policy["diff_content_visible_during_policy_freeze"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R13 policy exposes hidden evidence during policy freeze")
    expected_case_ids = [_case_id(*identity) for identity in SELECTORS]
    if policy["selected_case_ids"] != expected_case_ids:
        raise SystemExit("R13 policy selectors changed")

    already_selected = _selected_identities(source)
    prior_digests = []
    for path in args.prior_lock:
        prior = _read(path)
        already_selected.update(_selected_identities(prior))
        prior_digests.append(prior["selection_lock_sha256"])
    if prior_digests != policy["prior_selection_lock_sha256"]:
        raise SystemExit("R13 prior selection lock order or digest changed")

    cases: list[HistoricalPRCandidate] = []
    seen: set[tuple[str, int]] = set()
    for project, pull_number in SELECTORS:
        discovery = source["discoveries"][project]
        item = next(
            candidate for candidate in discovery["candidates"] if candidate["number"] == pull_number
        )
        identity = (project, pull_number)
        if identity in already_selected:
            raise SystemExit(f"R13 case was already scored in R1-R12: {identity}")
        if identity in seen or not item["eligible"]:
            raise SystemExit(f"R13 selector is duplicate or ineligible: {identity}")
        signal_text = (item["title"] + " " + " ".join(item["paths"])).lower()
        if not any(signal in signal_text for signal in TRAINING_SIGNALS):
            raise SystemExit(f"R13 selector lacks a training-path signal: {identity}")
        seen.add(identity)
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

    if len(cases) != policy["case_count"]:
        raise SystemExit("R13 allocation must contain exactly fourteen cases")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "source_outcome_free_discovery_sha256": source["selection_lock_sha256"],
        "prior_selection_lock_sha256": prior_digests,
        "preselection_policy_sha256": canonical_sha256(policy),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "candidate_body_visible_during_selection": False,
        "diff_content_visible_during_selection": False,
        "selection_basis": "title, changed paths, size, and exact SHA metadata only",
        "machine_policy_id": policy["machine_policy"]["policy_id"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "cases": [item.model_dump(mode="json") for item in cases],
    }
    payload = {
        "selection_material": material,
        "selection_lock_sha256": canonical_sha256(material),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps([item.case_id for item in cases], indent=2))
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
