#!/usr/bin/env python3
"""Freeze fifteen additional R13 training cases without outcome evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

SELECTORS = [
    ("megatron-core", 5819),
    ("megatron-core", 5761),
    ("megatron-core", 5724),
    ("megatron-core", 5714),
    ("megatron-core", 5710),
    ("slime", 2207),
    ("slime", 2205),
    ("slime", 2204),
    ("slime", 2198),
    ("slime", 2152),
    ("verl", 7012),
    ("verl", 7005),
    ("verl", 6996),
    ("verl", 6963),
    ("verl", 6960),
]

TRAINING_SIGNALS = (
    "backward",
    "grad",
    "loss",
    "trainer",
    "training",
    "optimizer",
    "checkpoint",
    "fsdp",
    "cuda graph",
    "autocast",
    "fp8",
    "thd",
    "reinforce",
    "reward",
    "ppo",
    "off-policy",
    "rollout",
    "logprob",
    "entropy",
    "distillation",
    "teacher",
    "weight export",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _case_id(project: str, number: int) -> str:
    prefix = {"megatron-core": "megatron", "slime": "slime", "verl": "verl"}[project]
    return f"{prefix}-pr-{number}"


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
            number = item.get("pull_number")
            if isinstance(project, str) and isinstance(number, int):
                identities.add((project, number))
    return identities


def _validate_lock(payload: dict[str, Any], digest_key: str) -> None:
    if digest_key == "selection_lock_sha256" and set(payload) == {
        "selection_material",
        "selection_lock_sha256",
    }:
        material = payload["selection_material"]
    else:
        material = {key: value for key, value in payload.items() if key != digest_key}
    if payload[digest_key] != canonical_sha256(material):
        raise SystemExit(f"digest mismatch for {digest_key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r8-discovery-lock", type=Path, required=True)
    parser.add_argument("--slime-discovery-lock", type=Path, required=True)
    parser.add_argument("--base-r13-selection", type=Path, required=True)
    parser.add_argument("--base-r13-plan", type=Path, required=True)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r8 = _read(args.r8_discovery_lock)
    _validate_lock(r8, "selection_lock_sha256")
    slime = _read(args.slime_discovery_lock)
    _validate_lock(slime, "discovery_sha256")
    base_selection = _read(args.base_r13_selection)
    _validate_lock(base_selection, "selection_lock_sha256")
    base_plan = _read(args.base_r13_plan)
    _validate_lock(base_plan, "test_plan_sha256")
    policy = _read(args.policy)

    if policy["source_r8_discovery_lock_sha256"] != r8["selection_lock_sha256"]:
        raise SystemExit("extension policy/R8 discovery binding mismatch")
    if slime["preselection_policy_sha256"] != canonical_sha256(policy):
        raise SystemExit("slime discovery/policy binding mismatch")
    if policy["base_r13_selection_lock_sha256"] != base_selection["selection_lock_sha256"]:
        raise SystemExit("extension/base R13 selection binding mismatch")
    if policy["base_r13_test_plan_sha256"] != base_plan["test_plan_sha256"]:
        raise SystemExit("extension/base R13 test-plan binding mismatch")
    if base_plan["selection_lock_sha256"] != base_selection["selection_lock_sha256"]:
        raise SystemExit("base R13 selection/plan binding mismatch")
    expected_ids = [_case_id(project, number) for project, number in SELECTORS]
    if policy["selected_case_ids"] != expected_ids:
        raise SystemExit("extension selectors changed after policy freeze")

    forbidden = (
        policy["merge_outcomes_visible_to_machine_judge"],
        policy["review_text_visible_to_machine_judge"],
        policy["ci_fields_visible_to_machine_judge"],
        policy["candidate_body_visible_during_policy_freeze"],
        policy["diff_content_visible_during_policy_freeze"],
        r8["outcome_fields_requested"],
        r8["review_fields_requested"],
        slime["outcome_fields_requested"],
        slime["review_text_requested"],
        slime["ci_fields_requested"],
        slime["candidate_body_requested"],
    )
    if any(value is not False for value in forbidden):
        raise SystemExit("extension source exposed forbidden evidence")

    already_selected = _selected_identities(r8) | _selected_identities(base_selection)
    prior_digests: list[str] = []
    for path in args.prior_lock:
        prior = _read(path)
        _validate_lock(prior, "selection_lock_sha256")
        already_selected.update(_selected_identities(prior))
        prior_digests.append(prior["selection_lock_sha256"])
    if prior_digests != base_selection["selection_material"]["prior_selection_lock_sha256"]:
        raise SystemExit("extension prior-lock order differs from base R13")

    discoveries = {
        "megatron-core": r8["discoveries"]["megatron-core"],
        "verl": r8["discoveries"]["verl"],
        "slime": {
            "repository": slime["repository"],
            "query": slime["query"],
            "candidates": slime["candidates"],
        },
    }
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for project, pull_number in SELECTORS:
        discovery = discoveries[project]
        item = next(
            candidate for candidate in discovery["candidates"] if candidate["number"] == pull_number
        )
        identity = (project, pull_number)
        if identity in already_selected or identity in seen:
            raise SystemExit(f"extension case is duplicate: {identity}")
        if int(item["changed_files"]) != len(item["paths"]):
            raise SystemExit(f"extension path list is incomplete: {identity}")
        signal_text = (item["title"] + " " + " ".join(item["paths"])).lower()
        if not any(signal in signal_text for signal in TRAINING_SIGNALS):
            raise SystemExit(f"extension case lacks training signal: {identity}")
        if not item.get("base_sha") or not item.get("head_sha"):
            raise SystemExit(f"extension case lacks exact revision metadata: {identity}")
        seen.add(identity)
        cases.append(
            {
                "schema_version": "0.5",
                "case_id": _case_id(project, pull_number),
                "project": project,
                "repository": discovery["repository"],
                "pull_number": pull_number,
                "title": item["title"],
                "created_at": item["created_at"],
                "base_ref": item["base_ref"],
                "base_tip_sha": item["base_ref_oid"],
                "base_sha": item["base_sha"],
                "base_derivation": "first-pr-commit-first-parent-path-parity",
                "head_sha": item["head_sha"],
                "pr_commit_shas": [item["head_sha"]],
                "changed_files": item["changed_files"],
                "additions": item["additions"],
                "deletions": item["deletions"],
                "paths": item["paths"],
                "acquisition_query": discovery["query"],
                "selection_policy_id": policy["protocol_id"],
                "outcome_fields_requested": False,
            }
        )

    allocation = {
        project: sum(case["project"] == project for case in cases)
        for project in ("megatron-core", "slime", "verl")
    }
    if len(cases) != 15 or allocation != policy["allocation"]:
        raise SystemExit("extension allocation must be exactly five cases per project")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "base_r13_selection_lock_sha256": base_selection["selection_lock_sha256"],
        "base_r13_test_plan_sha256": base_plan["test_plan_sha256"],
        "source_r8_discovery_lock_sha256": r8["selection_lock_sha256"],
        "source_slime_discovery_lock_sha256": slime["discovery_sha256"],
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
        "cases": cases,
    }
    payload = {
        "selection_material": material,
        "selection_lock_sha256": canonical_sha256(material),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps([case["case_id"] for case in cases], indent=2))
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
