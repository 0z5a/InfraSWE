#!/usr/bin/env python3
"""Freeze the outcome-blind policy for the 100-case inference R21 group."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_TERMINAL_SHA256 = (
    "sha256:aed8331c6099b99d99e9b0bc083057f24c19df4231207ab9bc13d6bff9064451"
)
EXPECTED_R20_POLICY_SHA256 = (
    "sha256:d24295227b1de65bb214193c361c1f3fad97c839e4b83deaff1a79489af38a99"
)


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def checked(path: Path, digest_field: str, expected: str) -> dict[str, Any]:
    payload = read(path)
    material = {key: value for key, value in payload.items() if key != digest_field}
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    if payload[digest_field] != expected:
        raise SystemExit(f"{path.name} identity changed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r20-terminal", type=Path, required=True)
    parser.add_argument("--r20-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    terminal = checked(
        args.r20_terminal, "recommendation_sha256", EXPECTED_TERMINAL_SHA256
    )
    prior = checked(args.r20_policy, "policy_sha256", EXPECTED_R20_POLICY_SHA256)
    if not terminal["release_gate"]["eligible_for_commit"]:
        raise SystemExit("R20 terminal policy did not clear the release gate")

    projects = copy.deepcopy(prior["projects"]["inference"])
    for project in projects.values():
        project["count"] = 25
    policy = {
        "schema_version": "0.1",
        "protocol_id": "inference-iterative-contract-v0.1-r21-100",
        "round": "R21",
        "case_count": 100,
        "domain_allocation": {"inference": 100},
        "r20_terminal_policy_sha256": terminal["recommendation_sha256"],
        "r20_policy_sha256": prior["policy_sha256"],
        "grouping_policy": {
            "requested_new_cases": 500,
            "groups": ["R21", "R22", "R23", "R24", "R25"],
            "group_size": 100,
            "iterate_only_after_reveal": True,
            "per_group_project_allocation": {
                "flashinfer": 25,
                "sglang": 25,
                "tensorrt_llm": 25,
                "vllm": 25,
            },
        },
        "created_at_window": copy.deepcopy(prior["created_at_window"]),
        "domains_in_order": ["inference"],
        "projects": {"inference": projects},
        "eligibility": copy.deepcopy(prior["eligibility"]),
        "domain_signals": copy.deepcopy(prior["domain_signals"]),
        "domain_anchor": copy.deepcopy(prior["domain_anchor"]),
        "selection_algorithm": copy.deepcopy(prior["selection_algorithm"]),
        "r21_disposition_rules": [rule["id"] for rule in terminal["terminal_rules"]],
        "blindness": copy.deepcopy(prior["blindness"]),
        "machine_policy": {
            "policy_id": "inference-contract-disposition-split-v0.1-r21",
            "technical_contract_and_disposition_are_separate": True,
            "check_requires_named_non_author_human_activity": True,
            "check_observability_without_allowed_human_activity": "unavailable",
            "recent_without_handoff_defaults_reject": True,
            "candidate_exact_boundary_failure_rejects": True,
            "unrelated_fixture_or_import_failure_is_neutral": True,
            "target_capability_gap_alone_does_not_reject": True,
            "candidate_cpp_and_integration_tests_are_first_class": True,
            "mature_accept_requires_route_or_independent_invariant": True,
            "merged_accept_recall_audited_before_release": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    if sum(project["count"] for project in projects.values()) != 100:
        raise SystemExit("R21 project allocation changed")
    payload = {**policy, "policy_sha256": canonical_sha256(policy)}
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "policy_sha256": payload["policy_sha256"],
        "domain_allocation": payload["domain_allocation"],
        "project_allocation": policy["grouping_policy"]["per_group_project_allocation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
