#!/usr/bin/env python3
"""Freeze the outcome-blind policy for the 30-case inference-only R19 group."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_ITERATION_SHA256 = (
    "sha256:e9686aa392e877a053990455c1a2ba84311e58fd60acabb2a5f910486bc792c1"
)
EXPECTED_R18_POLICY_SHA256 = (
    "sha256:23d6d5a5cee5d5b4fbb4e5c56e5e69a353f1f50b658b348bbb19a9883334db42"
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
    parser.add_argument("--r18-iteration", type=Path, required=True)
    parser.add_argument("--r18-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iteration = checked(args.r18_iteration, "iteration_sha256", EXPECTED_ITERATION_SHA256)
    prior = checked(args.r18_policy, "policy_sha256", EXPECTED_R18_POLICY_SHA256)
    allocation = iteration["r19_group"]["inference_project_allocation"]
    projects = copy.deepcopy(prior["projects"]["inference"])
    for name, project in projects.items():
        project["count"] = int(allocation[name])
    policy = {
        "schema_version": "0.1",
        "protocol_id": "inference-iterative-contract-v0.1-r19-30",
        "round": "R19",
        "case_count": 30,
        "domain_allocation": {"inference": 30},
        "r18_policy_iteration_sha256": iteration["iteration_sha256"],
        "r18_policy_sha256": prior["policy_sha256"],
        "grouping_policy": copy.deepcopy(prior["grouping_policy"]),
        "created_at_window": copy.deepcopy(prior["created_at_window"]),
        "domains_in_order": ["inference"],
        "projects": {"inference": projects},
        "eligibility": copy.deepcopy(prior["eligibility"]),
        "domain_signals": copy.deepcopy(prior["domain_signals"]),
        "domain_anchor": copy.deepcopy(prior["domain_anchor"]),
        "selection_algorithm": copy.deepcopy(prior["selection_algorithm"]),
        "r19_disposition_rules": [rule["id"] for rule in iteration["prospective_rules"]],
        "blindness": copy.deepcopy(prior["blindness"]),
        "machine_policy": {
            "policy_id": "inference-contract-disposition-split-v0.1-r19",
            "technical_contract_and_disposition_are_separate": True,
            "check_requires_explicit_external_handoff_proxy": True,
            "recent_without_handoff_defaults_reject": True,
            "target_skip_requires_target_receipt_for_accept": True,
            "small_unit_suite_is_not_disposition_proof": True,
            "final_head_evidence_overrides_stale_checklist": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    if sum(project["count"] for project in projects.values()) != 30:
        raise SystemExit("R19 project allocation changed")
    payload = {**policy, "policy_sha256": canonical_sha256(policy)}
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "policy_sha256": payload["policy_sha256"],
        "domain_allocation": payload["domain_allocation"],
        "project_allocation": allocation,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
