#!/usr/bin/env python3
"""Freeze the outcome-blind policy for the final 20-case inference R20 group."""

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
    "sha256:ac617b7676e462043655fd392e9cde39f106b8bb6a3ebf21327745bb143f802e"
)
EXPECTED_R19_POLICY_SHA256 = (
    "sha256:68b5e131407e222f3b1f60f1d894ac56934e60a27d58cf2aaf5fd0b2791cc224"
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
    parser.add_argument("--r19-iteration", type=Path, required=True)
    parser.add_argument("--r19-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iteration = checked(args.r19_iteration, "iteration_sha256", EXPECTED_ITERATION_SHA256)
    prior = checked(args.r19_policy, "policy_sha256", EXPECTED_R19_POLICY_SHA256)
    allocation = iteration["r20_group"]["inference_project_allocation"]
    projects = copy.deepcopy(prior["projects"]["inference"])
    for name, project in projects.items():
        project["count"] = int(allocation[name])
    policy = {
        "schema_version": "0.1",
        "protocol_id": "inference-iterative-contract-v0.1-r20-20",
        "round": "R20",
        "case_count": 20,
        "domain_allocation": {"inference": 20},
        "r19_policy_iteration_sha256": iteration["iteration_sha256"],
        "r19_policy_sha256": prior["policy_sha256"],
        "grouping_policy": copy.deepcopy(prior["grouping_policy"]),
        "created_at_window": copy.deepcopy(prior["created_at_window"]),
        "domains_in_order": ["inference"],
        "projects": {"inference": projects},
        "eligibility": copy.deepcopy(prior["eligibility"]),
        "domain_signals": copy.deepcopy(prior["domain_signals"]),
        "domain_anchor": copy.deepcopy(prior["domain_anchor"]),
        "selection_algorithm": copy.deepcopy(prior["selection_algorithm"]),
        "r20_disposition_rules": [rule["id"] for rule in iteration["prospective_rules"]],
        "blindness": copy.deepcopy(prior["blindness"]),
        "machine_policy": {
            "policy_id": "inference-contract-disposition-split-v0.1-r20",
            "technical_contract_and_disposition_are_separate": True,
            "check_requires_explicit_external_handoff_proxy": True,
            "recent_without_handoff_defaults_reject": True,
            "recent_technical_closure_can_directly_accept": False,
            "target_skip_requires_target_receipt_for_accept": True,
            "candidate_cpp_and_integration_tests_are_first_class": True,
            "final_head_evidence_overrides_stale_checklist": True,
            "self_report_without_executable_invariant_is_insufficient": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    if sum(project["count"] for project in projects.values()) != 20:
        raise SystemExit("R20 project allocation changed")
    payload = {**policy, "policy_sha256": canonical_sha256(policy)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "policy_sha256": payload["policy_sha256"],
                "domain_allocation": payload["domain_allocation"],
                "project_allocation": allocation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
