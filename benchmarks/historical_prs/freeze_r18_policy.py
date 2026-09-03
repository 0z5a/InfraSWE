#!/usr/bin/env python3
"""Freeze the outcome-blind policy for the 30-case inference-only R18 group."""

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
    "sha256:913fa960ae9a21287e22a5ca8d9663b5cec3069984e0b25dbee892f10b04930a"
)
EXPECTED_R17_POLICY_SHA256 = (
    "sha256:3bcd37c708d67618dd42ac767653eef715303a59bc38f641a8ad9299b9186a11"
)


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r17-iteration", type=Path, required=True)
    parser.add_argument("--r17-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iteration = read(args.r17_iteration)
    iteration_material = {
        key: value for key, value in iteration.items() if key != "iteration_sha256"
    }
    if iteration["iteration_sha256"] != canonical_sha256(iteration_material):
        raise SystemExit("R17 iteration digest mismatch")
    if iteration["iteration_sha256"] != EXPECTED_ITERATION_SHA256:
        raise SystemExit("R17 iteration identity changed")
    prior = read(args.r17_policy)
    prior_material = {key: value for key, value in prior.items() if key != "policy_sha256"}
    if prior["policy_sha256"] != canonical_sha256(prior_material):
        raise SystemExit("R17 policy digest mismatch")
    if prior["policy_sha256"] != EXPECTED_R17_POLICY_SHA256:
        raise SystemExit("R17 policy identity changed")

    allocation = iteration["r18_group"]["inference_project_allocation"]
    projects = copy.deepcopy(prior["projects"]["inference"])
    for name, project in projects.items():
        project["count"] = int(allocation[name])
    policy = {
        "schema_version": "0.1",
        "protocol_id": "inference-iterative-contract-v0.1-r18-30",
        "round": "R18",
        "case_count": 30,
        "domain_allocation": {"inference": 30},
        "r17_policy_iteration_sha256": iteration["iteration_sha256"],
        "r17_policy_sha256": prior["policy_sha256"],
        "grouping_policy": copy.deepcopy(prior["grouping_policy"]),
        "created_at_window": copy.deepcopy(prior["created_at_window"]),
        "domains_in_order": ["inference"],
        "projects": {"inference": projects},
        "eligibility": copy.deepcopy(prior["eligibility"]),
        "domain_signals": {
            "inference": copy.deepcopy(prior["domain_signals"]["inference"]),
            "weights": copy.deepcopy(prior["domain_signals"]["weights"]),
        },
        "domain_anchor": {
            "title_docs_prefix_is_excluded": True,
            "inference": copy.deepcopy(prior["domain_anchor"]["inference"]),
        },
        "selection_algorithm": copy.deepcopy(prior["selection_algorithm"]),
        "r18_disposition_rules": [rule["id"] for rule in iteration["prospective_rules"]],
        "blindness": copy.deepcopy(prior["blindness"]),
        "machine_policy": {
            "policy_id": "inference-contract-disposition-split-v0.1-r18",
            "technical_contract_and_disposition_are_separate": True,
            "recent_bounded_failure_can_be_check": True,
            "outcome_free_readiness_proxy_required_for_check": True,
            "target_gap_alone_is_not_reject": True,
            "candidate_test_path_can_close_by_target_skip": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    if sum(project["count"] for project in projects.values()) != policy["case_count"]:
        raise SystemExit("R18 project allocation changed")
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
