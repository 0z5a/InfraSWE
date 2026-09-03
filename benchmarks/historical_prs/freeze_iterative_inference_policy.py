#!/usr/bin/env python3
"""Freeze one prospective 100-case inference policy after the prior reveal."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def checked(path: Path, digest_field: str) -> dict[str, Any]:
    payload = read(path)
    material = {key: value for key, value in payload.items() if key != digest_field}
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--previous-policy", type=Path, required=True)
    parser.add_argument("--previous-audit", type=Path, required=True)
    parser.add_argument("--previous-iteration", type=Path, required=True)
    parser.add_argument("--exclude-project", action="append", default=[])
    parser.add_argument("--per-project-count", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    round_label = args.round.upper()
    if not round_label.startswith("R") or not round_label[1:].isdigit():
        raise SystemExit("--round must look like R22")
    previous_round = f"R{int(round_label[1:]) - 1}"
    previous_policy = checked(args.previous_policy, "policy_sha256")
    previous_audit = checked(args.previous_audit, "audit_sha256")
    iteration = checked(args.previous_iteration, "iteration_sha256")
    if iteration["source_audit_sha256"] != previous_audit["audit_sha256"]:
        raise SystemExit("iteration/audit binding mismatch")
    if not previous_audit["summary"]["target_check_reject_metric_improved"]:
        raise SystemExit(f"{previous_round} did not clear the iteration gate")

    projects = copy.deepcopy(previous_policy["projects"]["inference"])
    inherited_reordering = previous_policy.get("execution_reordering", {})
    inherited_deferred = set(inherited_reordering.get("deferred_to_tail_group", []))
    unknown_projects = set(args.exclude_project) - projects.keys() - inherited_deferred
    if unknown_projects:
        raise SystemExit(f"unknown --exclude-project values: {sorted(unknown_projects)}")
    for project_name in args.exclude_project:
        projects.pop(project_name, None)
    for project in projects.values():
        project["count"] = args.per_project_count
    case_count = sum(int(project["count"]) for project in projects.values())
    if case_count <= 0:
        raise SystemExit("project allocation selected no cases")
    machine_policy = copy.deepcopy(previous_policy["machine_policy"])
    machine_policy.update(
        {
            "policy_id": (
                f"inference-contract-disposition-split-v0.1-{round_label.lower()}"
            ),
            "outcome_free_review_activity_projection_allowed": bool(
                iteration.get("review_activity_projection_allowed", False)
            ),
            "outcome_free_review_state_metadata_projection_allowed": bool(
                iteration.get("review_state_metadata_projection_allowed", False)
            ),
            "review_text_visible": False,
            "long_running_normal_pr_seconds": 60,
            "long_running_tensorrt_llm_pr_seconds": 20,
            "timeout_disposition": "abandoned-time-budget-neutral",
            "prior_revealed_precedent_consensus_allowed": True,
        }
    )
    policy = {
        "schema_version": "0.1",
        "protocol_id": (
            f"inference-iterative-contract-v0.1-{round_label.lower()}-{case_count}"
        ),
        "round": round_label,
        "case_count": case_count,
        "domain_allocation": {"inference": case_count},
        "previous_round": previous_round,
        "previous_policy_sha256": previous_policy["policy_sha256"],
        "previous_audit_sha256": previous_audit["audit_sha256"],
        "previous_iteration_sha256": iteration["iteration_sha256"],
        "grouping_policy": copy.deepcopy(previous_policy["grouping_policy"]),
        "execution_reordering": {
            "excluded_projects": sorted(set(args.exclude_project) | inherited_deferred),
            "deferred_to_tail_group": sorted(
                set(args.exclude_project) | inherited_deferred
            ),
            "user_directed": bool(args.exclude_project) or bool(inherited_deferred),
            "inherited_from_previous_round": sorted(inherited_deferred),
        },
        "created_at_window": copy.deepcopy(previous_policy["created_at_window"]),
        "domains_in_order": ["inference"],
        "projects": {"inference": projects},
        "eligibility": copy.deepcopy(previous_policy["eligibility"]),
        "domain_signals": copy.deepcopy(previous_policy["domain_signals"]),
        "domain_anchor": copy.deepcopy(previous_policy["domain_anchor"]),
        "selection_algorithm": copy.deepcopy(previous_policy["selection_algorithm"]),
        "prospective_rule_ids": [rule["id"] for rule in iteration["prospective_rules"]],
        "blindness": copy.deepcopy(previous_policy["blindness"]),
        "allowed_post_selection_projection": {
            "review_activity_metadata": bool(
                iteration.get("review_activity_projection_allowed", False)
            ),
            "review_state_metadata": bool(
                iteration.get("review_state_metadata_projection_allowed", False)
            ),
            "review_or_comment_text": False,
            "state_or_merge": False,
            "ci_or_label": False,
        },
        "machine_policy": machine_policy,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    if sum(project["count"] for project in projects.values()) != case_count:
        raise SystemExit(f"{round_label} project allocation changed")
    payload = {**policy, "policy_sha256": canonical_sha256(policy)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "round": round_label,
                "policy_sha256": payload["policy_sha256"],
                "previous_iteration_sha256": iteration["iteration_sha256"],
                "case_count": policy["case_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
