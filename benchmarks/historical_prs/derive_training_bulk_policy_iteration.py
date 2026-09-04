#!/usr/bin/env python3
"""Derive the next training bulk policy from one revealed group."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from freeze_training_bulk_group import _decision

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

MERGED_ACCEPT_RECALL_MINIMUM = 0.99
MERGED_ACCEPT_RECALL_REPAIR_MARGIN = 0.005


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _checked(path: Path, digest_field: str) -> dict[str, Any]:
    payload = _read(path)
    material = {key: value for key, value in payload.items() if key != digest_field}
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path}: digest mismatch")
    return payload


def _label(value: str) -> str:
    return "accept" if value == "accept_with_scope" else value


def _should_promote_candidate(
    *,
    candidate_available: bool,
    current_merged_gate_satisfied: bool,
    candidate_exact_matches: int,
    current_exact_matches: int,
) -> bool:
    return candidate_available and (
        not current_merged_gate_satisfied or candidate_exact_matches > current_exact_matches
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_lock = _checked(args.input_lock, "group_input_sha256")
    judgment = _checked(args.judgment_locks, "lock_set_sha256")
    reveal = _checked(args.reveal, "reveal_sha256")
    audit = _checked(args.audit, "audit_sha256")
    summary = audit["summary"]
    target_metric_improved = bool(summary["target_metric_improved"])
    if reveal["judgment_lock_set_sha256"] != judgment["lock_set_sha256"]:
        raise SystemExit("reveal/judgment binding mismatch")

    # The frozen judgment embeds the previously verified policy, including its
    # transport digest.  A digest is never part of the material it authenticates;
    # carrying it forward would make the next policy self-inconsistent.
    old_policy = {key: value for key, value in judgment["policy"].items() if key != "policy_sha256"}
    current_group = int(judgment["group_index"])
    frozen_at = datetime.fromisoformat(judgment["frozen_at"].replace("Z", "+00:00"))
    input_by_id = {case["case_id"]: case for case in input_lock["cases"]}
    eligible_reveals = [
        case
        for case in reveal["cases"]
        if case.get("outcome", {}).get("availability", "available") == "available"
        and case["oracle_decision"] in {"accept", "check", "reject"}
    ]
    eligible_case_ids = {case["case_id"] for case in eligible_reveals}
    changes: list[dict[str, str]]
    updates: dict[str, Any]
    candidate_recall_target = MERGED_ACCEPT_RECALL_MINIMUM
    if old_policy.get("domain") in {"inference", "communication"} or current_group >= 2:
        candidates: list[dict[str, Any]] = [{}]
        for key in (
            "small_compile_accept_enabled",
            "explicit_revert_accept_enabled",
            "active_final_head_review_priority",
            "maintainer_precedes_review_without_approval",
            "maintainer_requires_runtime_source",
        ):
            candidates.append({key: not bool(old_policy.get(key, False))})
        for value in (7, 14, 30, 60):
            if value != int(old_policy["recent_pr_max_age_days"]):
                candidates.append({"recent_pr_max_age_days": value})
        for value in (3, 7, 14, 30):
            if value != int(old_policy["active_review_max_idle_days"]):
                candidates.append({"active_review_max_idle_days": value})
        for value in (60, 120, 240):
            if value != int(old_policy["small_change_max_lines"]):
                candidates.append({"small_change_max_lines": value})
        alternate_uncertain = (
            "accept_with_scope" if old_policy["uncertain_disposition"] == "reject" else "reject"
        )
        candidates.append({"uncertain_disposition": alternate_uncertain})

        # Search a deliberately bounded recall guard.  It can only use fields
        # present in the outcome-blind input lock, and is revalidated on the
        # next group before it contributes to aggregate claims.
        candidates.append({"merged_recall_guard_projects": []})
        observed_projects = sorted(
            {input_by_id[case["case_id"]]["project"] for case in eligible_reveals}
        )
        observed_associations = sorted(
            {input_by_id[case["case_id"]]["pr_author_association"] for case in eligible_reveals}
        )
        association_sets = {
            tuple(observed_associations),
            tuple(value for value in observed_associations if value != "NONE"),
            tuple(
                value
                for value in observed_associations
                if value in {"CONTRIBUTOR", "COLLABORATOR", "MEMBER", "OWNER"}
            ),
            tuple(
                value
                for value in observed_associations
                if value in {"COLLABORATOR", "MEMBER", "OWNER"}
            ),
        }
        association_sets.discard(())
        project_sets = [(project,) for project in observed_projects]
        if len(observed_projects) > 1:
            project_sets.append(tuple(observed_projects))
        for projects in project_sets:
            for associations in sorted(association_sets):
                for review_modes in (
                    ("reviewed",),
                    ("unreviewed",),
                    ("reviewed", "unreviewed"),
                ):
                    for max_changed_lines in (30, 60, 100, 120, 240, 500, 1000, None):
                        candidates.append(
                            {
                                "merged_recall_guard_projects": list(projects),
                                "merged_recall_guard_max_changed_lines": max_changed_lines,
                                "merged_recall_guard_author_associations": list(associations),
                                "merged_recall_guard_review_modes": list(review_modes),
                            }
                        )

        deduplicated_candidates: list[dict[str, Any]] = []
        seen_candidates: set[str] = set()
        for candidate in candidates:
            encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            if encoded not in seen_candidates:
                deduplicated_candidates.append(candidate)
                seen_candidates.add(encoded)
        candidates = deduplicated_candidates

        merged_cases = sum(case["outcome"]["merged"] for case in eligible_reveals)
        current_merged_accepts = int(summary["merged_machine_accepts"])
        hard_gate_merged_accepts = math.ceil(merged_cases * MERGED_ACCEPT_RECALL_MINIMUM)
        current_merged_gate_satisfied = current_merged_accepts >= hard_gate_merged_accepts
        current_merged_recall = current_merged_accepts / merged_cases if merged_cases else 1.0
        if current_merged_gate_satisfied:
            candidate_recall_target = max(
                MERGED_ACCEPT_RECALL_MINIMUM,
                current_merged_recall,
            )
        elif old_policy.get("merged_recall_guard_projects"):
            # A guard that missed prospectively needs headroom rather than another
            # knife-edge retrospective fit.  The hard release floor remains 99%.
            candidate_recall_target = min(
                1.0,
                MERGED_ACCEPT_RECALL_MINIMUM + MERGED_ACCEPT_RECALL_REPAIR_MARGIN,
            )
        else:
            candidate_recall_target = MERGED_ACCEPT_RECALL_MINIMUM
        minimum_merged_accepts = math.ceil(merged_cases * candidate_recall_target)
        evaluated: list[tuple[int, int, int, int, int, dict[str, Any]]] = []
        for candidate_updates in candidates:
            candidate_policy = {**old_policy, **candidate_updates}
            candidate_rows = []
            for revealed in eligible_reveals:
                decision, _ = _decision(
                    input_by_id[revealed["case_id"]],
                    revealed["technical_contract"],
                    candidate_policy,
                    frozen_at,
                )
                candidate_rows.append((_label(decision), revealed))
            exact_matches = sum(
                decision == revealed["oracle_decision"] for decision, revealed in candidate_rows
            )
            merged_accepts = sum(
                decision == "accept" and revealed["outcome"]["merged"]
                for decision, revealed in candidate_rows
            )
            reject_correct = sum(
                decision == "reject" and revealed["oracle_decision"] == "reject"
                for decision, revealed in candidate_rows
            )
            check_correct = sum(
                decision == "check" and revealed["oracle_decision"] == "check"
                for decision, revealed in candidate_rows
            )
            check_reject_correct = check_correct + reject_correct
            if merged_accepts >= minimum_merged_accepts:
                evaluated.append(
                    (
                        exact_matches,
                        check_reject_correct,
                        check_correct,
                        reject_correct,
                        merged_accepts,
                        candidate_updates,
                    )
                )
        best = max(evaluated, default=(0, 0, 0, 0, 0, {}), key=lambda item: item[:5])
        if _should_promote_candidate(
            candidate_available=bool(evaluated),
            current_merged_gate_satisfied=current_merged_gate_satisfied,
            candidate_exact_matches=best[0],
            current_exact_matches=int(summary["exact_label_matches"]),
        ):
            updates = best[5]
            if current_merged_gate_satisfied:
                evidence = (
                    f"Group {current_group} retrospective exact matches improve "
                    f"from {summary['exact_label_matches']} to {best[0]} while "
                    f"preserving at least {MERGED_ACCEPT_RECALL_MINIMUM:.0%} "
                    f"merged-PR accept recall: {updates}."
                )
                rule = "promote-bounded-policy-rule"
            else:
                evidence = (
                    f"Group {current_group} merged-PR accept coverage is below the hard "
                    f"{MERGED_ACCEPT_RECALL_MINIMUM:.0%} gate "
                    f"({current_merged_accepts}/{merged_cases}); select the "
                    f"highest-exact bounded repair at {best[4]}/{merged_cases} "
                    f"merged accepts and {best[0]} exact matches using a "
                    f"{candidate_recall_target:.1%} selection target: {updates}."
                )
                rule = "repair-merged-accept-recall-gate"
            changes = [
                {
                    "rule": rule,
                    "evidence": evidence,
                }
            ]
        else:
            updates = {}
            if current_merged_gate_satisfied:
                changes = [
                    {
                        "rule": "retain-current-policy-after-search",
                        "evidence": (
                            "No bounded candidate improved exact accuracy while "
                            f"preserving >={MERGED_ACCEPT_RECALL_MINIMUM:.0%} "
                            "merged-PR accept coverage."
                        ),
                    }
                ]
            else:
                raise SystemExit(
                    f"hard {MERGED_ACCEPT_RECALL_MINIMUM:.0%} merged-PR "
                    "accept-recall gate is unresolved: "
                    f"current={current_merged_accepts}/{merged_cases}; no bounded "
                    "outcome-blind candidate satisfies the gate"
                )
    elif current_group == 0:
        updates = {
            "small_compile_accept_enabled": False,
            "explicit_revert_accept_enabled": True,
            "active_final_head_review_priority": True,
        }
        changes = [
            {
                "rule": "disable-small-compile-only-auto-accept",
                "evidence": (
                    "Three of three group-0 accepts using only this rule were closed unmerged."
                ),
            },
            {
                "rule": "accept-explicit-revert-without-hard-failure",
                "evidence": (
                    "The sole group-0 explicit revert was merged despite absent review metadata."
                ),
            },
            {
                "rule": "prioritize-recent-final-head-review-as-check",
                "evidence": (
                    "The group-0 active recent final-head review oracle was otherwise "
                    "overcalled accept."
                ),
            },
        ]
    elif current_group == 1:
        updates = {
            "active_final_head_review_priority": False,
            "maintainer_precedes_review_without_approval": False,
            "maintainer_requires_runtime_source": True,
        }
        changes = [
            {
                "rule": "maintainer-auto-accept-requires-runtime-source",
                "evidence": (
                    "The sole group-1 maintainer auto-accept error changed only a "
                    "GitHub SSO action."
                ),
            },
            {
                "rule": "rollback-approved-final-head-check-priority",
                "evidence": (
                    "The group-1 recent final-head approved PR was merged; across two "
                    "groups this signal is tied and accept preserves merged recall."
                ),
            },
            {
                "rule": "retain-unapproved-review-reject-order",
                "evidence": (
                    "Across groups 0-1, review without approval had nine reject and "
                    "four accept oracles; maintainer precedence was only 60% correct."
                ),
            },
        ]
    elif target_metric_improved:
        updates = {}
        changes = [
            {
                "rule": "retain-current-policy",
                "evidence": (
                    "The current policy beat the same-cohort legacy baseline; "
                    "no additional hand-authored rule was activated."
                ),
            }
        ]
    else:
        updates = {}
        changes = [
            {
                "rule": "retain-current-policy-after-non-improving-group",
                "evidence": (
                    "The current group did not beat the same-cohort legacy "
                    "baseline, so no classifier rule change was promoted."
                ),
            }
        ]
    policy_domain = str(old_policy.get("domain", "training"))
    policy_prefix = f"{policy_domain}-bulk-disposition"
    policy = {
        **old_policy,
        **updates,
        "policy_id": f"{policy_prefix}-v0.1-g{current_group + 1:04d}",
        "derived_from_group_index": current_group,
        "source_audit_sha256": audit["audit_sha256"],
        "source_reveal_sha256": reveal["reveal_sha256"],
        "source_group_target_metric_improved": target_metric_improved,
        "changes": changes,
    }

    projected: list[dict[str, Any]] = []
    for revealed in reveal["cases"]:
        decision, rationale = _decision(
            input_by_id[revealed["case_id"]],
            revealed["technical_contract"],
            policy,
            frozen_at,
        )
        label = _label(decision)
        projected.append(
            {
                "case_id": revealed["case_id"],
                "previous_decision": _label(revealed["machine_decision"]),
                "projected_decision": label,
                "oracle_decision": revealed["oracle_decision"],
                "oracle_eligible": revealed["case_id"] in eligible_case_ids,
                "projected_exact_match": (
                    label == revealed["oracle_decision"]
                    if revealed["case_id"] in eligible_case_ids
                    else None
                ),
                "rationale_codes": rationale,
            }
        )
    eligible_projected = [item for item in projected if item["oracle_eligible"]]
    projection_matches = sum(bool(item["projected_exact_match"]) for item in eligible_projected)
    eligible_reveal_by_id = {item["case_id"]: item for item in eligible_reveals}
    projected_merged_cases = sum(
        eligible_reveal_by_id[item["case_id"]]["outcome"]["merged"] for item in eligible_projected
    )
    projected_merged_accepts = sum(
        item["projected_decision"] == "accept"
        and eligible_reveal_by_id[item["case_id"]]["outcome"]["merged"]
        for item in eligible_projected
    )
    projected_check_cases = sum(item["oracle_decision"] == "check" for item in eligible_projected)
    projected_check_correct = sum(
        item["oracle_decision"] == "check" and item["projected_decision"] == "check"
        for item in eligible_projected
    )
    projected_reject_cases = sum(item["oracle_decision"] == "reject" for item in eligible_projected)
    projected_reject_correct = sum(
        item["oracle_decision"] == "reject" and item["projected_decision"] == "reject"
        for item in eligible_projected
    )
    material = {
        **policy,
        "retrospective_projection": {
            "case_count": len(projected),
            "eligible_case_count": len(eligible_projected),
            "exact_matches": projection_matches,
            "exact_accuracy": (
                projection_matches / len(eligible_projected) if eligible_projected else None
            ),
            "changed_case_count": sum(
                item["previous_decision"] != item["projected_decision"] for item in projected
            ),
            "merged_case_count": projected_merged_cases,
            "merged_accept_count": projected_merged_accepts,
            "merged_accept_recall": (
                projected_merged_accepts / projected_merged_cases
                if projected_merged_cases
                else None
            ),
            "merged_accept_recall_minimum": MERGED_ACCEPT_RECALL_MINIMUM,
            "merged_accept_recall_selection_target": candidate_recall_target,
            "merged_accept_recall_gate_satisfied": (
                projected_merged_cases == 0
                or projected_merged_accepts / projected_merged_cases >= MERGED_ACCEPT_RECALL_MINIMUM
            ),
            "check_case_count": projected_check_cases,
            "check_correct_count": projected_check_correct,
            "reject_case_count": projected_reject_cases,
            "reject_correct_count": projected_reject_correct,
            "check_reject_exact_count": projected_check_correct + projected_reject_correct,
            "cases": projected,
            "not_a_replacement_for_next_group_validation": True,
        },
    }
    payload = {**material, "policy_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "policy_id": payload["policy_id"],
                "retrospective_exact_matches": projection_matches,
                "retrospective_exact_accuracy": (
                    projection_matches / len(eligible_projected) if eligible_projected else None
                ),
                "policy_sha256": payload["policy_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
