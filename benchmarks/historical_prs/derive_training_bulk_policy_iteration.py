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
from historical_bulk_quality_gates import (
    EXACT_ACCURACY_MINIMUM,
    MERGED_ACCEPT_RECALL_MINIMUM,
    exact_accuracy_gate_satisfied,
    merged_accept_recall_gate_satisfied,
    minimum_successes,
    release_quality_gate_satisfied,
)

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

MERGED_ACCEPT_RECALL_REPAIR_MARGIN = 0.005
CUMULATIVE_STRUCTURAL_MINIMUM_GROUPS = 6


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


def _cohort(
    *,
    input_lock: dict[str, Any],
    judgment: dict[str, Any],
    reveal: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    input_by_id = {case["case_id"]: case for case in input_lock["cases"]}
    rows = []
    for revealed in reveal["cases"]:
        if revealed.get("outcome", {}).get("availability", "available") != "available" or revealed[
            "oracle_decision"
        ] not in {"accept", "check", "reject"}:
            continue
        rows.append(
            {
                "case": input_by_id[revealed["case_id"]],
                "technical_contract": revealed["technical_contract"],
                "oracle_decision": revealed["oracle_decision"],
            }
        )
    return {
        "group_index": int(judgment["group_index"]),
        "frozen_at": datetime.fromisoformat(judgment["frozen_at"].replace("Z", "+00:00")),
        "audit_sha256": audit["audit_sha256"],
        "rows": rows,
    }


def _evaluate_cohort(policy: dict[str, Any], cohort: dict[str, Any]) -> dict[str, int]:
    metrics = {
        "eligible_cases": 0,
        "exact_matches": 0,
        "merged_cases": 0,
        "merged_accepts": 0,
    }
    for row in cohort["rows"]:
        decision, _ = _decision(
            row["case"],
            row["technical_contract"],
            policy,
            cohort["frozen_at"],
        )
        decision = _label(decision)
        oracle = row["oracle_decision"]
        metrics["eligible_cases"] += 1
        metrics["exact_matches"] += decision == oracle
        metrics["merged_cases"] += oracle == "accept"
        metrics["merged_accepts"] += oracle == "accept" and decision == "accept"
    return metrics


def _merged_recall_gate_satisfied(metrics: dict[str, int]) -> bool:
    # A cohort without an Accept oracle cannot disprove a candidate during
    # incremental search. Final release qualification remains fail-closed and
    # requires positive support for both gates.
    return metrics["merged_cases"] == 0 or merged_accept_recall_gate_satisfied(
        merged_accepts=metrics["merged_accepts"],
        merged_cases=metrics["merged_cases"],
    )


def _load_prior_cohorts(history_root: Path | None, current_group: int) -> list[dict[str, Any]]:
    if history_root is None or not history_root.is_dir():
        return []
    cohorts = []
    for group_dir in sorted(history_root.glob("group-[0-9][0-9][0-9][0-9]")):
        group_index = int(group_dir.name.removeprefix("group-"))
        if group_index >= current_group:
            continue
        required = {
            "input": group_dir / "input-lock.json",
            "judgment": group_dir / "judgment-locks.json",
            "reveal": group_dir / "outcome-reveal.json",
            "audit": group_dir / "oracle-audit.json",
        }
        if not all(path.is_file() for path in required.values()):
            continue
        input_lock = _checked(required["input"], "group_input_sha256")
        judgment = _checked(required["judgment"], "lock_set_sha256")
        reveal = _checked(required["reveal"], "reveal_sha256")
        audit = _checked(required["audit"], "audit_sha256")
        if any(
            int(payload["group_index"]) != group_index
            for payload in (input_lock, judgment, reveal, audit)
        ):
            raise SystemExit(f"{group_dir}: cumulative experience group-index mismatch")
        if judgment["group_input_sha256"] != input_lock["group_input_sha256"]:
            raise SystemExit(f"{group_dir}: cumulative judgment/input binding mismatch")
        if reveal["judgment_lock_set_sha256"] != judgment["lock_set_sha256"]:
            raise SystemExit(f"{group_dir}: cumulative reveal/judgment binding mismatch")
        if audit["reveal_sha256"] != reveal["reveal_sha256"]:
            raise SystemExit(f"{group_dir}: cumulative audit/reveal binding mismatch")
        cohorts.append(
            _cohort(
                input_lock=input_lock,
                judgment=judgment,
                reveal=reveal,
                audit=audit,
            )
        )
    return cohorts


def _existing_structural_pairs(policy: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for rule in policy.get("structural_reject_rules") or []:
        for project in rule.get("projects") or []:
            for association in rule.get("author_associations") or []:
                pairs.add((str(project), str(association)))
    return pairs


def _select_cumulative_structural_update(
    *,
    old_policy: dict[str, Any],
    prior_cohorts: list[dict[str, Any]],
    current_cohort: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]] | None:
    """Select on prior sealed groups, then require a fresh current-group confirmation."""

    if len(prior_cohorts) < CUMULATIVE_STRUCTURAL_MINIMUM_GROUPS:
        return None
    pairs = set()
    for cohort in prior_cohorts:
        for row in cohort["rows"]:
            decision, rationale = _decision(
                row["case"],
                row["technical_contract"],
                old_policy,
                cohort["frozen_at"],
            )
            if _label(decision) == "accept" and rationale[0] == "MERGED_RECALL_GUARD_PROJECT_SCOPE":
                pairs.add(
                    (
                        str(row["case"]["project"]),
                        str(row["case"]["pr_author_association"]),
                    )
                )
    pairs -= _existing_structural_pairs(old_policy)
    baselines = [_evaluate_cohort(old_policy, cohort) for cohort in prior_cohorts]
    ranked = []
    existing_rules = list(old_policy.get("structural_reject_rules") or [])
    for project, association in sorted(pairs):
        learned_rule = {
            "projects": [project],
            "author_associations": [association],
        }
        candidate_policy = {
            **old_policy,
            "structural_reject_rules": [*existing_rules, learned_rule],
        }
        candidate_metrics = [_evaluate_cohort(candidate_policy, cohort) for cohort in prior_cohorts]
        gains = [
            candidate["exact_matches"] - baseline["exact_matches"]
            for baseline, candidate in zip(baselines, candidate_metrics, strict=True)
        ]
        if (
            sum(gains) <= 0
            or min(gains) < 0
            or not all(_merged_recall_gate_satisfied(item) for item in candidate_metrics)
        ):
            continue
        ranked.append(
            (
                sum(gains),
                sum(gain > 0 for gain in gains),
                min(gains),
                project,
                association,
                learned_rule,
                candidate_policy,
                candidate_metrics,
            )
        )
    if not ranked:
        return None

    selected = max(ranked, key=lambda item: item[:5])
    (
        history_gain,
        improved_group_count,
        minimum_group_gain,
        project,
        association,
        learned_rule,
        candidate_policy,
        candidate_metrics,
    ) = selected
    current_baseline = _evaluate_cohort(old_policy, current_cohort)
    current_candidate = _evaluate_cohort(candidate_policy, current_cohort)
    current_gain = current_candidate["exact_matches"] - current_baseline["exact_matches"]
    if current_gain <= 0 or not _merged_recall_gate_satisfied(current_candidate):
        return None

    source_audits = [cohort["audit_sha256"] for cohort in prior_cohorts]
    history_eligible = sum(item["eligible_cases"] for item in baselines)
    history_before = sum(item["exact_matches"] for item in baselines)
    history_after = sum(item["exact_matches"] for item in candidate_metrics)
    experience_material = {
        "schema_version": "0.1",
        "provenance_tier": "reconstructed",
        "trajectory_status": "transcript-only",
        "policy_gradient_eligible": False,
        "allowed_uses": ["external-policy", "curriculum", "offline-retrieval", "audit"],
        "source_group_indices": [cohort["group_index"] for cohort in prior_cohorts],
        "source_audit_sha256s": source_audits,
        "selected_rule": learned_rule,
        "history_eligible_cases": history_eligible,
        "history_exact_matches_before": history_before,
        "history_exact_matches_after": history_after,
        "history_exact_match_gain": history_gain,
        "history_improved_group_count": improved_group_count,
        "history_minimum_group_gain": minimum_group_gain,
        "prospective_confirmation_group_index": current_cohort["group_index"],
        "prospective_exact_matches_before": current_baseline["exact_matches"],
        "prospective_exact_matches_after": current_candidate["exact_matches"],
        "prospective_exact_match_gain": current_gain,
        "prospective_merged_cases": current_candidate["merged_cases"],
        "prospective_merged_accepts": current_candidate["merged_accepts"],
        "merged_accept_recall_minimum": MERGED_ACCEPT_RECALL_MINIMUM,
    }
    experience = {
        **experience_material,
        "experience_sha256": canonical_sha256(experience_material),
    }
    updates = {
        "structural_reject_rules": [*existing_rules, learned_rule],
        "cumulative_experience": experience,
    }
    changes = [
        {
            "rule": "promote-cumulative-project-author-reject-rule",
            "evidence": (
                f"Prior sealed groups gained {history_gain} exact matches across "
                f"{history_eligible} eligible cases without regressing any group or "
                f"crossing the {MERGED_ACCEPT_RECALL_MINIMUM:.0%} merged-PR recall floor; "
                f"fresh group {current_cohort['group_index']} added {current_gain} exact "
                f"matches under the same floor for project={project}, "
                f"author_association={association}."
            ),
        }
    ]
    return updates, changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--history-root", type=Path)
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
    if audit["judgment_lock_set_sha256"] != judgment["lock_set_sha256"]:
        raise SystemExit("audit/judgment binding mismatch")
    if audit["reveal_sha256"] != reveal["reveal_sha256"]:
        raise SystemExit("audit/reveal binding mismatch")

    # The frozen judgment embeds the previously verified policy, including its
    # transport digest.  A digest is never part of the material it authenticates;
    # carrying it forward would make the next policy self-inconsistent.
    old_policy = {key: value for key, value in judgment["policy"].items() if key != "policy_sha256"}
    old_policy["uncertain_disposition"] = _label(str(old_policy["uncertain_disposition"]))
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
    history_root = args.history_root
    if (
        history_root is None
        and args.input_lock.parent.name.startswith("group-")
        and args.input_lock.parent.parent.name == "groups"
    ):
        history_root = args.input_lock.parent.parent
    prior_cohorts = _load_prior_cohorts(history_root, current_group)
    current_cohort = _cohort(
        input_lock=input_lock,
        judgment=judgment,
        reveal=reveal,
        audit=audit,
    )
    changes: list[dict[str, str]]
    updates: dict[str, Any]
    candidate_recall_target = MERGED_ACCEPT_RECALL_MINIMUM
    cumulative_update = None
    if old_policy.get("domain") in {"inference", "communication"} or current_group >= 2:
        cumulative_update = _select_cumulative_structural_update(
            old_policy=old_policy,
            prior_cohorts=prior_cohorts,
            current_cohort=current_cohort,
        )
    if cumulative_update is not None:
        updates, changes = cumulative_update
    elif old_policy.get("domain") in {"inference", "communication"} or current_group >= 2:
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
            "accept" if old_policy["uncertain_disposition"] == "reject" else "reject"
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
                changes = [
                    {
                        "rule": "retain-current-policy-to-collect-more-blind-evidence",
                        "evidence": (
                            f"The hard {MERGED_ACCEPT_RECALL_MINIMUM:.0%} merged-PR "
                            "accept-recall release gate remains unresolved at "
                            f"{current_merged_accepts}/{merged_cases}; no bounded "
                            "outcome-blind candidate satisfies the gate. Carry the "
                            "policy forward unchanged only to collect another sealed "
                            "blind cohort; retrospective_projection remains fail-closed "
                            "and this policy is not release-qualified."
                        ),
                    }
                ]
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
    exact_gate_satisfied = exact_accuracy_gate_satisfied(
        exact_matches=projection_matches,
        eligible_cases=len(eligible_projected),
    )
    merged_recall_gate_satisfied = merged_accept_recall_gate_satisfied(
        merged_accepts=projected_merged_accepts,
        merged_cases=projected_merged_cases,
    )
    release_gate_satisfied = release_quality_gate_satisfied(
        exact_matches=projection_matches,
        eligible_cases=len(eligible_projected),
        merged_accepts=projected_merged_accepts,
        merged_cases=projected_merged_cases,
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
            "exact_accuracy_minimum": EXACT_ACCURACY_MINIMUM,
            "exact_accuracy_required_matches": minimum_successes(
                len(eligible_projected), EXACT_ACCURACY_MINIMUM
            ),
            "exact_accuracy_gate_satisfied": exact_gate_satisfied,
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
            "merged_accept_recall_required_accepts": minimum_successes(
                projected_merged_cases, MERGED_ACCEPT_RECALL_MINIMUM
            ),
            "merged_accept_recall_selection_target": candidate_recall_target,
            "merged_accept_recall_gate_satisfied": merged_recall_gate_satisfied,
            "release_quality_gate_satisfied": release_gate_satisfied,
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
