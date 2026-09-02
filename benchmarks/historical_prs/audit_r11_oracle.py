#!/usr/bin/env python3
"""Audit frozen judgments against the disposition oracle and same-cohort baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.policy import (
    CHECK_ACTIVITY_MAX_IDLE_DAYS,
    CHECK_NEW_PR_MAX_AGE_DAYS,
    MERGE_ACCEPT_SCORE_FLOOR_100,
    STALE_REVIEWED_OPEN_MIN_AGE_DAYS,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_substantive_feedback(item: dict[str, Any], *, author: str) -> bool:
    body = str(item.get("body") or "").strip()
    return (
        not item["is_bot"]
        and item["author"] != author
        and len(body) >= 10
        and not body.startswith("/")
    )


def _label(decision: str) -> str:
    return {
        "accept_with_scope": "accept",
        "check": "check",
        "revise": "check",
        "reject": "reject",
        "unresolved": "unresolved",
    }[decision]


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main(*, round_label: str = "R11") -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r10-audit", type=Path)
    args = parser.parse_args()

    selection = _load(args.result_root / "selection-lock.json")
    locks = _load(args.result_root / "machine-judgment-locks.json")
    reveal = _load(args.result_root / "revealed-outcomes-reviews.json")
    lock_material = {key: value for key, value in locks.items() if key != "lock_set_sha256"}
    if locks["lock_set_sha256"] != canonical_sha256(lock_material):
        raise SystemExit(f"{round_label} machine lock-set digest mismatch")
    reveal_material = {key: value for key, value in reveal.items() if key != "reveal_sha256"}
    if reveal["reveal_sha256"] != canonical_sha256(reveal_material):
        raise SystemExit(f"{round_label} reveal digest mismatch")
    if locks["merge_outcomes_visible_during_machine_judgment"] is not False:
        raise SystemExit(f"{round_label} machine lock does not assert hidden outcomes")
    if locks["review_text_visible_during_machine_judgment"] is not False:
        raise SystemExit(f"{round_label} machine lock does not assert hidden reviews")
    if reveal["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit(f"{round_label} reveal is not bound to machine locks")

    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    locked = {item["material"]["case_id"]: item for item in locks["locks"]}
    for case_id, lock in locked.items():
        if lock["lock_sha256"] != canonical_sha256(lock["material"]):
            raise SystemExit(f"{case_id}: judgment lock digest mismatch")
    cases: list[dict[str, Any]] = []
    for item in reveal["cases"]:
        case_id = item["case_id"]
        source = selected[case_id]
        lock = locked[case_id]
        if item["judgment_lock_sha256"] != lock["lock_sha256"]:
            raise SystemExit(f"{case_id}: reveal/judgment lock mismatch")
        observed_at = _time(item["observed_at"])
        created_at = _time(source["created_at"])
        age_days = (observed_at - created_at).total_seconds() / 86_400
        feedback_times = [
            _time(feedback["created_at"])
            for feedback in item["feedback"]
            if feedback["created_at"] is not None
            and _is_substantive_feedback(feedback, author=item["pr_author"])
        ]
        last_review_at = max(feedback_times, default=None)
        review_idle_days = (
            (observed_at - last_review_at).total_seconds() / 86_400
            if last_review_at is not None
            else None
        )
        outcome = item["outcome"]
        if outcome["merged"]:
            oracle = "accept"
            reason = "MERGED_PR_ACCEPT_ORACLE"
        elif outcome["state"] == "closed":
            oracle = "reject"
            reason = "CLOSED_UNMERGED_REJECT_ORACLE"
        else:
            active_review = (
                age_days <= CHECK_NEW_PR_MAX_AGE_DAYS
                and item["final_head_explicit_human_feedback_count"] > 0
                and review_idle_days is not None
                and review_idle_days <= CHECK_ACTIVITY_MAX_IDLE_DAYS
            )
            if active_review:
                oracle = "check"
                reason = "ACTIVE_NEW_PR_REVIEW_CHECK_ORACLE"
            elif (
                age_days >= STALE_REVIEWED_OPEN_MIN_AGE_DAYS
                and item["human_non_author_review_count"] > 0
            ):
                oracle = "reject"
                reason = "STALE_REVIEWED_OPEN_REJECT_ORACLE"
            else:
                oracle = "reject"
                reason = "OPEN_PR_NOT_ACTIVE_NEW_REVIEW_REJECT_ORACLE"

        machine = lock["material"]["decision"]
        legacy = lock["material"]["legacy_r10_style_decision"]
        machine_exact = _label(machine)
        legacy_exact = _label(legacy)
        machine_binary = "accept" if machine == "accept_with_scope" else "reject"
        legacy_binary = "accept" if legacy == "accept_with_scope" else "reject"
        oracle_binary = "accept" if oracle == "accept" else "reject"
        cases.append(
            {
                "case_id": case_id,
                "frozen_machine_decision": _label(machine),
                "frozen_machine_decision_wire": machine,
                "legacy_r10_style_decision": _label(legacy),
                "legacy_r10_style_decision_wire": legacy,
                "machine_exact_label": machine_exact,
                "legacy_exact_label": legacy_exact,
                "machine_binary_label": machine_binary,
                "legacy_binary_label": legacy_binary,
                "oracle_decision": oracle,
                "oracle_binary_label": oracle_binary,
                "oracle_reason": reason,
                "exact_label_match": machine_exact == oracle,
                "legacy_exact_label_match": legacy_exact == oracle,
                "binary_direction_match": machine_binary == oracle_binary,
                "legacy_binary_direction_match": legacy_binary == oracle_binary,
                "repairability_label_changed": machine_exact != legacy_exact,
                "pr_age_days": age_days,
                "review_idle_days": review_idle_days,
                "merged_score_floor_100": MERGE_ACCEPT_SCORE_FLOOR_100,
                "merged_score_floor_status": (
                    f"not-applicable-{round_label.lower()}-is-explicitly-nonweighted"
                    if outcome["merged"]
                    else "not-applicable"
                ),
                "lock_sha256": lock["lock_sha256"],
            }
        )

    frozen_nonaccept = [row for row in cases if row["legacy_r10_style_decision"] != "accept"]
    oracle_nonaccept = [row for row in frozen_nonaccept if row["oracle_decision"] != "accept"]
    exact_matches = _count(cases, "exact_label_match")
    legacy_exact_matches = _count(cases, "legacy_exact_label_match")
    binary_matches = _count(cases, "binary_direction_match")
    legacy_binary_matches = _count(cases, "legacy_binary_direction_match")
    frozen_nonaccept_matches = _count(frozen_nonaccept, "exact_label_match")
    legacy_frozen_nonaccept_matches = _count(frozen_nonaccept, "legacy_exact_label_match")
    oracle_nonaccept_matches = _count(oracle_nonaccept, "exact_label_match")
    legacy_oracle_nonaccept_matches = _count(oracle_nonaccept, "legacy_exact_label_match")
    merged_ids = [item["case_id"] for item in reveal["cases"] if item["outcome"]["merged"]]
    by_id = {row["case_id"]: row for row in cases}
    machine_rejects = [row for row in cases if row["machine_exact_label"] == "reject"]
    machine_checks = [row for row in cases if row["machine_exact_label"] == "check"]

    r10_reference: dict[str, Any] | None = None
    if args.r10_audit:
        r10 = _load(args.r10_audit)
        r10_material = {key: value for key, value in r10.items() if key != "audit_sha256"}
        if r10["audit_sha256"] != canonical_sha256(r10_material):
            raise SystemExit("R10 reference audit digest mismatch")
        r10_cases = int(r10["summary"]["cases"])
        r10_exact = int(r10["summary"]["exact_label_matches"])
        r10_binary = int(r10["summary"]["binary_direction_matches"])
        r10_reference = {
            "path": str(args.r10_audit),
            "artifact_sha256": canonical_sha256(r10),
            "cases": r10_cases,
            "exact_matches": r10_exact,
            "exact_accuracy": _ratio(r10_exact, r10_cases),
            "binary_matches": r10_binary,
            "binary_accuracy": _ratio(r10_binary, r10_cases),
            "cross_cohort_exact_accuracy_delta": _ratio(exact_matches, len(cases))
            - (r10_exact / r10_cases),
            "cross_cohort_binary_accuracy_delta": _ratio(binary_matches, len(cases))
            - (r10_binary / r10_cases),
            "causal_comparison": False,
        }

    target_improved = (
        frozen_nonaccept_matches > legacy_frozen_nonaccept_matches
        and oracle_nonaccept_matches > legacy_oracle_nonaccept_matches
    )
    summary = {
        "cases": len(cases),
        "frozen_machine_decisions": {
            decision: sum(row["frozen_machine_decision"] == decision for row in cases)
            for decision in ("accept", "check", "reject", "unresolved")
        },
        "legacy_r10_style_decisions": {
            decision: sum(row["legacy_r10_style_decision"] == decision for row in cases)
            for decision in ("accept", "check", "reject", "unresolved")
        },
        "oracle_decisions": {
            decision: sum(row["oracle_decision"] == decision for row in cases)
            for decision in ("accept", "check", "reject")
        },
        "exact_label_matches": exact_matches,
        "exact_accuracy": _ratio(exact_matches, len(cases)),
        "legacy_exact_label_matches": legacy_exact_matches,
        "legacy_exact_accuracy": _ratio(legacy_exact_matches, len(cases)),
        "same_cohort_exact_accuracy_gain": _ratio(exact_matches, len(cases))
        - (legacy_exact_matches / len(cases)),
        "binary_direction_matches": binary_matches,
        "binary_accuracy": _ratio(binary_matches, len(cases)),
        "legacy_binary_direction_matches": legacy_binary_matches,
        "legacy_binary_accuracy": _ratio(legacy_binary_matches, len(cases)),
        "frozen_nonaccept_cases": len(frozen_nonaccept),
        "frozen_nonaccept_exact_matches": frozen_nonaccept_matches,
        "legacy_frozen_nonaccept_exact_matches": legacy_frozen_nonaccept_matches,
        "oracle_nonaccept_cases_with_frozen_nonaccept": len(oracle_nonaccept),
        "oracle_nonaccept_exact_matches": oracle_nonaccept_matches,
        "legacy_oracle_nonaccept_exact_matches": legacy_oracle_nonaccept_matches,
        "machine_reject_predictions": len(machine_rejects),
        "machine_reject_correct": sum(
            row["oracle_decision"] == "reject" for row in machine_rejects
        ),
        "machine_reject_precision": _ratio(
            sum(row["oracle_decision"] == "reject" for row in machine_rejects),
            len(machine_rejects),
        ),
        "machine_check_predictions": len(machine_checks),
        "machine_check_correct": sum(row["oracle_decision"] == "check" for row in machine_checks),
        "machine_check_precision": _ratio(
            sum(row["oracle_decision"] == "check" for row in machine_checks),
            len(machine_checks),
        ),
        "target_check_reject_metric_improved": target_improved,
        "merged_cases": len(merged_ids),
        "merged_machine_accepts": sum(
            by_id[case_id]["frozen_machine_decision"] == "accept" for case_id in merged_ids
        ),
        "merged_score_floor_auditable": False,
        "merged_score_floor_exclusion": f"{round_label} explicitly used no weighted score",
    }
    material = {
        "schema_version": "0.1",
        "policy_id": "historical-disposition-oracle-v0.1",
        "prospective_policy_only": True,
        "frozen_locks_rescored": False,
        "same_cohort_legacy_baseline": "R10-style accept-if-contract-passes, otherwise check",
        "terminology_migration": {
            "current_label": "check",
            "historical_wire_label": "revise",
            "partition_changed": False,
        },
        "slash_commands_are_substantive_feedback": False,
        "thresholds": {
            "merge_accept_score_floor_100": MERGE_ACCEPT_SCORE_FLOOR_100,
            "check_new_pr_max_age_days": CHECK_NEW_PR_MAX_AGE_DAYS,
            "check_activity_max_idle_days": CHECK_ACTIVITY_MAX_IDLE_DAYS,
            "stale_reviewed_open_min_age_days": STALE_REVIEWED_OPEN_MIN_AGE_DAYS,
        },
        "source_digests": {
            "selection_lock": canonical_sha256(selection),
            "machine_judgment_locks": canonical_sha256(locks),
            "reveal": canonical_sha256(reveal),
        },
        "cases": cases,
        "summary": summary,
        "r10_historical_reference": r10_reference,
    }
    payload = {**material, "audit_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"audit_sha256={payload['audit_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
