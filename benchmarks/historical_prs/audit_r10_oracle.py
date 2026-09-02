#!/usr/bin/env python3
"""Apply the frozen disposition oracle to an already revealed R10 result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.policy import (
    MERGE_ACCEPT_SCORE_FLOOR_100,
    REVISE_ACTIVITY_MAX_IDLE_DAYS,
    REVISE_NEW_PR_MAX_AGE_DAYS,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _load(args.result_root / "selection-lock.json")
    locks = _load(args.result_root / "machine-judgment-locks.json")
    reveal = _load(args.result_root / "revealed-outcomes-reviews.json")
    lock_material = {key: value for key, value in locks.items() if key != "lock_set_sha256"}
    if locks["lock_set_sha256"] != canonical_sha256(lock_material):
        raise SystemExit("R10 machine lock-set digest mismatch")
    reveal_material = {key: value for key, value in reveal.items() if key != "reveal_sha256"}
    if reveal["reveal_sha256"] != canonical_sha256(reveal_material):
        raise SystemExit("R10 reveal digest mismatch")
    if locks["merge_outcomes_visible_during_machine_judgment"] is not False:
        raise SystemExit("machine lock does not assert hidden outcomes")
    if locks["review_text_visible_during_machine_judgment"] is not False:
        raise SystemExit("machine lock does not assert hidden reviews")
    if reveal["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("reveal is not bound to this machine lock set")

    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    locked = {item["material"]["case_id"]: item for item in locks["locks"]}
    cases = []
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
                age_days <= REVISE_NEW_PR_MAX_AGE_DAYS
                and item["final_head_explicit_human_feedback_count"] > 0
                and review_idle_days is not None
                and review_idle_days <= REVISE_ACTIVITY_MAX_IDLE_DAYS
            )
            if active_review:
                oracle = "revise"
                reason = "ACTIVE_NEW_PR_REVIEW_REVISE_ORACLE"
            elif (
                age_days >= STALE_REVIEWED_OPEN_MIN_AGE_DAYS
                and item["human_non_author_review_count"] > 0
            ):
                oracle = "reject"
                reason = "STALE_REVIEWED_OPEN_REJECT_ORACLE"
            else:
                oracle = "reject"
                reason = "OPEN_PR_NOT_ACTIVE_NEW_REVIEW_REJECT_ORACLE"

        machine = item["machine_decision"]
        machine_exact = {
            "accept_with_scope": "accept",
            "revise": "revise",
            "reject": "reject",
            "unresolved": "unresolved",
        }[machine]
        machine_binary = "accept" if machine == "accept_with_scope" else "reject"
        oracle_binary = "accept" if oracle == "accept" else "reject"
        cases.append(
            {
                "case_id": case_id,
                "frozen_machine_decision": machine,
                "machine_exact_label": machine_exact,
                "machine_binary_label": machine_binary,
                "oracle_decision": oracle,
                "oracle_binary_label": oracle_binary,
                "oracle_reason": reason,
                "exact_label_match": machine_exact == oracle,
                "binary_direction_match": machine_binary == oracle_binary,
                "pr_age_days": age_days,
                "review_idle_days": review_idle_days,
                "merged_score_floor_100": MERGE_ACCEPT_SCORE_FLOOR_100,
                "merged_score_floor_status": (
                    "not-applicable-r10-is-explicitly-nonweighted"
                    if outcome["merged"]
                    else "not-applicable"
                ),
                "lock_sha256": lock["lock_sha256"],
            }
        )

    merged_case_ids = [item["case_id"] for item in reveal["cases"] if item["outcome"]["merged"]]
    case_by_id = {item["case_id"]: item for item in cases}
    material = {
        "schema_version": "0.1",
        "policy_id": "historical-disposition-oracle-v0.1",
        "prospective_policy_only": True,
        "frozen_locks_rescored": False,
        "slash_commands_are_substantive_feedback": False,
        "thresholds": {
            "merge_accept_score_floor_100": MERGE_ACCEPT_SCORE_FLOOR_100,
            "revise_new_pr_max_age_days": REVISE_NEW_PR_MAX_AGE_DAYS,
            "revise_activity_max_idle_days": REVISE_ACTIVITY_MAX_IDLE_DAYS,
            "stale_reviewed_open_min_age_days": STALE_REVIEWED_OPEN_MIN_AGE_DAYS,
        },
        "source_digests": {
            "selection_lock": canonical_sha256(selection),
            "machine_judgment_locks": canonical_sha256(locks),
            "reveal": canonical_sha256(reveal),
        },
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "frozen_machine_decisions": {
                decision: sum(case["frozen_machine_decision"] == decision for case in cases)
                for decision in ("accept_with_scope", "revise", "reject", "unresolved")
            },
            "oracle_decisions": {
                decision: sum(case["oracle_decision"] == decision for case in cases)
                for decision in ("accept", "revise", "reject")
            },
            "exact_label_matches": sum(case["exact_label_match"] for case in cases),
            "binary_direction_matches": sum(case["binary_direction_match"] for case in cases),
            "merged_cases": len(merged_case_ids),
            "merged_machine_accepts": sum(
                case_by_id[case_id]["frozen_machine_decision"] == "accept_with_scope"
                for case_id in merged_case_ids
            ),
            "merged_score_floor_auditable": False,
            "merged_score_floor_exclusion": "R10 explicitly used no weighted score",
        },
    }
    payload = {**material, "audit_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"audit_sha256={payload['audit_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
