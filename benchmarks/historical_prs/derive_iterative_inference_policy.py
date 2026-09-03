#!/usr/bin/env python3
"""Derive the next prospective inference policy from a sealed cohort audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

ADMIN_ASSOCIATIONS = {"COLLABORATOR", "MEMBER", "OWNER"}
FOREIGN_ACCELERATOR_PATH_MARKERS = (
    "/amd/",
    "/ascend/",
    "/npu/",
    "mi300",
    "mi325",
    "mi35",
    "rocm",
    "sm90",
    "sm100",
    "sm120",
    "blackwell",
    "h100",
    "h200",
    "b200",
    "gb200",
)


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def checked(
    path: Path, digest_field: str, *, material_field: str | None = None
) -> dict[str, Any]:
    payload = read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def label(decision: str) -> str:
    return {
        "accept_with_scope": "accept",
        "check": "check",
        "reject": "reject",
        "unresolved": "unresolved",
    }[decision]


def author_association(activity: dict[str, Any] | None) -> str:
    if not activity:
        return "UNKNOWN"
    explicit = str(activity.get("pr_author_association") or "UNKNOWN")
    if explicit in ADMIN_ASSOCIATIONS:
        return explicit
    author = activity["pr_author"]
    inferred = {
        str(event.get("author_association") or "UNKNOWN")
        for event in activity.get("projected_events", [])
        if event.get("author") == author
    }
    return sorted(inferred & ADMIN_ASSOCIATIONS)[0] if inferred & ADMIN_ASSOCIATIONS else explicit


def foreign_accelerator_target(selected: dict[str, Any]) -> bool:
    paths = [str(path).lower() for path in selected["paths"]]
    return any(
        marker in path
        for path in paths
        for marker in FOREIGN_ACCELERATOR_PATH_MARKERS
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--next-round", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = checked(args.audit, "audit_sha256")
    root = args.audit.parent
    locks = checked(root / "machine-judgment-locks.json", "lock_set_sha256")
    selection = checked(
        root / "selection-lock.json",
        "selection_lock_sha256",
        material_field="selection_material",
    )
    reveal = checked(root / "revealed-outcomes-reviews.json", "reveal_sha256")
    activity = checked(
        root / "review-activity-projection.json", "activity_projection_sha256"
    )
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("audit/lock artifact binding mismatch")
    if audit["source_digests"]["selection_lock"] != canonical_sha256(selection):
        raise SystemExit("audit/selection artifact binding mismatch")
    if audit["source_digests"]["reveal"] != canonical_sha256(reveal):
        raise SystemExit("audit/reveal artifact binding mismatch")

    source_round = activity["round"]
    if locks["protocol_id"] != selection["selection_material"]["protocol_id"]:
        raise SystemExit("lock/selection protocol mismatch")
    next_round = args.next_round.upper()
    expected_next = f"R{int(source_round[1:]) + 1}"
    if next_round != expected_next:
        raise SystemExit(f"next round must be {expected_next}")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit(f"{source_round} did not clear the user-gated improvement condition")

    selected = {
        item["case_id"]: item for item in selection["selection_material"]["cases"]
    }
    audited = {item["case_id"]: item for item in audit["cases"]}
    activities = {item["case_id"]: item for item in activity["cases"]}
    changes: list[dict[str, Any]] = []
    projected: dict[str, str] = {}
    for lock in locks["locks"]:
        material = lock["material"]
        case_id = material["case_id"]
        decision = material["decision"]
        reason = material["rationale_codes"][0]
        new_reason: str | None = None
        if reason == "RECENT_WITHOUT_ACTIVE_REVIEW":
            if audited[case_id].get(
                "oracle_final_head_explicit_human_feedback_count", 0
            ):
                decision = "check"
                new_reason = "REFRESHED_FINAL_HEAD_REVIEW_ACTIVITY"
            elif (
                material["technical_contract"] == "pass"
                and author_association(activities.get(case_id)) in ADMIN_ASSOCIATIONS
            ):
                decision = "accept_with_scope"
                new_reason = "RECENT_MAINTAINER_EXACT_TEST_PASS"
        elif reason == "MATURE_EVIDENCE_INCOMPLETE":
            decision = "accept_with_scope"
            new_reason = "MATURE_SOURCE_COMPLETE_RUNTIME_GAP"
        elif reason == "UNANIMOUS_NEGATIVE_PRECEDENT_CLUSTER" and len(
            material["precedent_consensus"]["matches"]
        ) < 3:
            decision = "accept_with_scope"
            new_reason = "NEGATIVE_PRECEDENT_BELOW_THREE_HIT_FLOOR"
        elif reason == "EXACT_CANDIDATE_FAILURE" and foreign_accelerator_target(
            selected[case_id]
        ):
            decision = "accept_with_scope"
            new_reason = "FOREIGN_ACCELERATOR_FAILURE_NEUTRALIZED"
        projected[case_id] = decision
        if new_reason is not None and decision != material["decision"]:
            changes.append(
                {
                    "case_id": case_id,
                    "from": material["decision"],
                    "to": decision,
                    "source_reason": reason,
                    "prospective_reason": new_reason,
                }
            )

    projected_exact = sum(
        label(projected[case_id]) == row["oracle_decision"]
        for case_id, row in audited.items()
    )
    projected_rejects = [
        case_id for case_id, decision in projected.items() if decision == "reject"
    ]
    projected_checks = [
        case_id for case_id, decision in projected.items() if decision == "check"
    ]
    merged_ids = {
        item["case_id"] for item in reveal["cases"] if item["outcome"]["merged"]
    }
    projected_merged_accepts = sum(
        projected[case_id] == "accept_with_scope" for case_id in merged_ids
    )
    projected_reject_correct = sum(
        audited[case_id]["oracle_decision"] == "reject" for case_id in projected_rejects
    )
    projected_check_correct = sum(
        audited[case_id]["oracle_decision"] == "check" for case_id in projected_checks
    )
    if projected_exact <= int(summary["exact_label_matches"]):
        raise SystemExit("prospective rules do not improve the sealed cohort")
    if projected_merged_accepts != len(merged_ids):
        raise SystemExit("prospective rules do not restore merged-PR accept recall")

    completed = (int(source_round[1:]) - 20) * 100
    remaining = max(0, 500 - completed)
    material = {
        "schema_version": "0.1",
        "protocol_id": (
            f"historical-pr-iterative-policy-{source_round.lower()}-to-"
            f"{next_round.lower()}-v0.1"
        ),
        "derived_after_source_reveal": True,
        "source_round": source_round,
        "next_round": next_round,
        "retrospective_source_locks_changed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "source_selection_lock_sha256": selection["selection_lock_sha256"],
        "observed_metrics": {
            key: summary[key]
            for key in (
                "cases",
                "exact_label_matches",
                "legacy_exact_label_matches",
                "same_cohort_exact_accuracy_gain",
                "binary_direction_matches",
                "machine_reject_precision",
                "machine_check_precision",
                "oracle_decisions",
                "merged_cases",
                "merged_machine_accepts",
            )
        },
        "retrospective_policy_projection": {
            "purpose": (
                f"validate only the general rules proposed for {next_round}; "
                f"{source_round} locks remain unchanged"
            ),
            "exact_label_matches": projected_exact,
            "cases": len(projected),
            "gain_over_frozen": projected_exact - int(summary["exact_label_matches"]),
            "gain_over_same_cohort_legacy": projected_exact
            - int(summary["legacy_exact_label_matches"]),
            "merged_machine_accepts": projected_merged_accepts,
            "merged_cases": len(merged_ids),
            "reject_predictions": len(projected_rejects),
            "reject_correct": projected_reject_correct,
            "reject_precision": (
                projected_reject_correct / len(projected_rejects)
                if projected_rejects
                else None
            ),
            "check_predictions": len(projected_checks),
            "check_correct": projected_check_correct,
            "check_precision": (
                projected_check_correct / len(projected_checks)
                if projected_checks
                else None
            ),
            "changed_cases": changes,
            "not_a_rescore": True,
        },
        "workload_ledger": {
            "requested": {"inference": 500, "groups": 5, "group_size": 100},
            f"completed_through_{source_round.lower()}": {"inference": completed},
            f"remaining_after_{source_round.lower()}": {"inference": remaining},
        },
        "next_group": {
            "round": next_round,
            "case_count": 100,
            "allocation": {"inference": 100},
            "inference_project_allocation": {
                "vllm": 25,
                "sglang": 25,
                "tensorrt_llm": 25,
                "flashinfer": 25,
            },
            "future_groups_preselected": False,
        },
        "review_activity_projection_allowed": True,
        "prospective_rules": [
            {
                "id": f"{next_round}-TIME-BUDGET-GATE",
                "rule": (
                    "Abandon normal PR tests after 60 seconds and TensorRT-LLM tests "
                    "after 20 seconds; timeout remains a neutral bounded gap."
                ),
            },
            {
                "id": f"{next_round}-JUST-IN-TIME-ACTIVITY-REFRESH",
                "rule": (
                    "Refresh the outcome-free review-activity projection immediately "
                    "before freezing judgments so review that arrives during execution "
                    "is observable without exposing state, outcome, CI, labels, or text."
                ),
            },
            {
                "id": f"{next_round}-CHECK-REQUIRES-ACTIVE-REVIEW",
                "rule": (
                    "Use check only for a <=7-day PR with substantive named non-author "
                    "final-head activity; otherwise never use check."
                ),
            },
            {
                "id": f"{next_round}-RECENT-MAINTAINER-PASS",
                "rule": (
                    "A recent maintainer-associated PR with an exact candidate-owned pass "
                    "may accept without non-author review; other recent PRs without active "
                    "review reject."
                ),
            },
            {
                "id": f"{next_round}-MATURE-RUNTIME-GAP-NEUTRAL",
                "rule": (
                    "Do not reject a source-complete mature PR solely because it lacks a "
                    "runnable candidate test or hits a bounded environment gap."
                ),
            },
            {
                "id": f"{next_round}-PLATFORM-MATCHED-FAILURE-ONLY",
                "rule": (
                    "A candidate assertion may trigger reject only when its declared "
                    "accelerator target matches the executor architecture recorded by "
                    "the test evidence; failures on foreign-only paths remain neutral."
                ),
            },
            {
                "id": f"{next_round}-NEGATIVE-PRECEDENT-THREE-HIT-FLOOR",
                "rule": (
                    "Negative precedent consensus requires at least three close "
                    "same-project, same-risk reject precedents and no accept neighbor."
                ),
            },
            {
                "id": f"{next_round}-EXPLICIT-READINESS-AND-INTEGRITY-VETOES",
                "rule": (
                    "Retain narrow bracketed-draft, self-declared-incomplete, and "
                    "source-integrity rejection vetoes."
                ),
            },
        ],
        "known_limitations": [
            "Historical merge disposition remains only partly identifiable from "
            "outcome-blind code and runtime evidence.",
            "The just-in-time activity projection stores metadata and derived booleans "
            "but no review text or terminal outcome fields.",
            "The retrospective projection validates prospective rules but never changes "
            "the sealed source-round judgments.",
            "The benchmark remains explicitly nonweighted, so the legacy numeric merged-"
            "score floor is not directly auditable.",
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["retrospective_policy_projection"], indent=2, sort_keys=True))
    print(f"iteration_sha256={payload['iteration_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
