#!/usr/bin/env python3
"""Derive the R25 policy from the sealed non-TensorRT R24 cohort."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from freeze_iterative_inference_judgments import TITLE_READINESS_RE, source_complete

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _checked(path: Path, digest_field: str, *, material_field: str | None = None) -> dict[str, Any]:
    payload = _read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def _label(decision: str) -> str:
    return {
        "accept_with_scope": "accept",
        "check": "check",
        "reject": "reject",
        "unresolved": "unresolved",
    }[decision]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review-state-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.audit.parent
    audit = _checked(args.audit, "audit_sha256")
    locks = _checked(root / "machine-judgment-locks.json", "lock_set_sha256")
    selection = _checked(
        root / "selection-lock.json",
        "selection_lock_sha256",
        material_field="selection_material",
    )
    reveal = _checked(root / "revealed-outcomes-reviews.json", "reveal_sha256")
    static = _checked(root / "static-evidence.json", "evidence_sha256")
    activity = _checked(root / "review-activity-projection.json", "activity_projection_sha256")
    review_state = _checked(args.review_state_metadata, "metadata_sha256")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("audit/judgment binding mismatch")
    if audit["source_digests"]["selection_lock"] != canonical_sha256(selection):
        raise SystemExit("audit/selection binding mismatch")
    if audit["source_digests"]["reveal"] != canonical_sha256(reveal):
        raise SystemExit("audit/reveal binding mismatch")
    if review_state["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("review-state/judgment binding mismatch")
    if review_state["reveal_sha256"] != reveal["reveal_sha256"]:
        raise SystemExit("review-state/reveal binding mismatch")
    if review_state["state_or_merge_fields_requested"] is not False:
        raise SystemExit("review-state artifact requested outcome fields")
    if review_state["review_text_requested"] is not False:
        raise SystemExit("review-state artifact requested review text")

    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    statics = {item["case_id"]: item for item in static["cases"]}
    review_states = {item["case_id"]: item for item in review_state["cases"]}
    audited = {item["case_id"]: item for item in audit["cases"]}
    revealed = {item["case_id"]: item for item in reveal["cases"]}

    projected: dict[str, str] = {}
    changes: list[dict[str, Any]] = []
    for lock in locks["locks"]:
        material = lock["material"]
        case_id = material["case_id"]
        case = selected[case_id]
        decision = material["decision"]
        reason = material["rationale_codes"][0]
        prospective_reason: str | None = None
        states = review_states[case_id]
        approvals = int(states["review_state_counts_at_lock"]["APPROVED"])
        review_records = int(states["review_record_count_at_lock"])
        explicit_readiness = TITLE_READINESS_RE.search(case["title"]) is not None
        if (
            case["temporal_band"] == "recent"
            and reason == "RECENT_MAINTAINER_EXACT_TEST_PASS"
            and case["additions"] + case["deletions"] > 120
        ):
            decision = "reject"
            prospective_reason = "RECENT_MAINTAINER_PASS_ABOVE_SMALL_CHANGE_BOUND"
        elif case["temporal_band"] == "mature" and source_complete(statics[case_id]):
            if explicit_readiness and material["technical_contract"] != "pass":
                decision = "reject"
                prospective_reason = "EXPLICIT_TITLE_READINESS_VETO"
            elif approvals > 0:
                decision = "accept_with_scope"
                prospective_reason = "MATURE_NON_AUTHOR_APPROVAL_RECEIPT"
            elif review_records > 0:
                decision = "reject"
                prospective_reason = "MATURE_REVIEW_WITHOUT_APPROVAL"
        projected[case_id] = decision
        if decision != material["decision"]:
            changes.append(
                {
                    "case_id": case_id,
                    "from": material["decision"],
                    "to": decision,
                    "source_reason": reason,
                    "prospective_reason": prospective_reason,
                }
            )

    projected_exact = sum(
        _label(projected[case_id]) == row["oracle_decision"] for case_id, row in audited.items()
    )
    projected_rejects = [case_id for case_id, decision in projected.items() if decision == "reject"]
    projected_checks = [case_id for case_id, decision in projected.items() if decision == "check"]
    merged_ids = {case_id for case_id, item in revealed.items() if item["outcome"]["merged"]}
    projected_merged_accepts = sum(
        projected[case_id] == "accept_with_scope" for case_id in merged_ids
    )
    projected_reject_correct = sum(
        audited[case_id]["oracle_decision"] == "reject" for case_id in projected_rejects
    )
    projected_check_correct = sum(
        audited[case_id]["oracle_decision"] == "check" for case_id in projected_checks
    )
    if projected_exact <= int(audit["summary"]["exact_label_matches"]):
        raise SystemExit("prospective R25 rules do not improve the sealed R24 cohort")
    if projected_merged_accepts != len(merged_ids):
        raise SystemExit("prospective R25 rules do not restore merged-PR accept recall")

    projection = {
        "purpose": "Validate general R25 rules; sealed R24 judgments remain unchanged.",
        "not_a_rescore": True,
        "cases": len(projected),
        "exact_label_matches": projected_exact,
        "exact_accuracy": projected_exact / len(projected),
        "gain_over_frozen": projected_exact - int(audit["summary"]["exact_label_matches"]),
        "gain_over_same_cohort_legacy": projected_exact
        - int(audit["summary"]["legacy_exact_label_matches"]),
        "merged_cases": len(merged_ids),
        "merged_machine_accepts": projected_merged_accepts,
        "reject_predictions": len(projected_rejects),
        "reject_correct": projected_reject_correct,
        "reject_precision": projected_reject_correct / len(projected_rejects),
        "check_predictions": len(projected_checks),
        "check_correct": projected_check_correct,
        "check_precision": (
            projected_check_correct / len(projected_checks) if projected_checks else None
        ),
        "changed_cases": changes,
    }
    summary = audit["summary"]
    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r24-to-r25-nontrt-v0.1",
        "prospective_policy_id": "inference-contract-disposition-cascade-v0.1-r25",
        "derived_after_source_reveal": True,
        "source_round": "R24",
        "next_round": "R25",
        "retrospective_source_locks_changed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "source_selection_lock_sha256": selection["selection_lock_sha256"],
        "source_activity_projection_sha256": activity["activity_projection_sha256"],
        "source_review_state_metadata_sha256": review_state["metadata_sha256"],
        "review_state_metadata_acquired_after_reveal_for_learning_only": True,
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
        "retrospective_policy_projection": projection,
        "workload_ledger": {
            "requested_inference_prs": 500,
            "completed_full_groups": 200,
            "completed_nontrt_tail_cases": 150,
            "completed_total_cases": 350,
            "next_nontrt_group_cases": 75,
            "deferred_tensorrt_llm_tail_cases_after_r25": 75,
        },
        "next_group": {
            "round": "R25",
            "case_count": 75,
            "inference_project_allocation": {
                "vllm": 25,
                "sglang": 25,
                "flashinfer": 25,
            },
            "tensorrt_llm_deferred_to_final_tail_group": True,
        },
        "review_activity_projection_allowed": True,
        "review_state_metadata_projection_allowed": True,
        "prospective_rules": [
            {
                "id": "R25-TIME-BUDGET-GATE",
                "rule": (
                    "Abandon normal PR tests after 60 seconds; timeout remains a "
                    "neutral bounded gap."
                ),
            },
            {
                "id": "R25-TEXT-FREE-REVIEW-STATE-PROJECTION",
                "rule": (
                    "Acquire non-author review states just before judgment while "
                    "keeping review text, PR state, merge outcome, CI, and labels hidden."
                ),
            },
            {
                "id": "R25-MATURE-APPROVAL-RECEIPT",
                "rule": (
                    "For an exact-source-complete mature PR without an explicit "
                    "draft/WIP veto, any pre-lock non-author approval is an accept receipt."
                ),
            },
            {
                "id": "R25-MATURE-REVIEW-WITHOUT-APPROVAL-REJECT",
                "rule": (
                    "For an exact-source-complete mature PR with review records but no "
                    "non-author approval, predict reject rather than treating comments "
                    "alone as merge readiness."
                ),
            },
            {
                "id": "R25-NO-REVIEW-DISPOSITION-CASCADE",
                "rule": (
                    "When no review record exists, retain the affiliation and precedent "
                    "cascade, including the SGLang maintainer self-merge workflow."
                ),
            },
            {
                "id": "R25-CHECK-REMAINS-STRICT",
                "rule": (
                    "Use check only for a <=7-day PR with substantive named non-author "
                    "final-head activity; never use check for mature ambiguity."
                ),
            },
            {
                "id": "R25-RECENT-MAINTAINER-SMALL-PASS",
                "rule": (
                    "A recent maintainer-associated PR without active review may accept "
                    "on an exact pass only when its total diff is at most 120 lines."
                ),
            },
            {
                "id": "R25-EXACT-SIDE-SOURCE-INTEGRITY",
                "rule": (
                    "Judge source integrity from the exact required base/head sides; a "
                    "missing rendered patch alone, including an empty added file, is neutral."
                ),
            },
            {
                "id": "R25-EXPLICIT-DRAFT-WIP-VETO",
                "rule": "Treat explicit title-level draft/WIP markers as readiness vetoes.",
            },
        ],
        "known_limitations": [
            "Approval state is a strong but imperfect merge-readiness proxy; R25 remains "
            "a prospective validation cohort.",
            "The review-state training artifact was acquired after R24 reveal, filtered "
            "to each pre-lock timestamp, and cannot change the sealed R24 locks.",
            "Terminal PR state and merge outcome remain unavailable during judgment.",
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(projection, indent=2, sort_keys=True))
    print(f"iteration_sha256={payload['iteration_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
