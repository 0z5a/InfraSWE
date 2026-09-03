#!/usr/bin/env python3
"""Derive an interim R23 policy from the sealed non-TensorRT fast cohort."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

ADMIN_ASSOCIATIONS = {"COLLABORATOR", "MEMBER", "OWNER"}
CRITICAL_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py")


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


def _has_critical_missing_patch(static: dict[str, Any]) -> bool:
    return any(
        not item["patch_available"]
        and (item["is_test"] or str(item["path"]).lower().endswith(CRITICAL_SUFFIXES))
        for item in static["files"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--author-associations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = _checked(args.audit, "audit_sha256")
    root = args.audit.parent
    locks = _checked(root / "machine-judgment-locks.json", "lock_set_sha256")
    selection = _checked(
        root / "selection-lock.json",
        "selection_lock_sha256",
        material_field="selection_material",
    )
    reveal = _checked(root / "revealed-outcomes-reviews.json", "reveal_sha256")
    static = _checked(args.static_evidence, "evidence_sha256")
    metadata = _checked(args.author_associations, "metadata_sha256")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("audit/judgment binding mismatch")
    if audit["source_digests"]["selection_lock"] != canonical_sha256(selection):
        raise SystemExit("audit/selection binding mismatch")
    if audit["source_digests"]["reveal"] != canonical_sha256(reveal):
        raise SystemExit("audit/reveal binding mismatch")
    if metadata["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("metadata/judgment binding mismatch")
    if metadata["reveal_sha256"] != reveal["reveal_sha256"]:
        raise SystemExit("metadata/reveal binding mismatch")

    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    statics = {item["case_id"]: item for item in static["cases"]}
    audited = {item["case_id"]: item for item in audit["cases"]}
    revealed = {item["case_id"]: item for item in reveal["cases"]}
    associations = {item["case_id"]: item for item in metadata["cases"]}

    projected: dict[str, str] = {}
    changes: list[dict[str, Any]] = []
    for lock in locks["locks"]:
        material = lock["material"]
        case_id = material["case_id"]
        case = selected[case_id]
        decision = material["decision"]
        new_reason: str | None = None
        if material["rationale_codes"][
            0
        ] == "SOURCE_INTEGRITY_FAILURE" and not _has_critical_missing_patch(statics[case_id]):
            decision = "accept_with_scope"
            new_reason = "ANCILLARY_PATCH_GAP_NEUTRALIZED"
        elif case["temporal_band"] == "mature":
            association = associations[case_id]["author_association"]
            human_reviews = revealed[case_id]["human_non_author_review_count"]
            if association == "NONE":
                decision = "reject"
                new_reason = "MATURE_UNAFFILIATED_AUTHOR"
            elif human_reviews == 0 and not (
                case["project"] == "sglang" and association in ADMIN_ASSOCIATIONS
            ):
                decision = "reject"
                new_reason = "MATURE_WITHOUT_HUMAN_REVIEW"
        projected[case_id] = decision
        if decision != material["decision"]:
            changes.append(
                {
                    "case_id": case_id,
                    "from": material["decision"],
                    "to": decision,
                    "source_reason": material["rationale_codes"][0],
                    "prospective_reason": new_reason,
                }
            )

    projected_exact = sum(
        _label(projected[case_id]) == row["oracle_decision"] for case_id, row in audited.items()
    )
    projected_rejects = [case_id for case_id, decision in projected.items() if decision == "reject"]
    projected_checks = [case_id for case_id, decision in projected.items() if decision == "check"]
    merged_ids = {item["case_id"] for item in reveal["cases"] if item["outcome"]["merged"]}
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
        raise SystemExit("interim prospective rules do not improve the sealed cohort")
    if projected_merged_accepts != len(merged_ids):
        raise SystemExit("interim prospective rules lose merged-PR accept recall")

    projection = {
        "purpose": (
            "Validate only the general rules proposed for the still-hidden TensorRT "
            "subcohort and later rounds; the fast-cohort locks remain unchanged."
        ),
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
    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-interim-policy-r23-fast-to-r23-trt-v0.1",
        "prospective_policy_id": "inference-contract-disposition-cascade-v0.1-r23-trt",
        "derived_after_fast_subcohort_reveal": True,
        "tensor_rt_outcomes_visible_during_derivation": False,
        "fast_subcohort_locks_changed": False,
        "review_activity_projection_allowed": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "source_selection_lock_sha256": selection["selection_lock_sha256"],
        "source_author_metadata_sha256": metadata["metadata_sha256"],
        "observed_metrics": {
            key: audit["summary"][key]
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
            "r23_selected": 100,
            "fast_subcohort_completed": 75,
            "tensorrt_llm_subcohort_pending": 25,
        },
        "prospective_rules": [
            {
                "id": "R23-TRT-MATURE-NONE-REJECT",
                "rule": (
                    "For a mature PR, an outcome-free authorAssociation of NONE predicts "
                    "reject; terminal state, merge outcome, CI, labels, and text remain hidden."
                ),
            },
            {
                "id": "R23-TRT-MATURE-NO-HUMAN-REVIEW",
                "rule": (
                    "For a mature PR with zero non-author human review records, predict "
                    "reject, except SGLang maintainer-associated self-merge workflows."
                ),
            },
            {
                "id": "R23-TRT-ANCILLARY-PATCH-GAP-NEUTRAL",
                "rule": (
                    "A missing patch for non-code ancillary metadata alone is not source "
                    "integrity failure; code and test patch gaps remain vetoes."
                ),
            },
            {
                "id": "R23-TRT-CHECK-REMAINS-STRICT",
                "rule": (
                    "Use check only for a recent PR with substantive named non-author "
                    "final-head activity; never manufacture check for mature ambiguity."
                ),
            },
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(projection, indent=2, sort_keys=True))
    print(f"iteration_sha256={payload['iteration_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
