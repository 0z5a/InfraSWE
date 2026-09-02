#!/usr/bin/env python3
"""Compare r4 reviews without treating stale/positive feedback as rejection labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--finality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locks = json.loads(args.judgment_locks.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    finality = json.loads(args.finality.read_text(encoding="utf-8"))
    if reviews["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("r4 reviews are not bound to the judgment locks")
    if finality["review_reveal_sha256"] != reviews["reveal_sha256"]:
        raise SystemExit("r4 finality audit is not bound to the review reveal")
    review_by_case = {item["case_id"]: item for item in reviews["cases"]}
    finality_by_case = {item["case_id"]: item for item in finality["cases"]}
    lock_by_case = {item["material"]["case_id"]: item for item in locks["locks"]}
    feedback_ids = {
        case_id: {item["feedback_id"] for item in case["feedback"]}
        for case_id, case in review_by_case.items()
    }

    cases = [
        {
            "case_id": "vllm-pr-8200",
            "judgment_lock_sha256": lock_by_case["vllm-pr-8200"]["lock_sha256"],
            "machine_decision": "revise",
            "close_classification": "stale-inactivity",
            "review_alignment_eligible": False,
            "verdict_correspondence": "not-scored",
            "exclusion_reason": (
                "All human review comments predate the final head and the PR was later closed "
                "automatically for inactivity; they are design guidance, not a final rejection."
            ),
            "historical_review_guidance": [
                {
                    "concept_id": "LAYOUT_IS_ENCODED_BY_TENSOR_STRIDES",
                    "feedback_ids": [
                        "review-comment:2031783575",
                        "review-comment:2031787982",
                        "review-comment:2039694118",
                        "review-comment:2039705133",
                    ],
                    "machine_correspondence": "exact-for-same-layout",
                    "explanation": (
                        "The final head implements this guidance and the machine confirms both "
                        "NHD/NHD and HND/HND. Its mixed key/value stride failure is additional."
                    ),
                }
            ],
            "machine_only_concepts": ["KEY_VALUE_CACHE_STRIDES_NOT_INDEPENDENT_OR_VALIDATED"],
        },
        {
            "case_id": "vllm-pr-13996",
            "judgment_lock_sha256": lock_by_case["vllm-pr-13996"]["lock_sha256"],
            "machine_decision": "revise",
            "close_classification": "author-close-without-explicit-reason",
            "review_alignment_eligible": True,
            "verdict_correspondence": "partial",
            "verdict_explanation": (
                "Reviewers required performance evidence and later reported a 5% H200 gain, "
                "with interest in picking up the work. The machine's revise decision is based "
                "on separate integration contracts, not reviewer opposition."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "DEEPGEMM_END_TO_END_PERFORMANCE_WIN",
                    "feedback_ids": [
                        "issue-comment:2693486627",
                        "issue-comment:2846141720",
                        "issue-comment:2852842104",
                    ],
                    "reviewer_concern": (
                        "Show an end-to-end throughput win on supported H200 hardware before "
                        "the integration is picked up."
                    ),
                    "correspondence": "partial",
                    "machine_rule_ids": [
                        "runtime-jit-has-a-precompile-path",
                        "unsupported-backend-early-exit-protects-benchmark-budget",
                    ],
                    "explanation": (
                        "The machine preserved benchmark/JIT boundaries but correctly did not "
                        "claim H200 performance from an A100; it therefore did not reproduce "
                        "the reported 5% gain."
                    ),
                }
            ],
            "machine_only_concepts": [
                "OPTIONAL_CAPABILITY_GATE_OR_FALLBACK_MISSING",
                "DEEPGEMM_LHS_SCALE_LAYOUT_CONTRACT_VIOLATED",
                "DEEPGEMM_RUNTIME_JIT_PRECOMPILE_CONTRACT_MISSING",
            ],
        },
    ]

    for case in cases:
        case_id = case["case_id"]
        if case["judgment_lock_sha256"] != review_by_case[case_id]["judgment_lock_sha256"]:
            raise SystemExit(f"r4 lock mismatch for {case_id}")
        if case["close_classification"] != finality_by_case[case_id]["close_classification"]:
            raise SystemExit(f"r4 close classification mismatch for {case_id}")
        concepts = [
            *case.get("historical_review_guidance", []),
            *case.get("reviewer_concepts", []),
        ]
        referenced = {
            feedback_id for concept in concepts for feedback_id in concept["feedback_ids"]
        }
        unknown = referenced - feedback_ids[case_id]
        if unknown:
            raise SystemExit(f"unknown r4 feedback ids for {case_id}: {sorted(unknown)}")

    eligible = [case for case in cases if case["review_alignment_eligible"]]
    eligible_concepts = [
        concept for case in eligible for concept in case.get("reviewer_concepts", [])
    ]
    concept_counts = {
        status: sum(concept["correspondence"] == status for concept in eligible_concepts)
        for status in ("exact", "partial", "missed")
    }
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-human-concept-comparison-v0.5-r4",
        "judgment_lock_set_sha256": locks["lock_set_sha256"],
        "review_reveal_sha256": reviews["reveal_sha256"],
        "review_finality_audit_sha256": finality["audit_sha256"],
        "post_lock_only": True,
        "rescore_machine_locks": False,
        "learned_model_used": False,
        "weighted_score_used": False,
        "cases": cases,
        "summary": {
            "closed_unmerged_cases": len(cases),
            "outcome_accuracy_computed": False,
            "outcome_accuracy_exclusion": (
                "Cases were preselected as closed-unmerged failures, so 2/2 is not an "
                "unbiased outcome-accuracy estimate."
            ),
            "review_alignment_eligible_cases": len(eligible),
            "closure_reason_attributable_cases": finality["summary"]["closure_reason_attributable"],
            "excluded_stale_or_unattributed_cases": len(cases) - len(eligible),
            "eligible_reviewer_concepts": len(eligible_concepts),
            "eligible_concept_correspondence": concept_counts,
        },
    }
    payload = {**material, "comparison_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"comparison_sha256={payload['comparison_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
