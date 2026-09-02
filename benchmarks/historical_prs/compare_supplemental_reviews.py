#!/usr/bin/env python3
"""Create a post-lock, nonweighted concept comparison for the r2 failure supplement."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locks = json.loads(args.judgment_locks.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    if reviews["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("review evidence is not bound to this judgment lock set")
    review_by_case = {item["case_id"]: item for item in reviews["cases"]}
    lock_by_case = {item["material"]["case_id"]: item for item in locks["locks"]}
    feedback_ids = {
        case_id: {item["feedback_id"] for item in case["feedback"]}
        for case_id, case in review_by_case.items()
    }

    cases = [
        {
            "case_id": "sglang-pr-3044",
            "judgment_lock_sha256": lock_by_case["sglang-pr-3044"]["lock_sha256"],
            "machine_decision": "revise",
            "verdict_correspondence": "exact",
            "verdict_explanation": (
                "The reviewer requested a placement change before merge; the machine also "
                "required revision, but for different reasons."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "TEST_DIRECTORY_TAXONOMY",
                    "feedback_ids": ["review-comment:1926053341"],
                    "reviewer_concern": (
                        "Kernel tests should live under test/srt/kernel rather than directly "
                        "under the broad test/srt root."
                    ),
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "The r2 rules checked discoverability, matrix size, and diagnostics but "
                        "did not ask whether the destination matched project test taxonomy."
                    ),
                }
            ],
            "machine_only_concepts": [
                "TEST_MATRIX_CONTRACTION_GT_50_PERCENT",
                "DIAGNOSTIC_SUPPRESSION_BROAD_USER_WARNING",
            ],
        },
        {
            "case_id": "vllm-pr-13140",
            "judgment_lock_sha256": lock_by_case["vllm-pr-13140"]["lock_sha256"],
            "machine_decision": "accept_with_scope",
            "verdict_correspondence": "exact",
            "verdict_explanation": (
                "The reviewer approved an interim correctness fix conditional on acceptable "
                "performance; the frozen machine decision had the same scope."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "INTERIM_CORRECTNESS_FIX",
                    "feedback_ids": ["review:2612809843"],
                    "reviewer_concern": "Land the barrier as an interim correctness repair.",
                    "correspondence": "exact",
                    "machine_rule_ids": [
                        "shared-initialization-happens-before-cross-warp-atomic",
                        "affected-kernel-sm80-correctness",
                    ],
                    "explanation": (
                        "The machine independently proved the happens-before gap and verified "
                        "the patched SM80 kernel."
                    ),
                },
                {
                    "concept_id": "BARRIER_PERFORMANCE_REGRESSION",
                    "feedback_ids": ["review:2612809843"],
                    "reviewer_concern": (
                        "Show that the added barrier does not materially regress performance."
                    ),
                    "correspondence": "exact",
                    "machine_rule_ids": ["steady-state-regression-below-ten-percent"],
                    "explanation": (
                        "The frozen check separated precompile from steady state and observed a "
                        "0.9343 head/base time ratio on A100."
                    ),
                },
                {
                    "concept_id": "KERNEL_ARCHITECTURE_RETHINK",
                    "feedback_ids": ["review:2612809843"],
                    "reviewer_concern": (
                        "The kernel needs a broader architectural rethink after the interim fix."
                    ),
                    "correspondence": "partial",
                    "machine_rule_ids": ["shared-initialization-happens-before-cross-warp-atomic"],
                    "explanation": (
                        "accept_with_scope encoded limited approval, but r2 did not surface the "
                        "nearby source TODO or name the follow-up ownership debt."
                    ),
                },
            ],
            "machine_only_concepts": [],
        },
        {
            "case_id": "vllm-pr-14027",
            "judgment_lock_sha256": lock_by_case["vllm-pr-14027"]["lock_sha256"],
            "machine_decision": "revise",
            "verdict_correspondence": "partial",
            "verdict_explanation": (
                "Both sides found unresolved work, but reviewers did not identify the frozen "
                "renormalize regression that drove the machine verdict."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "ROUTER_BIAS_DTYPE_CONSISTENCY",
                    "feedback_ids": [
                        "review-comment:1975332689",
                        "issue-comment:2803539213",
                    ],
                    "reviewer_concern": (
                        "Router correction bias initialization must have the intended FP32 dtype."
                    ),
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "r2 did not trace dtype consistency across weight, bias, and state dict."
                    ),
                },
                {
                    "concept_id": "STORAGE_DTYPE_VS_COMPUTE_DTYPE",
                    "feedback_ids": ["review-comment:2043332053"],
                    "reviewer_concern": (
                        "Keep stored weights in BF16 where possible and perform only computation "
                        "in FP32 to avoid extra memory."
                    ),
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "r2 tested numerical behavior but did not audit parameter storage growth."
                    ),
                },
                {
                    "concept_id": "ROUTER_PERFORMANCE_REGRESSION",
                    "feedback_ids": ["issue-comment:2690849736"],
                    "reviewer_concern": "Run performance benchmarks for the widened router path.",
                    "correspondence": "partial",
                    "machine_rule_ids": ["compile-free-minimal-probe"],
                    "explanation": (
                        "r2 measured the extracted fused_topk hot call, but not the widened gate "
                        "linear layer or an end-to-end workload."
                    ),
                },
                {
                    "concept_id": "MODEL_QUALITY_REGRESSION",
                    "feedback_ids": ["issue-comment:2811019420"],
                    "reviewer_concern": "Check task-level quality after changing router precision.",
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "r2 stopped at operator numerics and did not run model-level quality cases."
                    ),
                },
            ],
            "machine_only_concepts": ["CALLER_CONTRACT_PARAMETER_IGNORED:renormalize"],
        },
    ]

    for case in cases:
        case_id = case["case_id"]
        if case["judgment_lock_sha256"] != review_by_case[case_id]["judgment_lock_sha256"]:
            raise SystemExit(f"lock mismatch in comparison for {case_id}")
        referenced = {
            feedback_id
            for concept in case["reviewer_concepts"]
            for feedback_id in concept["feedback_ids"]
        }
        unknown = referenced - feedback_ids[case_id]
        if unknown:
            raise SystemExit(f"unknown feedback ids for {case_id}: {sorted(unknown)}")

    concepts = [concept for case in cases for concept in case["reviewer_concepts"]]
    counts = {
        status: sum(concept["correspondence"] == status for concept in concepts)
        for status in ("exact", "partial", "missed")
    }
    verdict_counts = {
        status: sum(case["verdict_correspondence"] == status for case in cases)
        for status in ("exact", "partial", "missed")
    }
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-human-concept-comparison-v0.5-r2",
        "judgment_lock_set_sha256": locks["lock_set_sha256"],
        "review_reveal_sha256": reviews["reveal_sha256"],
        "post_lock_only": True,
        "rescore_machine_locks": False,
        "learned_model_used": False,
        "weighted_score_used": False,
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "verdict_correspondence": verdict_counts,
            "reviewer_concepts": len(concepts),
            "concept_correspondence": counts,
            "exact_concept_coverage": counts["exact"] / len(concepts),
            "exact_or_partial_concept_coverage": (counts["exact"] + counts["partial"])
            / len(concepts),
            "machine_only_concepts": sum(len(case["machine_only_concepts"]) for case in cases),
        },
    }
    payload = {**material, "comparison_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"comparison_sha256={payload['comparison_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
