#!/usr/bin/env python3
"""Create the post-lock, nonweighted concept comparison for r3."""

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
        raise SystemExit("r3 review evidence is not bound to the judgment lock set")
    review_by_case = {item["case_id"]: item for item in reviews["cases"]}
    lock_by_case = {item["material"]["case_id"]: item for item in locks["locks"]}
    feedback_ids = {
        case_id: {item["feedback_id"] for item in case["feedback"]}
        for case_id, case in review_by_case.items()
    }

    cases = [
        {
            "case_id": "sglang-pr-3136",
            "judgment_lock_sha256": lock_by_case["sglang-pr-3136"]["lock_sha256"],
            "machine_decision": "revise",
            "verdict_correspondence": "exact",
            "verdict_explanation": (
                "The reviewer requested reverting the submodule change and challenged the "
                "test poke; both are frozen machine blockers."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "SUBMODULE_POINTER_SHOULD_NOT_CHANGE",
                    "feedback_ids": ["review-comment:1929654943"],
                    "reviewer_concern": (
                        "The base already points at the intended latest FlashInfer commit; "
                        "the proposed pointer change must be reverted."
                    ),
                    "correspondence": "exact",
                    "machine_rule_ids": ["submodule-pointer-does-not-move-backward"],
                    "explanation": (
                        "The machine independently proved that the proposed pointer is four "
                        "commits behind and quantified its 44-file hidden surface."
                    ),
                },
                {
                    "concept_id": "TEST_POKE_REQUIRES_PURPOSE",
                    "feedback_ids": ["review-comment:1929654999"],
                    "reviewer_concern": (
                        "The error-string-only test change needs a legitimate purpose."
                    ),
                    "correspondence": "exact",
                    "machine_rule_ids": ["test-poke-adds-observable-coverage"],
                    "explanation": (
                        "The machine found that the edit adds no case or assertion and makes "
                        "the error text less precise."
                    ),
                },
            ],
            "machine_only_concepts": [],
        },
        {
            "case_id": "vllm-pr-11531",
            "judgment_lock_sha256": lock_by_case["vllm-pr-11531"]["lock_sha256"],
            "machine_decision": "revise",
            "verdict_correspondence": "exact",
            "verdict_explanation": (
                "The frozen machine and final maintainer outcome both reject the current PR "
                "shape, although their reasons are complementary."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "PINNED_MEMORY_AND_UVA_CAPABILITY_GATE",
                    "feedback_ids": [
                        "review-comment:1907497549",
                        "review-comment:1907500113",
                    ],
                    "reviewer_concern": (
                        "Systems such as WSL lack pinned-memory/UVA support, so the optimized "
                        "path must be capability-gated."
                    ),
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "The machine exercised pinned CPU mappings on A100 but did not inspect "
                        "the unsupported-platform fallback contract."
                    ),
                },
                {
                    "concept_id": "PERFORMANCE_CLAIM_IS_REPRODUCIBLE",
                    "feedback_ids": [
                        "issue-comment:2609089960",
                        "issue-comment:2658719439",
                    ],
                    "reviewer_concern": (
                        "The reported 1250-page result and page/block terminology need a "
                        "reproducible benchmark method."
                    ),
                    "correspondence": "partial",
                    "machine_rule_ids": [
                        "performance-direction-is-measured-not-assumed",
                        "compilation-is-outside-steady-state",
                    ],
                    "explanation": (
                        "The machine independently swept 1/8/64/256 mappings with precompile "
                        "separation, but did not reproduce the author's exact 1250-page claim."
                    ),
                },
                {
                    "concept_id": "ENGINE_GENERATION_OWNS_OPTIMIZATION",
                    "feedback_ids": ["issue-comment:2773681751"],
                    "reviewer_concern": (
                        "The patch only optimizes the retired V0 engine while corresponding "
                        "V1 work is already in progress."
                    ),
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "The machine tested the changed code but did not map it to V0/V1 "
                        "architectural ownership."
                    ),
                },
            ],
            "machine_only_concepts": [
                "PAGE_COPY_DROPS_NON_INT64_TAIL_BYTES",
                "ASYNC_PINNED_MAPPING_BUFFER_LIFETIME_RACE",
                "ZERO_BLOCK_KERNEL_LAUNCH_UNGUARDED",
            ],
        },
        {
            "case_id": "vllm-pr-12111",
            "judgment_lock_sha256": lock_by_case["vllm-pr-12111"]["lock_sha256"],
            "machine_decision": "revise",
            "verdict_correspondence": "exact",
            "verdict_explanation": (
                "The reviewer prefers the established competing fix; the machine independently "
                "found blockers in the current patch."
            ),
            "reviewer_concepts": [
                {
                    "concept_id": "COMPETING_PR_IS_CANONICAL_FIX",
                    "feedback_ids": ["review-comment:1919871986"],
                    "reviewer_concern": (
                        "CPU CI is already addressed by PR 12150, which is the preferred fix."
                    ),
                    "correspondence": "missed",
                    "machine_rule_ids": [],
                    "explanation": (
                        "The machine did not search open/nearby PR ownership before testing."
                    ),
                },
                {
                    "concept_id": "DISTRIBUTED_TEST_LIFECYCLE_CLEANUP",
                    "feedback_ids": ["issue-comment:2598381043"],
                    "reviewer_concern": (
                        "Sequential LLM initialization requires explicit distributed environment "
                        "and memory cleanup."
                    ),
                    "correspondence": "partial",
                    "machine_rule_ids": ["test-process-environment-is-restored"],
                    "explanation": (
                        "The machine caught one leaked process-global environment mutation, but "
                        "did not name the broader distributed cleanup helper."
                    ),
                },
            ],
            "machine_only_concepts": ["CPU_CUSTOM_OP_ATTRIBUTE_UNINITIALIZED"],
        },
    ]

    for case in cases:
        case_id = case["case_id"]
        if case["judgment_lock_sha256"] != review_by_case[case_id]["judgment_lock_sha256"]:
            raise SystemExit(f"lock mismatch in r3 comparison for {case_id}")
        referenced = {
            feedback_id
            for concept in case["reviewer_concepts"]
            for feedback_id in concept["feedback_ids"]
        }
        unknown = referenced - feedback_ids[case_id]
        if unknown:
            raise SystemExit(f"unknown feedback ids for {case_id}: {sorted(unknown)}")

    concepts = [concept for case in cases for concept in case["reviewer_concepts"]]
    concept_counts = {
        status: sum(concept["correspondence"] == status for concept in concepts)
        for status in ("exact", "partial", "missed")
    }
    verdict_counts = {
        status: sum(case["verdict_correspondence"] == status for case in cases)
        for status in ("exact", "partial", "missed")
    }
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-human-concept-comparison-v0.5-r3",
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
            "concept_correspondence": concept_counts,
            "exact_concept_coverage": concept_counts["exact"] / len(concepts),
            "exact_or_partial_concept_coverage": (
                concept_counts["exact"] + concept_counts["partial"]
            )
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
