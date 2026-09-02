#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R19 inference policy from the locked R18 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:8433515dca3a7e10b894c89270c63f78bb5abb2bd36645920308c4a1f1f8869f"
EXPECTED_LOCK_SHA256 = "sha256:538ca41587b7ed4f5d24eec4e00a84fb3eb8886fe74d07e0327033af950b94ee"


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = read(args.audit)
    material = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if audit["audit_sha256"] != canonical_sha256(material):
        raise SystemExit("R18 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R18 audit identity changed")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R18 audit/lock artifact binding mismatch")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R18 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R18 did not meet the user-gated improvement condition")

    iteration = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r18-to-r19-v0.1",
        "derived_after_r18_reveal": True,
        "retrospective_r18_locks_changed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "observed_metrics": {
            key: summary[key]
            for key in (
                "cases",
                "exact_label_matches",
                "legacy_exact_label_matches",
                "same_cohort_exact_accuracy_gain",
                "binary_direction_matches",
                "frozen_nonaccept_exact_matches",
                "legacy_frozen_nonaccept_exact_matches",
                "machine_reject_precision",
                "machine_check_precision",
                "oracle_decisions",
            )
        },
        "workload_ledger": {
            "requested_after_r13": {"communication": 50, "training": 50, "inference": 100},
            "completed_through_r18": {"communication": 50, "training": 50, "inference": 50},
            "remaining_after_r18": {"communication": 0, "training": 0, "inference": 50},
        },
        "r19_group": {
            "case_count": 30,
            "allocation": {"inference": 30},
            "inference_project_allocation": {
                "vllm": 7,
                "sglang": 7,
                "tensorrt_llm": 8,
                "flashinfer": 8,
            },
            "reason": "rotate the eight-case shares across the four inference drafts; reserve five each for R20",
            "future_groups_preselected": False,
        },
        "prospective_rules": [
            {
                "id": "R19-CHECK-PRECISION-FIRST",
                "rule": "A recent PR is check only when outcome-free text names an external review or QA handoff, the response is visible in the candidate body or commits, and exactly one executable residual remains. Technical success, author-owned receipts, or adjacency to other proposals alone are not review-activity proxies.",
                "evidence": [
                    "All three R18 check predictions lacked post-lock human feedback and were oracle-reject.",
                    "TensorRT-LLM #18538 was the sole oracle-check, but its active review was unobservable before reveal; blind recall is sacrificed for check precision.",
                ],
            },
            {
                "id": "R19-RECENT-NO-HANDOFF-REJECT",
                "rule": "For the <=7-day band, no explicit external-review/QA handoff means reject unless target-functional closure is unusually complete enough for direct accept. Candidate unit passes alone do not imply check.",
                "evidence": [
                    "FlashInfer #4850, SGLang #37643, and vLLM #54979 were recent and technically plausible but had no qualifying human feedback and were oracle-reject.",
                ],
            },
            {
                "id": "R19-TARGET-SKIP-NEEDS-RECEIPT",
                "rule": "A target-architecture skip remains a bounded technical gap, but disposition accept additionally requires exact-head target-functional receipts or a complete algebraic/source invariant. A clean skip by itself cannot close the historical disposition prediction.",
                "evidence": [
                    "Merged FlashInfer #3355/#3357/#3393 supplied target matrices, while target-skipped #3457 remained stale-open.",
                ],
            },
            {
                "id": "R19-SMALL-UNIT-SUITE-NOT-DISPOSITION-PROOF",
                "rule": "For mature inference PRs, a small unit suite without base-distinguishing reproduction or production integration is technical evidence, not sufficient disposition evidence. Require an exact failure reproduction, end-to-end receipt, or exhaustive narrow invariant for accept.",
                "evidence": [
                    "FlashInfer #3503, TensorRT-LLM #14765, and vLLM #44431/#44584 passed small suites but were closed-unmerged or inactive-open.",
                    "vLLM #44450 paired two focused tests with a real Qwen-VL end-to-end reproduction and merged.",
                ],
            },
            {
                "id": "R19-FINAL-HEAD-EVIDENCE-OVERRIDES-STALE-CHECKLIST",
                "rule": "Unchecked or stale body checklist prose is not a reject veto when the exact frozen head contains candidate tests that close the title-scoped route. Reject only when the missing item remains technically observable in source or execution.",
                "evidence": [
                    "SGLang #27180 and vLLM #44456 merged despite stale not-ready prose, so R18 over-read narrative disposition cues.",
                ],
            },
            {
                "id": "R19-BROAD-NO-TEST-REJECT",
                "rule": "Broad backend, ownership, scheduler, cache, or quantization work without a candidate-owned production-route test remains reject even when benchmark tables or source plausibility are strong.",
                "evidence": [
                    "R18 correctly rejected SGLang #27159 and vLLM #44564.",
                ],
            },
            {
                "id": "R19-TECHNICAL-DISPOSITION-DUAL-OUTPUT",
                "rule": "Continue freezing technical_contract separately from accept/check/reject. Never rewrite a passing numeric or state contract merely to imitate historical closure outcomes.",
                "evidence": [
                    "R18 exact labels gained 6.7 points over the same-cohort legacy baseline while binary accuracy remained unchanged.",
                ],
            },
        ],
        "known_limitations": [
            "The check oracle directly depends on post-lock human review activity, so high blind check recall is impossible from allowed evidence.",
            "Technically strong mature PRs can be merged, superseded, abandoned, or closed for governance reasons that are absent from source and tests.",
            "R18 used no weighted score, so the historical >=85 merged-PR floor is not formally auditable.",
            "R19 remains inference-only and rotates the larger project shares to TensorRT-LLM and FlashInfer.",
        ],
    }
    payload = {**iteration, "iteration_sha256": canonical_sha256(iteration)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
