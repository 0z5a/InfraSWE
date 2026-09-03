#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R22 policy from the frozen R21 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = checked(args.audit, "audit_sha256")
    locks = checked(args.audit.parent / "machine-judgment-locks.json", "lock_set_sha256")
    selection = checked(
        args.audit.parent / "selection-lock.json",
        "selection_lock_sha256",
        material_field="selection_material",
    )
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R21 audit/lock artifact binding mismatch")
    if audit["source_digests"]["selection_lock"] != canonical_sha256(selection):
        raise SystemExit("R21 audit/selection artifact binding mismatch")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R21 did not meet the user-gated improvement condition")
    if summary["merged_machine_accepts"] != summary["merged_cases"]:
        raise SystemExit("R21 lost merged-PR accept recall")

    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r21-to-r22-v0.1",
        "derived_after_r21_reveal": True,
        "retrospective_r21_locks_changed": False,
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
        "retrospective_leave_one_out_check": {
            "purpose": "validate only the general rules proposed for R22; R21 locks remain unchanged",
            "exact_label_matches": 72,
            "cases": 100,
            "merged_machine_accepts": 63,
            "merged_cases": 63,
            "new_correct_rejects": [
                "flashinfer-pr-3015",
                "sglang-pr-26856",
                "sglang-pr-26857",
                "sglang-pr-26858",
                "tensorrt_llm-pr-14605",
            ],
            "new_incorrect_rejects": [],
            "not_a_rescore": True,
        },
        "workload_ledger": {
            "requested": {"inference": 500, "groups": 5, "group_size": 100},
            "completed_through_r21": {"inference": 100},
            "remaining_after_r21": {"inference": 400},
        },
        "next_group": {
            "round": "R22",
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
                "id": "R22-TIME-BUDGET-GATE",
                "rule": "Abandon an individual normal PR test after 60 seconds and a TensorRT-LLM PR test after 20 seconds. Record abandoned-time-budget as a neutral capability gap; timeout alone is never a candidate failure and does not trigger a retry loop.",
                "evidence": [
                    "The R21 execution gate bounded tail latency without converting timeouts into exact failures.",
                ],
            },
            {
                "id": "R22-CHECK-REQUIRES-ACTIVE-REVIEW",
                "rule": "Use check only for a <=7-day PR with outcome-free evidence of substantive named non-author activity after the frozen head. Review text, state, merge outcome, labels, and CI status remain hidden; absent qualifying activity, a recent PR predicts reject.",
                "evidence": [
                    "R21 contained one check oracle, TensorRT-LLM #18614, distinguished by fresh final-head collaborator activity; the other recent cases were reject.",
                ],
            },
            {
                "id": "R22-BRACKETED-DRAFT-VETO",
                "rule": "A title beginning with an explicit [Draft] marker predicts reject unless an exact evaluator-owned closure supersedes the declared readiness state. Incidental uses of draft for speculative decoding do not match this rule.",
                "evidence": [
                    "The two R21 bracketed-draft cases, FlashInfer #3015 and TensorRT-LLM #14605, were both oracle-reject; the narrow prefix rule avoids conflating speculative draft tokens.",
                ],
            },
            {
                "id": "R22-NEGATIVE-PRECEDENT-CONSENSUS",
                "rule": "For a mature otherwise-acceptable PR, reject on precedent only when at least two prior revealed cases from the same project and risk family pass a frozen close-neighbor gate and every such neighbor is reject. Any accepted close neighbor vetoes this negative inference.",
                "evidence": [
                    "Leave-one-out application to R21 identified the three abandoned Qwen3.5Opt stack members and introduced no merged false reject.",
                ],
            },
            {
                "id": "R22-PRECEDENT-NEIGHBOR-GATE",
                "rule": "A close neighbor requires title-token Jaccard >=0.35, non-test source-path Jaccard >=0.40, or source-directory-prefix Jaccard >=0.75. Retrieval ranking is evidence selection, not a weighted disposition score.",
                "evidence": [
                    "The frozen thresholds improved R21 leave-one-out exact labels from 67 to 70 before the independent bracketed-draft rule, with 63/63 merged recall preserved.",
                ],
            },
            {
                "id": "R22-MERGED-RECALL-GUARD",
                "rule": "Do not apply broad project, platform, test-count, or runtime-pass rejection heuristics. Exact candidate failure, explicit readiness veto, source-integrity failure, or unanimous negative precedent is required to overturn mature technical closure.",
                "evidence": [
                    "R21 preserved all 63 merged cases while project and runtime-pass rates were too mixed for a safe universal threshold.",
                ],
            },
            {
                "id": "R22-TECHNICAL-DISPOSITION-SPLIT",
                "rule": "Keep technical_contract separate from disposition. Abandoned, dependency-blocked, and unavailable-capability runs remain bounded-gap unless an exact candidate-owned assertion fails.",
                "evidence": [
                    "R21 runtime success was not itself disposition-predictive, while exact candidate failure remained a high-precision reject signal.",
                ],
            },
        ],
        "known_limitations": [
            "Historical merge disposition is only partly identifiable from code and test evidence.",
            "The activity projection observes event metadata and derived booleans but intentionally withholds review text and all terminal outcome fields.",
            "Precedent consensus is conservative and may abstain frequently when prior outcomes conflict.",
            "R21 and later rounds remain explicitly nonweighted, so the legacy numeric merged-score floor is not directly auditable.",
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
