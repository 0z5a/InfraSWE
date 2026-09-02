#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R17 policy from the locked R16 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:94dbccdbb136ee5f9fa5b4705590833e30360d1cf9b2adb928384ddd29459700"
EXPECTED_LOCK_SHA256 = "sha256:4b6884a5ed00919eec80db5960a8bb84750f6ab05c2476777f31fd9f6babdb44"


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
    audit_material = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if audit["audit_sha256"] != canonical_sha256(audit_material):
        raise SystemExit("R16 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R16 audit identity changed")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R16 audit/lock artifact binding mismatch")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R16 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R16 did not meet the user-gated improvement condition")

    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r16-to-r17-v0.1",
        "derived_after_r16_reveal": True,
        "retrospective_r16_locks_changed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "observed_metrics": {
            "cases": summary["cases"],
            "exact_label_matches": summary["exact_label_matches"],
            "legacy_exact_label_matches": summary["legacy_exact_label_matches"],
            "same_cohort_exact_accuracy_gain": summary["same_cohort_exact_accuracy_gain"],
            "binary_direction_matches": summary["binary_direction_matches"],
            "machine_reject_precision": summary["machine_reject_precision"],
            "machine_check_precision": summary["machine_check_precision"],
            "oracle_decisions": summary["oracle_decisions"],
            "merged_machine_accepts": summary["merged_machine_accepts"],
            "merged_cases": summary["merged_cases"],
        },
        "workload_ledger": {
            "requested_after_r13": {"communication": 50, "training": 50, "inference": 100},
            "completed_through_r16": {"communication": 50, "training": 40, "inference": 0},
            "remaining_after_r16": {"communication": 0, "training": 10, "inference": 100},
        },
        "r17_group": {
            "case_count": 30,
            "allocation": {"training": 10, "inference": 20},
            "inference_project_allocation": {"vllm": 5, "sglang": 5, "tensorrt_llm": 5, "flashinfer": 5},
            "reason": "finish the final ten training cases and begin the four-draft inference tranche without preselecting R18",
            "future_groups_preselected": False,
        },
        "prospective_rules": [
            {
                "id": "R17-CHECK-REQUIRES-READINESS-SIGNAL",
                "rule": "Keep the <=7-day and <=8-file check gates, but also require a candidate-owned exact core test, no explicit unreviewed/draft/CI-not-requested disclaimer, one executable integration residual, and evidence insufficient for direct accept. A target-functional body plus locally closed invariant should be accept rather than check.",
                "evidence": [
                    "R16 produced zero oracle checks; both machine checks missed.",
                    "Liger #1413 was merged and supplied detailed B300 functional results, so check understated a strong narrow repair.",
                    "slime #2345 had only py_compile/self-validation and no human review, so its fresh age alone did not justify check.",
                ],
            },
            {
                "id": "R17-EXPLICIT-NOT-READY-VETO",
                "rule": "When the outcome-free body explicitly says required human review or public CI has not occurred, predict reject for disposition unless an independent evaluator-owned end-to-end closure is complete; retain a separate technical result.",
                "evidence": [
                    "verl #7697 had strong functional unit tests but unchecked human-review and CI declarations; it remained open without review and oracle-rejected.",
                    "This signal is candidate-authored readiness evidence, not post-lock review/state leakage.",
                ],
            },
            {
                "id": "R17-MATURE-NO-TEST-DEFAULT",
                "rule": "A mature feature or algorithm change with no candidate-owned test defaults to disposition reject. Exceptions require an exact evaluator base/head oracle or one of the bounded structural migration/artifact rules below.",
                "evidence": [
                    "Liger #1244 passed an import probe but closed unmerged, and several mature open PRs with technically plausible code had no review activity.",
                    "Do not rewrite exact technical passes such as slime #2010 solely because history remained open.",
                ],
            },
            {
                "id": "R17-STRUCTURAL-MIGRATION-EXCEPTION",
                "rule": "A mature deprecation removal or configuration-container refactor may accept when a repository-wide old-symbol inventory reaches zero, every production call site has an explicit replacement, import/compile succeeds, and affected existing tests are updated; distributed topology is not required when runtime algorithms and state layout are unchanged.",
                "evidence": [
                    "Megatron #5134 and #5169 merged despite unavailable eight-rank or absent new-test closure.",
                    "Both were structural API/configuration migrations rather than new optimizer, collective, or numerical algorithms.",
                ],
            },
            {
                "id": "R17-ARTIFACT-BOUNDARY-EXCEPTION",
                "rule": "For checkpoint/export writer changes, exact tensor uniqueness, shard/index completeness, atomic finalization, and reload parity can close the title-scoped artifact contract without an optimizer step. A concurrency speedup may be accepted without a timing threshold only when work partitioning is structural and artifact parity is exact.",
                "evidence": [
                    "slime #1969 and #2020 merged; their candidate tests closed asset filtering, shard/index bytes, merged node-writer state, and incomplete final groups.",
                    "The exception does not cover parameter mapping omissions or a quantitative performance claim without parity.",
                ],
            },
            {
                "id": "R17-STACKED-SERIES-DIVERSITY",
                "rule": "At metadata-only selection time, cap highly overlapping same-subsystem candidates: retain at most two PRs per project when normalized title prefixes match and non-test path Jaccard overlap is >=0.5. Prefer distinct runtime paths and risk families.",
                "evidence": [
                    "R16 selected four tightly related TorchTitan graph-trainer patches; #3530/#3533 merged while #3534/#3538 closed unmerged despite passing focused tests.",
                    "Reducing correlated stacks improves coverage and avoids treating superseded series members as independent evidence.",
                ],
            },
            {
                "id": "R17-RESOURCE-CLAIM-NEEDS-RUNTIME",
                "rule": "Resource arithmetic or configuration tests do not close scheduler progress, memory, or concurrency claims; require an actual scheduler, allocator, or multi-worker execution unless the transformation is algebraically exact data motion.",
                "evidence": [
                    "verl #6574 passed eight sizing tests but closed unmerged without a Ray placement/progress run.",
                    "slime #2010 remains a technical pass because an exact base/head A100 allocation probe measured the claimed reduction with value/gradient parity.",
                ],
            },
            {
                "id": "R17-EXACT-HEAD-INTEGRITY-FIRST",
                "rule": "Syntax, conflict-marker, import, or candidate-owned exact failures reject immediately and override otherwise plausible scope or body claims.",
                "evidence": [
                    "verl #6558 contained an unclosed production parenthesis reproduced by static parsing and py_compile; its oracle was reject.",
                ],
            },
            {
                "id": "R17-DUAL-OUTPUT",
                "rule": "Continue freezing technical_contract and disposition_prediction separately. Closed/open governance mismatches never convert a passing numeric, gradient, or memory contract into a technical failure.",
                "evidence": [
                    "R16 gained 10.0 exact-label points over the same-cohort check-only fallback, but many technically strong training changes were closed or remained unreviewed.",
                ],
            },
        ],
        "known_limitations": [
            "A blind source evaluator cannot directly observe the human-review activity required by the disposition oracle; readiness signals are only proxies.",
            "Closed superseded stack members can be technically correct and remain indistinguishable without prohibited outcome/review evidence.",
            "R16 is nonweighted and cannot establish the formal merged-PR ProjectFit >=85 floor.",
            "Four merged R16 PRs were frozen reject because topology or structural exceptions were not yet modeled; frozen technical results remain unchanged.",
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
