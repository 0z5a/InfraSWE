#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R18 inference policy from the locked R17 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:fa61324983257f4679ba5c01b9b5e7347d9ed7720d287ee360ada82e80304750"
EXPECTED_LOCK_SHA256 = "sha256:c466a9e6cb56ad813ea53193a62a7cb37d96967acdac0bc00efb1beed8999d0d"


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
        raise SystemExit("R17 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R17 audit identity changed")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R17 audit/lock artifact binding mismatch")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R17 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R17 did not meet the user-gated improvement condition")

    iteration = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r17-to-r18-v0.1",
        "derived_after_r17_reveal": True,
        "retrospective_r17_locks_changed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "observed_metrics": {
            "cases": summary["cases"],
            "exact_label_matches": summary["exact_label_matches"],
            "legacy_exact_label_matches": summary["legacy_exact_label_matches"],
            "same_cohort_exact_accuracy_gain": summary["same_cohort_exact_accuracy_gain"],
            "binary_direction_matches": summary["binary_direction_matches"],
            "frozen_nonaccept_exact_matches": summary["frozen_nonaccept_exact_matches"],
            "legacy_frozen_nonaccept_exact_matches": summary["legacy_frozen_nonaccept_exact_matches"],
            "machine_reject_precision": summary["machine_reject_precision"],
            "machine_check_precision": summary["machine_check_precision"],
            "oracle_decisions": summary["oracle_decisions"],
        },
        "workload_ledger": {
            "requested_after_r13": {"communication": 50, "training": 50, "inference": 100},
            "completed_through_r17": {"communication": 50, "training": 50, "inference": 20},
            "remaining_after_r17": {"communication": 0, "training": 0, "inference": 80},
        },
        "r18_group": {
            "case_count": 30,
            "allocation": {"inference": 30},
            "inference_project_allocation": {"vllm": 8, "sglang": 8, "tensorrt_llm": 7, "flashinfer": 7},
            "reason": "continue the four-draft inference tranche; rotate the 8/8 shares in R19 and retain five each for R20",
            "future_groups_preselected": False,
        },
        "prospective_rules": [
            {
                "id": "R18-RECENT-BOUNDED-FAILURE-CHECK",
                "rule": "For a <=7-day, <=8-file PR with a candidate-owned exact core test and no integrity or safety failure, a title-scoped failing assertion is check when the repair direction and one residual are bounded; do not force reject merely because the author is still iterating.",
                "evidence": [
                    "FlashInfer #4861 had one exact per-row seed assertion failure but was actively reviewed and oracle-check.",
                    "verl #7685 was recent but broad ten-file distributed ownership work without review and remained reject.",
                ],
            },
            {
                "id": "R18-RECENT-READINESS-PROXY",
                "rule": "Predict check only when recent metadata gates combine with an outcome-free readiness proxy: addressed-review notes, a bounded QA follow-up, exact base/head reproduction, or independently executed core tests. Explicit cannot-run, happy-to-close, draft, CI-not-requested, or incomplete-checklist language vetoes check unless evaluator end-to-end closure exists.",
                "evidence": [
                    "TensorRT-LLM #18596 was correctly check from a bounded QA follow-up; TorchTitan #4398 also named addressed reviewer feedback and was oracle-check.",
                    "vLLM #54990 said its suite was not run and offered closure; despite local focused passes it had no active human review and oracle-rejected.",
                ],
            },
            {
                "id": "R18-RECENT-STRONG-PROOF-NOT-AUTO-ACCEPT",
                "rule": "A recent technically closed PR defaults to check, not accept, when the body or diff shows ongoing review/QA activity. Direct accept is reserved for unusually complete target-functional evidence without a named residual; technical pass remains separately recorded.",
                "evidence": [
                    "Megatron #7013 and TorchTitan #4398 passed exact tests but were active-review oracle checks.",
                    "Liger #1435 merged quickly with a complete target failure matrix and no residual, so the narrow direct-accept exception remains.",
                ],
            },
            {
                "id": "R18-TARGET-GAP-NOT-A-REJECT-VETO",
                "rule": "For mature inference work, unavailable accelerator architecture, generated bindings, or topology is a bounded technical gap rather than a disposition reject when candidate-owned tests exist, the changed production route is explicit, static integrity holds, and no exact reachable counterexample appears.",
                "evidence": [
                    "FlashInfer #3497, SGLang #27291, TensorRT-LLM #14922, and vLLM #44513/#44572 merged despite evaluator target gaps or incomplete local execution.",
                    "Exact reachable failures in SGLang #27203 and vLLM #44544 still correctly rejected.",
                ],
            },
            {
                "id": "R18-MATURE-TESTED-ROUTE-EXCEPTION",
                "rule": "A mature inference change may accept with bounded-gap when it changes a candidate test path and every reachable title-scoped check passes or skips only for the declared backend; absence of an added test function alone is not fatal if an existing exact matrix is extended. No-test production changes still default reject.",
                "evidence": [
                    "Several merged mature inference PRs extended existing parameter matrices rather than adding named tests.",
                    "slime #1959 remains evidence that an evaluator-only smoke probe cannot replace a candidate-owned production-route test.",
                ],
            },
            {
                "id": "R18-STACK-MARKER-IS-WEAK",
                "rule": "Use stack-overlap only for cohort diversity. Do not reject a selected case merely because its body says 1/N or supersedes another PR; judge whether that frozen member independently closes its contract.",
                "evidence": [
                    "slime #1930 was the first member of a series, passed all seven changed tests, and merged.",
                    "Stack membership remains unable to distinguish merged from closed-unmerged without forbidden outcome fields.",
                ],
            },
            {
                "id": "R18-EXACT-INTEGRITY-FIRST",
                "rule": "Syntax, conflict-marker, import-at-the-changed-boundary, safety, or mature candidate-owned title-scoped failures reject. Environment-only import failures and unsupported-device skips never masquerade as candidate failures.",
                "evidence": [
                    "R17 correctly rejected the exact SGLang TP-initialization and vLLM backend-availability failures.",
                ],
            },
            {
                "id": "R18-DUAL-OUTPUT",
                "rule": "Continue freezing technical_contract and disposition_prediction separately; historical governance cannot rewrite a valid numeric, state, or performance result.",
                "evidence": [
                    "R17 exact labels improved by 13.3 points while binary direction stayed 14/30, showing the gain came from check/reject separation rather than retrospective technical rewriting.",
                ],
            },
        ],
        "known_limitations": [
            "The blind evaluator cannot observe human review activity, which is a direct input to the check oracle.",
            "Mature merged and closed-unmerged PRs can have indistinguishable source/test evidence; project and scope priors are not outcomes.",
            "R17 did not use weighted scores, so the historical >=85 merged-PR floor is not formally auditable.",
            "R18 is inference-only and tests whether target-gap transfer rules generalize across the four main drafts.",
        ],
    }
    payload = {**iteration, "iteration_sha256": canonical_sha256(iteration)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
