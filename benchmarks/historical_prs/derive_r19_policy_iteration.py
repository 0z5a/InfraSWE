#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R20 inference policy from the locked R19 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:2c4aedf99053afd4a205b123ac0cec3839c8214b4ce9016136189e4758e543d9"
EXPECTED_LOCK_SHA256 = "sha256:da561d647a3cd1a9dd93d078412b5f79a6f7128924d905d8505fbab91a979958"


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
        raise SystemExit("R19 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R19 audit identity changed")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R19 audit/lock artifact binding mismatch")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R19 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R19 did not meet the user-gated improvement condition")

    iteration = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r19-to-r20-v0.1",
        "derived_after_r19_reveal": True,
        "retrospective_r19_locks_changed": False,
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
            "completed_through_r19": {"communication": 50, "training": 50, "inference": 80},
            "remaining_after_r19": {"communication": 0, "training": 0, "inference": 20},
        },
        "r20_group": {
            "case_count": 20,
            "allocation": {"inference": 20},
            "inference_project_allocation": {
                "vllm": 5,
                "sglang": 5,
                "tensorrt_llm": 5,
                "flashinfer": 5,
            },
            "reason": "complete the requested 100 inference PRs with 25 cases from each main draft",
            "future_groups_preselected": False,
        },
        "prospective_rules": [
            {
                "id": "R20-RECENT-NO-DIRECT-ACCEPT",
                "rule": "A <=7-day PR without an explicit external-review or QA handoff proxy is reject even when candidate tests and author-reported production validation are exceptionally strong. Direct technical closure is recorded separately and does not predict historical acceptance during the hot window.",
                "evidence": [
                    "All four R19 recent cases were oracle-reject, including SGLang #37638 with 270/270 local tests and a 44/44 production receipt.",
                ],
            },
            {
                "id": "R20-CHECK-EXTERNAL-HANDOFF-ONLY",
                "rule": "Predict check only for a recent PR whose outcome-free body or commits explicitly identify reviewer-requested or collaborator-owned follow-up, show the response, and leave exactly one bounded executable residual. Otherwise prefer reject over speculative check.",
                "evidence": [
                    "R19 made no check predictions and its oracle contained no check cases; R18 showed that technical success alone yielded zero check precision.",
                ],
            },
            {
                "id": "R20-TEST-LANGUAGE-NEUTRAL",
                "rule": "Count candidate-owned C++, integration-list, and existing-matrix changes as first-class test evidence. Python-only discovery is an execution convenience, not a readiness criterion.",
                "evidence": [
                    "TensorRT-LLM #14806/#14961/#14979 merged despite local generated-binding gaps or C++-only coverage.",
                ],
            },
            {
                "id": "R20-MATURE-FINAL-HEAD-OVER-BODY",
                "rule": "For mature PRs, final-head source and candidate tests outrank stale checklist, draft, or pending-CI prose. Accept when the title-scoped production route and its negative/control path are represented; reject only for an exact counterexample or a concrete uncovered invariant.",
                "evidence": [
                    "R19 false rejects TensorRT-LLM #14806/#14961/#14979 and vLLM #44514/#44577 later merged despite sparse or not-ready body text.",
                ],
            },
            {
                "id": "R20-SELF-REPORTED-RECEIPT-NOT-ENOUGH",
                "rule": "A body-only benchmark or base/head reproduction without candidate-owned test or independently executable narrow invariant cannot by itself justify mature accept.",
                "evidence": [
                    "FlashInfer #3434 and vLLM #44566 had detailed self-reported receipts but remained inactive-open.",
                ],
            },
            {
                "id": "R20-PREPARATORY-SCOPE-REQUIRES-OWN-CLOSURE",
                "rule": "A preparatory or stacked PR may accept only when its own title-scoped helper/API contract is exhaustively tested and useful independently; do not borrow closure from a follow-up integration.",
                "evidence": [
                    "TensorRT-LLM #14764 passed 39 helper tests but remained open as preparation for a follow-up, exposing the disposition limit of isolated helper correctness.",
                ],
            },
            {
                "id": "R20-EXACT-FAILURE-FIRST",
                "rule": "A reproducible candidate-owned assertion, signature, syntax, conflict, safety, or import failure at the changed boundary rejects before any receipt or project prior. Environment-only dependency failures stay capability gaps.",
                "evidence": [
                    "FlashInfer #4900 reproduced its 19-versus-20 plan signature failure in a unique JIT workspace and was correctly rejected.",
                ],
            },
            {
                "id": "R20-DUAL-OUTPUT",
                "rule": "Continue freezing technical_contract separately from disposition. The benchmark must expose technically passing but historically rejected PRs rather than falsifying their numeric or state results.",
                "evidence": [
                    "R19 exact labels improved by 20 points entirely through reject labeling while binary accept/non-accept accuracy stayed 15/30.",
                ],
            },
        ],
        "known_limitations": [
            "The check oracle consumes post-lock human review activity that allowed blind evidence cannot directly reveal.",
            "R19 showed that even exhaustive tests and production receipts cannot distinguish merged from abandoned recent work.",
            "Mature governance outcomes remain weakly identifiable: several sparse-body PRs merged while technically stronger peers stayed open or closed-unmerged.",
            "R20 is the final 20-case inference group, so its iteration will be summarized as a terminal recommendation rather than used to preselect another requested group.",
        ],
    }
    payload = {**iteration, "iteration_sha256": canonical_sha256(iteration)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
