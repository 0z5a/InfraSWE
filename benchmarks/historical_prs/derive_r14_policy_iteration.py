#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R15 policy amendment from the locked R14 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:137dcb29a541fc66bd13f58b4d7dc535db17e90d9fa992fda9f1b080d321340b"
EXPECTED_LOCK_SHA256 = "sha256:372c37f80c37aa39309dc6e210ce84530d914a328d24eb99caa2bd2844f2df0f"


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
        raise SystemExit("R14 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R14 audit identity changed")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(
        read(args.audit.parent / "machine-judgment-locks.json")
    ):
        raise SystemExit("R14 audit/lock artifact binding mismatch")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R14 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R14 did not meet the user-gated improvement condition")

    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r14-to-r15-v0.1",
        "derived_after_r14_reveal": True,
        "retrospective_r14_locks_changed": False,
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
        },
        "r15_group": {
            "case_count": 30,
            "allocation": {"communication": 20, "training": 10},
            "reason": "finish the remaining communication tranche, then begin the training tranche without violating the 30-case group boundary",
            "future_groups_preselected": False,
        },
        "prospective_rules": [
            {
                "id": "R15-MATURITY-BANDS",
                "rule": "At selection time, use either a <=30-day prospective case or a >=90-day mature case; exclude the 31-89-day resolution gray zone.",
                "evidence": [
                    "All four technically passing mature SGLang cases were still open at only 31-33 days.",
                    "The disposition oracle cannot distinguish project latency from rejection inside that gray zone.",
                ],
            },
            {
                "id": "R15-CHECK-NARROWING",
                "rule": "Predict check only for a <=30-day case with a demonstrated primary direction, <=4 changed files, candidate-owned closure coverage, exactly one executable residual, and no superseding or reachable counterexample.",
                "evidence": [
                    "R14 check precision was 0/2.",
                    "Megatron #6973 was the only frozen non-accept that became an active-review check; its gap was one unavailable four-rank parity closure.",
                    "A bounded technical gap alone does not establish active human review.",
                ],
            },
            {
                "id": "R15-NARROW-HARDWARE-BUGFIX-TRANSFER",
                "rule": "Do not reject a <=3-file mature bug fix solely because evaluator hardware is unavailable when candidate-owned target-hardware coverage exists, the source invariant is locally closed, and no exact counterexample is found.",
                "evidence": [
                    "FlashInfer #4139 and #4296 were merged despite A100-side target tests being skipped.",
                    "Keep broad new hardware features such as #4302 non-accept until evaluator-owned numeric evidence exists.",
                ],
            },
            {
                "id": "R15-REJECT-EVIDENCE-ORDER",
                "rule": "A technical reject requires legal production reachability followed by an exact semantic, progress, safety, or import failure; absence of local E2E evidence is recorded as unresolved, not fabricated as a failure.",
                "evidence": [
                    "verl #7045 produced an exact production import failure and was correctly rejected.",
                    "verl #7161 was merged, so an inferred version-gate regression without an executed counterexample was insufficient.",
                ],
            },
            {
                "id": "R15-DUAL-OUTPUT",
                "rule": "Continue locking technical_contract and disposition_prediction separately; never rewrite technical evidence from post-lock outcome or review data.",
                "evidence": [
                    "R14 exact improvement came only from check/reject disposition separation; binary accuracy remained 14/30.",
                ],
            },
        ],
        "known_limitations": [
            "The blind acquisition fields cannot directly prove active human review, so check remains intrinsically low-identifiability.",
            "Repository-specific merge latency is a governance signal, not a correctness score.",
            "R14 is nonweighted and cannot establish the official merged-PR ProjectFit >=85 floor.",
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
