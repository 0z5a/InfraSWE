#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the prospective R16 policy from the locked R15 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:d4e74cc4b4016e49c32cbe069174d58fc8079b0870549a8cf4082ae411a9821c"
EXPECTED_LOCK_SHA256 = "sha256:4fdc4fb7c37dd9bb056ca9e09bcd2d0334ee9b65adaebb899fc5f708fc0f0c31"


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
        raise SystemExit("R15 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R15 audit identity changed")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R15 audit/lock artifact binding mismatch")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R15 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R15 did not meet the user-gated improvement condition")

    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-iterative-policy-r15-to-r16-v0.1",
        "derived_after_r15_reveal": True,
        "retrospective_r15_locks_changed": False,
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
        "workload_ledger": {
            "requested_after_r13": {"communication": 50, "training": 50, "inference": 100},
            "completed_through_r15": {"communication": 50, "training": 10, "inference": 0},
            "remaining_after_r15": {"communication": 0, "training": 40, "inference": 100},
        },
        "r16_group": {
            "case_count": 30,
            "allocation": {"training": 30},
            "reason": "continue the training tranche with a full 30-case group; do not preselect later inference cases before R16 iteration",
            "future_groups_preselected": False,
        },
        "prospective_rules": [
            {
                "id": "R16-HOT-CHECK-WINDOW",
                "rule": "Use check only inside a <=7-day hot window. Permit up to 8 changed files when exact core-path evidence passes, the change is one coherent risk family, exactly one integration residual remains, and no exact counterexample exists.",
                "evidence": [
                    "Both correctly predicted checks were 0.9-2.5 days old with fresh final-head feedback.",
                    "The 11.9-day slime #2304 check had no human feedback and was an oracle reject.",
                    "Fresh FlashInfer #4795 and Megatron #7029 were active-review checks despite 8 and 5 paths; the four-file cap was too strict.",
                ],
            },
            {
                "id": "R16-TARGET-COVERAGE-MUST-BE-FUNCTIONAL",
                "rule": "For a target-only communication/training backend, compile/import evidence and a mocked orchestration probe do not qualify as target-hardware closure; require target functional numeric, progress, ordering, or memory evidence.",
                "evidence": [
                    "verl #6569 was technically plausible in an isolated asyncio probe but closed unmerged; its body supplied no functional HCCL/NPU run.",
                    "FlashInfer #3304 remains eligible because the representation invariant is bit-exact and its candidate target test exercises sentinel patterns rather than only importability.",
                ],
            },
            {
                "id": "R16-MECHANICAL-ADAPTER-PROPAGATION",
                "rule": "A mature multi-file change may be accept when it only propagates one backward-compatible value through adapters, every production implementation is enumerated, endpoint call tests pass, and no backend-specific semantics change.",
                "evidence": [
                    "verl #6507 was merged: most of its nine-file surface was uniform global_steps signature/call propagation, while both endpoint tests passed.",
                    "Do not apply this exception to algorithm, state-layout, collective, or memory-policy changes.",
                ],
            },
            {
                "id": "R16-DATA-MOTION-ONLY-TARGET-OPTIMIZATION",
                "rule": "A mature target optimization may transfer without evaluator hardware only when it removes a provably redundant copy/transpose, preserves tensor representation and control flow, has candidate target parity plus benchmark evidence, and introduces no new kernel or feature gate.",
                "evidence": [
                    "ROCm SGLang #27289 was merged after a cross-file removal of the same redundant FP8 scale transpose-copy.",
                    "This exception does not cover broad new Full-DTensor, dispatcher, or fused-kernel features.",
                ],
            },
            {
                "id": "R16-COUNTEREXAMPLE-OVERRIDES-DISPOSITION",
                "rule": "Keep technical reject when an exact candidate-owned test or measured resource claim fails, even if post-lock history shows merged; record the mismatch as oracle noise/governance debt rather than learning that failures are acceptable.",
                "evidence": [
                    "TorchTitan #3522 merged despite its exact disabled-replay candidate control raising ValueError.",
                    "verl #6593 merged despite backward peak memory exceeding baseline and scaling with full token count.",
                ],
            },
            {
                "id": "R16-CROSS-DOMAIN-TRAINING-CLOSURE",
                "rule": "For R16 training PRs, separate configuration plumbing tests from an optimizer-step, gradient, checkpoint, or measured-memory closure; broad training changes with only config mocks remain non-accept.",
                "evidence": [
                    "verl #6566 passed nine config tests but lacked an optimizer-step/state oracle and closed unmerged.",
                    "slime #2011 had strong correctness and measured memory evidence; its closed-unmerged outcome is a governance mismatch, not a reason to weaken training closure.",
                ],
            },
            {
                "id": "R16-DUAL-OUTPUT",
                "rule": "Continue locking technical_contract and disposition_prediction separately and keep the label check; never rewrite a frozen technical result from merge history.",
                "evidence": [
                    "R15 reached 19/30 exact versus 11/30 for the same-cohort legacy route, while exact failures still exposed five merged technical mismatches.",
                ],
            },
        ],
        "known_limitations": [
            "The hot-window rule is learned post-reveal and must be evaluated only prospectively from R16 onward.",
            "Superseded or duplicate correct patches such as vLLM #44495 are not identifiable from outcome-free candidate evidence alone.",
            "R15 is nonweighted and cannot establish the official merged-PR ProjectFit >=85 floor.",
            "Merged-as-accept is a disposition oracle and can conflict with exact technical failures; both outputs must remain visible.",
        ],
    }
    payload = {**material, "iteration_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
