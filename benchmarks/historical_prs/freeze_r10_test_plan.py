#!/usr/bin/env python3
"""Freeze case-specific R10 contracts before any source diff inspection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

CASE_PLANS: dict[str, dict[str, Any]] = {
    "cutlass-pr-3300": {
        "claim": "cute/util/print_tensor.hpp is a self-contained public include.",
        "execution_tier": "exact base/head minimal C++ compilation and include-order audit",
        "questions": [
            "Does a minimal translation unit including only print_tensor.hpp compile?",
            "Are declarations used by the header provided by direct, not accidental, includes?",
            "Is behavior independent of representative preceding include order?",
            "Does the self-contained-include test manifest cover this public header?",
        ],
        "decision_rule": (
            "Accept with scope only if exact head compilation fixes a base failure, direct include "
            "ownership is explicit, and the repository's self-contained-header test is wired. "
            "CUDA-only compilation remains unresolved when the required toolchain is absent."
        ),
    },
    "deepgemm-pr-310": {
        "claim": "pack_ue8m0_to_int enforces both exponent bounds with correct precedence.",
        "execution_tier": "exact helper extraction with a valid/invalid boundary matrix",
        "questions": [
            "Are lower and upper exponent bounds enforced for every input element?",
            "Are exact boundary values accepted while either-side violations are rejected?",
            "Are packed results for valid values unchanged?",
            "Is the corrected predicate covered by a direct or independent regression matrix?",
        ],
        "decision_rule": (
            "Accept with scope only if exact head execution rejects both invalid sides, preserves "
            "valid boundary packing, and removes the precedence ambiguity. Missing direct upstream "
            "coverage is revise even when the independent contract passes."
        ),
    },
    "flashattention-pr-2645": {
        "claim": "backward compilation keys include subtile_factor wherever it changes codegen.",
        "execution_tier": "exact AST/cache-key isolation probe plus changed regression audit",
        "questions": [
            (
                "Do otherwise identical configurations with different subtile_factor get "
                "distinct keys?"
            ),
            "Do identical configurations retain stable equal keys?",
            "Is every backward compilation path affected by subtile_factor keyed consistently?",
            "Do direct tests exercise both collision and stable-key controls?",
        ],
        "decision_rule": (
            "Accept with scope only if a base/head isolation matrix proves the collision is "
            "removed across affected backward paths without destabilizing identical keys, with "
            "direct tests. "
            "Full GPU kernel compilation remains separately unresolved if unavailable."
        ),
    },
    "flashinfer-pr-3918": {
        "claim": "the autotuner accepts non-tensor guarded arguments without tensor-only access.",
        "execution_tier": "exact autotuner guard/key probe with tensor and non-tensor arguments",
        "questions": [
            "Are tensor metadata operations restricted to actual tensor arguments?",
            "Are scalar, boolean, None, and structured non-tensor guards deterministic?",
            "Do tensor guards still distinguish shape, dtype, and device where declared?",
            "Does direct coverage include mixed tensor/non-tensor signatures and cache reuse?",
        ],
        "decision_rule": (
            "Accept with scope only if exact head execution handles the frozen non-tensor matrix, "
            "preserves tensor discrimination, and has direct mixed-signature regression coverage."
        ),
    },
    "liger-pr-1289": {
        "claim": "fused linear cross entropy handles a zero-width vocabulary deterministically.",
        "execution_tier": "exact control-flow and callable boundary probe over vocabulary sizes",
        "questions": [
            "Does vocabulary size zero avoid Python division or modulo by zero?",
            "Is zero-width behavior an explicit stable result or a precise domain error?",
            "Are positive vocabulary sizes and reduction modes unchanged?",
            "Does a direct regression cover zero plus neighboring nonzero sizes?",
        ],
        "decision_rule": (
            "Accept with scope only if zero width follows an explicit deterministic policy, no "
            "ZeroDivisionError leaks, positive-size behavior is preserved, and direct coverage "
            "exists. A silent semantically invalid result is revise."
        ),
    },
    "megatron-pr-5750": {
        "claim": "MambaLayer propagates the configured normalization epsilon.",
        "execution_tier": "exact constructor/configuration probe over default and custom epsilon",
        "questions": [
            "Does a custom configured epsilon reach every MambaLayer norm constructed from it?",
            "Is the default epsilon preserved when no override is supplied?",
            "Are unrelated normalization and layer configuration fields unchanged?",
            "Does a direct regression assert the instantiated norm epsilon?",
        ],
        "decision_rule": (
            "Accept with scope only if exact construction proves custom and default propagation at "
            "all affected norm sites, without a hard-coded fallback, and direct coverage is "
            "present."
        ),
    },
    "sglang-pr-31344": {
        "claim": (
            "prefill-only flashmla_auto is rejected as a DSA decode backend at CLI validation."
        ),
        "execution_tier": "exact argparse/validation truth table over DSA backend choices",
        "questions": [
            "Is flashmla_auto rejected specifically in the DSA decode-backend position?",
            "Does failure occur during normal argument parsing or validation before dispatch?",
            "Are valid DSA decode backends and prefill use of flashmla_auto preserved?",
            "Do direct tests cover the forbidden case and valid neighbors?",
        ],
        "decision_rule": (
            "Accept with scope only if the exact truth table rejects only the prefill-only decode "
            "selection, preserves valid neighboring configurations, and provides a precise early "
            "error."
        ),
    },
    "torchtitan-pr-3862": {
        "claim": "the Helion RoPE fake-tensor path produces valid stride metadata.",
        "execution_tier": "exact fake/meta and real-tensor shape/stride contract probe",
        "questions": [
            "Do fake/meta executions produce rank-consistent non-overlapping output strides?",
            "Do shape, dtype, and device metadata match the public RoPE contract?",
            "Are real-tensor layouts unchanged across representative shapes?",
            "Does a direct regression cover the fake path and a real/layout control?",
        ],
        "decision_rule": (
            "Accept with scope only if exact fake execution fixes the base stride failure across "
            "a shape matrix, real layouts remain valid, and the regression is not a single "
            "hard-coded shape."
        ),
    },
    "verl-pr-7044": {
        "claim": "the Qwen3 tool parser handles malformed XML without circular fallback.",
        "execution_tier": "exact CPU parser state-machine probe over valid and malformed streams",
        "questions": [
            (
                "Do truncated, mismatched, nested, and stray tags terminate without recursion or "
                "hangs?"
            ),
            "Are valid tool calls still parsed with identical names and arguments?",
            "Is ordinary text preserved rather than misclassified as a tool call?",
            "Do direct tests cover malformed cases plus valid and plain-text controls?",
        ],
        "decision_rule": (
            "Accept with scope only if the malformed-input matrix is bounded and deterministic, "
            "valid calls are preserved, plain text is not consumed, and direct CPU coverage exists."
        ),
    },
    "vllm-pr-48705": {
        "claim": "FunctionGemma streaming preserves argument content when a second key arrives.",
        "execution_tier": "exact CPU streaming-parser probe across JSON chunk boundaries",
        "questions": [
            "Are first-key values unchanged when the second and later keys begin?",
            "Are emitted deltas exactly-once and reconstruct the final argument object?",
            "Do split points inside escapes, Unicode, numbers, and delimiters remain correct?",
            "Do direct tests cover multiple keys, chunk boundaries, and incomplete input?",
        ],
        "decision_rule": (
            "Accept with scope only if exhaustive frozen chunk splits reconstruct multi-key "
            "arguments without corruption or duplicate deltas, while incomplete input fails or "
            "waits explicitly."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("R10 selection lock digest mismatch")
    if material["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R10 selection exposes review text")
    if material["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R10 selection exposes outcomes")
    cases = [HistoricalPRCandidate.model_validate(item) for item in material["cases"]]
    if {item.case_id for item in cases} != set(CASE_PLANS):
        raise SystemExit("R10 selection and plan case sets differ")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "review_text_requested": False,
        "frozen_before_source_diff_content_inspection": True,
        "scoring_policy": {
            "kind": "ordered explainable contract judgment",
            "weighted_score_used": False,
            "forced_polarization_used": False,
            "decisions": ["accept_with_scope", "revise", "reject", "unresolved"],
            "missing_environment_evidence": "unresolved, never candidate fail",
        },
        "cases": [
            {
                "case_id": item.case_id,
                "project": item.project,
                "repository": item.repository,
                "pull_number": item.pull_number,
                "base_sha": item.base_sha,
                "head_sha": item.head_sha,
                "changed_paths": item.paths,
                **CASE_PLANS[item.case_id],
            }
            for item in cases
        ],
    }
    payload = {**plan_material, "test_plan_sha256": canonical_sha256(plan_material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(cases)}")
    print(f"test_plan_sha256={payload['test_plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
