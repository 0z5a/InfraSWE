#!/usr/bin/env python3
"""Freeze case-specific R9 contracts before any source diff inspection."""

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
    "cutlass-pr-3332": {
        "claim": (
            "CuTeDSL lowers unsigned integer constants above INT64_MAX without signed overflow."
        ),
        "execution_tier": "exact-source boundary probe plus CuTeDSL runtime if available",
        "questions": [
            "Do 2^63 and UINT64_MAX retain exact integer values without float conversion?",
            "Are INT64_MAX and the signed negative boundary unchanged?",
            "Does the implementation choose an explicit signed/unsigned width rather than wrap?",
            "Do values outside the supported integer domain fail explicitly?",
            "Does the changed test cover both target and neighboring boundaries?",
        ],
        "decision_rule": (
            "Accept with scope only if exact source and boundary probes preserve all supported "
            "neighbors and the direct test covers the target. A missing outer-boundary policy is "
            "revise; inability to execute the full CuTeDSL compiler remains separately unresolved."
        ),
    },
    "flashinfer-pr-3950": {
        "claim": "MNNVL communicator flags survive checkpoint restore under inference mode.",
        "execution_tier": "exact-AST state-transition probe plus mocked checkpoint lifecycle",
        "questions": [
            "Are boolean/control flags restored even when tensor mutation is disabled?",
            "Are tensor payloads and immutable metadata still handled by their declared owners?",
            "Does restore preserve explicit user/runtime state outside the serialized fields?",
            "Do inference-mode and ordinary-mode restores converge on the same flag state?",
            "Does a direct regression exercise the actual checkpoint restore entrypoint?",
        ],
        "decision_rule": (
            "Accept with scope only if the same public restore path gives equal flag state in "
            "ordinary and inference modes, without weakening tensor safety or overwriting "
            "unowned state. Real multi-GPU MNNVL transport stays unresolved without its cell."
        ),
    },
    "sglang-pr-31346": {
        "claim": "CUDA TileLang DSA rejects fp8_e4m3 KV configuration before runtime dispatch.",
        "execution_tier": "exact validation-call probe with a backend/dtype truth table",
        "questions": [
            "Does the exact forbidden CUDA + TileLang DSA + fp8_e4m3 combination fail fast?",
            "Do supported KV dtypes remain allowed on the same backend?",
            "Are non-TileLang DSA and non-CUDA paths unaffected?",
            "Is the error precise and attached to the normal override/validation lifecycle?",
            "Does the changed test cover invalid and valid neighboring controls?",
        ],
        "decision_rule": (
            "Accept with scope only if the full truth table rejects exactly the unsupported "
            "combination at configuration time, retains supported neighbors, and has direct tests."
        ),
    },
    "sglang-pr-31349": {
        "claim": "FlashInfer CUDA-graph decode planning uses a bounded deterministic launch set.",
        "execution_tier": "exact helper/state probe over graph batch-size matrices",
        "questions": [
            "Is every planned decode launch bounded by the configured capture capacity?",
            "Is plan selection deterministic across repeated calls and orderings?",
            "Are zero, boundary, padding, and over-capacity inputs explicit?",
            "Do graph and non-graph paths preserve their separate lifecycle contracts?",
            "Does the direct test cover neighboring capture sizes instead of one public example?",
        ],
        "decision_rule": (
            "Accept with scope only if a repeated boundary matrix proves deterministic bounded "
            "planning with explicit over-capacity behavior and direct regression coverage. "
            "Actual CUDA graph replay stays unresolved if the required GPU stack is unavailable."
        ),
    },
    "vllm-pr-48695": {
        "claim": (
            "vLLM selects FlashInfer FP4 MoE only when the required native kernel is available."
        ),
        "execution_tier": "exact capability-gate probe with import/kernel/architecture controls",
        "questions": [
            "Does installed FlashInfer without the FP4 kernel report unavailable?",
            (
                "Does a supported kernel/profile report available without importing it during "
                "selection?"
            ),
            "Is unavailable behavior an explicit approved fallback rather than a later crash?",
            "Are cached capability results invalidated or scoped to the relevant environment?",
            "Does the direct test cover present, missing, and partial capability states?",
        ],
        "decision_rule": (
            "Accept with scope only if selection is based on the required kernel capability, "
            "partial installations take the approved fallback, supported configurations remain "
            "enabled, and the direct tests cover all three states."
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
        raise SystemExit("R9 selection lock digest mismatch")
    if material["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R9 selection exposes review text")
    if material["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R9 selection exposes outcomes")
    cases = [HistoricalPRCandidate.model_validate(item) for item in material["cases"]]
    if {item.case_id for item in cases} != set(CASE_PLANS):
        raise SystemExit("R9 selection and plan case sets differ")

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
