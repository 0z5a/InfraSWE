#!/usr/bin/env python3
"""Freeze r5 machine judgments before any selected reviewer text is revealed."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import (
    R4_POLICY_ID,
    compile_explainable_judgment,
    freeze_explainable_judgment,
)
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalHeuristicObservation

EXPECTED_SELECTION_FILE_SHA256 = "b4d3bef64cd61f8500124d94acfb75ab12f07ec27670d2805b37481bc1c8603e"
EXPECTED_TEST_PLAN_FILE_SHA256 = "c313c1c28bae1ce55861e453457db7841e70c5a5651df9748235d3068e43ba99"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pass(
    rule_id: str,
    question: str,
    conclusion: str,
    *evidence: str,
    counterevidence: list[str] | None = None,
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="pass",
        blocking=True,
        evidence=list(evidence),
        counterevidence=counterevidence or [],
        conclusion=conclusion,
    )


def _failure(
    rule_id: str,
    question: str,
    conclusion: str,
    failure_code: str,
    *evidence: str,
    counterevidence: list[str] | None = None,
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="fail",
        blocking=True,
        evidence=list(evidence),
        counterevidence=counterevidence or [],
        conclusion=conclusion,
        failure_code=failure_code,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection_file_sha = _file_sha256(args.selection_lock)
    test_plan_file_sha = _file_sha256(args.test_plan)
    if selection_file_sha != EXPECTED_SELECTION_FILE_SHA256:
        raise SystemExit("r5 selection-lock file digest mismatch")
    if test_plan_file_sha != EXPECTED_TEST_PLAN_FILE_SHA256:
        raise SystemExit("r5 test-plan file digest mismatch")

    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    if plan.get("review_feedback_hidden") is not True:
        raise SystemExit("r5 plan does not assert hidden reviewer feedback")
    if plan.get("frozen_before_probe_execution") is not True:
        raise SystemExit("r5 plan was not frozen before probe execution")
    if selection.get("review_bodies_accessed_before_selection") is not False:
        raise SystemExit("r5 selection does not assert hidden review bodies")
    if selection.get("issue_comment_bodies_accessed_before_selection") is not False:
        raise SystemExit("r5 selection does not assert hidden issue comments")

    selection_cases = {item["candidate_ref"]: item for item in selection["cases"]}
    plan_cases = {item["candidate_ref"]: item for item in plan["cases"]}
    if set(selection_cases) != set(plan_cases):
        raise SystemExit("r5 selection and test-plan case sets differ")
    for case_ref in selection_cases:
        if selection_cases[case_ref]["head_sha"] != plan_cases[case_ref]["head_sha"]:
            raise SystemExit(f"r5 head SHA mismatch for {case_ref}")

    probe_root = args.result_root / "supplemental-r5/probes"
    paths = {
        "vllm#50423": probe_root / "vllm-pr-50423-dequeue-lock.json",
        "vllm#52205": probe_root / "vllm-pr-52205-always-break.json",
        "vllm#53038": probe_root / "vllm-pr-53038-lora-int64.json",
    }
    evidence = {case_ref: _read(path) for case_ref, path in paths.items()}
    bindings = {
        case_ref: {str(path.relative_to(args.result_root)): canonical_sha256(evidence[case_ref])}
        for case_ref, path in paths.items()
    }

    dequeue = evidence["vllm#50423"]
    dequeue_digest = next(iter(bindings["vllm#50423"].values()))
    dequeue_observations = [
        _pass(
            "dequeue-receive-is-serialized",
            "Does the new lock prevent concurrent reader-cursor/socket receives?",
            (
                "The exact extracted head reaches at most one concurrent recv while the base "
                "reaches two, so the intended serialization is effective."
            ),
            dequeue_digest,
        ),
        _failure(
            "lock-wait-consumes-caller-timeout-budget",
            "Does lock acquisition consume the same end-to-end timeout budget as recv?",
            (
                "With an 80 ms lock hold and a 100 ms caller timeout, the head takes "
                f"{dequeue['timeout_budget']['head']['elapsed_seconds'] * 1000:.2f} ms because "
                "recv receives a fresh timeout after lock acquisition. The added test does not "
                "cover the total deadline."
            ),
            "DEQUEUE_TIMEOUT_BUDGET_RESTARTED_AFTER_LOCK_WAIT",
            dequeue_digest,
            counterevidence=["The serialization race itself is fixed."],
        ),
        _pass(
            "uncontended-overhead-remains-sub-microsecond",
            "Is the uncontended absolute lock overhead small enough to retain the approach?",
            (
                "The median rises from "
                f"{dequeue['uncontended_overhead']['base']['median_nanoseconds_per_call']:.1f} "
                "ns to "
                f"{dequeue['uncontended_overhead']['head']['median_nanoseconds_per_call']:.1f} "
                "ns; the relative ratio is large but the absolute increase remains below one "
                "microsecond."
            ),
            dequeue_digest,
            counterevidence=[
                f"head/base ratio={dequeue['uncontended_overhead']['head_to_base_ratio']:.3f}"
            ],
        ),
    ]

    always_break_digest = next(iter(bindings["vllm#52205"].values()))
    always_break_observations = [
        _pass(
            "always-break-routes-full-capture-to-eager",
            "Does always_break route FULL capture through an eager segment?",
            (
                "Extracted decorator behavior shows the new opt-in breaks FULL capture while "
                "legacy FULL and PIECEWISE behavior remains unchanged."
            ),
            always_break_digest,
        ),
        _pass(
            "amd-kda-opt-in-is-scoped",
            "Is the new behavior limited to the intended AMD KDA call site?",
            "Only the AMD Kimi-K3 KDA call site opts in; the NVIDIA path does not.",
            always_break_digest,
        ),
        _failure(
            "amd-full-cudagraph-path-has-direct-evidence",
            (
                "Does the changed AMD FULL capture/replay path have direct tests and "
                "performance evidence?"
            ),
            (
                "The patch changes no test, no test references always_break, and no runtime "
                "benchmark is included, so extracted decorator semantics cannot establish AMD "
                "FULL capture/replay correctness or performance."
            ),
            "AMD_FULL_CUDAGRAPH_CORRECTNESS_AND_PERFORMANCE_EVIDENCE_MISSING",
            always_break_digest,
            counterevidence=["The isolated routing semantics and backend scope are correct."],
        ),
    ]

    lora = evidence["vllm#53038"]
    lora_digest = next(iter(bindings["vllm#53038"].values()))
    ratio = lora["steady_latency_ms"]["head_to_base_ratio"]
    lora_observations = [
        _pass(
            "int64-offset-crosses-int32-boundary",
            "Does the int64 change fix pointer arithmetic beyond 2**31 elements?",
            (
                "At row 174763 and stride 12288 the base wraps to "
                f"{lora['boundary_arithmetic']['base_int32_result']}, while the head produces "
                f"the expected {lora['boundary_arithmetic']['expected_int64']}."
            ),
            lora_digest,
        ),
        _pass(
            "timed-shapes-are-precompiled",
            "Are both boundary and common-range variants compiled before steady timing?",
            (
                f"Precompile created {lora['precompile']['new_cache_files']} cache files and "
                "steady timing created zero additional files."
            ),
            lora_digest,
        ),
        _pass(
            "common-range-latency-within-three-percent",
            "Does common-range steady latency avoid a regression greater than 3%?",
            f"The seven-pair median head/base latency ratio is {ratio:.4f}.",
            lora_digest,
        ),
        _failure(
            "each-changed-kernel-family-has-direct-large-offset-coverage",
            "Does every changed ordinary and FP8 kernel family have a direct large-offset test?",
            (
                "Four ordinary/FP8 expand/shrink files change, but the new large test is BF16 "
                "and the FP8 path is not directly exercised."
            ),
            "FP8_LORA_INT64_DIRECT_TEST_MISSING",
            lora_digest,
            counterevidence=[
                "Standalone arithmetic proves the int64 mechanism and common-range timing passes."
            ],
        ),
    ]

    observations = {
        "vllm#50423": dequeue_observations,
        "vllm#52205": always_break_observations,
        "vllm#53038": lora_observations,
    }
    frozen_at = datetime.now(UTC)
    test_plan_sha = canonical_sha256(plan)
    locks = []
    for case_ref in ("vllm#53038", "vllm#50423", "vllm#52205"):
        candidate_sha = canonical_sha256(
            {
                "selection": selection_cases[case_ref],
                "test_plan": plan_cases[case_ref],
            }
        )
        material = compile_explainable_judgment(
            case_id=case_ref,
            candidate_sha256=candidate_sha,
            test_plan_sha256=test_plan_sha,
            evidence_sha256=canonical_sha256(bindings[case_ref]),
            observations=observations[case_ref],
            frozen_at=frozen_at,
            policy_id=R4_POLICY_ID,
        )
        locks.append(freeze_explainable_judgment(material).model_dump(mode="json"))

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-reviewed-failure-alignment-v0.5-r5",
        "review_text_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "selection_lock_file_sha256": "sha256:" + selection_file_sha,
        "selection_lock_sha256": canonical_sha256(selection),
        "test_plan_file_sha256": "sha256:" + test_plan_file_sha,
        "test_plan_sha256": test_plan_sha,
        "frozen_at": frozen_at.isoformat(),
        "evidence_bindings": bindings,
        "locks": locks,
    }
    payload = {**material, "lock_set_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
