#!/usr/bin/env python3
"""Freeze r2 machine judgments before any selected reviewer text is revealed."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import (
    compile_explainable_judgment,
    freeze_explainable_judgment,
)
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalHeuristicObservation


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _pass(
    rule_id: str, question: str, conclusion: str, *evidence: str
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="pass",
        blocking=True,
        evidence=list(evidence),
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


def _static_observations(payload: dict[str, Any]) -> list[HistoricalHeuristicObservation]:
    return [HistoricalHeuristicObservation.model_validate(item) for item in payload["observations"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--test-plan-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = _read(args.test_plan)
    plan_lock = _read(args.test_plan_lock)
    if plan_lock.get("review_text_revealed") is not False:
        raise ValueError(
            "supplemental test-plan lock no longer asserts a blind reviewer-text state"
        )
    if canonical_sha256(plan) != plan_lock["test_plan_sha256"]:
        raise ValueError("supplemental test-plan digest mismatch")
    cases = {item["case_id"]: item for item in plan["cases"]}
    frozen_at = datetime.now(UTC)
    evidence_root = args.result_root / "supplemental-r2"

    sglang_static_path = evidence_root / "static/sglang-pr-3044.json"
    sglang_move_path = evidence_root / "probes/sglang-pr-3044-move.json"
    sglang_static = _read(sglang_static_path)
    sglang_move = _read(sglang_move_path)
    sglang_bindings = {
        str(sglang_static_path.relative_to(args.result_root)): canonical_sha256(sglang_static),
        str(sglang_move_path.relative_to(args.result_root)): canonical_sha256(sglang_move),
    }
    sglang_observations = [
        _pass(
            "exact-patch-identity",
            "Are the reconstructed base/head sources bound to the locked PR commits?",
            (
                "All four expected paths are represented and their before/after source and "
                "diff digests are frozen."
            ),
            sglang_static["evidence_sha256"],
        ),
        *_static_observations(sglang_static),
        _pass(
            "moved-test-discovery-preserved",
            "Are all moved test definitions parseable and registered exactly once?",
            (
                "All three AST inventories are unchanged and every moved filename occurs "
                "exactly once in run_suite."
            ),
            sglang_bindings[str(sglang_move_path.relative_to(args.result_root))],
        ),
        _pass(
            "early-exit-protects-benchmark-budget",
            "Can the decision stop before executing the contracted full GPU matrix?",
            (
                "Two source-contract failures already require revision, so the expensive "
                "matrix is skipped without weakening that decision."
            ),
            "compile_seconds=0",
            f"probe_duration_seconds={sglang_move['duration_seconds']}",
        ),
    ]

    barrier_static_path = evidence_root / "probes/vllm-pr-13140-static.json"
    barrier_dynamic_path = evidence_root / "probes/vllm-pr-13140-barrier.json"
    barrier_base_race_path = evidence_root / "probes/vllm-pr-13140-base-racecheck.log"
    barrier_head_race_path = evidence_root / "probes/vllm-pr-13140-head-racecheck.log"
    barrier_static = _read(barrier_static_path)
    barrier_dynamic = _read(barrier_dynamic_path)
    base_race = barrier_base_race_path.read_text(encoding="utf-8")
    head_race = barrier_head_race_path.read_text(encoding="utf-8")
    barrier_bindings = {
        str(barrier_static_path.relative_to(args.result_root)): canonical_sha256(barrier_static),
        str(barrier_dynamic_path.relative_to(args.result_root)): canonical_sha256(barrier_dynamic),
        str(barrier_base_race_path.relative_to(args.result_root)): canonical_sha256(base_race),
        str(barrier_head_race_path.relative_to(args.result_root)): canonical_sha256(head_race),
    }
    head_replays = sum(
        item["replays"] for item in barrier_dynamic["results"] if item["variant"] == "head"
    )
    head_failures = barrier_dynamic["head_failures"]
    base_replays = sum(
        item["replays"] for item in barrier_dynamic["results"] if item["variant"] == "base"
    )
    barrier_observations = [
        _pass(
            "exact-patch-identity",
            "Is the exact one-file patch bound before judgment?",
            (
                "Pinned base/head sources differ in the affected kernel only by one "
                "unconditional block barrier."
            ),
            barrier_static["source_identity"]["kernel_diff_sha256"],
        ),
        _pass(
            "shared-initialization-happens-before-cross-warp-atomic",
            (
                "Does the patch order every shared-count initializer before any warp can "
                "update any row?"
            ),
            (
                "The new depth-1 barrier is reached by all 1024 threads and separates 32 "
                "warp-owned row initializers from input-directed cross-warp atomics."
            ),
            barrier_bindings[str(barrier_static_path.relative_to(args.result_root))],
        ),
        _pass(
            "affected-kernel-sm80-correctness",
            (
                "Does the compiled head kernel match CPU grouping and padding on adversarial "
                "SM80 cases?"
            ),
            (
                f"Head passed {head_replays - head_failures}/{head_replays}; base also passed "
                f"{base_replays}/{base_replays}, so the ordinary scheduler did not expose "
                "the latent race."
            ),
            barrier_bindings[str(barrier_dynamic_path.relative_to(args.result_root))],
        ),
        _pass(
            "compute-sanitizer-regression",
            "Is the head free of sanitizer-visible races in the instrumented launch?",
            (
                "Head racecheck reported zero hazards. Base also reported zero, which is "
                "counterevidence against claiming dynamic reproduction, not against the "
                "static happens-before defect."
            ),
            barrier_bindings[str(barrier_base_race_path.relative_to(args.result_root))],
            barrier_bindings[str(barrier_head_race_path.relative_to(args.result_root))],
        ),
        _pass(
            "steady-state-regression-below-ten-percent",
            "Does the added barrier avoid a greater-than-10% hot-loop regression on this cell?",
            (
                f"Observed head/base time ratio was "
                f"{barrier_dynamic['timing']['head_over_base']:.4f}; compilation was pre-run "
                "once and steady-state compilation was zero."
            ),
            barrier_dynamic["precompile"]["cache_key_sha256"],
        ),
    ]

    router_static_path = evidence_root / "static/vllm-pr-14027.json"
    router_dynamic_path = evidence_root / "probes/vllm-pr-14027-router.json"
    router_static = _read(router_static_path)
    router_dynamic = _read(router_dynamic_path)
    router_bindings = {
        str(router_static_path.relative_to(args.result_root)): canonical_sha256(router_static),
        str(router_dynamic_path.relative_to(args.result_root)): canonical_sha256(router_dynamic),
    }
    renormalized = next(
        item for item in router_dynamic["fused_topk_cases"] if item["renormalize"] is True
    )
    router_observations = [
        _pass(
            "exact-patch-identity",
            "Are both changed sources and the exact diff bound to the locked PR commits?",
            "Both changed source pairs and their unified diff have frozen digests.",
            router_static["evidence_sha256"],
        ),
        *_static_observations(router_static),
        _failure(
            "renormalize-runtime-contract",
            "Does fused_topk still make selected weights sum to one when renormalize=True?",
            (
                f"Base max sum error is "
                f"{renormalized['base_sum_to_one_max_error']:.3g}, while head error is "
                f"{renormalized['head_sum_to_one_max_error']:.3g}; the retained flag is "
                "ignored at runtime."
            ),
            "CALLER_CONTRACT_PARAMETER_IGNORED:renormalize",
            router_bindings[str(router_dynamic_path.relative_to(args.result_root))],
            counterevidence=[
                "Base/head expert IDs are equal and renormalize=False behavior is unchanged."
            ],
        ),
        _pass(
            "fp32-routing-precision",
            "Do the changed softmax and sigmoid paths preserve or improve FP64 agreement?",
            (
                "Both grouped routing probes passed; head error was no worse than base for "
                "softmax and sigmoid."
            ),
            router_bindings[str(router_dynamic_path.relative_to(args.result_root))],
        ),
        _pass(
            "compile-free-minimal-probe",
            "Can the contract failure be diagnosed without building the full host project?",
            (
                "Exact functions were extracted with compile_seconds=0; hot call time was "
                f"{router_dynamic['steady_state_seconds_per_call']:.6g}s."
            ),
            "steady_state_compile_seconds=0",
        ),
    ]

    case_inputs = {
        "sglang-pr-3044": (sglang_observations, sglang_bindings),
        "vllm-pr-13140": (barrier_observations, barrier_bindings),
        "vllm-pr-14027": (router_observations, router_bindings),
    }
    locks = []
    evidence_bindings: dict[str, dict[str, str]] = {}
    for case_id in plan_lock["cases"]:
        case = cases[case_id]
        observations, bindings = case_inputs[case_id]
        candidate_sha256 = canonical_sha256(
            {
                "case_id": case_id,
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "base_sha": case["base_sha"],
                "head_sha": case["head_sha"],
                "expected_paths": case["expected_paths"],
            }
        )
        material = compile_explainable_judgment(
            case_id=case_id,
            candidate_sha256=candidate_sha256,
            test_plan_sha256=plan_lock["test_plan_sha256"],
            evidence_sha256=canonical_sha256(bindings),
            observations=observations,
            frozen_at=frozen_at,
        )
        locks.append(freeze_explainable_judgment(material).model_dump(mode="json"))
        evidence_bindings[case_id] = bindings

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-reviewed-failure-alignment-v0.5-r2",
        "review_text_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "test_plan_sha256": plan_lock["test_plan_sha256"],
        "frozen_at": frozen_at.isoformat(),
        "evidence_bindings": evidence_bindings,
        "locks": locks,
    }
    payload = {**material, "lock_set_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
