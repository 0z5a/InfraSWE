#!/usr/bin/env python3
"""Freeze r4 machine judgments before revealing selected reviewer text."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import (
    R4_POLICY_ID,
    analyze_integration_preflight,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = _read(args.test_plan)
    stored_plan_sha = plan.pop("test_plan_sha256")
    if canonical_sha256(plan) != stored_plan_sha:
        raise SystemExit("r4 test-plan digest mismatch")
    selection = _read(args.selection_lock)
    if canonical_sha256(selection["selection_material"]) != selection["selection_lock_sha256"]:
        raise SystemExit("r4 selection-lock digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("r4 plan is not bound to the selection lock")
    if plan["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("r4 plan does not assert hidden reviewer text")

    probe_root = args.result_root / "supplemental-r4/probes"
    oracle_path = args.result_root / ("supplemental-r4/oracles/deepgemm-20250227-contract.json")
    paths = {
        "vllm-pr-8200": [
            probe_root / "vllm-pr-8200-cache-layout.json",
            probe_root / "vllm-pr-8200-cache-layout-stable.json",
        ],
        "vllm-pr-13996": [
            probe_root / "vllm-pr-13996-deepgemm.json",
            oracle_path,
        ],
    }
    evidence = {
        case_id: [(path, _read(path)) for path in case_paths]
        for case_id, case_paths in paths.items()
    }

    cache_first = evidence["vllm-pr-8200"][0][1]
    cache_stable = evidence["vllm-pr-8200"][1][1]
    cache_observations = [
        *analyze_integration_preflight(
            capability_contracts={},
            targeted_architecture_generation=None,
            default_architecture_generation=None,
            successor_generation_covered=False,
        ),
        _pass(
            "same-layout-cache-strides-are-correct",
            "Do NHD and HND key/value caches match a stride-aware oracle?",
            (
                "The exact head formula passes both NHD/NHD and HND/HND; the base fails HND, "
                "so the intended feature is genuinely implemented."
            ),
            cache_stable["evidence_sha256"],
        ),
        _failure(
            "key-and-value-cache-strides-are-independent-or-validated",
            (
                "Are key/value page and head strides passed independently, or rejected when "
                "they differ?"
            ),
            (
                "The head uses key-cache page/head strides for both tensors. Mixed HND/NHD "
                "cases preserve key values but silently corrupt value-cache placement, and no "
                "stride-equality guard exists."
            ),
            "KEY_VALUE_CACHE_STRIDES_NOT_INDEPENDENT_OR_VALIDATED",
            cache_stable["evidence_sha256"],
            counterevidence=["Same-layout NHD and HND cases both pass."],
        ),
        _pass(
            "existing-layout-performance-is-preserved",
            "Does stride-aware indexing preserve NHD steady-state speed?",
            (
                "On 1024 tokens, 32 heads, and head size 128, the head/base NHD median "
                f"ratio is {cache_stable['performance']['nhd_head_over_base']:.3f}."
            ),
            cache_stable["evidence_sha256"],
        ),
        _pass(
            "compilation-is-outside-steady-state",
            "Is unavoidable CUDA compilation excluded from benchmark timing?",
            (
                f"First precompile took {cache_first['phases']['precompile']['seconds']:.2f}s; "
                "cached load took "
                f"{cache_stable['phases']['precompile']['seconds']:.3f}s and steady-state "
                "compile time was zero."
            ),
            cache_first["evidence_sha256"],
            cache_stable["evidence_sha256"],
        ),
    ]

    deepgemm = evidence["vllm-pr-13996"][0][1]
    oracle = evidence["vllm-pr-13996"][1][1]
    deepgemm_observations = [
        *analyze_integration_preflight(
            capability_contracts={
                "deepgemm_hopper_sm90a": (
                    deepgemm["facts"]["architecture_gate_present"],
                    False,
                    "historical upstream supports Hopper/sm90a only",
                ),
                "deepgemm_optional_package": (
                    deepgemm["facts"]["optional_dependency_availability_gate_present"],
                    False,
                    "selected branch imports deep_gemm directly",
                ),
            },
            targeted_architecture_generation=None,
            default_architecture_generation=None,
            successor_generation_covered=False,
        ),
        _failure(
            "backend-scale-layout-contract-is-satisfied",
            "Does the supplied LHS scale use DeepGEMM's TMA-aligned column-major layout?",
            (
                "Historical DeepGEMM requires a TMA-aligned transposed LHS scale, while the "
                "branch calls per_token_group_quant_fp8 without column_major_scales=True."
            ),
            "DEEPGEMM_LHS_SCALE_LAYOUT_CONTRACT_VIOLATED",
            deepgemm["evidence_sha256"],
            oracle["oracle_sha256"],
        ),
        _failure(
            "runtime-jit-has-a-precompile-path",
            "Can all shape-keyed DeepGEMM JIT work finish before timed benchmark cases?",
            (
                "The historical backend compiles shape-specific kernels at runtime, but the "
                "patch exposes only an execution-time environment flag and no warmup/precompile "
                "entry point."
            ),
            "DEEPGEMM_RUNTIME_JIT_PRECOMPILE_CONTRACT_MISSING",
            deepgemm["evidence_sha256"],
            oracle["oracle_sha256"],
        ),
        _pass(
            "unsupported-backend-early-exit-protects-benchmark-budget",
            "Can an unsupported A100 stop before import or JIT compilation?",
            (
                "The machine rejected the missing capability contract before importing or "
                "compiling DeepGEMM; observed steady-state compile time is zero."
            ),
            "device_compute_capability=8.0",
            "steady_state_compile_seconds=0",
        ),
    ]

    observations = {
        "vllm-pr-8200": cache_observations,
        "vllm-pr-13996": deepgemm_observations,
    }
    cases = {item["case_id"]: item for item in plan["cases"]}
    selected_case_ids = selection["selection_material"]["case_ids"]
    if set(cases) != set(selected_case_ids) or set(observations) != set(cases):
        raise SystemExit("r4 cases, observations, and selection do not match")

    frozen_at = datetime.now(UTC)
    locks: list[dict[str, Any]] = []
    bindings_by_case: dict[str, dict[str, str]] = {}
    for case_id in selected_case_ids:
        case = cases[case_id]
        bindings = {
            str(path.relative_to(args.result_root)): canonical_sha256(payload)
            for path, payload in evidence[case_id]
        }
        candidate_sha256 = canonical_sha256(
            {
                key: case[key]
                for key in (
                    "case_id",
                    "repository",
                    "pull_number",
                    "base_sha",
                    "base_derivation",
                    "head_sha",
                    "paths",
                )
            }
        )
        material = compile_explainable_judgment(
            case_id=case_id,
            candidate_sha256=candidate_sha256,
            test_plan_sha256=stored_plan_sha,
            evidence_sha256=canonical_sha256(bindings),
            observations=observations[case_id],
            frozen_at=frozen_at,
            policy_id=R4_POLICY_ID,
        )
        locks.append(freeze_explainable_judgment(material).model_dump(mode="json"))
        bindings_by_case[case_id] = bindings

    lock_material = {
        "schema_version": "0.5",
        "protocol_id": "historical-reviewed-failure-alignment-v0.5-r4",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": stored_plan_sha,
        "review_text_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "frozen_at": frozen_at.isoformat(),
        "evidence_bindings": bindings_by_case,
        "locks": locks,
    }
    payload = {**lock_material, "lock_set_sha256": canonical_sha256(lock_material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "lock_set_sha256": payload["lock_set_sha256"],
                "cases": [
                    {
                        "case_id": item["material"]["case_id"],
                        "decision": item["material"]["decision"],
                        "rationale_codes": item["material"]["rationale_codes"],
                        "lock_sha256": item["lock_sha256"],
                    }
                    for item in locks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
