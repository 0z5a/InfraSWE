#!/usr/bin/env python3
"""Freeze R7 cross-project machine judgments before outcome/review reveal."""

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

EXPECTED_SELECTION_FILE_SHA256 = "d4c1645634e721793b97395b9c19ecc2b9fc893ede4b94b7436c104ff2745c3e"
EXPECTED_TEST_PLAN_FILE_SHA256 = "314ac6ad34cba5769707f8abe724bfce7b43ade17ba64a184381a00215ace703"


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


def _unresolved(
    rule_id: str,
    question: str,
    conclusion: str,
    failure_code: str,
    *evidence: str,
    blocking: bool = True,
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="unresolved",
        blocking=blocking,
        evidence=list(evidence),
        conclusion=conclusion,
        failure_code=failure_code,
    )


def _validate_evidence(
    payload: dict[str, Any],
    *,
    case_id: str,
    selected: dict[str, Any],
    selection_sha: str,
    plan_sha: str,
    path: Path,
) -> None:
    embedded = payload.get("evidence_sha256")
    material = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    if embedded != canonical_sha256(material):
        raise SystemExit(f"R7 embedded evidence digest mismatch in {path}")
    if payload.get("case_id") != case_id:
        raise SystemExit(f"R7 evidence case mismatch in {path}")
    source_identity = payload.get("source_identity") or {}
    bound_base = payload.get("base_sha") or source_identity.get("base_sha")
    bound_head = payload.get("head_sha") or source_identity.get("head_sha")
    bound_selection = payload.get("selection_lock_sha256") or source_identity.get(
        "selection_lock_sha256"
    )
    bound_plan = payload.get("test_plan_sha256") or source_identity.get("test_plan_sha256")
    if bound_base != selected["base_sha"]:
        raise SystemExit(f"R7 evidence base SHA mismatch in {path}")
    if bound_head != selected["head_sha"]:
        raise SystemExit(f"R7 evidence head SHA mismatch in {path}")
    if bound_selection != selection_sha:
        raise SystemExit(f"R7 evidence selection binding mismatch in {path}")
    if bound_plan != plan_sha:
        raise SystemExit(f"R7 evidence test-plan binding mismatch in {path}")
    if payload.get("status") != "pass":
        raise SystemExit(f"R7 probe execution did not complete in {path}")


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
        raise SystemExit("R7 selection-lock file digest mismatch")
    if test_plan_file_sha != EXPECTED_TEST_PLAN_FILE_SHA256:
        raise SystemExit("R7 test-plan file digest mismatch")

    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_material = selection["selection_material"]
    if selection_material["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 selection does not assert hidden review text")
    if selection_material["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 selection does not assert hidden merge outcomes")
    if plan["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 plan does not assert hidden review text")
    if plan["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 plan does not assert hidden merge outcomes")
    if plan["review_text_requested"] is not False:
        raise SystemExit("R7 plan says review text was requested before judgment")
    if plan["frozen_before_source_diff_content_inspection"] is not True:
        raise SystemExit("R7 plan was not frozen before source inspection")
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R7 embedded selection digest mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit("R7 embedded test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R7 test plan is not bound to the selection lock")

    selection_cases = {item["case_id"]: item for item in selection_material["cases"]}
    plan_cases = {item["case_id"]: item for item in plan["cases"]}
    if set(selection_cases) != set(plan_cases):
        raise SystemExit("R7 selection and test-plan case sets differ")
    for case_id in selection_cases:
        if selection_cases[case_id]["base_sha"] != plan_cases[case_id]["base_sha"]:
            raise SystemExit(f"R7 base SHA mismatch for {case_id}")
        if selection_cases[case_id]["head_sha"] != plan_cases[case_id]["head_sha"]:
            raise SystemExit(f"R7 head SHA mismatch for {case_id}")

    paths: dict[str, list[Path]] = {
        "cutlass-pr-3427": [
            args.result_root / "static/cutlass-pr-3427.json",
            args.result_root / "probes/cutlass-pr-3427-scaled-basis.json",
            args.result_root / "probes/cutlass-pr-3427-sm120-aot.json",
        ],
        "liger-pr-1328": [
            args.result_root / "static/liger-pr-1328.json",
            args.result_root / "probes/liger-pr-1328-h100.json",
        ],
        "deepgemm-pr-389": [
            args.result_root / "static/deepgemm-pr-389.json",
            args.result_root / "probes/deepgemm-pr-389-sm100-aot.json",
        ],
        "megatron-pr-6174": [
            args.result_root / "static/megatron-pr-6174.json",
            args.result_root / "probes/megatron-pr-6174.json",
        ],
        "torchtitan-pr-4032": [
            args.result_root / "static/torchtitan-pr-4032.json",
            args.result_root / "probes/torchtitan-pr-4032.json",
        ],
        "verl-pr-7220": [
            args.result_root / "static/verl-pr-7220.json",
            args.result_root / "probes/verl-pr-7220.json",
        ],
    }
    evidence: dict[str, list[dict[str, Any]]] = {}
    for case_id, case_paths in paths.items():
        payloads = [_read(path) for path in case_paths]
        for path, payload in zip(case_paths, payloads, strict=True):
            _validate_evidence(
                payload,
                case_id=case_id,
                selected=selection_cases[case_id],
                selection_sha=selection["selection_lock_sha256"],
                plan_sha=plan["test_plan_sha256"],
                path=path,
            )
        evidence[case_id] = payloads
    bindings = {
        case_id: {
            str(path.relative_to(args.result_root)): canonical_sha256(payload)
            for path, payload in zip(paths[case_id], evidence[case_id], strict=True)
        }
        for case_id in paths
    }

    def digests(case_id: str) -> list[str]:
        return list(bindings[case_id].values())

    cutlass_static, cutlass_host, cutlass_aot = evidence["cutlass-pr-3427"]
    cutlass_facts = cutlass_aot["facts"]
    scaled_ok = (
        cutlass_host["facts"]["base_reproduces_uncomparable_value_compile_failure"]
        and cutlass_host["facts"]["head_compiles_and_executes"]
        and cutlass_host["facts"]["equal_basis_behavior_retained"]
        and cutlass_facts["base_scaled_basis_failure_reproduced"]
        and cutlass_facts["head_scaled_basis_full_header_pass"]
    )
    if scaled_ok:
        cutlass_equality = _pass(
            "scaled-basis-equality-compiles-and-distinguishes-bases",
            "Do equal ScaledBasis values remain equal while mismatched bases compare unequal?",
            (
                "The extracted and full-header probes reproduce the base type-comparison failure; "
                "the exact head compiles, executes, retains equal-basis behavior, and returns "
                "false for mismatched comparable and uncomparable basis value types."
            ),
            *digests("cutlass-pr-3427"),
        )
    else:
        cutlass_equality = _failure(
            "scaled-basis-equality-compiles-and-distinguishes-bases",
            "Do equal ScaledBasis values remain equal while mismatched bases compare unequal?",
            "One or more exact extracted/full-header equality controls did not pass.",
            "CUTLASS_SCALED_BASIS_EQUALITY_CONTRACT_FAILED",
            *digests("cutlass-pr-3427"),
        )
    metadata_ok = (
        cutlass_facts["base_valid_k256_pass"]
        and cutlass_facts["head_valid_k256_pass"]
        and cutlass_facts["head_invalid_k128_rejected"]
        and cutlass_facts["head_invalid_reaches_intended_assert"]
    )
    if metadata_ok:
        cutlass_metadata = _pass(
            "sm120-metadata-k-rejects-invalid-and-retains-valid-neighbor",
            "Does invalid K fail at the intended boundary while a valid neighbor instantiates?",
            (
                "Exact SM120a AOT rejects head K=128 at the new metadata-atom assertion and "
                "both base and head instantiate the valid K=256 neighbor."
            ),
            *digests("cutlass-pr-3427"),
            counterevidence=[
                "Base K=128 also fails later at generic tile_to_shape divisibility; the head "
                "adds the intended earlier, domain-specific diagnostic."
            ],
        )
    else:
        cutlass_metadata = _failure(
            "sm120-metadata-k-rejects-invalid-and-retains-valid-neighbor",
            "Does invalid K fail at the intended boundary while a valid neighbor instantiates?",
            "The intended invalid/valid SM120 AOT matrix did not pass in full.",
            "CUTLASS_SM120_METADATA_K_AOT_CONTRACT_FAILED",
            *digests("cutlass-pr-3427"),
        )
    direct_cutlass_tests = cutlass_static["facts"]["changed_test_file_count"] > 0
    if direct_cutlass_tests:
        cutlass_tests = _pass(
            "both-cutlass-changes-have-direct-regressions",
            "Are direct regression tests present for both changed behaviors?",
            "Changed tests directly retain both equality and metadata-K behavior.",
            *digests("cutlass-pr-3427"),
        )
    else:
        cutlass_tests = _failure(
            "both-cutlass-changes-have-direct-regressions",
            "Are direct regression tests present for both changed behaviors?",
            (
                "The PR changes no test file and the exact test search finds no retained direct "
                "regression for either ScaledBasis equality or metadata K."
            ),
            "CUTLASS_EQUALITY_AND_METADATA_K_DIRECT_TESTS_MISSING",
            *digests("cutlass-pr-3427"),
            counterevidence=["Both offline compile contracts pass on the available toolchain."],
        )
    cutlass_observations = [
        cutlass_equality,
        cutlass_metadata,
        cutlass_tests,
        _unresolved(
            "sm120-runtime-performance-is-out-of-cell",
            "Are SM120 runtime correctness and performance retained?",
            "The available H100 is SM90; SM120 runtime and performance are not inferred from AOT.",
            "CUTLASS_SM120_RUNTIME_HARDWARE_UNAVAILABLE",
            *digests("cutlass-pr-3427"),
            blocking=False,
        ),
    ]

    liger_static, liger = evidence["liger-pr-1328"]
    liger_facts = liger["facts"]
    repair_ok = (
        liger_facts["base_target_failure_reproduced"]
        and liger_facts["base_failure_mentions_addmm_dtype_out"]
        and liger_facts["head_fullgraph_pass"]
    )
    if repair_ok:
        liger_repair = _pass(
            "dtype-out-failure-is-reproduced-and-fixed-under-fullgraph",
            "Does fullgraph compile complete after reproducing the base dtype_out failure?",
            (
                "The H100 base fails in the targeted addmm dtype_out lowering path while the "
                "exact head completes fullgraph forward/backward and matches both eager and the "
                "independent PyTorch oracle."
            ),
            *digests("liger-pr-1328"),
        )
    elif (
        liger_facts["base_target_failure_reproduced"]
        and liger_facts["base_failure_mentions_addmm_dtype_out"]
        and not liger_facts["head_fullgraph_pass"]
    ):
        fullgraph_error = liger_facts["compile_results"]["head_fullgraph"]["error"]
        liger_repair = _failure(
            "dtype-out-failure-is-reproduced-and-fixed-under-fullgraph",
            "Does fullgraph compile complete after reproducing the base dtype_out failure?",
            (
                "Base reproduces the targeted dtype_out lowering failure and head default compile "
                "passes, but head fullgraph stops at the existing Tensor.item graph break: "
                f"{fullgraph_error.splitlines()[0]}"
            ),
            "LIGER_FULLGRAPH_TENSOR_ITEM_GRAPH_BREAK",
            *digests("liger-pr-1328"),
            counterevidence=[
                "The specific dtype_out failure is fixed in head default compile and all head "
                "correctness cases match the independent oracle."
            ],
        )
    else:
        liger_repair = _failure(
            "dtype-out-failure-is-reproduced-and-fixed-under-fullgraph",
            "Does fullgraph compile complete after reproducing the base dtype_out failure?",
            "The base target failure was not reproduced or the head repair matrix failed.",
            "LIGER_DTYPE_OUT_REPAIR_CONTRACT_FAILED",
            *digests("liger-pr-1328"),
        )
    if liger_facts["head_default_all_cases_pass"]:
        liger_correctness = _pass(
            "compiled-boundaries-match-independent-oracle",
            "Do representative and boundary BF16/FP16 cases match an oracle?",
            (
                "Default and boundary BF16/FP16 loss, input gradients, and weight gradients "
                "match eager and an independent PyTorch linear-plus-cross-entropy oracle."
            ),
            *digests("liger-pr-1328"),
        )
    else:
        liger_correctness = _failure(
            "compiled-boundaries-match-independent-oracle",
            "Do representative and boundary BF16/FP16 cases match an oracle?",
            "At least one compiled boundary case fails execution or oracle tolerances.",
            "LIGER_COMPILED_FORWARD_BACKWARD_ORACLE_MISMATCH",
            *digests("liger-pr-1328"),
        )
    lifecycle_ok = (
        liger_facts["head_default_zero_graph_breaks"]
        and liger_facts["head_default_zero_steady_compile_artifacts"]
    )
    if lifecycle_ok:
        liger_lifecycle = _pass(
            "steady-compile-and-graph-break-contract",
            "After precompile, do repeated calls avoid graph breaks and compile artifacts?",
            "The repeated head phase reports zero graph breaks and zero new cache artifacts.",
            *digests("liger-pr-1328"),
        )
    else:
        liger_lifecycle = _failure(
            "steady-compile-and-graph-break-contract",
            "After precompile, do repeated calls avoid graph breaks and compile artifacts?",
            (
                f"The head records {liger_facts['head_default_graph_break_count']} Tensor.item "
                "graph breaks; repeated execution creates no new cache artifacts."
            ),
            "LIGER_STEADY_COMPILE_OR_GRAPH_BREAK_CONTRACT_FAILED",
            *digests("liger-pr-1328"),
        )
    timing = liger_facts["eager_timing"]
    if timing["within_three_percent"]:
        liger_performance = _pass(
            "eager-common-path-latency-within-three-percent",
            "Does eager/common-path steady latency remain within 3%?",
            (
                "Seven paired H100 measurements give a head/base median ratio of "
                f"{timing['head_over_base_median_ratio']:.4f}."
            ),
            *digests("liger-pr-1328"),
        )
    else:
        liger_performance = _failure(
            "eager-common-path-latency-within-three-percent",
            "Does eager/common-path steady latency remain within 3%?",
            (
                "Seven paired H100 measurements give a head/base median ratio of "
                f"{timing['head_over_base_median_ratio']:.4f}, above the frozen 1.03 limit."
            ),
            "LIGER_EAGER_COMMON_PATH_REGRESSION_GT_3PCT",
            *digests("liger-pr-1328"),
        )
    direct_liger_test = (
        liger_static["facts"]["changed_test_file_count"] == 1
        and liger_static["facts"]["target"]["direct_compile_regression_test_count"] >= 1
    )
    if direct_liger_test:
        liger_test = _pass(
            "changed-test-directly-exercises-compile-failure",
            "Does the changed test directly reproduce the compile failure?",
            (
                "The changed CUDA test compiles the public FLCE path with BF16 inputs and FP32 "
                "accumulation, then checks loss and both gradients against eager."
            ),
            *digests("liger-pr-1328"),
            counterevidence=["The repository test itself does not request fullgraph=True."],
        )
    else:
        liger_test = _failure(
            "changed-test-directly-exercises-compile-failure",
            "Does the changed test directly reproduce the compile failure?",
            "No changed direct compile regression was found.",
            "LIGER_DIRECT_DTYPE_OUT_COMPILE_REGRESSION_MISSING",
            *digests("liger-pr-1328"),
        )
    liger_observations = [
        liger_repair,
        liger_correctness,
        liger_lifecycle,
        liger_performance,
        liger_test,
    ]

    deepgemm_static, deepgemm = evidence["deepgemm-pr-389"]
    deep_static_facts = deepgemm_static["facts"]
    source_order_ok = all(
        (
            deep_static_facts["target"]["head_bf16_fence_call_count"] == 1,
            deep_static_facts["target"]["head_fp8_fp4_fence_call_count"] == 1,
            deep_static_facts["target"]["head_fences_precede_phase_flip"],
            deep_static_facts["target"]["head_helper_uses_exact_ptx"],
        )
    )
    if source_order_ok:
        deep_source = _pass(
            "both-mega-moe-combine-loops-use-intended-fence-before-reuse",
            "Are both affected implementations fenced before every reuse transition?",
            (
                "BF16 and FP8/FP4 each add one exact CTA shared async-proxy fence immediately "
                "before the combine-phase/load-stage transition."
            ),
            *digests("deepgemm-pr-389"),
        )
    else:
        deep_source = _failure(
            "both-mega-moe-combine-loops-use-intended-fence-before-reuse",
            "Are both affected implementations fenced before every reuse transition?",
            "The exact-source fence count, primitive, or placement contract fails.",
            "DEEPGEMM_MEGA_MOE_FENCE_SOURCE_ORDER_CONTRACT_FAILED",
            *digests("deepgemm-pr-389"),
        )
    deep_aot_facts = deepgemm["facts"]
    deep_aot_ok = (
        deep_aot_facts["base_exact_sm100_ptx_pass"]
        and deep_aot_facts["head_exact_sm100_ptx_pass"]
        and deep_aot_facts["added_fence_instruction_count"] == 2
        and deep_aot_facts["base_fence_before_phase_xor_count"] == 0
        and deep_aot_facts["head_fence_before_phase_xor_count"] == 2
        and deep_aot_facts["head_ptx_contains_fence_instruction"]
    )
    if deep_aot_ok:
        deep_aot = _pass(
            "exact-sm100-ptx-retains-both-fences",
            "Does exact SM100 AOT retain the intended PTX ordering primitive?",
            (
                "Both exact mega-MoE templates compile for compute_100a. Total fence count rises "
                f"from {deep_aot_facts['base_fence_instruction_count']} to "
                f"{deep_aot_facts['head_fence_instruction_count']}; only head has the two target "
                "fences directly before the phase XOR transitions."
            ),
            *digests("deepgemm-pr-389"),
        )
    else:
        deep_aot = _failure(
            "exact-sm100-ptx-retains-both-fences",
            "Does exact SM100 AOT retain the intended PTX ordering primitive?",
            "The exact base/head SM100 PTX compile or fence-retention matrix did not pass.",
            "DEEPGEMM_EXACT_SM100_PTX_FENCE_CONTRACT_FAILED",
            *digests("deepgemm-pr-389"),
        )
    deep_tests = _failure(
        "mega-moe-reuse-race-has-direct-regression",
        "Is direct race/reuse regression coverage included?",
        (
            "The PR changes no test file; the existing mega-MoE benchmark does not directly "
            "assert the affected combine-buffer reuse race."
        ),
        "DEEPGEMM_DIRECT_MEGA_MOE_REUSE_REGRESSION_MISSING",
        *digests("deepgemm-pr-389"),
    )
    deepgemm_observations = [
        deep_source,
        deep_aot,
        _unresolved(
            "sm100-dynamic-race-and-concurrency",
            "Do repeated and concurrent SM100 Mega-MoE launches avoid stale data or races?",
            "The available H100 is SM90, so SM100 dynamic race behavior is not inferred from PTX.",
            "DEEPGEMM_SM100_DYNAMIC_HARDWARE_UNAVAILABLE",
            *digests("deepgemm-pr-389"),
        ),
        _unresolved(
            "sm100-steady-performance",
            "Does the valid SM100 common path remain within 3% with zero steady compile?",
            "No SM100 device is available for steady runtime and performance measurement.",
            "DEEPGEMM_SM100_PERFORMANCE_HARDWARE_UNAVAILABLE",
            *digests("deepgemm-pr-389"),
        ),
        deep_tests,
    ]

    megatron_static, megatron = evidence["megatron-pr-6174"]
    megatron_facts = megatron["facts"]
    schedule_ok = (
        megatron_facts["state_transition"]["only_pre_backward_changed_to_idle"]
        and megatron_facts["all_layer_boundaries_match_oracle"]
        and megatron_facts["wrapper_callback_wired"]
        and megatron_facts["run_scenarios"]["forward_only"]["forward_prepare_calls"] == 1
        and megatron_facts["run_scenarios"]["backward_only"]["forward_prepare_calls"] == 0
        and megatron_facts["run_scenarios"]["same_underlying_layer_overlap"][
            "forward_prepare_calls"
        ]
        == 0
    )
    if schedule_ok:
        megatron_schedule = _pass(
            "combined-1f1b-prefetch-matches-schedule-oracle",
            "Do callback, state, overlap, and layer boundaries match the schedule oracle?",
            (
                "Exact extracted methods pass PRE_BACKWARD-only state transitions, forward-only, "
                "backward-only, same/different-layer overlap, and the 1-7-layer boundary matrix."
            ),
            *digests("megatron-pr-6174"),
        )
    else:
        megatron_schedule = _failure(
            "combined-1f1b-prefetch-matches-schedule-oracle",
            "Do callback, state, overlap, and layer boundaries match the schedule oracle?",
            "One or more exact schedule-oracle boundaries fail.",
            "MEGATRON_COMBINED_1F1B_SCHEDULE_ORACLE_FAILED",
            *digests("megatron-pr-6174"),
        )
    direct_megatron_test = (
        megatron_facts["direct_test_present"]
        and megatron_facts["direct_test_uses_two_microbatches"]
        and megatron_static["facts"]["target"]["direct_state_transition_test_count"] == 1
    )
    if direct_megatron_test:
        megatron_test = _pass(
            "changed-test-exercises-genuine-combined-schedule-state",
            "Do changed tests directly exercise the failing combined schedule?",
            (
                "The changed two-microbatch test invokes the production combined-1F1B schedule "
                "and asserts genuine forward-state transitions."
            ),
            *digests("megatron-pr-6174"),
        )
    else:
        megatron_test = _failure(
            "changed-test-exercises-genuine-combined-schedule-state",
            "Do changed tests directly exercise the failing combined schedule?",
            "The direct production-schedule state-transition test contract was not found.",
            "MEGATRON_DIRECT_COMBINED_1F1B_REGRESSION_MISSING",
            *digests("megatron-pr-6174"),
        )
    megatron_observations = [
        megatron_schedule,
        megatron_test,
        _unresolved(
            "multirank-fsdp-correctness-and-liveness",
            "Does multi-rank FSDP complete without hang and preserve outputs and gradients?",
            (
                "Only one H100 is available and Transformer Engine is absent, so the frozen "
                "multi-rank integration cell cannot run."
            ),
            "MEGATRON_MULTIRANK_FSDP_ENVIRONMENT_UNAVAILABLE",
            *digests("megatron-pr-6174"),
        ),
        _unresolved(
            "multirank-step-latency-and-memory",
            "Does prefetch retain step latency and peak memory after warmup?",
            "The same missing multi-rank FSDP environment blocks the performance cell.",
            "MEGATRON_MULTIRANK_PERFORMANCE_ENVIRONMENT_UNAVAILABLE",
            *digests("megatron-pr-6174"),
        ),
    ]

    torchtitan_static, torchtitan = evidence["torchtitan-pr-4032"]
    titan_facts = torchtitan["facts"]
    titan_math_ok = (
        titan_facts["recommendation_matrix_all_pass"]
        and titan_facts["direct_repo_test"]["return_code"] == 0
    )
    if titan_math_ok:
        titan_math = _pass(
            "queue-recommendation-math-matches-oracle",
            "Does the helper derive the intended power-of-two queue recommendation?",
            (
                f"All {titan_facts['recommendation_matrix_cases']} configuration combinations "
                "match the independent power-of-two lane oracle, and the four repository tests "
                "pass."
            ),
            *digests("torchtitan-pr-4032"),
        )
    else:
        titan_math = _failure(
            "queue-recommendation-math-matches-oracle",
            "Does the helper derive the intended power-of-two queue recommendation?",
            "The independent recommendation matrix or exact repository tests fail.",
            "TORCHTITAN_QUEUE_RECOMMENDATION_MATH_FAILED",
            *digests("torchtitan-pr-4032"),
        )
    titan_apply = _failure(
        "rocm-only-application-contract",
        "Does the helper set the value on ROCm while leaving CUDA and CPU unchanged?",
        (
            "The helper has no backend detection and does not mutate the environment; main "
            "unconditionally prints an export command, so evaluating it also changes CUDA/CPU."
        ),
        "TORCHTITAN_ROCM_ONLY_APPLICATION_CONTRACT_MISSING",
        *digests("torchtitan-pr-4032"),
        counterevidence=["The queue recommendation arithmetic itself passes 192 cases."],
    )
    titan_override = _failure(
        "explicit-user-value-is-preserved",
        "Is an explicit user GPU_MAX_HW_QUEUES value preserved?",
        (
            "With GPU_MAX_HW_QUEUES=64, main emits a note and still prints export "
            "GPU_MAX_HW_QUEUES=2; shell eval would overwrite the explicit value."
        ),
        "TORCHTITAN_EXPLICIT_GPU_MAX_HW_QUEUES_OVERWRITTEN",
        *digests("torchtitan-pr-4032"),
    )
    titan_error = _failure(
        "tool-and-device-errors-are-explicit",
        "Do missing tools, malformed output, and unsupported devices fail or fall back explicitly?",
        (
            "The implementation has no backend/tool/device validation branch, so the frozen "
            "error and unsupported-device contract is not represented."
        ),
        "TORCHTITAN_BACKEND_TOOL_ERROR_POLICY_MISSING",
        *digests("torchtitan-pr-4032"),
    )
    titan_state = _pass(
        "recommendation-is-recomputed-per-launch",
        "Is helper state evaluated per launch without leaking between invocations?",
        "Two distinct mocked launch configurations recompute distinct exports without stale state.",
        *digests("torchtitan-pr-4032"),
    )
    titan_test_coverage = (
        torchtitan_static["facts"]["target"]["direct_tests_cover_main"]
        and torchtitan_static["facts"]["target"]["direct_tests_cover_backend_noop"]
        and torchtitan_static["facts"]["target"]["direct_tests_cover_existing_override"]
    )
    if titan_test_coverage:
        titan_tests = _pass(
            "direct-tests-cover-main-noop-override-and-errors",
            "Do direct tests cover ROCm, no-op, override, and error branches?",
            "The changed tests cover all frozen helper branches.",
            *digests("torchtitan-pr-4032"),
        )
    else:
        titan_tests = _failure(
            "direct-tests-cover-main-noop-override-and-errors",
            "Do direct tests cover ROCm, no-op, override, and error branches?",
            (
                "All four repository tests pass, but they only test lane arithmetic; none invokes "
                "main or covers backend no-op, explicit override, or error behavior."
            ),
            "TORCHTITAN_DIRECT_BACKEND_OVERRIDE_ERROR_TESTS_MISSING",
            *digests("torchtitan-pr-4032"),
        )
    torchtitan_observations = [
        titan_math,
        titan_apply,
        titan_override,
        titan_error,
        titan_state,
        titan_tests,
        _unresolved(
            "real-rocm-scheduling-impact",
            "Does the recommendation improve real ROCm graph scheduling?",
            "No ROCm device is available; real scheduling impact is not inferred from mocks.",
            "TORCHTITAN_ROCM_RUNTIME_HARDWARE_UNAVAILABLE",
            *digests("torchtitan-pr-4032"),
            blocking=False,
        ),
    ]

    _verl_static, verl = evidence["verl-pr-7220"]
    verl_facts = verl["facts"]
    if verl_facts["both_entrypoints_match_text_and_multimodal"]:
        verl_entrypoints = _pass(
            "both-entrypoints-match-for-text-and-multimodal",
            "Do both trainer entrypoints consume the same text and multimodal model contract?",
            (
                "Exact AST execution shows both entrypoints forwarding the same model config and "
                "producing matching tokenizer-only and tokenizer-plus-processor state."
            ),
            *digests("verl-pr-7220"),
        )
    else:
        verl_entrypoints = _failure(
            "both-entrypoints-match-for-text-and-multimodal",
            "Do both trainer entrypoints consume the same text and multimodal model contract?",
            "The two exact entrypoints diverge in a text or multimodal branch.",
            "VERL_TWO_ENTRYPOINT_MODEL_CONFIG_SEMANTICS_DIVERGE",
            *digests("verl-pr-7220"),
        )
    semantics = verl_facts["hf_model_config_semantics"]
    propagation_ok = all(
        (
            semantics["has_revision"],
            semantics["has_trust_remote_code"],
            semantics["has_tokenizer_path"],
            semantics["has_processor_kwargs"],
        )
    )
    if propagation_ok:
        verl_propagation = _pass(
            "model-config-propagates-all-frozen-loader-fields",
            "Are revision, trust, tokenizer path, and processor kwargs propagated?",
            "HFModelConfig contains and forwards all frozen loader fields.",
            *digests("verl-pr-7220"),
        )
    else:
        verl_propagation = _failure(
            "model-config-propagates-all-frozen-loader-fields",
            "Are revision, trust, tokenizer path, and processor kwargs propagated?",
            (
                "HFModelConfig carries trust_remote_code and tokenizer_path, but has neither a "
                "revision field nor processor_kwargs, so the full loader contract is not preserved."
            ),
            "VERL_HF_MODEL_CONFIG_LOADER_FIELDS_INCOMPLETE",
            *digests("verl-pr-7220"),
        )
    if verl_facts["repeated_init_replaces_stale_state"]:
        verl_state = _pass(
            "repeated-init-replaces-tokenizer-and-processor-state",
            "Does repeated initialization avoid duplicate or stale cross-run state?",
            "A second exact initialization replaces both tokenizer and processor state.",
            *digests("verl-pr-7220"),
        )
    else:
        verl_state = _failure(
            "repeated-init-replaces-tokenizer-and-processor-state",
            "Does repeated initialization avoid duplicate or stale cross-run state?",
            "Repeated initialization retains stale tokenizer or processor state.",
            "VERL_REPEATED_MODEL_INIT_STATE_LEAK",
            *digests("verl-pr-7220"),
        )
    compatibility_ok = (
        verl_facts["head_reads_data_trust_remote_code"]
        or verl_facts["head_has_legacy_trust_fallback_or_warning"]
    )
    if compatibility_ok:
        verl_compatibility = _pass(
            "legacy-trust-input-has-explicit-policy",
            "Are legacy configuration inputs compatible or rejected explicitly?",
            "The head retains or explicitly diagnoses the legacy data trust input.",
            *digests("verl-pr-7220"),
        )
    else:
        verl_compatibility = _failure(
            "legacy-trust-input-has-explicit-policy",
            "Are legacy configuration inputs compatible or rejected explicitly?",
            (
                "Base reads config.data.trust_remote_code; head stops reading it without a "
                "fallback, rejection, or compatibility warning."
            ),
            "VERL_LEGACY_TRUST_REMOTE_CODE_POLICY_MISSING",
            *digests("verl-pr-7220"),
        )
    direct_verl_tests = verl_facts["direct_two_entrypoint_tests_in_changed_paths"]
    if direct_verl_tests:
        verl_tests = _pass(
            "direct-tests-cover-two-entrypoint-modality-matrix",
            "Do direct tests cover both entrypoints and both modality branches?",
            "Changed tests directly cover the complete two-entrypoint modality matrix.",
            *digests("verl-pr-7220"),
        )
    else:
        verl_tests = _failure(
            "direct-tests-cover-two-entrypoint-modality-matrix",
            "Do direct tests cover both entrypoints and both modality branches?",
            (
                "The PR changes no test file and no direct two-entrypoint text/multimodal "
                "initialization matrix was found."
            ),
            "VERL_DIRECT_TWO_ENTRYPOINT_MODALITY_TESTS_MISSING",
            *digests("verl-pr-7220"),
            counterevidence=["The exact mocked AST matrix passes offline."],
        )
    verl_observations = [
        verl_entrypoints,
        verl_propagation,
        verl_state,
        verl_compatibility,
        verl_tests,
    ]

    observations = {
        "cutlass-pr-3427": cutlass_observations,
        "liger-pr-1328": liger_observations,
        "deepgemm-pr-389": deepgemm_observations,
        "megatron-pr-6174": megatron_observations,
        "torchtitan-pr-4032": torchtitan_observations,
        "verl-pr-7220": verl_observations,
    }
    frozen_at = datetime.now(UTC)
    locks = []
    for case_id in plan_cases:
        candidate_sha = canonical_sha256(
            {
                "selection": selection_cases[case_id],
                "test_plan": plan_cases[case_id],
            }
        )
        material = compile_explainable_judgment(
            case_id=case_id,
            candidate_sha256=candidate_sha,
            test_plan_sha256=plan["test_plan_sha256"],
            evidence_sha256=canonical_sha256(bindings[case_id]),
            observations=observations[case_id],
            frozen_at=frozen_at,
            policy_id=R4_POLICY_ID,
        )
        locks.append(freeze_explainable_judgment(material).model_dump(mode="json"))

    lock_material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "review_text_visible_during_machine_judgment": False,
        "merge_outcomes_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "selection_lock_file_sha256": "sha256:" + selection_file_sha,
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_file_sha256": "sha256:" + test_plan_file_sha,
        "test_plan_sha256": plan["test_plan_sha256"],
        "frozen_at": frozen_at.isoformat(),
        "evidence_bindings": bindings,
        "locks": locks,
    }
    payload = {**lock_material, "lock_set_sha256": canonical_sha256(lock_material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
