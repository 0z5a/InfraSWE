#!/usr/bin/env python3
"""Freeze R6 cross-project machine judgments before outcome/review reveal."""

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

EXPECTED_SELECTION_FILE_SHA256 = "a9f3647c23d32a5a55097548e5fa69f6ecc292aa57c767413a9de951e817419d"
EXPECTED_TEST_PLAN_FILE_SHA256 = "2844b23b3075f062d01dbe7cad9cc0d898102a47460779802fc96fde572d223c"


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
        raise SystemExit("R6 selection-lock file digest mismatch")
    if test_plan_file_sha != EXPECTED_TEST_PLAN_FILE_SHA256:
        raise SystemExit("R6 test-plan file digest mismatch")

    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_material = selection["selection_material"]
    if selection_material["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R6 selection does not assert hidden review text")
    if plan["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R6 plan does not assert hidden review text")
    if plan["review_text_requested"] is not False:
        raise SystemExit("R6 plan says review text was requested before judgment")
    if plan["frozen_before_source_diff_content_inspection"] is not True:
        raise SystemExit("R6 plan was not frozen before source inspection")
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R6 embedded selection lock digest mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit("R6 embedded test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R6 test plan is not bound to the selection lock")

    selection_cases = {item["case_id"]: item for item in selection_material["cases"]}
    plan_cases = {item["case_id"]: item for item in plan["cases"]}
    if set(selection_cases) != set(plan_cases):
        raise SystemExit("R6 selection and test-plan case sets differ")
    for case_id in selection_cases:
        if selection_cases[case_id]["head_sha"] != plan_cases[case_id]["head_sha"]:
            raise SystemExit(f"R6 head SHA mismatch for {case_id}")

    paths: dict[str, list[Path]] = {
        "cutlass-pr-2275": [
            args.result_root / "static/cutlass-pr-2275.json",
            args.result_root / "probes/cutlass-pr-2275-k128-v3.json",
            args.result_root / "probes/cutlass-pr-2275-mixed-k.json",
        ],
        "liger-pr-804": [
            args.result_root / "static/liger-pr-804.json",
            args.result_root / "probes/liger-pr-804-int64.json",
        ],
        "deepgemm-pr-55": [
            args.result_root / "static/deepgemm-pr-55.json",
            args.result_root / "probes/deepgemm-pr-55-cuda-graph.json",
        ],
        "megatron-pr-5608": [
            args.result_root / "static/megatron-pr-5608.json",
            args.result_root / "probes/megatron-pr-5608-runtime-shapes-confirm.json",
        ],
        "torchtitan-pr-2717": [
            args.result_root / "static/torchtitan-pr-2717.json",
            args.result_root / "probes/torchtitan-pr-2717-router.json",
        ],
        "verl-pr-1688": [
            args.result_root / "static/verl-pr-1688.json",
            args.result_root / "probes/verl-pr-1688-temperature.json",
        ],
    }
    evidence = {
        case_id: [_read(path) for path in case_paths] for case_id, case_paths in paths.items()
    }
    bindings = {
        case_id: {
            str(path.relative_to(args.result_root)): canonical_sha256(payload)
            for path, payload in zip(paths[case_id], evidence[case_id], strict=True)
        }
        for case_id in paths
    }

    def digests(case_id: str) -> list[str]:
        return list(bindings[case_id].values())

    cutlass_static, cutlass, _cutlass_mixed = evidence["cutlass-pr-2275"]
    cutlass_timing = cutlass["facts"]["timing"]
    cutlass_observations = [
        _pass(
            "k128-final-scale-tile-is-guarded",
            "Does the head guard the final scale load for a one-tile K dimension?",
            (
                "The exact SM90 FP8 blockwise collective changes from zero to one "
                "k_tile_count == 1 guard, and both exact base and head AOT builds completed."
            ),
            *digests("cutlass-pr-2275"),
        ),
        _pass(
            "head-k128-and-neighbor-execute-correctly",
            "Do head K=128 and neighboring K=256 runs match the CUTLASS reference?",
            (
                "Head K=128, head K=256, and the documented mixed set with four K=128 "
                "problems all report Disposition: Passed."
            ),
            *digests("cutlass-pr-2275"),
            counterevidence=[
                "Both the uniform and mixed base controls also pass on this H100, so this "
                "environment does not dynamically distinguish the claimed defect."
            ],
        ),
        _pass(
            "neighbor-latency-within-three-percent",
            "Does K=256 retain steady latency after AOT compilation?",
            (
                "The seven-pair K=256 median head/base ratio is "
                f"{cutlass_timing['head_over_base_median_ratio']:.4f}, with no JIT path."
            ),
            *digests("cutlass-pr-2275"),
        ),
        _failure(
            "changed-collective-has-direct-k128-regression",
            "Is the exact changed collective protected by a direct K=128 regression test?",
            (
                "The PR changes no test file. K=128 appears in example documentation, but the "
                "changed collective has no new direct regression that would retain this fix."
            ),
            "CUTLASS_DIRECT_K128_REGRESSION_TEST_MISSING",
            *digests("cutlass-pr-2275"),
            counterevidence=[
                f"The offline H100 head run passes; changed_test_file_count="
                f"{cutlass_static['facts']['changed_test_file_count']}."
            ],
        ),
    ]

    liger_static, liger = evidence["liger-pr-804"]
    boundary = liger["facts"]["boundary_results"]
    liger_ratio = liger["facts"]["latency"]["head_over_base_median_ratio"]
    liger_observations = [
        _pass(
            "all-changed-offsets-widen-program-id-before-multiplication",
            "Are all changed RMSNorm and RoPE offsets widened before multiplication?",
            "The exact head contains two RMSNorm and one RoPE int64 program-ID casts.",
            *digests("liger-pr-804"),
        ),
        _pass(
            "large-offset-arithmetic-crosses-int32-boundary",
            "Do forward/RoPE and RMSNorm-backward offsets avoid int32 wraparound?",
            (
                f"At program {boundary[0]['target_program']} the base wraps to "
                f"{boundary[0]['base']}, while both head paths produce the expected "
                f"{boundary[0]['expected']}."
            ),
            *digests("liger-pr-804"),
        ),
        _pass(
            "ordinary-range-behavior-and-latency-are-retained",
            "Are common-range outputs and steady latency retained after precompile?",
            (
                f"Common outputs are equal, the seven-pair head/base ratio is {liger_ratio:.4f}, "
                "and steady timing creates zero cache files."
            ),
            *digests("liger-pr-804"),
        ),
        _failure(
            "both-kernel-families-have-direct-large-offset-tests",
            "Do direct tests cover large-offset RMSNorm and RoPE?",
            (
                "The PR changes no test file and the exact head has no existing large-offset "
                "RMSNorm or RoPE match, so both changed families lack retained boundary coverage."
            ),
            "LIGER_RMSNORM_AND_ROPE_LARGE_OFFSET_TESTS_MISSING",
            *digests("liger-pr-804"),
            counterevidence=[
                f"Offline arithmetic and common-range timing pass; changed_test_file_count="
                f"{liger_static['facts']['changed_test_file_count']}."
            ],
        ),
    ]

    deepgemm_static, deepgemm = evidence["deepgemm-pr-55"]
    graph = deepgemm["facts"]["cuda_graph"]
    deepgemm_observations = [
        _pass(
            "head-cuda-graph-capture-and-replay-completes",
            "Can the exact changed template mapping capture once and replay repeatedly?",
            (
                "Head capture and replay pass with the expected changed replay output and an "
                "unchanged tensor pointer."
            ),
            *digests("deepgemm-pr-55"),
            counterevidence=[
                f"The base control also reports {graph['base']['status']}, so the target failure "
                "is not reproduced in this environment."
            ],
        ),
        _pass(
            "common-ctype-and-stream-mapping-is-retained",
            "Are supported tensors, scalars, and CUDA streams still mapped correctly?",
            (
                "Common tensor/scalar mappings pass, the generated runtime source is unchanged, "
                "and the runtime dtype assertion remains present."
            ),
            *digests("deepgemm-pr-55"),
        ),
        _failure(
            "map-ctype-retains-unsupported-dtype-rejection",
            "Does the new capability mapping preserve ordinary unsupported-dtype validation?",
            (
                "Head map_ctype accepts an unregistered int8 pointer and float16 even though the "
                "generator map lacks float16, broadening the base mapping contract without a "
                "direct validation test."
            ),
            "DEEPGEMM_UNSUPPORTED_POINTER_DTYPE_VALIDATION_BROADENED",
            *digests("deepgemm-pr-55"),
            counterevidence=["The unchanged generated runtime still contains a dtype assertion."],
        ),
        _failure(
            "changed-template-has-direct-cuda-graph-test",
            "Is capture/replay protected by a direct lifecycle regression?",
            (
                "The PR changes no test file; existing Capture tests capture stdout rather than a "
                "CUDA graph, so the changed lifecycle has no direct retained coverage."
            ),
            "DEEPGEMM_DIRECT_CUDA_GRAPH_REGRESSION_TEST_MISSING",
            *digests("deepgemm-pr-55"),
            counterevidence=[
                f"The offline graph probe passes; changed_test_file_count="
                f"{deepgemm_static['facts']['changed_test_file_count']}."
            ],
        ),
    ]

    megatron_static, megatron = evidence["megatron-pr-5608"]
    compile_reduction = megatron["facts"]["runtime_shape_compile_reduction"]
    megatron_latency = megatron["facts"]["latency"]
    megatron_observations = [
        _pass(
            "runtime-batch-specialization-is-reduced",
            "Does the head reduce compilation caused by runtime batch-size variation?",
            (
                "All tested outputs match and runtime-shape variation drops new cubins from "
                f"{compile_reduction['base_new_compiled_kernels']} to "
                f"{compile_reduction['head_new_compiled_kernels']}."
            ),
            *digests("megatron-pr-5608"),
        ),
        _pass(
            "stable-shape-launches-create-zero-cache-files",
            "After precompile, do stable launches avoid steady compilation?",
            "The confirmed stable-shape timing phase creates zero new Triton cache artifacts.",
            *digests("megatron-pr-5608"),
        ),
        _failure(
            "steady-latency-is-within-three-percent",
            "Does the common stable-shape path stay within the frozen 3% latency limit?",
            (
                "The confirmed seven-pair medians are "
                f"{megatron_latency['base_median_us']:.3f} us base and "
                f"{megatron_latency['head_median_us']:.3f} us head, ratio "
                f"{megatron_latency['head_over_base_median_ratio']:.4f}."
            ),
            "MEGATRON_STEADY_LATENCY_REGRESSION_GT_3PCT",
            *digests("megatron-pr-5608"),
            counterevidence=["The patch reduces runtime-shape compilation variants by two thirds."],
        ),
        _failure(
            "production-lifecycle-has-direct-test",
            "Is compile/autotune lifecycle behavior directly covered?",
            (
                "Existing tests check tensor-operation outputs, but the PR changes no test file "
                "and no direct cache/compile lifecycle assertion was found."
            ),
            "MEGATRON_DIRECT_COMPILE_LIFECYCLE_TEST_MISSING",
            *digests("megatron-pr-5608"),
            counterevidence=[
                f"Offline cache accounting passes; changed_test_file_count="
                f"{megatron_static['facts']['changed_test_file_count']}."
            ],
        ),
    ]

    torchtitan_static, torchtitan = evidence["torchtitan-pr-2717"]
    router_perf = torchtitan["facts"]["performance"]
    noncontiguous = torchtitan["facts"]["noncontiguous_behavior"]
    torchtitan_observations = [
        _pass(
            "bf16-forward-and-training-gradients-match-oracle",
            "Do BF16 forward outputs and required gradients match the eager oracle?",
            (
                "All representative and non-power-of-two BF16 cases pass forward, routed-input "
                "gradient, and score-gradient tolerances."
            ),
            *digests("torchtitan-pr-2717"),
        ),
        _pass(
            "fused-router-improves-h100-latency-and-memory",
            "Does the fused path improve steady H100 performance after precompile?",
            (
                f"Median latency falls from {router_perf['base_median_us']:.3f} us to "
                f"{router_perf['head_median_us']:.3f} us (ratio "
                f"{router_perf['head_over_base_median_ratio']:.4f}), peak allocation falls, "
                "and steady compilation is zero."
            ),
            *digests("torchtitan-pr-2717"),
        ),
        _failure(
            "layout-assumption-is-explicitly-gated",
            "Are unsupported layouts rejected or routed to a safe fallback?",
            (
                "The head has no contiguity gate and silently executes a noncontiguous BF16 "
                f"input with max absolute error {noncontiguous['error']['max_abs']:.3f}."
            ),
            "TORCHTITAN_ROUTER_CONTIGUITY_GATE_MISSING",
            *digests("torchtitan-pr-2717"),
            counterevidence=["Contiguous BF16 forward and gradient cases pass."],
        ),
        _failure(
            "fused-router-and-call-gate-have-direct-tests",
            "Are the fused kernel and its call-site gate protected by direct tests?",
            "The PR changes no test file and no direct fused-router test was found.",
            "TORCHTITAN_DIRECT_FUSED_ROUTER_TESTS_MISSING",
            *digests("torchtitan-pr-2717"),
            counterevidence=[
                f"Offline correctness and performance pass; changed_test_file_count="
                f"{torchtitan_static['facts']['changed_test_file_count']}."
            ],
        ),
    ]

    verl_static, verl = evidence["verl-pr-1688"]
    branches = verl["facts"]["branch_results"]
    verl_observations = [
        _pass(
            "temperature-keyword-is-capability-gated",
            "Does head pass temperature only on the fused path?",
            (
                f"Head fused is {branches['head_fused']['status']} with temperature and head "
                f"non-fused is {branches['head_nonfused']['status']} without it; the base "
                "non-fused control rejects the unexpected keyword."
            ),
            *digests("verl-pr-1688"),
        ),
        _pass(
            "other-keywords-and-per-call-locality-are-retained",
            "Are other call semantics preserved without stale capability state?",
            (
                "All other keywords are preserved, both exact head call sites expand the gate, "
                "and the gate dictionary is created per call."
            ),
            *digests("verl-pr-1688"),
        ),
        _failure(
            "both-call-signature-branches-have-direct-tests",
            "Are fused and non-fused signatures both protected by direct tests?",
            (
                "The PR changes no test file and no exact two-branch signature test was found, "
                "so the one-sided keyword regression is not retained in the repository suite."
            ),
            "VERL_DIRECT_TWO_BRANCH_CALL_SIGNATURE_TEST_MISSING",
            *digests("verl-pr-1688"),
            counterevidence=[
                f"The exact AST offline probe passes both head branches; changed_test_file_count="
                f"{verl_static['facts']['changed_test_file_count']}."
            ],
        ),
    ]

    observations = {
        "cutlass-pr-2275": cutlass_observations,
        "liger-pr-804": liger_observations,
        "deepgemm-pr-55": deepgemm_observations,
        "megatron-pr-5608": megatron_observations,
        "torchtitan-pr-2717": torchtitan_observations,
        "verl-pr-1688": verl_observations,
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
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
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
