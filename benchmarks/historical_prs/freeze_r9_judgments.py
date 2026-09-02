#!/usr/bin/env python3
"""Freeze R9 case-contract judgments before outcome and review reveal."""

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

EXPECTED_SELECTION_FILE_SHA256 = "c75c205f016dd03a446dc6ddd546fac6b3dc531e79ec4b438a4787e377c1f62c"
EXPECTED_TEST_PLAN_FILE_SHA256 = "0e95ebe34f1f9a841ad155530c8a75e6e162af9763a218cfbf669b3973fffb23"


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
    blocking: bool = False,
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
        raise SystemExit(f"R9 embedded evidence digest mismatch in {path}")
    if payload.get("case_id") != case_id:
        raise SystemExit(f"R9 evidence case mismatch in {path}")
    if payload.get("base_sha") != selected["base_sha"]:
        raise SystemExit(f"R9 evidence base SHA mismatch in {path}")
    if payload.get("head_sha") != selected["head_sha"]:
        raise SystemExit(f"R9 evidence head SHA mismatch in {path}")
    if payload.get("selection_lock_sha256") != selection_sha:
        raise SystemExit(f"R9 evidence selection binding mismatch in {path}")
    if payload.get("test_plan_sha256") != plan_sha:
        raise SystemExit(f"R9 evidence test-plan binding mismatch in {path}")
    if payload.get("path_parity") is not True:
        raise SystemExit(f"R9 exact changed-path parity failed in {path}")
    if payload.get("status") != "pass":
        raise SystemExit(f"R9 probe execution did not complete in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection_file_sha = _file_sha256(args.selection_lock)
    plan_file_sha = _file_sha256(args.test_plan)
    if selection_file_sha != EXPECTED_SELECTION_FILE_SHA256:
        raise SystemExit("R9 selection-lock file digest mismatch")
    if plan_file_sha != EXPECTED_TEST_PLAN_FILE_SHA256:
        raise SystemExit("R9 test-plan file digest mismatch")

    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R9 embedded selection digest mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit("R9 embedded test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R9 test plan is not bound to the selection lock")
    if any(
        item is not False
        for item in (
            selection_material["review_text_visible_to_machine_judge"],
            selection_material["merge_outcomes_visible_to_machine_judge"],
            plan["review_text_visible_to_machine_judge"],
            plan["merge_outcomes_visible_to_machine_judge"],
            plan["review_text_requested"],
        )
    ):
        raise SystemExit("R9 source locks do not preserve the blind boundary")
    if plan["frozen_before_source_diff_content_inspection"] is not True:
        raise SystemExit("R9 plan was not frozen before source inspection")
    if plan["scoring_policy"]["weighted_score_used"] is not False:
        raise SystemExit("R9 unexpectedly enables weighted scoring")
    if plan["scoring_policy"]["forced_polarization_used"] is not False:
        raise SystemExit("R9 unexpectedly enables forced polarization")

    selected = {item["case_id"]: item for item in selection_material["cases"]}
    planned = {item["case_id"]: item for item in plan["cases"]}
    if selected.keys() != planned.keys():
        raise SystemExit("R9 selection and test-plan case sets differ")
    for case_id in selected:
        if selected[case_id]["base_sha"] != planned[case_id]["base_sha"]:
            raise SystemExit(f"R9 base SHA mismatch for {case_id}")
        if selected[case_id]["head_sha"] != planned[case_id]["head_sha"]:
            raise SystemExit(f"R9 head SHA mismatch for {case_id}")

    paths = {case_id: args.result_root / "probes" / f"{case_id}.json" for case_id in selected}
    evidence = {case_id: _read(path) for case_id, path in paths.items()}
    for case_id, payload in evidence.items():
        _validate_evidence(
            payload,
            case_id=case_id,
            selected=selected[case_id],
            selection_sha=selection["selection_lock_sha256"],
            plan_sha=plan["test_plan_sha256"],
            path=paths[case_id],
        )
    bindings = {
        case_id: {
            "path": str(paths[case_id].relative_to(args.result_root)),
            "evidence_sha256": payload["evidence_sha256"],
            "artifact_sha256": canonical_sha256(payload),
        }
        for case_id, payload in evidence.items()
    }

    def digests(case_id: str) -> list[str]:
        binding = bindings[case_id]
        return [binding["evidence_sha256"], binding["artifact_sha256"]]

    cutlass = evidence["cutlass-pr-3332"]["facts"]
    cutlass_observations = [
        _pass(
            "uint64-boundaries-lower-with-exact-two-complement-values",
            "Do 2^63 through UINT64_MAX lower without float conversion or lost bits?",
            (
                "The exact head helper maps 2^63, 2^63+12345, and UINT64_MAX to the "
                "corresponding signed two's-complement MLIR values; the base lacks the helper."
            ),
            *digests("cutlass-pr-3332"),
        )
        if (
            not cutlass["base_target_helper_present"]
            and cutlass["head_target_helper_present"]
            and cutlass["target_outputs"] == [-9223372036854775808, -9223372036854763463, -1]
        )
        else _failure(
            "uint64-boundaries-lower-with-exact-two-complement-values",
            "Do 2^63 through UINT64_MAX lower without float conversion or lost bits?",
            "The exact target boundary matrix does not match the required values.",
            "CUTLASS_UINT64_BOUNDARY_LOWERING_FAILED",
            *digests("cutlass-pr-3332"),
        ),
        _pass(
            "neighbor-and-type-controls-are-preserved",
            "Are neighboring, signed, bool, and wider-integer controls unchanged?",
            "Exact source execution preserves all frozen neighboring and type controls.",
            *digests("cutlass-pr-3332"),
        )
        if (
            cutlass["neighbor_values_preserved"]
            and cutlass["signed_bool_and_128bit_controls_preserved"]
        )
        else _failure(
            "neighbor-and-type-controls-are-preserved",
            "Are neighboring, signed, bool, and wider-integer controls unchanged?",
            "At least one frozen neighboring or type control changes.",
            "CUTLASS_UNSIGNED_LOWERING_NEIGHBOR_REGRESSION",
            *digests("cutlass-pr-3332"),
        ),
        _pass(
            "direct-test-covers-conversion-and-binary-paths",
            "Does the changed regression cover target boundaries and binary operations?",
            "The changed test covers both direct conversion and binary-operation paths.",
            *digests("cutlass-pr-3332"),
        )
        if cutlass["direct_test_covers_conversion_and_binary_paths"]
        else _failure(
            "direct-test-covers-conversion-and-binary-paths",
            "Does the changed regression cover target boundaries and binary operations?",
            "The changed test omits a required target path.",
            "CUTLASS_UINT64_DIRECT_REGRESSION_INCOMPLETE",
            *digests("cutlass-pr-3332"),
        ),
        _unresolved(
            "full-cutedsl-compiler-runtime",
            "Does the complete CuTeDSL compiler retain the same boundary behavior?",
            "The exact helper contract passed; the complete CuTeDSL stack was not installed.",
            "CUTLASS_FULL_CUTEDSL_ENVIRONMENT_UNAVAILABLE",
            *digests("cutlass-pr-3332"),
        ),
    ]

    flashinfer = evidence["flashinfer-pr-3950"]["facts"]
    flashinfer_observations = [
        _pass(
            "checkpoint-restore-repairs-inference-tensor-flag-path",
            "Does the real checkpoint_restore path repair the inference-mode failure?",
            (
                "On torch 2.8.0+cu128, base reproduces the inference-tensor in-place failure; "
                "the exact head checkpoint_restore path completes and restores the flag vector."
            ),
            *digests("flashinfer-pr-3950"),
        )
        if (
            flashinfer["base_checkpoint_restore_reproduces_inference_tensor_failure"]
            and flashinfer["head_checkpoint_restore_passes"]
            and flashinfer["head_flags_match_protocol"]
        )
        else _failure(
            "checkpoint-restore-repairs-inference-tensor-flag-path",
            "Does the real checkpoint_restore path repair the inference-mode failure?",
            "The exact base/head checkpoint lifecycle did not meet the frozen contract.",
            "FLASHINFER_MNNVL_CHECKPOINT_RESTORE_CONTRACT_FAILED",
            *digests("flashinfer-pr-3950"),
        ),
        _pass(
            "restore-preserves-ordinary-repeat-and-mode-lifecycle",
            "Are ordinary tensors, repeated restore, and global inference state preserved?",
            (
                "Ordinary base/head restore passes, repeated restore is a no-op, and the probe "
                "leaves global inference mode restored."
            ),
            *digests("flashinfer-pr-3950"),
        )
        if (
            flashinfer["ordinary_tensor_base_and_head_pass"]
            and flashinfer["repeated_restore_is_noop"]
            and flashinfer["global_inference_mode_restored"]
        )
        else _failure(
            "restore-preserves-ordinary-repeat-and-mode-lifecycle",
            "Are ordinary tensors, repeated restore, and global inference state preserved?",
            "At least one restore lifecycle control fails.",
            "FLASHINFER_MNNVL_RESTORE_LIFECYCLE_REGRESSION",
            *digests("flashinfer-pr-3950"),
        ),
        _pass(
            "changed-and-existing-tests-cover-protocol-and-public-lifecycle",
            "Is the repaired protocol state exercised through a retained public lifecycle?",
            (
                "The changed unit regression directly exercises _initialize_protocol, while the "
                "exact independent probe executes checkpoint_restore end to end."
            ),
            *digests("flashinfer-pr-3950"),
            counterevidence=[
                "The newly changed test does not itself invoke checkpoint_restore; that path is "
                "covered by the frozen independent probe and retained multi-GPU integration test."
            ],
        )
        if flashinfer["direct_new_test_calls_initialize_protocol"]
        else _failure(
            "changed-and-existing-tests-cover-protocol-and-public-lifecycle",
            "Is the repaired protocol state exercised through a retained public lifecycle?",
            "No direct protocol regression was found.",
            "FLASHINFER_MNNVL_PROTOCOL_DIRECT_REGRESSION_MISSING",
            *digests("flashinfer-pr-3950"),
        ),
        _unresolved(
            "real-two-gpu-mnnvl-transport",
            "Does real two-GPU MNNVL transport restore and communicate correctly?",
            "The A100 cell validated PyTorch state transitions but cannot attest MNNVL transport.",
            "FLASHINFER_MNNVL_TRANSPORT_CELL_UNAVAILABLE",
            *digests("flashinfer-pr-3950"),
        ),
    ]

    sglang_validation = evidence["sglang-pr-31346"]["facts"]
    sglang_validation_observations = [
        _pass(
            "tilelang-dsa-fp8-truth-table-is-exact",
            "Is exactly the unsupported CUDA + TileLang DSA + fp8_e4m3 set rejected?",
            (
                "All 96 frozen backend/dtype/platform combinations match the oracle, with "
                f"{sglang_validation['truth_table_rejections']} precise rejections."
            ),
            *digests("sglang-pr-31346"),
        )
        if (
            not sglang_validation["base_validator_present"]
            and sglang_validation["head_validator_present"]
            and sglang_validation["exact_truth_table_pass"]
            and sglang_validation["truth_table_cases"] == 96
        )
        else _failure(
            "tilelang-dsa-fp8-truth-table-is-exact",
            "Is exactly the unsupported CUDA + TileLang DSA + fp8_e4m3 set rejected?",
            "The exact 96-case truth table did not pass.",
            "SGLANG_TILELANG_DSA_FP8_TRUTH_TABLE_FAILED",
            *digests("sglang-pr-31346"),
        ),
        _pass(
            "validation-is-wired-once-into-normal-resolution",
            "Does fail-fast validation run exactly once in normal resolution?",
            "The exact validator has one call in the normal backend-resolution lifecycle.",
            *digests("sglang-pr-31346"),
        )
        if sglang_validation["normal_resolution_lifecycle_call_count"] == 1
        else _failure(
            "validation-is-wired-once-into-normal-resolution",
            "Does fail-fast validation run exactly once in normal resolution?",
            "The validator is missing or duplicated in normal resolution.",
            "SGLANG_TILELANG_DSA_VALIDATION_LIFECYCLE_INVALID",
            *digests("sglang-pr-31346"),
        ),
        _pass(
            "direct-tests-cover-invalid-and-valid-neighbors",
            "Do changed tests cover the rejection and supported neighbors?",
            "Five direct changed tests cover invalid and valid neighboring configurations.",
            *digests("sglang-pr-31346"),
        )
        if sglang_validation["direct_test_methods"] == 5
        else _failure(
            "direct-tests-cover-invalid-and-valid-neighbors",
            "Do changed tests cover the rejection and supported neighbors?",
            "The direct validation matrix is incomplete.",
            "SGLANG_TILELANG_DSA_DIRECT_TEST_MATRIX_INCOMPLETE",
            *digests("sglang-pr-31346"),
        ),
    ]

    sglang_plan = evidence["sglang-pr-31349"]["facts"]
    sglang_plan_observations = [
        _pass(
            "decode-launch-plan-is-deterministic-bounded-and-pure",
            "Is the decode launch set deterministic, bounded, and input-preserving?",
            (
                "All 112 frozen matrices pass twice, the bound is exact, invalid controls fail, "
                "and the input plan is not mutated."
            ),
            *digests("sglang-pr-31349"),
        )
        if (
            not sglang_plan["base_narrow_helper_present"]
            and sglang_plan["head_narrow_helper_present"]
            and sglang_plan["boundary_matrix_pass"]
            and sglang_plan["boundary_matrix_cases"] == 112
            and sglang_plan["same_bound_returns_original"]
            and sglang_plan["input_not_mutated"]
        )
        else _failure(
            "decode-launch-plan-is-deterministic-bounded-and-pure",
            "Is the decode launch set deterministic, bounded, and input-preserving?",
            "The exact deterministic planning matrix did not pass.",
            "SGLANG_FLASHINFER_DECODE_PLAN_CONTRACT_FAILED",
            *digests("sglang-pr-31349"),
        ),
        _pass(
            "narrowing-is-confined-to-intended-cuda-graph-path",
            "Is narrowing guarded by the intended graph/backend conditions?",
            "The integration guard retains split-KV, CUDA-graph, tensor-core, and FA2 conditions.",
            *digests("sglang-pr-31349"),
        )
        if sglang_plan["integration_guard_has_graph_tensorcore_fa2_splitkv_conditions"]
        else _failure(
            "narrowing-is-confined-to-intended-cuda-graph-path",
            "Is narrowing guarded by the intended graph/backend conditions?",
            "One or more required integration guards are absent.",
            "SGLANG_FLASHINFER_DECODE_PLAN_GUARD_INCOMPLETE",
            *digests("sglang-pr-31349"),
        ),
        _pass(
            "direct-tests-cover-plan-boundaries",
            "Do changed tests cover neighboring plan boundaries and invalid controls?",
            (
                f"The changed file contains {sglang_plan['direct_test_methods']} direct test "
                "methods, and all seven invalid controls are rejected."
            ),
            *digests("sglang-pr-31349"),
        )
        if sglang_plan["direct_test_methods"] >= 10
        and sglang_plan["invalid_controls_rejected"] == 7
        else _failure(
            "direct-tests-cover-plan-boundaries",
            "Do changed tests cover neighboring plan boundaries and invalid controls?",
            "The direct boundary/control matrix is incomplete.",
            "SGLANG_FLASHINFER_DECODE_PLAN_TEST_MATRIX_INCOMPLETE",
            *digests("sglang-pr-31349"),
        ),
        _unresolved(
            "actual-cuda-graph-replay",
            "Does a real CUDA graph replay retain correctness and launch bounds?",
            "The exact planning contract passed; this run did not execute CUDA graph replay.",
            "SGLANG_FLASHINFER_CUDA_GRAPH_RUNTIME_UNAVAILABLE",
            *digests("sglang-pr-31349"),
        ),
    ]

    vllm = evidence["vllm-pr-48695"]["facts"]
    vllm_observations = [
        _pass(
            "capability-probe-handles-toolchain-failure-matrix",
            "Does probing fail closed across absent, old, malformed, error, and timeout states?",
            (
                "The exact ten-scenario matrix passes; nvcc calls use check and timeout, and "
                "the quantization scheme consults the gate at both required sites."
            ),
            *digests("vllm-pr-48695"),
        )
        if (
            not vllm["base_fp4_specific_gate_present"]
            and vllm["head_fp4_specific_gate_present"]
            and vllm["scenario_matrix_pass"]
            and vllm["nvcc_calls_use_check_and_timeout"]
            and vllm["quant_scheme_gate_call_count"] == 2
        )
        else _failure(
            "capability-probe-handles-toolchain-failure-matrix",
            "Does probing fail closed across absent, old, malformed, error, and timeout states?",
            "The capability/toolchain matrix or gate integration fails.",
            "VLLM_FP4_CAPABILITY_GATE_MECHANICS_FAILED",
            *digests("vllm-pr-48695"),
        ),
        _failure(
            "artifact-proof-is-fp4-kernel-specific",
            "Does an artifact hit prove the required FP4 kernel specifically?",
            (
                "The head accepts generic FlashInfer cubin or JIT-cache presence as sufficient; "
                "neither proves that the required FP4 MoE kernel is available."
            ),
            "VLLM_FP4_KERNEL_SPECIFIC_ATTESTATION_MISSING",
            *digests("vllm-pr-48695"),
            counterevidence=[
                "Absent, old, malformed, error, and timeout nvcc paths fail closed correctly."
            ],
        )
        if (
            not vllm["generic_cubin_presence_proves_specific_fp4_kernel"]
            or not vllm["generic_jit_cache_presence_proves_specific_fp4_kernel"]
        )
        else _pass(
            "artifact-proof-is-fp4-kernel-specific",
            "Does an artifact hit prove the required FP4 kernel specifically?",
            "Artifact evidence is scoped to the required FP4 kernel.",
            *digests("vllm-pr-48695"),
        ),
        _failure(
            "direct-tests-cover-present-missing-and-partial-capabilities",
            "Do direct tests cover present, missing, and partial installation states?",
            (
                "The changed tests omit the base-capability-missing, JIT-cache, malformed nvcc, "
                "tool error, and timeout branches."
            ),
            "VLLM_FP4_CAPABILITY_BRANCH_TEST_MATRIX_INCOMPLETE",
            *digests("vllm-pr-48695"),
        )
        if not all(
            (
                vllm["direct_test_covers_base_capability_missing"],
                vllm["direct_test_covers_jit_cache_branch"],
                vllm["direct_test_covers_malformed_or_failed_nvcc"],
            )
        )
        else _pass(
            "direct-tests-cover-present-missing-and-partial-capabilities",
            "Do direct tests cover present, missing, and partial installation states?",
            "Direct tests cover the full frozen capability matrix.",
            *digests("vllm-pr-48695"),
        ),
    ]

    observations = {
        "cutlass-pr-3332": cutlass_observations,
        "flashinfer-pr-3950": flashinfer_observations,
        "sglang-pr-31346": sglang_validation_observations,
        "sglang-pr-31349": sglang_plan_observations,
        "vllm-pr-48695": vllm_observations,
    }
    frozen_at = datetime.now(UTC)
    locks = []
    for case_id in planned:
        candidate_sha = canonical_sha256(
            {"selection": selected[case_id], "test_plan": planned[case_id]}
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
        "schema_version": "0.1",
        "protocol_id": "historical-pr-blind-case-contract-v0.1-r9-5",
        "review_text_visible_during_machine_judgment": False,
        "merge_outcomes_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "forced_polarization_used": False,
        "selection_lock_file_sha256": "sha256:" + selection_file_sha,
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_file_sha256": "sha256:" + plan_file_sha,
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
