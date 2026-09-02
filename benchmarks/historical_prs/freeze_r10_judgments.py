#!/usr/bin/env python3
"""Freeze R10 case-contract judgments before outcome and review reveal."""

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

EXPECTED_SELECTION_FILE_SHA256 = "5cffa1ddd1641d7621b75886ea6c22377b0dc9ebcfe2579af0416c4c24d2793f"
EXPECTED_TEST_PLAN_FILE_SHA256 = "ecd92b9ef44a39851febd7fadb8f74f7653d4de62c254291c5bcc1bc87c2c4cd"


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


def _check(
    condition: bool,
    *,
    rule_id: str,
    question: str,
    pass_conclusion: str,
    fail_conclusion: str,
    failure_code: str,
    evidence: list[str],
    counterevidence: list[str] | None = None,
) -> HistoricalHeuristicObservation:
    if condition:
        return _pass(
            rule_id,
            question,
            pass_conclusion,
            *evidence,
            counterevidence=counterevidence,
        )
    return _failure(
        rule_id,
        question,
        fail_conclusion,
        failure_code,
        *evidence,
        counterevidence=counterevidence,
    )


def _unresolved(
    rule_id: str,
    question: str,
    conclusion: str,
    failure_code: str,
    *evidence: str,
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="unresolved",
        blocking=False,
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
        raise SystemExit(f"R10 embedded evidence digest mismatch in {path}")
    if payload.get("case_id") != case_id:
        raise SystemExit(f"R10 evidence case mismatch in {path}")
    if payload.get("base_sha") != selected["base_sha"]:
        raise SystemExit(f"R10 evidence base SHA mismatch in {path}")
    if payload.get("head_sha") != selected["head_sha"]:
        raise SystemExit(f"R10 evidence head SHA mismatch in {path}")
    if payload.get("selection_lock_sha256") != selection_sha:
        raise SystemExit(f"R10 evidence selection binding mismatch in {path}")
    if payload.get("test_plan_sha256") != plan_sha:
        raise SystemExit(f"R10 evidence test-plan binding mismatch in {path}")
    if payload.get("path_parity") is not True:
        raise SystemExit(f"R10 exact changed-path parity failed in {path}")
    if payload.get("status") != "pass":
        raise SystemExit(f"R10 probe execution did not complete in {path}")


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
        raise SystemExit("R10 selection-lock file digest mismatch")
    if plan_file_sha != EXPECTED_TEST_PLAN_FILE_SHA256:
        raise SystemExit("R10 test-plan file digest mismatch")

    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R10 embedded selection digest mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit("R10 embedded test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R10 test plan is not bound to the selection lock")
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
        raise SystemExit("R10 source locks do not preserve the blind boundary")
    if plan["frozen_before_source_diff_content_inspection"] is not True:
        raise SystemExit("R10 plan was not frozen before source inspection")
    if plan["scoring_policy"]["weighted_score_used"] is not False:
        raise SystemExit("R10 unexpectedly enables weighted scoring")
    if plan["scoring_policy"]["forced_polarization_used"] is not False:
        raise SystemExit("R10 unexpectedly enables forced polarization")

    selected = {item["case_id"]: item for item in selection_material["cases"]}
    planned = {item["case_id"]: item for item in plan["cases"]}
    if selected.keys() != planned.keys():
        raise SystemExit("R10 selection and test-plan case sets differ")
    for case_id in selected:
        if selected[case_id]["base_sha"] != planned[case_id]["base_sha"]:
            raise SystemExit(f"R10 base SHA mismatch for {case_id}")
        if selected[case_id]["head_sha"] != planned[case_id]["head_sha"]:
            raise SystemExit(f"R10 head SHA mismatch for {case_id}")

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

    cutlass = evidence["cutlass-pr-3300"]["facts"]
    cutlass_evidence = digests("cutlass-pr-3300")
    cutlass_observations = [
        _check(
            cutlass["native_compile_executed"]
            and cutlass["base_failure_head_success"]
            and cutlass["base_compile"]["returncode"] != 0
            and cutlass["head_compile"]["returncode"] == 0,
            rule_id="self-contained-header-compiles-after-direct-dependency",
            question="Does the exact minimal print_tensor.hpp translation unit compile?",
            pass_conclusion=(
                "CUDA 12.8 nvcc reproduces the base smem_ptr_flag_bits compile failure and "
                "compiles the exact head successfully."
            ),
            fail_conclusion="The exact base/head self-contained compilation contract did not pass.",
            failure_code="CUTLASS_PRINT_TENSOR_SELF_CONTAINED_COMPILE_FAILED",
            evidence=cutlass_evidence,
        ),
        _check(
            cutlass["base_direct_pointer_flagged_include_count"] == 0
            and cutlass["head_direct_pointer_flagged_include_count"] == 1
            and cutlass["head_uses_smem_ptr_flag_bits"]
            and not cutlass["base_manifest_lists_header"]
            and cutlass["head_manifest_lists_header"],
            rule_id="header-dependency-and-self-contained-test-have-one-owner",
            question="Are the declaration owner and self-contained regression wired directly?",
            pass_conclusion=(
                "The head directly includes pointer_flagged.hpp and adds print_tensor.hpp to the "
                "self-contained include manifest; the base has neither."
            ),
            fail_conclusion="Direct include ownership or manifest coverage is incomplete.",
            failure_code="CUTLASS_PRINT_TENSOR_INCLUDE_OWNERSHIP_INCOMPLETE",
            evidence=cutlass_evidence,
        ),
    ]

    deepgemm = evidence["deepgemm-pr-310"]["facts"]
    deepgemm_evidence = digests("deepgemm-pr-310")
    deepgemm_observations = [
        _check(
            not deepgemm["base_head_function_ast_equal"],
            rule_id="precedence-change-alters-executable-predicate",
            question="Does the parenthesization change the executable validation predicate?",
            pass_conclusion="The head predicate has a distinct executable AST from the base.",
            fail_conclusion=(
                "Python parses the base and head pack_ue8m0_to_int assertions to the exact same "
                "AST, so the patch is a semantic no-op."
            ),
            failure_code="DEEPGEMM_PRECEDENCE_PATCH_SEMANTIC_NOOP",
            evidence=deepgemm_evidence,
        ),
        _check(
            deepgemm["head_enforces_both_exponent_bounds"],
            rule_id="frozen-ue8m0-exponent-boundaries-are-enforced",
            question="Does the head reject values outside both frozen exponent bounds?",
            pass_conclusion="The exact head rejects both lower- and upper-bound violations.",
            fail_conclusion=(
                "The exact head accepts both the zero exponent and all-ones exponent controls; "
                "the frozen two-sided exponent contract is not enforced."
            ),
            failure_code="DEEPGEMM_UE8M0_EXPONENT_BOUNDS_UNENFORCED",
            evidence=deepgemm_evidence,
        ),
        _check(
            deepgemm["base_head_valid_outputs_equal"]
            and deepgemm["base_and_head_reject_nonzero_mantissa"],
            rule_id="valid-pack-and-mantissa-neighbors-are-preserved",
            question="Are valid pack outputs and the existing mantissa check preserved?",
            pass_conclusion=(
                "Real PyTorch execution gives byte-identical valid packing and both revisions "
                "reject nonzero mantissas."
            ),
            fail_conclusion="A valid packing or mantissa neighbor changed unexpectedly.",
            failure_code="DEEPGEMM_UE8M0_VALID_NEIGHBOR_REGRESSION",
            evidence=deepgemm_evidence,
        ),
        _check(
            deepgemm["changed_direct_test_present"],
            rule_id="direct-regression-covers-pack-boundary-matrix",
            question="Does the PR include a direct regression for the claimed predicate repair?",
            pass_conclusion="A changed direct regression covers the predicate repair.",
            fail_conclusion="The PR changes only math.py and adds no direct regression.",
            failure_code="DEEPGEMM_PACK_UE8M0_DIRECT_REGRESSION_MISSING",
            evidence=deepgemm_evidence,
        ),
    ]

    flashattention = evidence["flashattention-pr-2645"]["facts"]
    flashattention_evidence = digests("flashattention-pr-2645")
    flashattention_observations = [
        _check(
            flashattention["compile_key_branch_count"] == 2
            and flashattention["base_q_subtile_factor_counts"] == [0, 0]
            and flashattention["head_q_subtile_factor_counts"] == [1, 1]
            and flashattention["all_affected_head_keys_include_factor_once"]
            and flashattention["identical_factor_keys_stable"],
            rule_id="all-backward-codegen-keys-own-subtile-factor",
            question="Do all affected backward compilation keys isolate subtile_factor?",
            pass_conclusion=(
                "Both exact backward key branches add q_subtile_factor exactly once; identical "
                "factors remain stable while different factors separate."
            ),
            fail_conclusion="At least one affected backward key still aliases subtile_factor.",
            failure_code="FLASHATTENTION_SUBTILE_COMPILE_KEY_INCOMPLETE",
            evidence=flashattention_evidence,
        ),
        _check(
            flashattention["direct_test_has_two_sparse_block_q_values"]
            and flashattention["direct_test_compares_dq_dk_dv_to_dense"]
            and len(flashattention["changed_test_methods"]) == 1,
            rule_id="direct-regression-exercises-key-collision-and-gradients",
            question="Does a direct same-process regression exercise two factors and correctness?",
            pass_conclusion=(
                "The changed same-process SM90 regression uses sparse block sizes 128 and 256 and "
                "compares dq, dk, and dv against dense references."
            ),
            fail_conclusion=(
                "The direct compile-key collision/correctness regression is incomplete."
            ),
            failure_code="FLASHATTENTION_SUBTILE_DIRECT_REGRESSION_INCOMPLETE",
            evidence=flashattention_evidence,
        ),
        _unresolved(
            "full-sm90-backward-kernel-runtime",
            "Does the full changed SM90 backward test execute in this cell?",
            "The exact key contract and direct test were audited, but the available GPU is SM80.",
            "FLASHATTENTION_SM90_RUNTIME_UNAVAILABLE",
            *flashattention_evidence,
        ),
    ]

    flashinfer = evidence["flashinfer-pr-3918"]["facts"]
    flashinfer_evidence = digests("flashinfer-pr-3918")
    flashinfer_observations = [
        _check(
            flashinfer["base_mixed_cold_l2_path_fails"]
            and flashinfer["head_batch_count"] > 1
            and flashinfer["head_non_tensor_identity_preserved"]
            and flashinfer["head_tensor_clones_are_distinct"],
            rule_id="mixed-cold-l2-batches-clone-only-tensors",
            question="Does cold-L2 batching clone tensors while preserving non-tensors?",
            pass_conclusion=(
                "The base crashes on mixed inputs; the exact head creates five batches, clones "
                "only tensors, and preserves every non-tensor object."
            ),
            fail_conclusion="The exact mixed tensor/non-tensor cold-L2 contract failed.",
            failure_code="FLASHINFER_AUTOTUNER_MIXED_INPUT_BATCHING_FAILED",
            evidence=flashinfer_evidence,
        ),
        _check(
            flashinfer["head_no_tensor_path_returns_single_original_batch"]
            and flashinfer["head_size_guard_marks_non_tensors"],
            rule_id="non-tensor-size-and-zero-buffer-controls-are-deterministic",
            question="Are non-tensor size guards and no-tensor batches deterministic?",
            pass_conclusion=(
                "Scalar, boolean, None, and structured controls map to the non-tensor size marker; "
                "an all-non-tensor input returns one unchanged batch."
            ),
            fail_conclusion="A non-tensor size or zero-buffer control is unstable.",
            failure_code="FLASHINFER_AUTOTUNER_NON_TENSOR_GUARD_UNSTABLE",
            evidence=flashinfer_evidence,
        ),
        _check(
            flashinfer["direct_test_covers_dtype_and_none"]
            and flashinfer["direct_test_covers_cloning_and_identity"],
            rule_id="direct-test-covers-mixed-signature-copy-semantics",
            question="Does direct coverage exercise mixed inputs and identity/copy behavior?",
            pass_conclusion=(
                "The changed regression covers dtype and None non-tensors and asserts tensor clone "
                "versus non-tensor identity behavior."
            ),
            fail_conclusion="The direct mixed-signature regression is incomplete.",
            failure_code="FLASHINFER_AUTOTUNER_MIXED_INPUT_TEST_INCOMPLETE",
            evidence=flashinfer_evidence,
        ),
        _unresolved(
            "actual-gpu-autotuner-profile",
            "Does a full GPU autotuner profile preserve the same mixed-input behavior?",
            (
                "The exact batching contract passed; a full kernel profiling run was not required "
                "here."
            ),
            "FLASHINFER_GPU_AUTOTUNER_PROFILE_NOT_EXECUTED",
            *flashinfer_evidence,
        ),
    ]

    liger = evidence["liger-pr-1289"]["facts"]
    liger_evidence = digests("liger-pr-1289")
    liger_observations = [
        _check(
            liger["base_zero_exception"] == "ZeroDivisionError"
            and liger["head_zero_exception"] == "AssertionError"
            and liger["head_zero_message_is_actionable"]
            and liger["head_zero_fails_before_triton_calls"],
            rule_id="zero-vocab-fails-early-with-actionable-domain-error",
            question="Does zero vocabulary fail before Triton division with actionable context?",
            pass_conclusion=(
                "The base reaches ZeroDivisionError; normal head execution fails before any Triton "
                "call with a materialization/all-gather explanation."
            ),
            fail_conclusion="Normal execution does not provide the required early domain failure.",
            failure_code="LIGER_ZERO_VOCAB_EARLY_DOMAIN_ERROR_MISSING",
            evidence=liger_evidence,
        ),
        _check(
            liger["head_zero_policy_survives_python_optimized_mode"],
            rule_id="zero-vocab-policy-survives-optimized-python",
            question=(
                "Is the zero-vocabulary policy stable when Python assertions are optimized out?"
            ),
            pass_conclusion="The explicit zero-vocabulary policy survives optimized execution.",
            fail_conclusion=(
                "With optimized bytecode, the assertion disappears and the exact head returns to "
                f"{liger['head_optimized_zero_exception']}; the domain policy is not stable."
            ),
            failure_code="LIGER_ZERO_VOCAB_VALIDATION_USES_REMOVABLE_ASSERT",
            evidence=liger_evidence,
        ),
        _check(
            liger["positive_vocab_prefix_preserved"]
            and liger["retained_positive_test_method_count"] >= 10,
            rule_id="positive-vocabulary-neighbors-are-preserved",
            question="Are positive vocabulary sizes and retained regression coverage unchanged?",
            pass_conclusion=(
                "Five positive vocabulary boundaries take bytecode-equivalent prefixes in base and "
                "head, and twelve retained positive tests remain."
            ),
            fail_conclusion="A positive vocabulary neighbor or retained test contract regressed.",
            failure_code="LIGER_POSITIVE_VOCAB_NEIGHBOR_REGRESSION",
            evidence=liger_evidence,
        ),
        _check(
            liger["direct_zero_regression_present"],
            rule_id="direct-test-covers-zero-vocabulary",
            question="Does a direct regression cover the zero-width vocabulary path?",
            pass_conclusion=(
                "The changed regression calls the exact forward helper with weight[0] == 0."
            ),
            fail_conclusion="No direct zero-vocabulary regression was found.",
            failure_code="LIGER_ZERO_VOCAB_DIRECT_REGRESSION_MISSING",
            evidence=liger_evidence,
        ),
    ]

    megatron = evidence["megatron-pr-5750"]["facts"]
    megatron_evidence = digests("megatron-pr-5750")
    megatron_observations = [
        _check(
            megatron["base_omits_epsilon"] and megatron["head_propagates_every_epsilon"],
            rule_id="mamba-norm-receives-configured-epsilon-matrix",
            question="Does every frozen custom/default epsilon reach MambaLayer's norm?",
            pass_conclusion=(
                "The base omits epsilon; the exact head forwards 1e-5, 1e-6, and 1e-8 unchanged "
                "to the norm builder."
            ),
            fail_conclusion="At least one configured epsilon is omitted or rewritten.",
            failure_code="MEGATRON_MAMBA_NORM_EPSILON_PROPAGATION_FAILED",
            evidence=megatron_evidence,
        ),
        _check(
            megatron["head_uses_explicit_config_hidden_size_eps_keywords"],
            rule_id="norm-builder-contract-is-explicit",
            question="Are config, hidden_size, and eps owned explicitly at the builder boundary?",
            pass_conclusion=(
                "All exact head invocations use the explicit config, hidden_size, and eps keyword "
                "contract."
            ),
            fail_conclusion="The norm-builder boundary still relies on an implicit epsilon.",
            failure_code="MEGATRON_MAMBA_NORM_BUILDER_CONTRACT_INCOMPLETE",
            evidence=megatron_evidence,
        ),
        _check(
            megatron["direct_test_sets_and_asserts_epsilon"],
            rule_id="direct-test-asserts-instantiated-norm-epsilon",
            question="Does a direct test assert the instantiated norm's configured epsilon?",
            pass_conclusion=(
                "The changed unit test sets 1e-6 and checks the instantiated norm value."
            ),
            fail_conclusion="The direct configured-epsilon regression is missing.",
            failure_code="MEGATRON_MAMBA_NORM_EPSILON_TEST_MISSING",
            evidence=megatron_evidence,
        ),
    ]

    sglang = evidence["sglang-pr-31344"]["facts"]
    sglang_evidence = digests("sglang-pr-31344")
    sglang_observations = [
        _check(
            not sglang["base_has_distinct_decode_choices"]
            and sglang["head_decode_excludes_exactly_flashmla_auto"]
            and sglang["argparse_rejected"] == ["flashmla_auto"]
            and set(sglang["argparse_accepted"]) == set(sglang["head_decode_choices"]),
            rule_id="decode-backend-truth-table-excludes-only-prefill-auto",
            question="Does CLI validation reject exactly flashmla_auto in decode position?",
            pass_conclusion=(
                "The exact seven-choice matrix rejects only flashmla_auto and accepts all six "
                "declared decode backends."
            ),
            fail_conclusion="The decode backend truth table rejects too much or too little.",
            failure_code="SGLANG_DSA_DECODE_BACKEND_TRUTH_TABLE_FAILED",
            evidence=sglang_evidence,
        ),
        _check(
            sglang["both_primary_and_deprecated_decode_args_use_narrow_choices"]
            and sglang["prefill_keeps_full_choices"],
            rule_id="primary-alias-and-prefill-lifecycles-are-consistent",
            question="Are the primary/alias decode flags narrowed while prefill stays supported?",
            pass_conclusion=(
                "Both decode flags use the narrow list, while prefill retains flashmla_auto."
            ),
            fail_conclusion="A CLI alias or the prefill lifecycle is inconsistent.",
            failure_code="SGLANG_DSA_BACKEND_CLI_LIFECYCLE_INCONSISTENT",
            evidence=sglang_evidence,
        ),
        _check(
            sglang["runtime_fallback_changed_from_assert_to_value_error"]
            and len(sglang["direct_test_methods"]) == 5,
            rule_id="runtime-fallback-and-direct-neighbor-tests-are-explicit",
            question="Is the fallback a real exception and is the full neighbor matrix tested?",
            pass_conclusion=(
                "The residual runtime guard uses ValueError instead of removable assert, and five "
                "direct tests cover primary, alias, prefill, and valid neighbors."
            ),
            fail_conclusion="The runtime fallback or direct test matrix is incomplete.",
            failure_code="SGLANG_DSA_DECODE_BACKEND_GUARD_TEST_INCOMPLETE",
            evidence=sglang_evidence,
        ),
    ]

    torchtitan = evidence["torchtitan-pr-3862"]["facts"]
    torchtitan_evidence = digests("torchtitan-pr-3862")
    torchtitan_observations = [
        _check(
            torchtitan["real_torch_cpu_executed"]
            and torchtitan["real_torch_base_outputs_contiguous"] == [False, False]
            and torchtitan["real_torch_head_outputs_contiguous"] == [True, True]
            and torchtitan["real_torch_shapes_preserved"],
            rule_id="real-torch-fake-kernel-matches-contiguous-output-contract",
            question="Does real PyTorch reproduce the base stride mismatch and head repair?",
            pass_conclusion=(
                "Real PyTorch non-contiguous gradients yield non-contiguous base fake outputs and "
                "contiguous shape-preserving head outputs."
            ),
            fail_conclusion="The real PyTorch base/head stride contract did not pass.",
            failure_code="TORCHTITAN_HELION_ROPE_REAL_TORCH_STRIDE_FAILED",
            evidence=torchtitan_evidence,
        ),
        _check(
            torchtitan["head_produces_contiguous_fake_strides"]
            and torchtitan["head_preserves_shapes"]
            and torchtitan["real_backward_explicitly_contiguates_inputs"],
            rule_id="fake-and-real-rope-layout-contracts-agree",
            question="Do fake outputs agree with the real backward contiguous layout?",
            pass_conclusion=(
                "All three frozen shape/stride cases produce contiguous head fake outputs, "
                "matching the real backward's explicit contiguous conversion."
            ),
            fail_conclusion="Fake output layout still diverges from the real backward contract.",
            failure_code="TORCHTITAN_HELION_ROPE_FAKE_REAL_LAYOUT_MISMATCH",
            evidence=torchtitan_evidence,
        ),
        _check(
            torchtitan["direct_opcheck_uses_noncontiguous_grads"],
            rule_id="direct-opcheck-covers-noncontiguous-gradients",
            question="Does direct opcheck coverage exercise non-contiguous incoming gradients?",
            pass_conclusion=(
                "The changed opcheck constructs transposed q/k gradients and asserts both are "
                "non-contiguous before invoking the custom backward op."
            ),
            fail_conclusion="Direct non-contiguous fake-kernel coverage is missing.",
            failure_code="TORCHTITAN_HELION_ROPE_NONCONTIGUOUS_OPCHECK_MISSING",
            evidence=torchtitan_evidence,
        ),
        _unresolved(
            "actual-helion-gpu-opcheck",
            "Does the complete Helion GPU custom-op stack pass opcheck?",
            (
                "Real PyTorch stride semantics passed; the Helion GPU dependency stack was "
                "unavailable."
            ),
            "TORCHTITAN_HELION_GPU_STACK_UNAVAILABLE",
            *torchtitan_evidence,
        ),
    ]

    verl = evidence["verl-pr-7044"]["facts"]
    verl_evidence = digests("verl-pr-7044")
    required_verl_tests = {
        "test_malformed_function_header_returns_none",
        "test_malformed_parameter_is_skipped_but_valid_parameter_is_preserved",
        "test_missing_tools_defaults_to_empty_list",
    }
    verl_observations = [
        _check(
            verl["base_truncated_parameter_raises"]
            and verl["head_truncated_parameter_preserves_prior_valid"]
            and verl["malformed_matrix_has_no_exceptions"]
            and verl["missing_tools_defaults_to_empty"],
            rule_id="malformed-xml-matrix-is-bounded-and-loss-local",
            question=(
                "Do malformed headers/parameters terminate without discarding prior valid data?"
            ),
            pass_conclusion=(
                "The base raises on a truncated parameter; the head completes six malformed cases, "
                "preserves prior valid data, and tolerates missing tool schemas."
            ),
            fail_conclusion=(
                "A malformed XML case still escapes or discards prior valid parameters."
            ),
            failure_code="VERL_QWEN3_MALFORMED_XML_CONTRACT_FAILED",
            evidence=verl_evidence,
            counterevidence=[
                "The parser remains intentionally permissive: some mismatched or nested fragments "
                "are returned as string values rather than rejected semantically."
            ],
        ),
        _check(
            verl["valid_call_preserved"]
            and verl["head_valid_arguments"] == {"count": 2, "items": "valid"}
            and verl["ordinary_text_produces_no_function_calls"],
            rule_id="valid-calls-and-plain-text-controls-are-preserved",
            question="Are valid calls identical while ordinary text stays ordinary?",
            pass_conclusion=(
                "Base and head produce the same typed valid call, and ordinary text yields no "
                "function call."
            ),
            fail_conclusion="A valid call or plain-text control changes unexpectedly.",
            failure_code="VERL_QWEN3_VALID_OR_PLAIN_TEXT_REGRESSION",
            evidence=verl_evidence,
        ),
        _check(
            required_verl_tests <= set(verl["direct_test_methods"]),
            rule_id="direct-tests-cover-three-new-malformed-boundaries",
            question="Do direct CPU tests cover header, parameter, and missing-tools boundaries?",
            pass_conclusion=(
                "All three newly repaired malformed-input boundaries have direct tests."
            ),
            fail_conclusion="The direct malformed-input regression matrix is incomplete.",
            failure_code="VERL_QWEN3_MALFORMED_XML_TEST_MATRIX_INCOMPLETE",
            evidence=verl_evidence,
        ),
    ]

    vllm = evidence["vllm-pr-48705"]["facts"]
    vllm_evidence = digests("vllm-pr-48705")
    vllm_matrices = vllm["streaming_matrices"]
    vllm_improves_all = all(
        row["head"]["passed"] > row["base"]["passed"] for row in vllm_matrices.values()
    )
    vllm_counts = ", ".join(
        f"{name}: {row['base']['passed']}->{row['head']['passed']}/{row['head']['chunking_count']}"
        for name, row in sorted(vllm_matrices.items())
    )
    vllm_observations = [
        _check(
            vllm["base_has_at_least_one_corrupt_chunking_per_case"] and vllm_improves_all,
            rule_id="streaming-repair-improves-every-frozen-multi-key-case",
            question="Does the head reduce corruption for every frozen multi-key payload?",
            pass_conclusion=f"Every case improves its valid reconstruction count ({vllm_counts}).",
            fail_conclusion="At least one multi-key payload does not improve over the base.",
            failure_code="VLLM_FUNCTIONGEMMA_STREAM_REPAIR_NO_IMPROVEMENT",
            evidence=vllm_evidence,
        ),
        _check(
            vllm["head_all_frozen_chunkings_pass"],
            rule_id="all-frozen-json-chunk-boundaries-reconstruct-exactly-once",
            question="Do all frozen key, escape, numeric, and delimiter splits reconstruct JSON?",
            pass_conclusion="Every frozen streaming split reconstructs the exact final arguments.",
            fail_conclusion=(
                "The head still emits unretractable partial argument fragments at several frozen "
                f"boundaries ({vllm_counts}); concatenated deltas are invalid or corrupted."
            ),
            failure_code="VLLM_FUNCTIONGEMMA_STREAM_BOUNDARY_CORRUPTION_REMAINS",
            evidence=vllm_evidence,
            counterevidence=[
                "The upstream two-key and three-key chunk patterns pass and the frozen matrix is "
                "substantially better than base."
            ],
        ),
        _check(
            not vllm["incomplete_stream_finalized_json"]
            and vllm["incomplete_stream_emissions"] == []
            and vllm["direct_test_has_two_and_three_key_regressions"]
            and len(vllm["direct_test_methods"]) == 2,
            rule_id="incomplete-input-waits-and-direct-regressions-exist",
            question="Does incomplete input wait, with direct two/three-key coverage?",
            pass_conclusion=(
                "Incomplete input emits no finalized arguments, and two direct regressions cover "
                "the demonstrated multi-key failure."
            ),
            fail_conclusion="Incomplete-input handling or direct regression coverage is missing.",
            failure_code="VLLM_FUNCTIONGEMMA_INCOMPLETE_OR_TEST_CONTRACT_FAILED",
            evidence=vllm_evidence,
        ),
    ]

    observations = {
        "cutlass-pr-3300": cutlass_observations,
        "deepgemm-pr-310": deepgemm_observations,
        "flashattention-pr-2645": flashattention_observations,
        "flashinfer-pr-3918": flashinfer_observations,
        "liger-pr-1289": liger_observations,
        "megatron-pr-5750": megatron_observations,
        "sglang-pr-31344": sglang_observations,
        "torchtitan-pr-3862": torchtitan_observations,
        "verl-pr-7044": verl_observations,
        "vllm-pr-48705": vllm_observations,
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
        "protocol_id": plan["protocol_id"],
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
    print(
        json.dumps(
            {
                "lock_set_sha256": payload["lock_set_sha256"],
                "decisions": {
                    lock["material"]["case_id"]: lock["material"]["decision"] for lock in locks
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
