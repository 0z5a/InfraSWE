#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the expanded 29-case R13 training cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.triage import CaseContractTriageEvidence, classify_case_contract
from infraswe.io import atomic_write_json

EXPECTED_SELECTION_FILE_SHA256 = "aec3143512d7e3a1e6959345828ce65ac0aaeec9ae2a59d25e621ef9b6e73741"
EXPECTED_TEST_PLAN_FILE_SHA256 = "6efde3d21be30233f70da8abc3491828d43ec9396cecc56917126425297797b2"
EXPECTED_SOURCE_BUNDLE_SHA256 = "sha256:dc0cab095afa903e0780196b8511b4a8a935d41aeff85ca400293c4fbb525ee7"
POLICY_ID = "training-case-contract-v0.1-r13"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_digest(payload: dict[str, Any], field: str, label: str) -> None:
    material = {key: value for key, value in payload.items() if key != field}
    _require(payload.get(field) == canonical_sha256(material), f"{label} digest mismatch")


def _at(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _all_zero(rows: list[dict[str, Any]], key: str) -> bool:
    return all(float(row[key]) == 0.0 for row in rows)


def _all_true(payload: dict[str, Any]) -> bool:
    return all(value is True for value in payload.values())


FACT_CHECKS: dict[str, tuple[tuple[str, Any], ...]] = {
    "flashattention-pr-2654": (
        ("candidate_says_local_uncommitted_prerequisite_fixes", True),
        ("candidate_mentions_ignored_errors", True),
        ("changed_test_count", 0),
        ("head_interface_sm80_or_sm120_score_mod_rejection_present", False),
    ),
    "liger-pr-1274": (
        ("head_uses_torch_where", True),
        ("head_uses_boolean_index_assignment", False),
        ("fullgraph_compile.head.fullgraph_succeeded", True),
        ("fullgraph_compile.base.fullgraph_succeeded", True),
    ),
    "liger-pr-1268": (
        ("max_base_head_gradient_error", 0.0),
        ("max_independent_oracle_gradient_error", 0.0),
        ("target_correction_barrier_present", True),
        ("ignore_index_returns_before_target_correction", True),
    ),
    "liger-pr-1230": (
        ("base_plain_model_redirection_fails", True),
        ("head_plain_model_direct_result.0", "finite-loss"),
        ("fsdp_branch_still_calls_redirection", True),
        ("trl_new_location_with_old_fallback", True),
    ),
    "megatron-pr-5808": (
        ("root_hooks_restored_exactly_once", True),
        ("child_hooks_exactly_once_in_both", True),
        ("head_uses_module_call_protocol", True),
        ("gradient_max_abs", 0.0),
    ),
    "megatron-pr-5798": (
        ("base_distinguishes_mbs4", True),
        ("head_all_layouts_partition_invariant", True),
        ("changed_test_count", 1),
    ),
    "megatron-pr-5743": (
        ("analytic_eager_deferred_max_abs", 0.0),
        ("outer_finalize_is_last_gated", True),
        ("last_microbatch_threaded_from_context", True),
        ("partial_dtensor_roundtrip_supported", True),
    ),
    "megatron-pr-5742": (
        ("continuous_resume_moment_max_abs", 0.0),
        ("continuous_resume_param_max_abs", 0.0),
        ("lion_state_key_property_present", True),
        ("constructor_accepts_init_state_fn", True),
    ),
    "torchtitan-pr-3841": (
        ("graphs_linted_and_recompiled", True),
        ("no_input_grad_returns_none", True),
        ("candidate_has_explicit_residual_control", True),
        ("candidate_has_explicit_shared_parameter_control", False),
    ),
    "torchtitan-pr-3897": (
        ("head_conflict_marker_free", False),
        ("head_python_syntax_ok", False),
        ("head_conflict_marker_count", 3),
        ("integration_file_has_unresolved_conflict", True),
    ),
    "torchtitan-pr-3867": (
        ("post_load_fused_roundtrip_max_abs", 0.0),
        ("generator_reloads_received_state", True),
        ("candidate_test_covers_fused_qkv", True),
        ("candidate_test_covers_fused_swiglu", False),
        ("candidate_test_exercises_malformed_or_missing_keys", False),
    ),
    "verl-pr-7014": (
        ("head_streamed_export_max_abs_from_merged", 0.0),
        ("head_post_context_restore_max_abs", 0.0),
        ("old_delayed_export_max_abs_from_merged", 4.0),
        ("early_return_precedes_later_qat_or_quantization_references", True),
    ),
    "verl-pr-7013": (
        ("all_three_trainers_save_and_load", True),
        ("resume_trajectory_abs_error", 0.0),
        ("malformed_and_nonfinite_fail_closed", True),
        ("missing_state_compatibility_present", True),
    ),
    "verl-pr-6984": (
        ("base_retained_model_output_tensor_count", 8),
        ("head_retained_model_output_tensor_count", 0),
        ("pop_occurs_only_after_backward", True),
        ("forward_only_guard_preserves_output", True),
    ),
    "megatron-pr-5819": (
        ("base_destructive_consumer_corrupts_static_cache", True),
        ("head_destructive_consumer_preserves_static_cache", True),
        ("head_nested_dict_is_distinct", True),
        ("head_tensor_identity_is_preserved", True),
    ),
    "megatron-pr-5761": (
        ("base_enters_autocast", False),
        ("head_enters_both_contexts", True),
        ("head_source_uses_multi_context_with", True),
        ("changed_test_count", 0),
    ),
    "megatron-pr-5724": (
        ("head_function_matches_expected_branches", True),
        ("graph_without_capacity_fails_closed", True),
        ("eager_default_preserves_dynamic_cu_seqlens", True),
        ("changed_test_count", 2),
    ),
    "megatron-pr-5714": (
        ("all_even_divisors_supported", True),
        ("odd_dp3_rejected_by_mapping", True),
        ("candidate_test_topology_mentions_tp2_dp4", True),
        ("source_asserts_two_halves_cover_dp", True),
    ),
    "megatron-pr-5710": (
        ("all_frozen_pre_post_balanced", True),
        ("candidate_has_frozen_outside_backward_graph_control", True),
        ("full_backward_hook_only_for_zero_trainable", True),
    ),
    "slime-pr-2207": (
        ("all_valid_inputs_semantically_identical", True),
        ("helper_adds_only_post_assignment_validation", True),
        ("helper_assigns_same_response_length_mask", True),
        ("head_invalid_metadata_fails_closed", True),
    ),
    "slime-pr-2205": (
        ("candidate_function_present", True),
        ("max_bfloat16_error", 1.15625),
        ("paired_timing_ms_per_call.speedup", lambda value: float(value) > 10.0),
        ("changed_test_count", 1),
    ),
    "slime-pr-2204": (
        ("permutation_invariance_max_abs", 0.0),
        ("singleton_groups_zero", True),
        ("missing_group_identity_fails_closed", True),
        ("uses_explicit_positions_by_group", True),
    ),
    "slime-pr-2198": (
        ("all_policy_ratio_call_sites_changed", True),
        ("clamp_bound", 20.0),
        ("nan_remains_visible", True),
        ("low_variance_kl_only_is_clamped", True),
    ),
    "slime-pr-2152": (
        ("backward_mutates_saved_logprob_softmax_inplace", True),
        ("metric_only_entropy_avoids_full_vocab_saved_tensors", True),
        ("metric_only_entropy_marked_non_differentiable", True),
        ("tp1_value_gradient_and_repeat_matrix", lambda rows: all(row["repeat_backward_exception"] for row in rows)),
    ),
    "verl-pr-7012": (
        ("all_integer_local_lengths_align", True),
        ("forced_length_applied_to_both_teacher_values_and_ids", True),
        ("forced_length_derived_from_student_and_cp", True),
        ("post_split_shape_assertion_present", True),
    ),
    "verl-pr-7005": (
        ("analytic_elements_live_peak.whole_shard_staging_then_materialize", 33),
        ("analytic_elements_live_peak.per_tensor_stream_only", 18),
        ("skip_scope_is_fsdp2_non_peft_only", True),
        ("offload_is_symmetric_with_skipped_load", True),
    ),
    "verl-pr-6996": (
        ("value_branch_broadcast_tensor_device", "cpu"),
        ("value_branch_uses_default_process_group", True),
        ("value_branch_has_backend_aware_cpu_group", False),
        ("only_rank0_builds_full_state", True),
    ),
    "verl-pr-6963": (
        ("changed_path_guard_matrix", _all_true),
        ("present_empty_not_rejected_by_presence_checks", True),
        ("request_config_checks_both_producer_and_actor_flags", True),
        ("candidate_changed_test_count", 0),
    ),
    "verl-pr-6960": (
        ("both_kernel_grad_inputs_normalized", True),
        ("normalization_precedes_fused_kernel_call", True),
        ("contiguous_copy_value_error", 0.0),
        ("upstream_grad_layouts.dlogprobs_contiguous_after", True),
        ("upstream_grad_layouts.dentropy_contiguous_after", True),
    ),
}


def _validate_facts(case_id: str, facts: dict[str, Any]) -> None:
    for path, expected in FACT_CHECKS[case_id]:
        actual = _at(facts, path)
        if callable(expected):
            _require(bool(expected(actual)), f"{case_id}: fact predicate failed for {path}")
        else:
            _require(actual == expected, f"{case_id}: {path} changed: {actual!r}")
    _require(facts["head_conflict_marker_free"] is (case_id != "torchtitan-pr-3897"), f"{case_id}: conflict-marker status changed")


def _validate_probe(
    payload: dict[str, Any],
    *,
    case_id: str,
    selected: dict[str, Any],
    selection_sha256: str,
    test_plan_sha256: str,
) -> None:
    _validate_digest(payload, "evidence_sha256", f"R13 probe {case_id}")
    expected = {
        "case_id": case_id,
        "base_sha": selected["base_sha"],
        "head_sha": selected["head_sha"],
        "selection_lock_sha256": selection_sha256,
        "test_plan_sha256": test_plan_sha256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "probe_status": "pass",
        "failure_codes": [],
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"{case_id}: probe {key} mismatch")
    environment = payload["environment"]
    _require(environment["torch_cuda_available"] is True, f"{case_id}: CUDA unavailable")
    _require(environment["gpu_count"] == 2, f"{case_id}: expected two GPUs")
    _require(environment["gpu_names"] == ["NVIDIA A100-SXM4-40GB"] * 2, f"{case_id}: GPU identity changed")
    _validate_facts(case_id, payload["facts"])


def _assessment(
    triage: CaseContractTriageEvidence,
    findings: list[str],
    residual: str | None,
) -> dict[str, Any]:
    return {"triage": triage, "findings": findings, "residual": residual}


ACCEPT = CaseContractTriageEvidence(True, True, True, closure_test="frozen-probe")
CHECK = CaseContractTriageEvidence(
    False,
    True,
    True,
    remediation_scope="single-site",
    closure_test="frozen-probe",
    residual_failure_families=1,
)


ASSESSMENTS: dict[str, dict[str, Any]] = {
    "flashattention-pr-2654": _assessment(
        CaseContractTriageEvidence(False, False, False),
        [
            "Head removes the architecture guard and extends score-mod data flow, but the exact SM80 runtime could not be imported because the native extension and CuTe stack were unavailable.",
            "The candidate report explicitly depended on two local fixes not contained in this PR and ignored another error family, so its report cannot substitute for evaluator execution.",
        ],
        "Build the exact head with its declared CuTe dependencies and run the frozen SM80 forward/backward identity and position-dependent score-mod matrix without external source edits.",
    ),
    "liger-pr-1274": _assessment(
        CHECK,
        [
            "The torch.where rewrite is eager-equivalent in FP32 and BF16, and the native SAPO matrix passes.",
            "Both exact base and head compile under the available Torch version, so this environment does not reproduce the claimed base graph break and the candidate adds no distinguishing regression.",
        ],
        "Retain an exact version-pinned fullgraph regression that fails on the base indexing assignment and passes on head.",
    ),
    "liger-pr-1268": _assessment(
        CHECK,
        [
            "The single-shot correction is algebraically exact across smoothing, weighting, softcap, reduction, and ignore-index controls; 56 native option-matrix tests pass.",
            "Evaluator execution did not include a paired kernel timing, so the performance half of the title-scoped claim remains bounded but unverified.",
        ],
        "Run a paired A100 timing with warmup, spread, and a non-regression threshold for the changed correction path.",
    ),
    "liger-pr-1230": _assessment(
        ACCEPT,
        [
            "The base no-FSDP path enters an invalid redirection owner, while head directly returns the finite model result.",
            "The FSDP branch and TRL compatibility fallback remain explicit, and the exact candidate test passes after installing its optional dependency.",
        ],
        None,
    ),
    "megatron-pr-5808": _assessment(
        ACCEPT,
        [
            "Calling through the module protocol restores root pre/post hooks exactly once without duplicating child hooks.",
            "The deterministic traversal probe preserves output and gradients; the native suite is separately marked environment-blocked on missing Transformer Engine.",
        ],
        None,
    ),
    "megatron-pr-5798": _assessment(
        ACCEPT,
        [
            "Head removes micro-batch-size dependence for uniform and uneven valid-token partitions while the base changes by a factor of four.",
            "The exact candidate sequence-level regression passes and the source keeps the intended token weighting.",
        ],
        None,
    ),
    "megatron-pr-5743": _assessment(
        ACCEPT,
        [
            "The last-microbatch flag reaches post-backward and gates one DP-outer finalization while preserving the partial accumulation placement.",
            "On two A100 ranks, deferred and eager reductions are exact while outer collective count falls from three to one.",
        ],
        None,
    ),
    "megatron-pr-5742": _assessment(
        ACCEPT,
        [
            "Lion is routed through the distributed optimizer with dynamic state-key typing instead of Adam-only assumptions.",
            "Continuous and resumed parameter/moment trajectories are exact, and the native optimizer suite passes 12 tests.",
        ],
        None,
    ),
    "torchtitan-pr-3841": _assessment(
        CHECK,
        [
            "The split pass keeps graph constants and symbolic live-ins, lints/recompiles, and passes the direct dI/dW cases including residual and no-input-grad controls.",
            "The frozen shared-parameter alias control is absent, leaving one bounded ownership case for a graph-rewriting change.",
        ],
        "Add and pass a shared-parameter graph whose dI/dW recomposition is compared with independent autograd gradients.",
    ),
    "torchtitan-pr-3897": _assessment(
        CaseContractTriageEvidence(
            False,
            True,
            False,
            remediation_scope="cross-cutting",
            closure_test="missing",
            baseline_regression=True,
            safety_or_integrity_failure=True,
            residual_failure_families=2,
        ),
        [
            "The exact candidate head contains three unresolved merge-conflict markers in a changed integration test and fails Python compilation.",
            "A compound FP16 training claim cannot be evaluated or integrated from a syntactically invalid head even though several component edits are directionally plausible.",
        ],
        "Resolve the head, then execute the full FP16 loss/optimizer/attention/generator multi-step matrix.",
    ),
    "torchtitan-pr-3867": _assessment(
        CHECK,
        [
            "Reloading the received split state triggers fused-QKV merge hooks and exact live-parameter/forward reconstruction; all nine native tests pass.",
            "FusedSwiGLU and malformed-key behavior are not directly covered, but the synthetic missing-key case is not promoted to a production failure without reachability evidence.",
        ],
        "Add a fused-SwiGLU TorchStore round trip and a reachable schema-mismatch assertion that inspects missing/unexpected keys.",
    ),
    "verl-pr-7014": _assessment(
        CaseContractTriageEvidence(
            False,
            True,
            True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            baseline_regression=True,
            residual_failure_families=1,
        ),
        [
            "The generator materializes LoRA-merged tensors before context exit and then restores base weights exactly.",
            "Its new early return bypasses the existing QAT quantization branch even though module construction permits LoRA and QAT together, regressing a reachable neighboring configuration.",
        ],
        "Route the merged-weight generator through the existing QAT transform before returning and add the LoRA+QAT intersection regression.",
    ),
    "verl-pr-7013": _assessment(
        ACCEPT,
        [
            "All three changed trainer families save and restore the adaptive-KL controller through an atomic tensor-only payload with compatibility and validation.",
            "The uninterrupted/resumed trajectory is exact, and all 12 candidate CPU tests pass.",
        ],
        None,
    ),
    "verl-pr-6984": _assessment(
        CHECK,
        [
            "model_output is popped only after backward and retained graph tensors fall from eight to zero while forward-only output remains available.",
            "The liveness direction is correct, but no paired full training gradient/metric and large-model A100 peak comparison was executed.",
        ],
        "Run multi-microbatch base/head gradient, metric, and peak-memory parity on the title-scoped actor update.",
    ),
    "megatron-pr-5819": _assessment(
        CHECK,
        [
            "Recursive container copying prevents a destructive iterator consumer from corrupting the static cache while intentionally preserving tensor identity.",
            "No candidate regression executes full-iteration CUDA graph capture/replay, so exhaustion and multi-chunk iterator lifecycles remain bounded gaps.",
        ],
        "Execute the frozen eager/capture/replay batch-identity sequence across repeat capture, exhaustion, and multi-chunk iterators.",
    ),
    "megatron-pr-5761": _assessment(
        CHECK,
        [
            "Head enters autocast then FP8 and exits in reverse order; base omits autocast in the combined-1F1B region.",
            "The exact context lifecycle is closed, but real FP8 numeric execution is unavailable because Transformer Engine has no ready wheel for this environment.",
        ],
        "Run the same joint-context control with Transformer Engine and compare finite loss, gradients, and updates to the non-combined schedule.",
    ),
    "megatron-pr-5724": _assessment(
        CHECK,
        [
            "The metadata branch preserves dynamic eager lengths, resolves explicit capacity identically for eager/graph, and fails closed when graph capacity is absent.",
            "All three changed padding tests pass; the real Transformer-Engine CUDA graph output/gradient path remains environment-blocked.",
        ],
        "Execute eager versus CUDA-graph THD outputs and gradients with Transformer Engine on aligned and non-aligned packed batches.",
    ),
    "megatron-pr-5714": _assessment(
        CHECK,
        [
            "The SwiGLU half/shard mapping is coherent for even DP sizes, but rejects DP=1 and odd DP sizes.",
            "The candidate test is written for TP2/DP4 yet fails on the available TP2/DP1 topology with two shards where it asserts one.",
        ],
        "Define and test DP=1/odd-DP support explicitly, then run an actual save/load/resume round trip rather than only factory offsets.",
    ),
    "megatron-pr-5710": _assessment(
        ACCEPT,
        [
            "All-frozen modules install a paired full-backward completion path while trainable modules retain parameter-gradient completion.",
            "Repeated hook events balance exactly, and both ranks pass the two direct frozen-parameter regressions.",
        ],
        None,
    ),
    "slime-pr-2207": _assessment(
        CaseContractTriageEvidence(
            False,
            True,
            False,
            remediation_scope="single-site",
            closure_test="existing",
            semantic_noop=True,
            residual_failure_families=1,
        ),
        [
            "For every valid response length, base and head assign the identical all-zero response mask; the production replacement only moves that assignment into a helper.",
            "Head newly rejects inconsistent metadata, but that hardening does not demonstrate the title's claimed valid-input off-policy alignment fix.",
        ],
        "Implement the intended generated-token-span mask and make the tokenwise loss/gradient regression distinguish base from head.",
    ),
    "slime-pr-2205": _assessment(
        CHECK,
        [
            "FP32 recurrence error stays below 1.7e-5, all upstream FP32/FP64 tests pass, and the A100 microbenchmark is roughly 59x faster.",
            "The frozen BF16 matrix is finite and dtype-preserving but diverges from the scalar recurrence by as much as 1.15625 on long high-discount sequences, which candidate tests omit.",
        ],
        "Accumulate the scan in FP32 or declare/verify a BF16 error contract with long-sequence return and policy-loss impact tests.",
    ),
    "slime-pr-2204": _assessment(
        ACCEPT,
        [
            "Explicit group positions make irregular/interleaved normalization permutation-invariant with exact per-group zero means and safe singleton groups.",
            "Missing group identity fails closed, the legacy fallback is explicit, and all five candidate reward tests pass.",
        ],
        None,
    ),
    "slime-pr-2198": _assessment(
        ACCEPT,
        [
            "All policy ratio exponentiation call sites clamp in FP32; extreme outputs and gradients are finite while NaNs remain visible.",
            "Healthy-range FP32 is exact, BF16 remains within its representational tolerance, and all four candidate numeric tests pass.",
        ],
        None,
    ),
    "slime-pr-2152": _assessment(
        CHECK,
        [
            "Single-GPU and TP2 candidate suites pass; two A100 ranks show exact log-probability, sub-0.003 BF16 gradient error, and about 66.7% lower modeled peak scratch memory.",
            "Backward negates the saved softmax in place, so a second backward on the retained graph fails a version-counter check in both entropy modes.",
        ],
        "Use non-mutating or separately owned backward scratch and add a repeated-backward regression to the existing TP1/TP2 matrix.",
    ),
    "verl-pr-7012": _assessment(
        CHECK,
        [
            "Teacher values and IDs derive the forced total length from student local length and CP size; all integer CP layouts align.",
            "On TP2 the base length mismatches on both ranks and head matches, but the frozen distributed top-k loss/gradient oracle was not executed.",
        ],
        "Run the two-rank top-k loss and student-gradient comparison, including odd lengths and padding boundaries.",
    ),
    "verl-pr-7005": _assessment(
        CaseContractTriageEvidence(False, False, True),
        [
            "Source removes whole-shard staging only for FSDP2 non-PEFT and keeps load/offload ownership symmetric; the analytic live-element peak falls from 33 to 18.",
            "No exact two-rank export/reload or measured peak-memory execution was captured, so the required distributed evidence is incomplete.",
        ],
        "Execute exact FSDP2 base/head export, reload, repeated export, and peak allocation on two A100 ranks.",
    ),
    "verl-pr-6996": _assessment(
        CaseContractTriageEvidence(
            False,
            True,
            False,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            safety_or_integrity_failure=True,
            residual_failure_families=1,
        ),
        [
            "Only rank zero constructs the value-model full state, then head broadcasts a CPU tensor through the default process group.",
            "Both A100 ranks reproduce `No backend type associated with device type cpu` under default NCCL while the CUDA control broadcast succeeds.",
        ],
        "Use a backend-aware CPU group or stage the payload on CUDA, then add a two-rank value-model load and first-forward regression.",
    ),
    "verl-pr-6963": _assessment(
        ACCEPT,
        [
            "Assembly, post-balance, rollouter, and separation-debug owners all fail closed when requested rollout log-probabilities are absent, and assembly also validates shape.",
            "Producer and actor request flags are checked consistently while present-empty and not-requested states remain distinct.",
        ],
        None,
    ),
    "verl-pr-6960": _assessment(
        ACCEPT,
        [
            "Both noncontiguous upstream gradient inputs become contiguous before their fused backward kernels with zero value error.",
            "The change is local to the consumer boundary and leaves the fused loss algebra untouched.",
        ],
        None,
    ),
}


def _validate_execution_artifact(
    payload: dict[str, Any], selection_sha256: str, test_plan_sha256: str, label: str
) -> None:
    _validate_digest(payload, "evidence_sha256", label)
    _require(payload["selection_lock_sha256"] == selection_sha256, f"{label}: selection binding mismatch")
    _require(payload["test_plan_sha256"] == test_plan_sha256, f"{label}: plan binding mismatch")
    _require(payload["outcome_review_ci_fields_requested"] is False, f"{label}: blind boundary changed")


def _artifact_binding(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": path.name,
        "evidence_sha256": payload["evidence_sha256"],
        "artifact_sha256": canonical_sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--dual-gpu-evidence", type=Path, required=True)
    parser.add_argument("--upstream-matrix", type=Path, required=True)
    parser.add_argument("--upstream-corrections", type=Path, required=True)
    parser.add_argument("--upstream-verl7013", type=Path, required=True)
    parser.add_argument("--upstream-slime2152", type=Path, required=True)
    parser.add_argument("--upstream-liger1230", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require(_file_sha256(args.selection_lock) == EXPECTED_SELECTION_FILE_SHA256, "R13 selection file digest mismatch")
    _require(_file_sha256(args.test_plan) == EXPECTED_TEST_PLAN_FILE_SHA256, "R13 test-plan file digest mismatch")
    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    _require(selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]), "R13 embedded selection digest mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    _require(plan["test_plan_sha256"] == canonical_sha256(plan_material), "R13 embedded test-plan digest mismatch")
    _require(plan["selection_lock_sha256"] == selection["selection_lock_sha256"], "R13 plan/selection binding mismatch")
    blind_flags = (
        selection["selection_material"]["review_text_visible_to_machine_judge"],
        selection["selection_material"]["merge_outcomes_visible_to_machine_judge"],
        selection["selection_material"]["ci_fields_visible_to_machine_judge"],
        plan["review_text_visible_to_machine_judge"],
        plan["merge_outcomes_visible_to_machine_judge"],
        plan["ci_fields_visible_to_machine_judge"],
        plan["review_text_requested"],
    )
    _require(all(value is False for value in blind_flags), "R13 blind boundary is not intact")
    _require(plan["scoring_policy"]["weighted_score_used"] is False, "R13 unexpectedly uses weighted scoring")
    _require(plan["scoring_policy"]["forced_polarization_used"] is False, "R13 unexpectedly forces polarization")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    _require(len(selected) == 29, "R13 expanded cohort is not 29 cases")
    _require(selected.keys() == planned.keys() == ASSESSMENTS.keys() == FACT_CHECKS.keys(), "R13 case sets differ")

    evidence: dict[str, dict[str, Any]] = {}
    evidence_bindings: dict[str, dict[str, Any]] = {}
    for case_id, selected_case in selected.items():
        _require(selected_case["base_sha"] == planned[case_id]["base_sha"], f"{case_id}: base SHA differs")
        _require(selected_case["head_sha"] == planned[case_id]["head_sha"], f"{case_id}: head SHA differs")
        path = args.result_root / "probes" / f"{case_id}.json"
        payload = _read(path)
        _validate_probe(
            payload,
            case_id=case_id,
            selected=selected_case,
            selection_sha256=selection["selection_lock_sha256"],
            test_plan_sha256=plan["test_plan_sha256"],
        )
        evidence[case_id] = payload
        evidence_bindings[case_id] = {
            "path": f"probes/{case_id}.json",
            "evidence_sha256": payload["evidence_sha256"],
            "artifact_sha256": canonical_sha256(payload),
        }

    dual = _read(args.dual_gpu_evidence)
    _validate_digest(dual, "evidence_sha256", "R13 dual-GPU evidence")
    _require(dual["selection_lock_sha256"] == selection["selection_lock_sha256"], "R13 dual selection mismatch")
    _require(dual["test_plan_sha256"] == plan["test_plan_sha256"], "R13 dual plan mismatch")
    _require(dual["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256, "R13 dual source mismatch")
    dual_facts = dual["facts"]
    _require(
        dual_facts["two_rank_nccl_smoke"]
        and dual_facts["megatron_5743_deferred_matches_eager"]
        and dual_facts["slime_2152_memory_reduction_on_all_ranks"]
        and dual_facts["verl_7012_base_shape_mismatch_on_all_ranks"]
        and dual_facts["verl_7012_head_shape_matches_on_all_ranks"]
        and dual_facts["verl_6996_cpu_broadcast_fails_on_all_nccl_ranks"]
        and dual_facts["verl_6996_cuda_broadcast_control_passes"],
        "R13 dual-GPU facts changed",
    )
    dual_binding = _artifact_binding(args.dual_gpu_evidence, dual)

    execution_paths = (
        args.upstream_matrix,
        args.upstream_corrections,
        args.upstream_verl7013,
        args.upstream_slime2152,
        args.upstream_liger1230,
    )
    execution_payloads: dict[str, dict[str, Any]] = {}
    execution_bindings: dict[str, dict[str, Any]] = {}
    for path in execution_paths:
        payload = _read(path)
        _validate_execution_artifact(payload, selection["selection_lock_sha256"], plan["test_plan_sha256"], path.name)
        execution_payloads[path.name] = payload
        execution_bindings[path.name] = _artifact_binding(path, payload)

    authoritative_artifact = {
        "liger-pr-1268": args.upstream_corrections.name,
        "megatron-pr-5724": args.upstream_corrections.name,
        "verl-pr-7013": args.upstream_verl7013.name,
        "slime-pr-2152": args.upstream_slime2152.name,
        "liger-pr-1230": args.upstream_liger1230.name,
    }
    native_case_bindings: dict[str, list[dict[str, Any]]] = {}
    for artifact_name, payload in execution_payloads.items():
        for index, record in enumerate(payload["records"]):
            case_id = record["case_id"]
            expected_artifact = authoritative_artifact.get(case_id, args.upstream_matrix.name)
            if artifact_name != expected_artifact:
                continue
            native_case_bindings.setdefault(case_id, []).append(
                {
                    "artifact": execution_bindings[artifact_name],
                    "record_index": index,
                    "scope": record["scope"],
                    "returncode": record["returncode"],
                    "output_sha256": record["output_sha256"],
                }
            )
    expected_passing_native = {
        "slime-pr-2205",
        "slime-pr-2204",
        "slime-pr-2198",
        "slime-pr-2207",
        "slime-pr-2152",
        "torchtitan-pr-3841",
        "torchtitan-pr-3867",
        "liger-pr-1274",
        "liger-pr-1268",
        "liger-pr-1230",
        "megatron-pr-5798",
        "megatron-pr-5742",
        "megatron-pr-5724",
        "megatron-pr-5710",
        "verl-pr-7013",
    }
    for case_id in expected_passing_native:
        _require(case_id in native_case_bindings, f"{case_id}: missing authoritative upstream record")
        _require(all(record["returncode"] == 0 for record in native_case_bindings[case_id]), f"{case_id}: authoritative upstream test failed")
    for case_id, expected_code in {
        "torchtitan-pr-3897": 1,
        "megatron-pr-5808": 2,
        "megatron-pr-5714": 1,
        "flashattention-pr-2654": 1,
    }.items():
        _require(len(native_case_bindings[case_id]) == 1, f"{case_id}: unexpected native record count")
        _require(native_case_bindings[case_id][0]["returncode"] == expected_code, f"{case_id}: native status changed")

    frozen_at = datetime.now(UTC).isoformat()
    locks: list[dict[str, Any]] = []
    dual_cases = {"megatron-pr-5743", "slime-pr-2152", "verl-pr-7012", "verl-pr-6996"}
    for case_id, selected_case in selected.items():
        assessment = ASSESSMENTS[case_id]
        triage = assessment["triage"]
        result = classify_case_contract(triage)
        legacy_decision = "accept_with_scope" if triage.contract_satisfied else "check"
        supplemental: list[dict[str, Any]] = []
        if case_id in dual_cases:
            supplemental.append(dual_binding)
        supplemental.extend(native_case_bindings.get(case_id, []))
        material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256({"selection": selected_case, "test_plan": planned[case_id]}),
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "evidence_binding_sha256": canonical_sha256(evidence_bindings[case_id]),
            "supplemental_evidence_binding_sha256": canonical_sha256(supplemental),
            "triage_input": asdict(triage),
            "decision": result.decision,
            "rationale_codes": list(result.rationale_codes),
            "technical_findings": assessment["findings"],
            "residual_contract": assessment["residual"],
            "legacy_r10_style_decision": legacy_decision,
            "frozen_at": frozen_at,
        }
        locks.append({"material": material, "lock_sha256": canonical_sha256(material)})

    decision_counts: dict[str, int] = {}
    legacy_counts: dict[str, int] = {}
    for lock in locks:
        decision = lock["material"]["decision"]
        legacy = lock["material"]["legacy_r10_style_decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        legacy_counts[legacy] = legacy_counts.get(legacy, 0) + 1
    lock_material = {
        "schema_version": "0.1",
        "protocol_id": plan["protocol_id"],
        "policy_id": POLICY_ID,
        "review_text_visible_during_machine_judgment": False,
        "merge_outcomes_visible_during_machine_judgment": False,
        "ci_fields_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "forced_polarization_used": False,
        "terminology": "check",
        "selection_lock_file_sha256": "sha256:" + EXPECTED_SELECTION_FILE_SHA256,
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_file_sha256": "sha256:" + EXPECTED_TEST_PLAN_FILE_SHA256,
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "dual_gpu_evidence_binding": dual_binding,
        "upstream_execution_bindings": execution_bindings,
        "frozen_at": frozen_at,
        "decision_counts": decision_counts,
        "legacy_r10_style_decision_counts": legacy_counts,
        "evidence_bindings": evidence_bindings,
        "locks": locks,
    }
    output = {**lock_material, "lock_set_sha256": canonical_sha256(lock_material)}
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "lock_set_sha256": output["lock_set_sha256"],
                "decision_counts": decision_counts,
                "legacy_r10_style_decision_counts": legacy_counts,
                "decisions": {lock["material"]["case_id"]: lock["material"]["decision"] for lock in locks},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
