#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze R11 repairability judgments before outcome and review reveal."""

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

EXPECTED_SELECTION_FILE_SHA256 = "7ba335693e343e3a610a42577d9f7e2e16167a3b5e57fb6316b5929a623890f5"
EXPECTED_TEST_PLAN_FILE_SHA256 = "dd28993913b265dab79e04c21828f77f1d14ae4e62b2150e3a894e28313b93f5"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "sha256:e25489c5cdb780c07083e041f44b4b07b6cdc0f285c6beb0f1f0708cf110d9a2"
)
EXPECTED_ENVIRONMENT_SHA256 = (
    "sha256:4f544cefe550a17a058d6ed6870b655d1b2ca9551f32827565a26139e6c53e7b"
)
POLICY_ID = "case-contract-repairability-v0.1"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _validate_probe(
    payload: dict[str, Any],
    *,
    case_id: str,
    selected: dict[str, Any],
    selection_sha256: str,
    test_plan_sha256: str,
    path: Path,
) -> None:
    material = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    _require(
        payload.get("evidence_sha256") == canonical_sha256(material),
        f"R11 embedded evidence digest mismatch in {path}",
    )
    expected = {
        "case_id": case_id,
        "base_sha": selected["base_sha"],
        "head_sha": selected["head_sha"],
        "selection_lock_sha256": selection_sha256,
        "test_plan_sha256": test_plan_sha256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "environment_sha256": EXPECTED_ENVIRONMENT_SHA256,
        "probe_status": "pass",
        "failure_codes": [],
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"R11 {key} mismatch in {path}")
    environment = payload.get("environment", {})
    _require(environment.get("torch_cuda_available") is True, f"R11 CUDA unavailable in {path}")
    _require(
        environment.get("cuda_dot_product") == 32.0,
        f"R11 CUDA arithmetic smoke failed in {path}",
    )


def _validate_case_facts(case_id: str, facts: dict[str, Any]) -> None:
    """Fail closed if a hand-authored blind assessment no longer matches its probe."""

    if case_id == "cutlass-pr-3352":
        _require(facts["head_cache_declaration_count"] == 6, "CUTLASS 3352 cache count changed")
        _require(
            facts["head_unique_value_cache_growth_is_unbounded_by_source"],
            "CUTLASS 3352 no longer demonstrates unbounded cache growth",
        )
        _require(
            facts["integer_pointer_runs"]["base"]["cache_entries_after_2048_unique_values"] == 0
            and facts["integer_pointer_runs"]["head"]["cache_entries_after_2048_unique_values"]
            == 2048,
            "CUTLASS 3352 growth control changed",
        )
    elif case_id == "cutlass-pr-3380":
        _require(facts["head_arch_truth_table_matches_contract"], "CUTLASS 3380 arch gate failed")
        _require(
            facts["copy_gate"]["head_has_leader_and_elect"],
            "CUTLASS 3380 leader gate failed",
        )
    elif case_id == "deepgemm-pr-327":
        _require(
            facts["head_is_finite"] and not facts["head_is_negative_infinity"],
            "DeepGEMM 327 finite counterexample changed",
        )
    elif case_id == "deepgemm-pr-337":
        _require(
            facts["base_oracle_mismatch_count"] > 0
            and facts["head_oracle_mismatch_count"] == 0
            and not facts["changed_direct_test"],
            "DeepGEMM 337 mask matrix changed",
        )
    elif case_id == "flashattention-pr-2662":
        _require(
            facts["base_boundary_mismatch_count"] == 6
            and facts["head_int64_cast_count"] == 3
            and facts["head_bounds_quotient_before_int32_cast"],
            "FlashAttention 2662 boundary matrix changed",
        )
    elif case_id == "flashattention-pr-2678":
        runtime = facts["runtime"]
        _require(
            runtime["base"]["compile_error"] is not None
            and runtime["head"]["compiled_fullgraph"] is True
            and runtime["head"]["eager_fake_context"] is True
            and runtime["head"]["eager_normal"] is False,
            "FlashAttention 2678 traceability matrix changed",
        )
    elif case_id == "flashinfer-pr-3930":
        _require(
            not facts["head_exact_match_contract_satisfied"]
            and len(facts["head_false_positive_lookalike_families"]) == 2,
            "FlashInfer 3930 look-alike counterexamples changed",
        )
    elif case_id == "flashinfer-pr-3990":
        head = facts["nvlink_matrix"]["head"]
        _require(
            head["all_active"]["need_all_up_true"] is True
            and head["one_inactive"]["need_all_up_true"] is False
            and head["all_unsupported_state"]["need_all_up_true"] is False
            and facts["head_no_capable_links_is_false"],
            "FlashInfer 3990 topology matrix changed",
        )
    elif case_id == "liger-pr-1251":
        _require(
            facts["head_clones_only_when_not_inplace"]
            and facts["head_captures_requires_grad_before_clone"]
            and facts["head_passes_inplace_through_public_api"]
            and facts["direct_test_compares_safe_path_to_torch_reference"]
            and facts["runtime_control"]["safe_working_copy_preserves_caller"],
            "Liger 1251 safe-path evidence changed",
        )
    elif case_id == "liger-pr-1283":
        rows = facts["cuda_matrix"]
        _require(
            len(rows) == 2
            and all(row["base_error"] for row in rows)
            and all(row["head_close_to_reference"] for row in rows)
            and all(row["same_dtype_base_head_equal"] for row in rows)
            and all(row["repeated_accumulation_matches_control"] for row in rows)
            and facts["head_keeps_fp32_accumulator"],
            "Liger 1283 AMP matrix changed",
        )
    elif case_id == "megatron-pr-5726":
        _require(
            facts["default_modes_matrix"]["head"]["sequence_modes"] == [0, 0]
            and facts["explicit_modes_matrix"]["head"]["sequence_modes"] == [4, 5]
            and not facts["head_validates_modes_length"]
            and facts["mismatched_modes_head"]["error"] is None,
            "Megatron 5726 cardinality matrix changed",
        )
    elif case_id == "megatron-pr-5759":
        _require(
            facts["public_entrypoint_matrix"]["base"]["error"] is not None
            and facts["public_entrypoint_matrix"]["head"]["error"] is None
            and facts["direct_test_reenabled"]
            and facts["direct_test_retains_common_state_shard"],
            "Megatron 5759 public API evidence changed",
        )
    elif case_id == "sglang-pr-31339":
        _require(
            facts["second_hop_metrics_disabled_state"]["base"] == {}
            and facts["second_hop_metrics_disabled_state"]["head"]["has_timing_data"] is True
            and facts["head_preserves_timing_on_second_hop"]
            and facts["head_default_disabled_state_is_empty"],
            "SGLang 31339 serialization matrix changed",
        )
    elif case_id == "sglang-pr-31351":
        detectors = facts["detectors"].values()
        _require(
            all(row["base"]["leak_count"] == 19 for row in detectors)
            and all(row["head"]["leak_count"] == 0 for row in detectors)
            and facts["head_all_partial_prefixes_suppressed"]
            and facts["head_plain_unicode_preserved"],
            "SGLang 31351 streaming matrix changed",
        )
    elif case_id == "torchtitan-pr-3861":
        _require(
            facts["base_silently_converts_wrapped_buffers"]
            and facts["head_preserves_all_wrapped_buffers"]
            and facts["head_still_converts_parameters"],
            "TorchTitan 3861 canonical-name matrix changed",
        )
    elif case_id == "torchtitan-pr-3869":
        table = facts["truth_table"]
        _require(
            table["head"]["valid"]["symmetric_memory_calls"] == ["frozen-tp-group"]
            and table["head"]["disabled"] == table["base"]["disabled"]
            and table["head"]["missing_compile"] == table["base"]["missing_compile"],
            "TorchTitan 3869 prerequisite table changed",
        )
    elif case_id == "verl-pr-7010":
        schedule = facts["bounded_concurrency_schedule"]
        _require(
            not schedule["base"]["lock_acquired_during_capacity_wait"]
            and schedule["head"]["lock_acquired_during_capacity_wait"]
            and schedule["head"]["sample_processed"]
            and schedule["head"]["remaining_active_tasks"] == 0,
            "verl 7010 bounded schedule changed",
        )
    elif case_id == "verl-pr-7046":
        matrix = facts["constructor_matrix"]
        _require(
            matrix["base"]["subclass_fallback"]["error"] is None
            and matrix["head"]["subclass_fallback"]["error"] is not None
            and facts["head_breaks_subclass_fallback"],
            "verl 7046 subclass regression changed",
        )
    elif case_id == "vllm-pr-48754":
        _require(
            facts["head_fixes_versioned_local_name"]
            and facts["head_preserves_custom_class"]
            and facts["head_misclassifies_identifier_only_registered_local_name"]
            and not facts["classifier_has_registry_or_filesystem_context"],
            "vLLM 48754 classifier matrix changed",
        )
    elif case_id == "vllm-pr-48755":
        matrices = facts["streaming_matrices"]
        _require(
            not facts["head_all_frozen_chunkings_pass"]
            and all(not row["head"]["all_passed"] for row in matrices.values())
            and facts["direct_test_asserts_nonempty_not_exact_reconstruction"],
            "vLLM 48755 streaming matrix changed",
        )
    else:
        raise SystemExit(f"R11 has no fact validator for {case_id}")


ASSESSMENTS: dict[str, dict[str, Any]] = {
    "cutlass-pr-3352": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=False,
            remediation_scope="bounded-multi-site",
            closure_test="frozen-probe",
            baseline_regression=True,
            safety_or_integrity_failure=True,
            design_change_required=True,
            residual_failure_families=1,
        ),
        "findings": [
            "Base retains zero entries after 2,048 distinct integer-pointer constructions; head retains 2,048.",
            "Head adds six process-lifetime dictionaries without weak ownership, eviction, or a size bound.",
            "Live pointer dereference remains correct, but the claimed leak repair introduces unbounded retained values.",
        ],
        "residual": "Replace process-lifetime value caches with bounded or lifetime-coupled ownership across six constructors.",
    },
    "cutlass-pr-3380": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "Head excludes sm80/sm89 and retains sm90/sm100 for i8-to-bf16 conversion.",
            "The s2t side effect executes only when both leader-CTA and elected-thread predicates hold.",
        ],
        "residual": None,
    },
    "deepgemm-pr-327": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=False,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            safety_or_integrity_failure=True,
            residual_failure_families=1,
        ),
        "findings": [
            "Head replaces IEEE negative infinity with finite -1e38f.",
            "The required numeric semantics are false independent of whether the compatibility expression compiles.",
        ],
        "residual": "Use a CUDA-12.8-compatible expression that still produces negative infinity.",
    },
    "deepgemm-pr-337": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            residual_failure_families=1,
        ),
        "findings": [
            "Head matches the independent exponent-mask oracle for all 24 frozen bit patterns; base misses 17.",
            "Equal exponents with different mantissas become equal as required.",
            "The PR contains no direct distinguishing regression despite the frozen acceptance requirement.",
        ],
        "residual": "Add one direct nonzero-mantissa regression using the demonstrated matrix.",
    },
    "flashattention-pr-2662": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "Base wraps at six of eight frozen forward/backward 2^31 boundary rows.",
            "Head widens all three affected products and narrows only after bounded division.",
        ],
        "residual": None,
    },
    "flashattention-pr-2678": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "The exact base function fails torch.compile(fullgraph=True); head compiles and executes.",
            "Head preserves false in eager mode and true under FakeTensorMode.",
        ],
        "residual": None,
    },
    "flashinfer-pr-3930": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            safety_or_integrity_failure=True,
            residual_failure_families=2,
        ),
        "findings": [
            "Head rejects libcudart_stub.so and selects the later canonical runtime.",
            "It still selects libcudart.evil.so and libcudart.so.stub.so ahead of a real runtime.",
        ],
        "residual": "Replace prefix matching with an exact supported libcudart filename grammar covering both look-alike families.",
    },
    "flashinfer-pr-3990": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "Head reports true for complete active links and false for an inactive link, no capable links, and all state-unsupported slots.",
            "A state-unsupported enumerated slot is excluded from the usable-link denominator; at least one queryable active link remains required.",
        ],
        "residual": None,
    },
    "liger-pr-1251": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        ),
        "findings": [
            "The new opt-in safe path clones after capturing requires_grad and before destructive workspace reuse.",
            "The public API forwards inplace=False and a direct branched-gradient regression compares it with PyTorch.",
            "The explicit/default inplace=True compatibility path is retained.",
        ],
        "residual": None,
    },
    "liger-pr-1283": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "On A100, base addmm raises for both FP16/FP32 and BF16/FP32 operand pairs.",
            "Head aligns only the input operand, keeps FP32 accumulation, and matches the FP32 reference tolerance.",
            "Same-dtype and repeated-accumulation controls remain stable.",
        ],
        "residual": None,
    },
    "megatron-pr-5726": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            residual_failure_families=1,
        ),
        "findings": [
            "Head derives two default modes from two lengths and preserves valid explicit modes.",
            "A malformed one-mode/two-length input is still accepted and leaves metadata cardinalities misaligned.",
        ],
        "residual": "Add a local modes-versus-lengths cardinality guard and its frozen malformed-input regression.",
    },
    "megatron-pr-5759": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        ),
        "findings": [
            "The base public entrypoint dispatches to an absent Save-strategy method; head dispatches to Load successfully.",
            "Head uses the resolved metadata filename consistently and re-enables a direct test retaining unrelated common state.",
        ],
        "residual": None,
    },
    "sglang-pr-31339": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "Head preserves all populated timing fields across a second metrics-disabled IPC hop; base drops them.",
            "A default metrics-disabled record remains an empty compatibility payload.",
        ],
        "residual": None,
    },
    "sglang-pr-31351": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        ),
        "findings": [
            "Both detectors suppress all 19 strict bot-token prefixes; base leaks every prefix.",
            "Plain Unicode text is preserved and direct split regressions cover both detectors.",
        ],
        "residual": None,
    },
    "torchtitan-pr-3861": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "Base compares raw wrapper-qualified names and silently downcasts two wrapped buffers.",
            "Head canonicalizes both sides, preserves float32/float64/int buffers, and still converts parameters.",
        ],
        "residual": None,
    },
    "torchtitan-pr-3869": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "For valid AsyncTP, head enables symmetric memory for the resolved TP group before the micro-pipeline flag.",
            "Disabled and missing-compile prerequisite controls remain unchanged and fail safely.",
        ],
        "residual": None,
    },
    "verl-pr-7010": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        ),
        "findings": [
            "A deterministic bounded schedule cannot acquire the base lock during capacity wait but can acquire head's lock.",
            "Head still processes the sample, removes completed tasks under lock, and leaves no active-task leak.",
        ],
        "residual": None,
    },
    "verl-pr-7046": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            baseline_regression=True,
            residual_failure_families=1,
        ),
        "findings": [
            "Head replaces the base missing-schema AttributeError with an actionable assertion.",
            "It also rejects a previously valid subclass property fallback, regressing constructor precedence.",
        ],
        "residual": "Preserve subclass-provided schema fallback while preventing recursion for the base default.",
    },
    "vllm-pr-48754": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="cross-cutting",
            closure_test="frozen-probe",
            design_change_required=True,
            residual_failure_families=1,
        ),
        "findings": [
            "Head fixes Qwen3.5 and preserves pkg.MyProposer as a custom class.",
            "It still classifies an identifier-only registered dotted local name as a class.",
            "The syntax-only helper has neither registry nor filesystem context to resolve that intrinsic ambiguity.",
        ],
        "residual": "Move the decision to a resolver with registry/filesystem context or require an explicit class marker.",
    },
    "vllm-pr-48755": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="cross-cutting",
            closure_test="frozen-probe",
            design_change_required=True,
            residual_failure_families=3,
        ),
        "findings": [
            "Head improves frozen two-key, Unicode-escape, and nested matrices over base.",
            "It still reconstructs only 26/66, 38/71, and 24/80 chunkings respectively.",
            "The direct test checks nonempty output rather than exact once-only reconstruction.",
        ],
        "residual": "Redesign parser state so name and raw-argument deltas are tracked independently across all three boundary families.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require(
        _file_sha256(args.selection_lock) == EXPECTED_SELECTION_FILE_SHA256,
        "R11 selection-lock file digest mismatch",
    )
    _require(
        _file_sha256(args.test_plan) == EXPECTED_TEST_PLAN_FILE_SHA256,
        "R11 test-plan file digest mismatch",
    )
    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_material = selection["selection_material"]
    _require(
        selection["selection_lock_sha256"] == canonical_sha256(selection_material),
        "R11 embedded selection digest mismatch",
    )
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    _require(
        plan["test_plan_sha256"] == canonical_sha256(plan_material),
        "R11 embedded test-plan digest mismatch",
    )
    _require(
        plan["selection_lock_sha256"] == selection["selection_lock_sha256"],
        "R11 test plan is not bound to selection lock",
    )
    blind_flags = (
        selection_material["review_text_visible_to_machine_judge"],
        selection_material["merge_outcomes_visible_to_machine_judge"],
        selection_material["ci_fields_visible_to_machine_judge"],
        plan["review_text_visible_to_machine_judge"],
        plan["merge_outcomes_visible_to_machine_judge"],
        plan["review_text_requested"],
    )
    _require(all(value is False for value in blind_flags), "R11 blind boundary is not intact")
    _require(
        plan["frozen_before_source_diff_content_inspection"] is True,
        "R11 test plan was not frozen before source inspection",
    )
    _require(
        plan["scoring_policy"]["weighted_score_used"] is False
        and plan["scoring_policy"]["forced_polarization_used"] is False,
        "R11 unexpectedly enables weighting or forced polarization",
    )

    selected = {item["case_id"]: item for item in selection_material["cases"]}
    planned = {item["case_id"]: item for item in plan["cases"]}
    _require(selected.keys() == planned.keys(), "R11 selection/test-plan case sets differ")
    _require(selected.keys() == ASSESSMENTS.keys(), "R11 assessment case set differs")
    evidence_dir = args.evidence_dir or args.result_root / "probes"
    evidence: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for case_id, selected_case in selected.items():
        _require(
            selected_case["base_sha"] == planned[case_id]["base_sha"]
            and selected_case["head_sha"] == planned[case_id]["head_sha"],
            f"R11 selected/planned SHA mismatch for {case_id}",
        )
        path = evidence_dir / f"{case_id}.json"
        payload = _read(path)
        _validate_probe(
            payload,
            case_id=case_id,
            selected=selected_case,
            selection_sha256=selection["selection_lock_sha256"],
            test_plan_sha256=plan["test_plan_sha256"],
            path=path,
        )
        _validate_case_facts(case_id, payload["facts"])
        evidence[case_id] = payload
        bindings[case_id] = {
            "path": f"probes/{case_id}.json",
            "evidence_sha256": payload["evidence_sha256"],
            "artifact_sha256": canonical_sha256(payload),
        }

    frozen_at = datetime.now(UTC).isoformat()
    locks = []
    for case_id, selected_case in selected.items():
        assessment = ASSESSMENTS[case_id]
        triage = assessment["triage"]
        result = classify_case_contract(triage)
        # R11 was hash-frozen before the terminology migration. Preserve its historical wire
        # label while all new classifiers expose the same state as ``check``.
        frozen_decision = "revise" if result.decision == "check" else result.decision
        legacy_decision = "accept_with_scope" if triage.contract_satisfied else "revise"
        material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256(
                {"selection": selected_case, "test_plan": planned[case_id]}
            ),
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "evidence_binding_sha256": canonical_sha256(bindings[case_id]),
            "triage_input": asdict(triage),
            "decision": frozen_decision,
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
        "selection_lock_file_sha256": "sha256:" + EXPECTED_SELECTION_FILE_SHA256,
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_file_sha256": "sha256:" + EXPECTED_TEST_PLAN_FILE_SHA256,
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "environment_sha256": EXPECTED_ENVIRONMENT_SHA256,
        "frozen_at": frozen_at,
        "decision_counts": decision_counts,
        "legacy_r10_style_decision_counts": legacy_counts,
        "evidence_bindings": bindings,
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
