#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze R12 communication judgments before revealing outcomes or reviews."""

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

EXPECTED_SELECTION_FILE_SHA256 = "bd2dae7e7d1153ad44e83e580b5e6d416ded915f52697c751edd75068527541f"
EXPECTED_TEST_PLAN_FILE_SHA256 = "24574ab89d532f23b3e71d995394dae8ce6d466565ddf4e2aa5a206d112fbbb3"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "sha256:c25d5a59d67fe0979b1ea6210ab39aa109c3bf167830afbb2c619bcb1e0a92a6"
)
EXPECTED_PROBE_ENVIRONMENT_SHA256 = (
    "sha256:35ce10fa020d509d7110e8d6cd7c03a54d34d16652781d0cc087781b42424120"
)
POLICY_ID = "communication-case-contract-v0.1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _validate_payload_digest(payload: dict[str, Any], field: str, label: str) -> None:
    material = {key: value for key, value in payload.items() if key != field}
    _require(payload.get(field) == canonical_sha256(material), f"{label} digest mismatch")


def _validate_probe(
    payload: dict[str, Any],
    *,
    case_id: str,
    selected: dict[str, Any],
    selection_sha256: str,
    test_plan_sha256: str,
    path: Path,
) -> None:
    _validate_payload_digest(payload, "evidence_sha256", f"R12 probe {path}")
    expected = {
        "case_id": case_id,
        "base_sha": selected["base_sha"],
        "head_sha": selected["head_sha"],
        "selection_lock_sha256": selection_sha256,
        "test_plan_sha256": test_plan_sha256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "environment_sha256": EXPECTED_PROBE_ENVIRONMENT_SHA256,
        "probe_status": "pass",
        "failure_codes": [],
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"R12 {key} mismatch in {path}")
    environment = payload.get("environment", {})
    _require(environment.get("torch_cuda_available") is True, f"R12 CUDA unavailable in {path}")
    _require(environment.get("gpu_count") == 2, f"R12 did not record two GPUs in {path}")
    _require(
        environment.get("gpu_names") == ["NVIDIA A100-SXM4-40GB"] * 2,
        f"R12 GPU identity changed in {path}",
    )


def _validate_case_facts(case_id: str, facts: dict[str, Any]) -> None:
    if case_id == "cutlass-pr-3294":
        _require(
            facts["changed_example_count"] == 7
            and facts["all_examples_support_both_cuda_core_generations"]
            and facts["patch_is_import_only"]
            and facts["blackwell_runtime_is_non_blocking_for_frozen_claim"],
            "CUTLASS 3294 import matrix changed",
        )
    elif case_id == "flashinfer-pr-3939":
        _require(
            facts["checkpoint_api_count"] == {"base": 4, "head": 0}
            and facts["checkpoint_test_removed"]
            and not facts["head_named_checkpoint_recovery_supported"]
            and facts["head_fail_closes_checkpoint_calls_by_absent_api"],
            "FlashInfer 3939 checkpoint removal changed",
        )
    elif case_id == "flashinfer-pr-3931":
        _require(
            facts["binding_kernel_call_count"] == 2
            and facts["binding_kernel_calls_forward_do_finalize"] == [False, False]
            and facts["kernel_default_for_omitted_do_finalize_is_true"]
            and not facts["effective_skip_reached_from_python_false"]
            and not facts["direct_test_detects_finalize_was_skipped"],
            "FlashInfer 3931 deferred-finalize data flow changed",
        )
    elif case_id == "flashinfer-pr-3880":
        runtime = facts["cuda_runtime"]
        _require(
            facts["head_all_partial_warps_mask_safe"]
            and facts["head_unsafe_partial_size_count"] == 0
            and facts["changed_direct_test"]
            and facts["direct_test_checks_sum_and_max"]
            and facts["required_cuda_runtime_executed"]
            and runtime["safe_partial_all_match"],
            "FlashInfer 3880 partial-warp matrix changed",
        )
    elif case_id == "flashinfer-pr-3879":
        _require(
            not facts["head_all_partial_warps_mask_safe"]
            and facts["head_unsafe_partial_size_count"] == 10
            and facts["decisive_static_failure"]
            and facts["required_cuda_runtime_executed"]
            and not facts["changed_direct_test"],
            "FlashInfer 3879 unsafe partial-warp evidence changed",
        )
    elif case_id == "megatron-pr-5720":
        _require(
            facts["one_sum_to_destination_leader_with_group"]
            and facts["both_backward_entrypoints_reduce_before_send"]
            and facts["direct_cp2_vs_cp1_numeric_test"]
            and facts["direct_test_covers_steady_and_cooldown"]
            and facts["title_scope_is_destination_cp_only"],
            "Megatron 5720 destination-CP evidence changed",
        )
    elif case_id == "sglang-pr-31311":
        _require(
            facts["head_named_allreduce_cardinality_matches"]
            and facts["base_double_reduces_real_a2a"]
            and facts["valid_named_geometry_aligns"]
            and facts["helper_silently_short_slices_non_divisible_geometry"]
            and not facts["candidate_owned_closure_test"],
            "SGLang 31311 call/shape matrix changed",
        )
    elif case_id == "sglang-pr-31290":
        _require(
            facts["all_modeled_ranks_use_identical_order"]
            and facts["coordinator_validates_shape_and_sizes"]
            and facts["direct_test_checks_uneven_values"]
            and not facts["direct_test_has_zero_size_rank"]
            and not facts["required_xpu_runtime_available"]
            and not facts["required_xpu_runtime_executed"],
            "SGLang 31290 XPU evidence boundary changed",
        )
    elif case_id == "torchtitan-pr-3827":
        _require(
            facts["head_logprob_gradient_matches_oracle"]
            and facts["non_tp_groups_misrouted_to_tp"]
            and facts["wrapper_ignores_supplied_group_when_intercepting"]
            and facts["root_module_keys_have_leading_dot"]
            and len(facts["residual_failure_families"]) == 2,
            "TorchTitan 3827 compound evidence changed",
        )
    elif case_id == "torchtitan-pr-3821":
        _require(
            facts["head_recognizes_all_reduce_start_and_wait"]
            and facts["source_change_is_scheduler_type_extension_only"]
            and facts["direct_test_has_ddp_two_gpu_case"]
            and facts["direct_test_distinguishes_call_counts"]
            and facts["direct_test_checks_graph_lint"]
            and not facts["direct_test_checks_numeric_equivalence"],
            "TorchTitan 3821 graph evidence changed",
        )
    elif case_id == "verl-pr-6958":
        _require(
            facts["modeled_resize_reuse_sequence"]["all_accepted"]
            and facts["receiver_synchronizes_before_ack"]
            and facts["sender_synchronizes_before_each_publish"]
            and not facts["sender_cache_checks_device"]
            and not facts["cache_has_explicit_eviction_or_process_teardown"]
            and not facts["candidate_owned_closure_test"],
            "verl 6958 ownership evidence changed",
        )
    elif case_id == "vllm-pr-48763":
        _require(
            facts["generic_sequence_parallel_algebra_equal"]
            and facts["nvidia_rowwise_norm_commutes_with_row_gather"]
            and facts["head_gathers_once_after_residual_add"]
            and facts["modeled_payload_reduction_fraction"] == 0.5
            and not facts["candidate_contains_paired_benchmark"]
            and not facts["required_two_gpu_performance_executed"],
            "vLLM 48763 source/performance boundary changed",
        )
    else:
        raise SystemExit(f"R12 has no fact validator for {case_id}")


ASSESSMENTS: dict[str, dict[str, Any]] = {
    "cutlass-pr-3294": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="frozen-probe",
        ),
        "findings": [
            "All seven changed distributed examples prefer cuda.core.Device and fall back only on ImportError.",
            "Replacing the guarded import with the former experimental import reproduces the exact base sources; collective code is unchanged.",
        ],
        "residual": None,
    },
    "flashinfer-pr-3939": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=False,
            remediation_scope="cross-cutting",
            closure_test="missing",
            design_change_required=True,
            residual_failure_families=1,
        ),
        "findings": [
            "Head removes all four checkpoint prepare/restore entrypoints, the graph-replay regression, and the stable-VA remap state.",
            "The named checkpoint recovery sequence therefore cannot run; absence fails closed but does not satisfy recovery compatibility.",
        ],
        "residual": "Reintroduce a safe, single-owner checkpoint/remap protocol and a two-rank capture/replay/restore/cleanup regression.",
    },
    "flashinfer-pr-3931": {
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
            "Python forwards do_finalize=False to the binding, but both binding-to-kernel calls omit the flag and inherit true.",
            "The new early return is unreachable from the public false path, so the promised pre-reduce buffer boundary is not established.",
            "The direct test compares a manual finalize but has no assertion proving that kernel finalization was actually skipped.",
        ],
        "residual": "Forward do_finalize through both kernel calls and add an assertion that distinguishes pre-reduce from already-finalized storage.",
    },
    "flashinfer-pr-3880": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        ),
        "findings": [
            "Head adds active-mask partial-warp sum/max primitives and uses the ceil warp count.",
            "The A100 CUDA matrix passes all 15 partial/full-warp sizes, and candidate tests directly cover sum and max at non-warp-multiple sizes.",
        ],
        "residual": None,
    },
    "flashinfer-pr-3879": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=False,
            remediation_scope="single-site",
            closure_test="missing",
            safety_or_integrity_failure=True,
            residual_failure_families=1,
        ),
        "findings": [
            "Head changes floor to ceil warp count but still uses full-mask shuffles in a partial warp.",
            "Ten frozen sizes retain invalid-lane participation, and the candidate adds no direct regression.",
        ],
        "residual": "Use an active mask plus initialized-lane bounds for both reduction stages and retain the shared partial-warp matrix as a direct test.",
    },
    "megatron-pr-5720": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        ),
        "findings": [
            "Destination CP ranks reduce one contiguous gradient onto the destination leader's CP group before either backward send entrypoint.",
            "The title is destination-CP scoped; source CP remains explicitly unsupported rather than silently misrouted.",
            "Candidate tests encode CP2-vs-CP1 numeric reconstruction for steady and cooldown paths; a two-rank NCCL reduce control reaches the leader exactly.",
        ],
        "residual": None,
    },
    "sglang-pr-31311": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            residual_failure_families=1,
        ),
        "findings": [
            "The real-A2A truth table removes exactly the redundant model-level all-reduce while preserving non-A2A cardinality.",
            "Named divisible gather/slice geometries align, but the unconditional helper silently returns too few rows for a non-divisible shrink on the last rank.",
            "No candidate-owned regression covers the new collective and row-alignment branches.",
        ],
        "residual": "Guard divisibility/rank bounds in _scmoe_align_rows and add the frozen non-divisible plus real-A2A call-count matrix.",
    },
    "sglang-pr-31290": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=False,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="existing",
            residual_failure_families=1,
        ),
        "findings": [
            "Source and modeled schedules agree on rank order, offsets, global-rank destinations, and preallocated shapes for world sizes 2, 3, and 4.",
            "Candidate tests cover an uneven two-rank value oracle and input immutability but not a zero-size rank.",
            "The frozen XPU numeric runtime is unavailable on the CUDA host, so backend-specific XCCL behavior remains infrastructure-unresolved.",
        ],
        "residual": "Run the existing uneven tests plus a zero-tail case on two or more Intel XPUs.",
    },
    "torchtitan-pr-3827": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=False,
            remediation_scope="bounded-multi-site",
            closure_test="missing",
            safety_or_integrity_failure=True,
            residual_failure_families=2,
        ),
        "findings": [
            "Identity placement fixes the trainer logprob gradient scale.",
            "The redistribute wrapper intercepts every no-grad replicate/identity destination and routes it to global TP even when a different process group was supplied.",
            "Root-module FusedSwiGLU keys are emitted with a leading dot, leaving the compound weight-sync claim incomplete.",
        ],
        "residual": "Make interception group-aware, fix root-module key joining, and add direct regressions for both independent failure families.",
    },
    "torchtitan-pr-3821": {},
    "verl-pr-6958": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="bounded-multi-site",
            closure_test="missing",
            safety_or_integrity_failure=True,
            residual_failure_families=2,
        ),
        "findings": [
            "Same-size generations reuse one synchronized sender/receiver mapping and reject a missing or stale generation.",
            "The sender cache key checks endpoint and byte count but not device, and neither process-global cache has explicit eviction or teardown ownership.",
            "The patch adds no lifecycle regression for device change, cancellation, restart, or final release.",
        ],
        "residual": "Bind cache compatibility to device ownership and add explicit paired teardown/eviction with deterministic restart and cancellation tests.",
    },
    "vllm-pr-48763": {
        "triage": CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="frozen-probe",
            residual_failure_families=1,
        ),
        "findings": [
            "Both changed model paths add or normalize locally before one gather; row-wise RMSNorm commutes with row gather and non-sequence all-reduce remains.",
            "On two A100s, old 2H and new H gather paths are exactly equal for four sizes; the new path is faster on every rank with median reductions of roughly 25% to 53%.",
            "The candidate contains no paired benchmark, and the evaluator microbenchmark does not establish the title's 5% full-model E2E gain.",
        ],
        "residual": "Add a paired full-model TP2 benchmark with the claimed workload, repeated spread, and an explicit no-regression threshold.",
    },
}


def _torchtitan_3821_assessment(exact: dict[str, Any]) -> dict[str, Any]:
    facts = exact["facts"]
    passed = facts["execution_status"] == "pass" and facts["all_ranks_candidate_contract_passed"]
    if passed:
        triage = CaseContractTriageEvidence(
            contract_satisfied=True,
            evidence_complete=True,
            primary_claim_demonstrated=True,
            closure_test="existing",
        )
        residual = None
    else:
        triage = CaseContractTriageEvidence(
            contract_satisfied=False,
            evidence_complete=False,
            primary_claim_demonstrated=True,
            remediation_scope="single-site",
            closure_test="existing",
        )
        residual = "Run the exact DDP graph rewrite on the PyTorch nightly ABI that supplies all-reduce bucketing."
    return {
        "triage": triage,
        "findings": [
            "The candidate change only teaches the scheduler to recognize bucketed all-reduce start/wait nodes; upstream grouping invariants remain shared.",
            "A two-rank NCCL control proves six compatible SUM reductions are numerically identical when flattened into one bucket.",
            f"Exact candidate DDP execution status: {facts['execution_status']}.",
        ],
        "residual": residual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--dual-gpu-evidence", type=Path, required=True)
    parser.add_argument("--torchtitan-exact-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require(
        _file_sha256(args.selection_lock) == EXPECTED_SELECTION_FILE_SHA256,
        "R12 selection-lock file digest mismatch",
    )
    _require(
        _file_sha256(args.test_plan) == EXPECTED_TEST_PLAN_FILE_SHA256,
        "R12 test-plan file digest mismatch",
    )
    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_material = selection["selection_material"]
    _require(
        selection["selection_lock_sha256"] == canonical_sha256(selection_material),
        "R12 embedded selection digest mismatch",
    )
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    _require(
        plan["test_plan_sha256"] == canonical_sha256(plan_material),
        "R12 embedded test-plan digest mismatch",
    )
    _require(
        plan["selection_lock_sha256"] == selection["selection_lock_sha256"],
        "R12 test plan is not bound to selection lock",
    )
    blind_flags = (
        selection_material["review_text_visible_to_machine_judge"],
        selection_material["merge_outcomes_visible_to_machine_judge"],
        selection_material["ci_fields_visible_to_machine_judge"],
        plan["review_text_visible_to_machine_judge"],
        plan["merge_outcomes_visible_to_machine_judge"],
        plan["review_text_requested"],
    )
    _require(all(value is False for value in blind_flags), "R12 blind boundary is not intact")
    _require(
        plan["frozen_before_source_diff_content_inspection"] is True,
        "R12 plan was not frozen before source inspection",
    )
    _require(
        plan["scoring_policy"]["weighted_score_used"] is False
        and plan["scoring_policy"]["forced_polarization_used"] is False,
        "R12 unexpectedly enables weighting or forced polarization",
    )

    selected = {item["case_id"]: item for item in selection_material["cases"]}
    planned = {item["case_id"]: item for item in plan["cases"]}
    _require(selected.keys() == planned.keys(), "R12 selection/test-plan case sets differ")
    _require(selected.keys() == ASSESSMENTS.keys(), "R12 assessment case set differs")

    evidence_dir = args.evidence_dir or args.result_root / "probes"
    evidence: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for case_id, selected_case in selected.items():
        _require(
            selected_case["base_sha"] == planned[case_id]["base_sha"]
            and selected_case["head_sha"] == planned[case_id]["head_sha"],
            f"R12 selected/planned SHA mismatch for {case_id}",
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

    dual = _read(args.dual_gpu_evidence)
    _validate_payload_digest(dual, "evidence_sha256", "R12 dual-GPU evidence")
    _require(
        dual["selection_lock_sha256"] == selection["selection_lock_sha256"]
        and dual["test_plan_sha256"] == plan["test_plan_sha256"]
        and dual["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256,
        "R12 dual-GPU evidence binding mismatch",
    )
    dual_facts = dual["facts"]
    _require(
        dual_facts["nccl_two_rank_smoke_passed"]
        and dual_facts["megatron_5720_destination_cp_sum_reaches_leader"]
        and dual_facts["torchtitan_3821_bucket_equivalence_all_ranks"]
        and dual_facts["vllm_48763_all_shapes_equivalent"]
        and dual_facts["vllm_48763_all_shapes_faster"]
        and "not full-model E2E" in dual_facts["vllm_48763_scope"],
        "R12 dual-GPU facts changed",
    )
    dual_binding = {
        "path": args.dual_gpu_evidence.name,
        "evidence_sha256": dual["evidence_sha256"],
        "artifact_sha256": canonical_sha256(dual),
    }

    exact = _read(args.torchtitan_exact_evidence)
    _validate_payload_digest(exact, "evidence_sha256", "R12 TorchTitan exact evidence")
    exact_facts = exact["facts"]
    _require(
        exact["case_id"] == "torchtitan-pr-3821"
        and exact["selection_lock_sha256"] == selection["selection_lock_sha256"]
        and exact["test_plan_sha256"] == plan["test_plan_sha256"]
        and exact_facts["exact_head_sha"] == selected["torchtitan-pr-3821"]["head_sha"]
        and exact_facts["world_size"] == 2,
        "R12 TorchTitan exact evidence binding mismatch",
    )
    exact_binding = {
        "path": args.torchtitan_exact_evidence.name,
        "evidence_sha256": exact["evidence_sha256"],
        "artifact_sha256": canonical_sha256(exact),
    }
    ASSESSMENTS["torchtitan-pr-3821"] = _torchtitan_3821_assessment(exact)

    frozen_at = datetime.now(UTC).isoformat()
    locks = []
    for case_id, selected_case in selected.items():
        assessment = ASSESSMENTS[case_id]
        triage = assessment["triage"]
        result = classify_case_contract(triage)
        legacy_decision = "accept_with_scope" if triage.contract_satisfied else "check"
        supplemental = []
        if case_id in {"megatron-pr-5720", "torchtitan-pr-3821", "vllm-pr-48763"}:
            supplemental.append(dual_binding)
        if case_id == "torchtitan-pr-3821":
            supplemental.append(exact_binding)
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
        "probe_environment_sha256": EXPECTED_PROBE_ENVIRONMENT_SHA256,
        "dual_gpu_evidence_binding": dual_binding,
        "torchtitan_exact_evidence_binding": exact_binding,
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
