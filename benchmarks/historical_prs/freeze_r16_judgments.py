#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the 30-case R16 training cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.triage import CaseContractTriageEvidence, classify_case_contract
from infraswe.io import atomic_write_json

EXPECTED_FILE_SHA256 = {
    "selection": "825ea8f67c7316e2cd930c259b19cc414b9c510cf16182f763d5f4e044d905c9",
    "plan": "484c4a7bd38278fb3607565143a47c43a973dac801b3ef89ea5d8e386fa01692",
    "manifest": "86f945e77b2b7663749bd3b98622af5f850993fe5af354504cac080484ac7dca",
    "static": "f357d41a14c208dcb7839b6d8118ef59e61711985788c411786652ebed6f5ab3",
    "initial": "1712e2b978ca942b65d8f2f9d665705fa49475f9a23a2fec260aaf5ba171993c",
    "followup": "8cb5a49fa61709d1e85037efac61539aab5580ba4bb01369be87944423e8de41",
    "orpo_retry": "d86b23615da15cb6267058246d204c046179772ce97242772503d5a0cc2a2d96",
    "checkpoint_retry": "f25ce5ca82a27cd4dfabd232cd19c9b08871d1a9b51db9194eb4e36ba0eeeb94",
    "checkpoint_focused": "f96ba5d65224891b2553a8e5c6be63b3549d951f306daf387782c13551e8532a",
}
EXPECTED_SELECTION_SHA256 = "sha256:e62dfd65af89ec1b75847d007c2a3f1d4ff0d37e8ac5f6ed792df7193e4c24fa"
EXPECTED_TEST_PLAN_SHA256 = "sha256:1bf58d5803699b63323b7ab09feb2f5c599fa4368efa7527025c8f6172d5f2a0"
EXPECTED_SOURCE_BUNDLE_SHA256 = "sha256:09e279a10bb85a5dfe9dbf56a4009f1f2c987380e8f137e168a5d60be9d1dbbf"
POLICY_ID = "training-contract-disposition-split-v0.1-r16"


@dataclass(frozen=True, slots=True)
class Assessment:
    triage: CaseContractTriageEvidence
    technical_contract: str
    findings: tuple[str, ...]
    residual: str | None = None


ACCEPT = CaseContractTriageEvidence(True, True, True, closure_test="frozen-probe")
CHECK = CaseContractTriageEvidence(
    False,
    True,
    True,
    remediation_scope="single-site",
    closure_test="frozen-probe",
    residual_failure_families=1,
)
REJECT_UNPROVEN = CaseContractTriageEvidence(
    False,
    True,
    False,
    remediation_scope="unknown",
    closure_test="missing",
    residual_failure_families=1,
)
REJECT_BROAD = CaseContractTriageEvidence(
    False,
    True,
    True,
    remediation_scope="cross-cutting",
    closure_test="missing",
    design_change_required=True,
    residual_failure_families=2,
)
REJECT_FAILURE = CaseContractTriageEvidence(
    False,
    True,
    False,
    remediation_scope="single-site",
    closure_test="frozen-probe",
    baseline_regression=True,
    safety_or_integrity_failure=True,
    residual_failure_families=1,
)


def assessment(
    triage: CaseContractTriageEvidence,
    technical_contract: str,
    *findings: str,
    residual: str | None = None,
) -> Assessment:
    return Assessment(triage, technical_contract, findings, residual)


ASSESSMENTS: dict[str, Assessment] = {
    "liger-pr-1413": assessment(
        CHECK,
        "bounded-gap",
        "The one-file change separates the two fp32 dot accumulators exactly as its Blackwell miscompile hypothesis requires; six A100 fused-MoE cases passed before the bounded suite timeout.",
        "The evaluator has no B300/sm_103, so the title-scoped dx correction remains unexecuted locally despite detailed candidate target results.",
        residual="Run the frozen multi-expert dx reference matrix on B300/sm_103 and confirm both dot contributions across top-k and dtype variants.",
    ),
    "liger-pr-1244": assessment(
        ACCEPT,
        "pass",
        "The exact head imports LigerORPOTrainer through TRL 1.12's experimental ORPO export, while the guarded fallback preserves the former import location.",
        "The two-site change is restricted to import compatibility and the current-path probe succeeds.",
    ),
    "liger-pr-1204": assessment(
        ACCEPT,
        "pass",
        "All 322 selected candidate DPO formula/value/gradient cases pass across the five added losses, dtype, bias, reference, and normalization variants.",
        "Validation rejects invalid loss and label-smoothing inputs, and no existing loss-mode counterexample appears.",
    ),
    "liger-pr-1202": assessment(
        ACCEPT,
        "pass",
        "Sixty-nine candidate GRPO value/gradient and normalization cases pass, with only explicitly unsupported VESPO/LUSPO combinations skipped.",
        "The six-path patch has direct fused/unfused reference coverage at the altered loss boundaries.",
    ),
    "liger-pr-1253": assessment(
        ACCEPT,
        "pass",
        "All five GroupNorm forward/backward shape and precision cases pass on A100.",
        "The one-file fix maps FP16 accumulators to FP16 rather than BF16 and leaves FP32/BF16 branches explicit.",
    ),
    "liger-pr-1208": assessment(
        ACCEPT,
        "pass",
        "All 24 DyT forward/backward numeric cases pass after real Triton autotuning on A100.",
        "The DA buffer is reset across variable-grid tuning trials, while DG/DB coverage is fully overwritten for every chosen block width.",
    ),
    "megatron-pr-7021": assessment(
        ACCEPT,
        "pass",
        "Argument validation and the exact parameter/router ownership probe pass; the pre-wrap hook is idempotent and excludes frozen parameters before distributed wrapping.",
        "Candidate functional evidence covers MTP forward/backward, recomputation, gradients, and optimizer state on one GPU; the missing local TE package is not an observed counterexample.",
    ),
    "megatron-pr-5145": assessment(
        ACCEPT,
        "pass",
        "The candidate LatentMoE accounting test passes and the formula separately counts routed latent dimensions, full-width shared experts, and both latent projections.",
        "The two-file mature correction has a locally closed component inventory.",
    ),
    "megatron-pr-5169": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "The refactor adds a 363-line inference configuration surface and rewires seven production/tool paths without a matching candidate test.",
        "Importability alone would not close training field projection, distributed construction, or a forward/backward step.",
        residual="Enumerate all moved fields and run equivalent MCore/MBridge training construction plus forward/backward and inference controls.",
    ),
    "megatron-pr-5134": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source consistently replaces deprecated strategy factories with concrete torch strategies and updates affected imports.",
        "Every changed checkpoint suite that reaches the altered behavior requires an eight-rank topology; one- and two-rank attempts stop at the declared world-size precondition, leaving save/load compatibility unexecuted.",
        residual="Run the changed serialization, integrity, fully-parallel, and MSC suites at their eight-rank topology and verify checkpoint round trips.",
    ),
    "megatron-pr-5131": assessment(
        ACCEPT,
        "pass",
        "The source makes CUDA-graph capture and replay preserve the dataloader's absent-mask signature while retaining masks for required attention types.",
        "Three target tests are correctly skipped without TransformerEngine, while the candidate supplies detailed H100/CP memory measurements and target-path coverage with no contrary evidence.",
    ),
    "megatron-pr-5162": assessment(
        ACCEPT,
        "pass",
        "Both exact tests pass: programmatic ModelParallelConfig construction warns but succeeds, while training-argument validation still rejects unsafe TE cross entropy fusion.",
        "The two-file change matches the title-scoped ownership boundary.",
    ),
    "slime-pr-2345": assessment(
        CHECK,
        "bounded-gap",
        "The exact production coroutine sorts nested sample groups as 2,5,7 and avoids treating list.index as a numeric field.",
        "The recent one-file fix has one residual because no candidate regression test preserves nested/empty/stable-order behavior in the repository.",
        residual="Add the executed nested-group matrix as a candidate-owned test, including empty groups and equal-index stable ordering.",
    ),
    "slime-pr-2010": assessment(
        ACCEPT,
        "pass",
        "The base/head probe preserves response slices and gradients while reducing first-yield peak allocation from 163,577,856 to 46,137,344 bytes.",
        "Scalar temperature division commutes with each frozen slicing path and the one-file change removes the full-logits temporary.",
    ),
    "slime-pr-2015": assessment(
        ACCEPT,
        "pass",
        "All three candidate lifecycle tests pass, covering pause, flush, release, restore, and continue ordering.",
        "The production boundary now owns quiescence before offload memory release.",
    ),
    "slime-pr-2014": assessment(
        ACCEPT,
        "pass",
        "All seven candidate plugin rollout contracts pass after installing the declared lightweight parser dependency.",
        "Filtering is applied once after validation and before manager ownership, while the all-samples provider hook remains intentionally upstream.",
    ),
    "slime-pr-1969": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Three candidate tests close asset copying, stale-weight cleanup, and standalone safetensor shard/index writing.",
        "The ten-path raw-mode feature also changes model iteration, direct/bridge mappings, arguments, and distributed save ownership without an HF reload, logits, or continued-training closure.",
        residual="Save a sharded raw-mode model through each iterator, reload it with Transformers, compare logits, and continue one optimizer step.",
    ),
    "slime-pr-2020": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Two candidate node-writer tests pass for merged writer state and incomplete final groups.",
        "No multiprocess node execution or elapsed-time comparison demonstrates the title's acceleration, atomic publication, or writer-failure behavior.",
        residual="Run serial versus multi-node writer saves with checksum/reload parity, failure injection, and measured elapsed time.",
    ),
    "torchtitan-pr-4358": assessment(
        ACCEPT,
        "pass",
        "Eight focused trainer/config replay tests pass, including failure-before-optimizer, first-microbatch scope, checkpoint rearming, and determinism guards.",
        "The candidate also registers one- and two-GPU mismatch/cudagraph integration controls, providing functional training closure beyond configuration plumbing.",
    ),
    "torchtitan-pr-3523": assessment(
        ACCEPT,
        "pass",
        "Both candidate CPU-offload dependency tests pass and explicitly cover non-Tensor consumers.",
        "The three-path change retains the wait edge through the graph schema boundary without broad state changes.",
    ),
    "torchtitan-pr-3538": assessment(
        ACCEPT,
        "pass",
        "All three exact call sites changed to rely on the forward default pass, including a real CUDA compile/capture replay.",
        "The earlier whole-file 23-versus-24 metadata count is outside the changed call sites and is not treated as a candidate failure.",
    ),
    "torchtitan-pr-3534": assessment(
        ACCEPT,
        "pass",
        "Both candidate cudagraph staging tests pass for the altered non-static input path.",
        "The one-production-file change provides shared-pool capture/replay parity and rejects unsafe alias behavior.",
    ),
    "torchtitan-pr-3533": assessment(
        ACCEPT,
        "pass",
        "Both candidate per-node capture-safety tests pass.",
        "The predicate and mixed fallback are title-scoped and have direct graph behavior coverage.",
    ),
    "torchtitan-pr-3530": assessment(
        ACCEPT,
        "pass",
        "The candidate rematerialization test passes and proves duplicate custom metadata has distinct identity while preserving values.",
        "The two-file fix is numerically neutral and closes the exact mutable-alias invariant.",
    ),
    "verl-pr-7697": assessment(
        ACCEPT,
        "pass",
        "Twenty-nine selected agent-loop and configuration tests pass, covering invalid categories, ordered parallel results, resets, thresholds, and default-off behavior.",
        "The functional behavior is tested beyond the generated configuration propagation.",
    ),
    "verl-pr-6558": assessment(
        REJECT_FAILURE,
        "fail",
        "The exact production head has an unclosed parenthesis at ray_trainer.py:400 and cannot be imported.",
        "Both static parsing and py_compile independently reproduce the candidate-owned collection failure.",
        residual="Restore syntactically complete OmegaConf selection and pass the exhaustion boundary plus parameter/state immutability test.",
    ),
    "verl-pr-6564": assessment(
        ACCEPT,
        "pass",
        "All three candidate packed-boundary tests pass without FlashAttention.",
        "Labels are shifted in padded layout then gathered by the existing indices, preventing cross-sequence leakage in both torch and Megatron paths.",
    ),
    "verl-pr-6574": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "All eight CPU sizing and validation tests pass for default, constrained, explicit, and invalid reservations.",
        "The ten-path scheduling change has no actual Ray actor placement/progress or restart resource-release test, so helper arithmetic alone does not close the nonprogress claim.",
        residual="Start the fully-async actor set on a tight local Ray cluster, prove progress, and verify resources are released and reacquired on restart.",
    ),
    "verl-pr-6560": assessment(
        ACCEPT,
        "pass",
        "Seven selected config/serializer tests pass locally and the candidate reports nineteen across defaults, validation, docs, and CLI projection.",
        "The eleven paths are mechanical propagation of three backward-compatible values through enumerated adapters, with no tool-execution semantics added.",
    ),
    "verl-pr-6598": assessment(
        REJECT_BROAD,
        "unresolved",
        "The six-path patch introduces a new cross-node NCCL group, synchronous event-loop wrapper, rank mapping, broadcast, and HTTP activation flow.",
        "No candidate test runs, the body is an untouched template, and neither two-rank progress nor coherent version activation is demonstrated.",
        residual="Add and run multi-rank group initialization, sharded weight broadcast, slow/failing receiver, version activation, and repeated-update tests.",
    ),
}


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_digest(payload: dict[str, Any], field: str, label: str) -> None:
    material = {key: value for key, value in payload.items() if key != field}
    require(payload.get(field) == canonical_sha256(material), f"{label} digest mismatch")


def binding(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": path.name,
        "evidence_sha256": payload.get("evidence_sha256") or payload.get("evidence_manifest_sha256"),
        "artifact_sha256": canonical_sha256(payload),
    }


def record_bindings(path: Path, payload: dict[str, Any], case_id: str) -> list[dict[str, Any]]:
    result = []
    for index, record in enumerate(payload.get("records", [])):
        if record.get("case_id") != case_id:
            continue
        result.append(
            {
                "artifact": binding(path, payload),
                "record_index": index,
                "returncode": record.get("returncode"),
                "status": record.get("status"),
                "output_sha256": record.get("output_sha256"),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(file_sha256(args.selection_lock) == EXPECTED_FILE_SHA256["selection"], "R16 selection file digest mismatch")
    require(file_sha256(args.test_plan) == EXPECTED_FILE_SHA256["plan"], "R16 test-plan file digest mismatch")
    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    require(selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]), "R16 embedded selection digest mismatch")
    require(selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R16 selection identity changed")
    require(plan["test_plan_sha256"] == canonical_sha256({key: value for key, value in plan.items() if key != "test_plan_sha256"}), "R16 embedded test-plan digest mismatch")
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R16 test-plan identity changed")
    require(plan["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R16 plan/selection binding mismatch")
    require(plan["disposition_policy"]["weighted_score_used"] is False, "R16 unexpectedly uses weighted scoring")
    require(plan["disposition_policy"]["forced_polarization_used"] is False, "R16 unexpectedly forces polarization")
    blind_flags = (
        selection["selection_material"]["review_or_comment_visible"],
        selection["selection_material"]["merge_outcomes_visible"],
        selection["selection_material"]["ci_or_label_visible"],
        selection["selection_material"]["candidate_body_visible"],
        selection["selection_material"]["diff_content_visible"],
        plan["review_or_comment_requested"],
        plan["merge_outcome_or_state_requested"],
        plan["ci_or_label_requested"],
    )
    require(all(value is False for value in blind_flags), "R16 blind boundary is not intact")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 30, "R16 cohort is not 30 cases")
    require(selected.keys() == planned.keys() == ASSESSMENTS.keys(), "R16 case sets differ")

    filenames = {
        "manifest": "source-evidence-manifest.json",
        "static": "static-evidence.json",
        "initial": "upstream-test-matrix.json",
        "followup": "upstream-followup-tests.json",
        "orpo_retry": "upstream-followup-orpo-retry.json",
        "checkpoint_retry": "upstream-followup-checkpoint-retry.json",
        "checkpoint_focused": "upstream-followup-checkpoint-focused.json",
    }
    evidence_paths = {name: args.result_root / filename for name, filename in filenames.items()}
    evidence = {name: read(path) for name, path in evidence_paths.items()}
    for name, path in evidence_paths.items():
        require(file_sha256(path) == EXPECTED_FILE_SHA256[name], f"{name} file digest mismatch")
    validate_digest(evidence["manifest"], "evidence_manifest_sha256", "manifest")
    for name in ("static", "initial", "followup", "orpo_retry", "checkpoint_retry", "checkpoint_focused"):
        validate_digest(evidence[name], "evidence_sha256", name)
    require(evidence["manifest"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256, "R16 source bundle identity changed")
    require(evidence["static"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256, "R16 static/source binding mismatch")

    initial_by_id = {record["case_id"]: record for record in evidence["initial"]["records"]}
    followup_by_id = {record["case_id"]: record for record in evidence["followup"]["records"]}
    require(initial_by_id["liger-pr-1204"]["returncode"] == 0 and "322 passed" in initial_by_id["liger-pr-1204"]["output_tail"], "R16 DPO evidence changed")
    require(initial_by_id["liger-pr-1202"]["returncode"] == 0 and "69 passed" in initial_by_id["liger-pr-1202"]["output_tail"], "R16 GRPO evidence changed")
    require(followup_by_id["liger-pr-1208"]["returncode"] == 0 and "24 passed" in followup_by_id["liger-pr-1208"]["output_tail"], "R16 DyT evidence changed")
    require('"mode": "base", "first_yield_peak_bytes": 163577856' in followup_by_id["slime-pr-2010"]["output_tail"], "R16 slime base peak changed")
    require('"mode": "head", "first_yield_peak_bytes": 46137344' in followup_by_id["slime-pr-2010"]["output_tail"], "R16 slime head peak changed")
    require("8 passed" in followup_by_id["torchtitan-pr-4358"]["output_tail"], "R16 TorchTitan replay evidence changed")
    require("SyntaxError: '(' was never closed" in followup_by_id["verl-pr-6558"]["output_tail"], "R16 verl syntax counterexample changed")
    require(evidence["orpo_retry"]["records"][0]["returncode"] == 0 and "current_trl_orpo_import=pass" in evidence["orpo_retry"]["records"][0]["output_tail"], "R16 ORPO retry changed")

    execution_artifacts = ("initial", "followup", "orpo_retry", "checkpoint_retry", "checkpoint_focused")
    execution_bindings: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in selected}
    for name in execution_artifacts:
        for case_id in selected:
            execution_bindings[case_id].extend(record_bindings(evidence_paths[name], evidence[name], case_id))
    require(all(execution_bindings.values()), "R16 case missing execution record")

    common_evidence = {name: binding(evidence_paths[name], evidence[name]) for name in evidence_paths}
    frozen_at = datetime.now(UTC).isoformat()
    locks: list[dict[str, Any]] = []
    for case_id, selected_case in selected.items():
        planned_case = planned[case_id]
        require(selected_case["base_sha"] == planned_case["base_sha"] and selected_case["head_sha"] == planned_case["head_sha"], f"{case_id}: selection/test-plan SHA mismatch")
        assessed = ASSESSMENTS[case_id]
        result = classify_case_contract(assessed.triage)
        if result.decision == "check":
            require(selected_case["temporal_band"] == "recent", f"{case_id}: mature case cannot be check")
            require(len(selected_case["paths"]) <= 8, f"{case_id}: check exceeds eight changed paths")
        legacy = "accept_with_scope" if assessed.triage.contract_satisfied else "check"
        supplemental = execution_bindings[case_id]
        lock_material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256({"selection": selected_case, "test_plan": planned_case}),
            "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
            "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
            "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
            "common_evidence_binding_sha256": canonical_sha256(common_evidence),
            "supplemental_evidence_binding_sha256": canonical_sha256(supplemental),
            "technical_contract": assessed.technical_contract,
            "triage_input": asdict(assessed.triage),
            "decision": result.decision,
            "rationale_codes": list(result.rationale_codes),
            "technical_findings": list(assessed.findings),
            "residual_contract": assessed.residual,
            "hot_window_check_eligible": selected_case["temporal_band"] == "recent" and len(selected_case["paths"]) <= 8,
            "legacy_r10_style_decision": legacy,
            "frozen_at": frozen_at,
        }
        locks.append({"material": lock_material, "lock_sha256": canonical_sha256(lock_material)})

    decision_counts = {
        decision: sum(lock["material"]["decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check", "reject", "unresolved")
    }
    output_material = {
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
        "selection_lock_file_sha256": "sha256:" + EXPECTED_FILE_SHA256["selection"],
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "test_plan_file_sha256": "sha256:" + EXPECTED_FILE_SHA256["plan"],
        "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "candidate_body_integrity_note": "All bodies were acquired after the plan lock; no outcome-bearing block required redaction in R16.",
        "common_evidence_bindings": common_evidence,
        "frozen_at": frozen_at,
        "decision_counts": decision_counts,
        "legacy_r10_style_decision_counts": {
            "accept_with_scope": sum(lock["material"]["legacy_r10_style_decision"] == "accept_with_scope" for lock in locks),
            "check": sum(lock["material"]["legacy_r10_style_decision"] == "check" for lock in locks),
        },
        "locks": locks,
    }
    output = {**output_material, "lock_set_sha256": canonical_sha256(output_material)}
    atomic_write_json(args.output, output)
    print(json.dumps({
        "lock_set_sha256": output["lock_set_sha256"],
        "decision_counts": decision_counts,
        "decisions": {lock["material"]["case_id"]: lock["material"]["decision"] for lock in locks},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
