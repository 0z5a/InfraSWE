#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze training-specific R13 contracts before source diff or PR body inspection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

CASE_PLANS: dict[str, dict[str, Any]] = {
    "flashattention-pr-2654": {
        "claim": "CuTe FlashAttention backward supports the title-scoped score_mod derivative path on SM80 and SM120 without changing the identity path.",
        "training_risk": "custom attention-score autograd and backward-kernel architecture routing",
        "execution_tier": "exact source data-flow plus SM80 forward/backward comparison against a PyTorch autograd oracle",
        "required_hardware": "SM80 CUDA GPU for the runnable title-scoped architecture; SM120 is a corroborating architecture",
        "questions": [
            "For identity and representative differentiable score_mod functions, do output and dQ/dK/dV match an independent autograd oracle?",
            "Is the derivative applied at the correct pre-softmax value with the correct softmax scale and masking semantics?",
            "Does architecture routing make the named SM80 path reachable without changing non-score_mod behavior?",
            "Is there a candidate-owned regression that distinguishes base from head on a supported architecture?",
        ],
        "decision_rule": "Accept if the reachable SM80 score_mod backward matrix and identity controls pass. One bounded modifier or retained-test gap with a runnable closure test is check; wrong gradients, wrong masking, compile failure in the named path, or an identity regression is reject. Missing required dependencies is unresolved.",
    },
    "liger-pr-1274": {
        "claim": "The SAPO loss path avoids a torch.compile graph break while preserving eager forward and backward semantics.",
        "training_risk": "compiled RL loss graph capture and gradient equivalence",
        "execution_tier": "exact base/head function isolation under eager and torch.compile(fullgraph=True)",
        "required_hardware": "CUDA GPU for the normal training path; CPU graph isolation is diagnostic",
        "questions": [
            "Does head compile as one full graph on representative positive/negative log-probability inputs while base distinguishes the failure?",
            "Do eager and compiled loss values and gradients agree across dtypes and boundary coefficients?",
            "Does the change avoid data-dependent Python branching or scalar extraction?",
            "Does a candidate-owned test exercise both compile closure and numeric gradients?",
        ],
        "decision_rule": "Accept if head closes the named graph break and matches eager values/gradients. One bounded missing compile regression with evaluator closure is check; a retained break, wrong loss, wrong gradient, or eager regression is reject. Environment-only compiler failure is unresolved.",
    },
    "liger-pr-1268": {
        "claim": "The single-shot cross-entropy dx_y correction preserves exact loss and input gradients while reducing correction overhead.",
        "training_risk": "fused cross-entropy backward gradient accumulation and performance",
        "execution_tier": "exact algebra/source isolation plus A100 dtype, reduction, ignore-index, and shape matrix with paired timing",
        "required_hardware": "CUDA GPU",
        "questions": [
            "Do loss and logits gradients match an independent torch cross_entropy oracle for mean, sum, none, ignore-index, and label smoothing controls?",
            "Is the target-column correction applied exactly once for every valid row and skipped for ignored rows?",
            "Are FP32 accumulation and FP16/BF16 output semantics unchanged?",
            "Does paired timing show no material regression, and is candidate benchmark evidence available?",
        ],
        "decision_rule": "Accept if all reachable numeric controls pass and the performance direction is non-regressive. A correct kernel with only bounded missing/noisy performance evidence is check; any reachable gradient error, ignored-row write, dtype regression, or measured material slowdown is reject. Missing CUDA is unresolved.",
    },
    "liger-pr-1230": {
        "claim": "LigerORPOTrainer works when FSDP is not configured and retains the existing FSDP setup path.",
        "training_risk": "trainer configuration branching and distributed-plugin ownership",
        "execution_tier": "exact constructor/setup isolation across no-FSDP, FSDP1, and neighboring trainer controls",
        "required_hardware": "none for configuration ownership proof; a CUDA training step is corroborating",
        "questions": [
            "Does a no-FSDP trainer initialize without dereferencing or mutating an absent FSDP plugin?",
            "Does the FSDP path still set every intended option on the actual plugin owner?",
            "Are accelerator state and unrelated trainer configurations unchanged?",
            "Does a candidate-owned regression distinguish no-FSDP base from head and retain an FSDP control?",
        ],
        "decision_rule": "Accept if no-FSDP initialization is safe and the reachable FSDP branch is preserved. One bounded missing FSDP control with a closure test is check; an absent-plugin crash, wrong owner mutation, or FSDP regression is reject.",
    },
    "megatron-pr-5808": {
        "claim": "MegatronFSDP dispatches root-module hooks through the correct root state without skipping or duplicating child hooks.",
        "training_risk": "FSDP module-hook traversal, root ownership, and exactly-once execution",
        "execution_tier": "exact base/head hook state machine plus two-rank root/child module execution",
        "required_hardware": "two CUDA GPUs for distributed corroboration; deterministic traversal proof is required locally",
        "questions": [
            "Does the root hook run exactly once against the root state for root-only and nested module trees?",
            "Do child hooks retain their owner, order, arguments, and exactly-once cardinality?",
            "Are forward/backward values and parameter gradients unchanged relative to an unhooked reference?",
            "Does the changed test distinguish base and include a nested control?",
        ],
        "decision_rule": "Accept if reachable root and child hook dispatch is owner-correct, exactly once, and numerically neutral. One bounded untested nesting pattern with a closure test is check; skipped/duplicate hooks, wrong owner state, rank divergence, or gradient change is reject.",
    },
    "megatron-pr-5798": {
        "claim": "Sequence-level auxiliary MoE loss is invariant to batch packing and batch-size partition while preserving token weighting and gradients.",
        "training_risk": "auxiliary loss normalization and MoE router gradient scale",
        "execution_tier": "exact formula extraction plus batch partition, padding, sequence-length, and gradient oracle matrix",
        "required_hardware": "CUDA GPU for training-tensor execution; formula proof is architecture-independent",
        "questions": [
            "Does one logical token/sequence set yield the same auxiliary loss under equivalent batch partitions?",
            "Are padding, sequence boundaries, expert counts, and variable lengths normalized over the intended denominator?",
            "Do router-logit gradients match an independently grouped reference and retain expected scale?",
            "Does a direct candidate test include at least two batch sizes that distinguish base from head?",
        ],
        "decision_rule": "Accept if loss and gradients are partition-invariant for all supported layouts. One bounded unsupported padding layout with a direct closure test is check; batch-dependent loss, wrong gradient scale, NaN, or a neighboring token-level regression is reject.",
    },
    "megatron-pr-5743": {
        "claim": "HSDP can defer the DP-outer gradient reduction and later execute exactly one equivalent reduction without stale or overwritten buffers.",
        "training_risk": "deferred distributed gradient reduction, buffer lifecycle, and rank agreement",
        "execution_tier": "exact state-machine/source proof plus two-rank NCCL eager-versus-deferred numeric control",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Does defer suppress only the outer reduction while preserving inner shard semantics and accumulation?",
            "Does the later flush reduce every intended gradient exactly once on the correct group and op?",
            "Are buffers owned until completion across microbatches, no_sync, zero-grad, and exception/empty cases?",
            "Do candidate tests distinguish eager and deferred paths and compare resulting gradients?",
        ],
        "decision_rule": "Accept if deferred and eager reachable paths are numerically equivalent with exactly-once reduction and safe ownership. One bounded lifecycle case with a runnable closure test is check; missing/duplicate reduction, wrong group, stale buffer, deadlock, or gradient divergence is reject. Missing two-GPU execution is unresolved.",
    },
    "megatron-pr-5742": {
        "claim": "Lion is routed through DistributedOptimizer and its single-moment state can be saved and restored without Adam-only assumptions.",
        "training_risk": "distributed optimizer dispatch and checkpoint state-schema compatibility",
        "execution_tier": "exact optimizer construction/state serialization plus continuous-versus-resumed update trajectory",
        "required_hardware": "none for optimizer/checkpoint proof; two-rank execution is corroborating",
        "questions": [
            "Does selecting Lion construct the intended distributed wrapper rather than an Adam-specific path?",
            "Does checkpoint save/load preserve the single moment, step, group metadata, dtype, and shard mapping without requiring a second moment?",
            "Do uninterrupted and save/resume parameter trajectories match for multiple steps?",
            "Do Adam and non-distributed Lion controls remain unchanged with direct tests?",
        ],
        "decision_rule": "Accept if Lion dispatch and one-moment checkpoint/resume exactly match a continuous reference while neighboring optimizers pass. One bounded metadata compatibility gap with a closure test is check; wrong optimizer routing, state loss, second-moment assumption, or resumed divergence is reject.",
    },
    "torchtitan-pr-3841": {
        "claim": "Graph pipeline parallelism splits backward into dI and dW graphs without losing dependencies, gradients, or supported graph nodes.",
        "training_risk": "AOT backward graph partitioning and dependency preservation",
        "execution_tier": "exact FX graph pass over linear, residual, shared-parameter, and control graphs with independent autograd comparison",
        "required_hardware": "none for FX partition proof; CUDA execution is corroborating",
        "questions": [
            "Does the split assign input-gradient and weight-gradient nodes to the correct graph exactly once?",
            "Are shared ancestors, saved tensors, ordering edges, outputs, and graph lint preserved?",
            "Do recomposed dI/dW results match the unsplit backward for representative modules?",
            "Do direct tests include a base-distinguishing graph plus residual/shared dependency controls?",
        ],
        "decision_rule": "Accept if every supported graph partitions, lints, recomposes, and matches autograd. One explicitly guarded unsupported node family with a runnable closure test is check; omitted/duplicate nodes, broken dependencies, wrong gradients, or silent fallback is reject.",
    },
    "torchtitan-pr-3897": {
        "claim": "The RL training pipeline supports FP16 with correct loss, optimizer, attention, and generator dtype behavior.",
        "training_risk": "mixed-precision training stability and cross-component dtype contracts",
        "execution_tier": "exact compound dtype/data-flow matrix plus A100 multi-step FP16 training smoke against FP32/BF16 controls",
        "required_hardware": "CUDA GPU with FP16 support",
        "questions": [
            "Are logits, loss reductions, optimizer parameters/states, masks, and attention intermediates in safe intended dtypes?",
            "Does loss scaling or equivalent overflow handling prevent silent zero/inf gradients over multiple steps?",
            "Do FP16 updates track an FP32 reference within tolerance and leave BF16 behavior unchanged?",
            "Do candidate tests exercise every title-implied component rather than only configuration parsing?",
        ],
        "decision_rule": "Accept only if the compound FP16 pipeline trains stably and every named component satisfies dtype and gradient controls. Exactly one bounded local component gap with a direct closure test is check; NaN/inf, silent underflow, wrong dtype, wrong update, or multiple residual components is reject. Missing compatible CUDA execution is unresolved.",
    },
    "torchtitan-pr-3867": {
        "claim": "TorchStore weight synchronization correctly transfers models with fused parameters using the public split state-dict contract.",
        "training_risk": "RL trainer-to-generator parameter naming, fusion mapping, shape, and version ownership",
        "execution_tier": "exact fused-QKV state mapping and sender/receiver round-trip with unfused controls",
        "required_hardware": "none for state mapping; two-rank CUDA execution is corroborating",
        "questions": [
            "Are every fused source parameter and exposed split destination key mapped exactly once with correct slice, shape, dtype, and order?",
            "Does a sync round-trip reproduce all receiver weights for fused QKV and neighboring unfused modules?",
            "Are missing, duplicate, extra, and version-mismatched keys rejected rather than silently skipped?",
            "Does the candidate direct test distinguish base and cover both fused and unfused models?",
        ],
        "decision_rule": "Accept if reachable fused/unfused round-trips are exact and key validation fails closed. One bounded additional fusion family with a direct closure test is check; missing/duplicate/wrong slices, silent skip, or receiver divergence is reject.",
    },
    "verl-pr-7014": {
        "claim": "FSDP synchronizes merged LoRA weights before leaving the merge context so exported base weights contain the adapter update.",
        "training_risk": "LoRA merge/unmerge lifetime and FSDP weight-export ordering",
        "execution_tier": "exact context-manager/source ordering plus two-rank merge-sync-unmerge value model",
        "required_hardware": "two CUDA GPUs for distributed corroboration; ownership/order proof is required locally",
        "questions": [
            "Is synchronization invoked while merged weights are still live and before unmerge restores local base storage?",
            "Do all ranks/export consumers observe base plus LoRA delta exactly once?",
            "Are exception, nested context, no-adapter, and repeated-sync paths ownership-safe?",
            "Does a candidate test distinguish the former stale export and verify post-context restoration?",
        ],
        "decision_rule": "Accept if reachable sync observes the merged value exactly once and context exit restores the intended local state. One bounded lifecycle case with a closure test is check; stale base export, double delta, rank divergence, or unsafe exception ownership is reject.",
    },
    "verl-pr-7013": {
        "claim": "Adaptive KL controller state persists across checkpoint resume for every changed trainer path and produces the same subsequent trajectory.",
        "training_risk": "stateful RL control checkpoint completeness and resume determinism",
        "execution_tier": "exact state-schema/source matrix plus uninterrupted-versus-resumed controller trajectory",
        "required_hardware": "none",
        "questions": [
            "Are controller value, target, horizon, update count, and variant-specific fields serialized and restored at the correct step?",
            "Do all changed trainer implementations use the same authoritative state owner without double restore?",
            "Does a save/resume sequence produce the same next KL coefficients as uninterrupted execution?",
            "Do old checkpoints without the field have an explicit compatible default and direct test?",
        ],
        "decision_rule": "Accept if every title-scoped trainer resumes an identical controller trajectory and compatibility is explicit. One bounded legacy-schema case with a closure test is check; state loss, wrong step ordering, divergent trajectory, or multiple unhandled trainers is reject.",
    },
    "verl-pr-6984": {
        "claim": "Dropping per-micro-batch model_output releases training graph memory without changing accumulated gradients, metrics, or required post-step consumers.",
        "training_risk": "autograd graph lifetime, microbatch accumulation, and actor-update peak memory",
        "execution_tier": "exact use/lifetime analysis plus A100 multi-microbatch weak-reference, memory, and gradient control",
        "required_hardware": "CUDA GPU for the OOM/memory claim; source liveness proof is required locally",
        "questions": [
            "Is model_output dead after each backward, with no later metric, auxiliary loss, logging, or hook consumer?",
            "Do weak references and allocated-memory traces show graphs become reclaimable between microbatches?",
            "Do accumulated gradients, loss, metrics, and optimizer updates exactly match the retained-output baseline?",
            "Does candidate evidence include a workload that previously grows or OOMs and now stays bounded?",
        ],
        "decision_rule": "Accept if output is provably dead, memory is bounded/reduced, and all training results match. A correct lifetime fix with only bounded missing large-model peak evidence is check; a later consumer break, gradient/metric change, retained graph, or measured memory regression is reject. Missing CUDA is unresolved.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("R13 selection lock digest mismatch")
    hidden = (
        material["review_text_visible_to_machine_judge"],
        material["merge_outcomes_visible_to_machine_judge"],
        material["ci_fields_visible_to_machine_judge"],
        material["candidate_body_visible_during_selection"],
        material["diff_content_visible_during_selection"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R13 selection exposes hidden evidence")
    cases = [HistoricalPRCandidate.model_validate(item) for item in material["cases"]]
    if {item.case_id for item in cases} != set(CASE_PLANS):
        raise SystemExit("R13 selection and plan case sets differ")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "machine_policy_id": material["machine_policy_id"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "review_text_requested": False,
        "candidate_body_requested": False,
        "frozen_before_candidate_body_acquisition": True,
        "frozen_before_source_diff_content_inspection": True,
        "claim_scope_policy": {
            "title_scoped_acceptance_is_blocking": True,
            "unclaimed_hardening_is_blocking": False,
            "compound_title_claims_are_all_blocking": True,
            "synthetic_failure_requires_production_reachability_to_block": True,
        },
        "candidate_evidence_policy": {
            "body_may_be_acquired_after_this_lock": True,
            "candidate_authored_tests_and_benchmarks_are_admissible": True,
            "body_claim_is_not_evaluator_execution": True,
            "outcome_review_ci_fields_remain_forbidden": True,
        },
        "scoring_policy": {
            "kind": "ordered exact training-contract judgment with repairability triage",
            "weighted_score_used": False,
            "forced_polarization_used": False,
            "decisions": ["accept_with_scope", "check", "reject", "unresolved"],
            "check_is_bounded_repairability_or_bounded_claim-evidence_gap": True,
            "missing_required_environment_evidence": "unresolved, never candidate fail",
            "technical_contract_and_disposition_are_reported_separately": True,
        },
        "cross_case_controls": [
            "A synthetic shape, module root, process group, or state transition cannot block unless a supported training configuration reaches it.",
            "Performance or memory improvement cannot compensate for wrong loss, gradient, optimizer state, rank agreement, or checkpoint trajectory.",
            "Candidate PR-body evidence is recorded separately from evaluator execution and cannot expose state, merge, review, label, or CI fields.",
            "Base and head use the same numeric, lifecycle, and distributed matrix; head does not receive an easier oracle.",
            "Static source evidence cannot be promoted to runtime evidence when the frozen contract requires CUDA or two-rank execution.",
        ],
        "cases": [
            {
                "case_id": item.case_id,
                "project": item.project,
                "repository": item.repository,
                "pull_number": item.pull_number,
                "base_sha": item.base_sha,
                "head_sha": item.head_sha,
                "changed_paths": item.paths,
                **CASE_PLANS[item.case_id],
            }
            for item in cases
        ],
    }
    payload = {**plan_material, "test_plan_sha256": canonical_sha256(plan_material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(cases)}")
    print(f"test_plan_sha256={payload['test_plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
