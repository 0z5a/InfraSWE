#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze communication-specific R12 contracts before source diff inspection."""

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
    "cutlass-pr-3294": {
        "claim": "Every changed distributed CuTeDSL example imports CUDA Device from a supported module and can reach argument setup without a Device NameError.",
        "communication_risk": "distributed example launch portability",
        "execution_tier": "exact base/head Python import, syntax, and launch-prefix isolation",
        "required_hardware": "none for the title-scoped import claim; Blackwell execution is non-blocking hardening",
        "questions": [
            "Do all seven changed distributed examples resolve the same supported Device symbol?",
            "Does Python compilation and an import-isolated launch prefix succeed for every example?",
            "Are collective algorithms and launch arguments otherwise unchanged?",
            "Is there a direct smoke path that distinguishes the broken import from head?",
        ],
        "decision_rule": "Accept if all affected examples resolve Device and the patch is import-only. One missed local example with a runnable closure probe is check; an invalid replacement module or a collective semantic change is reject.",
    },
    "flashinfer-pr-3939": {
        "claim": "Reverting graph-VA preservation restores safe all-reduce checkpoint recovery without stale mappings, double release, or replay corruption.",
        "communication_risk": "CUDA graph virtual-address and communicator resource lifecycle",
        "execution_tier": "exact ownership/state model plus CUDA-graph checkpoint/replay matrix",
        "required_hardware": "two CUDA GPUs; MNNVL-only coverage may be reported separately when unavailable",
        "questions": [
            "After restore, are graph buffers and IPC mappings reconstructed from one live owner rather than stale preserved VAs?",
            "Do capture, replay, restore, recapture, and repeated close avoid use-after-free and double-unmap states?",
            "Do eager and non-checkpoint all-reduce paths retain their prior lifecycle?",
            "Do changed tests exercise the failure-producing restore sequence and repeated cleanup?",
        ],
        "decision_rule": "Accept only if the named checkpoint sequence is safe and numerically correct. A single bounded backend gap with an executable closure test is check; stale VA reuse, double release, wrong reduction, or a cross-backend lifecycle redesign is reject. Missing required hardware is unresolved.",
    },
    "flashinfer-pr-3931": {
        "claim": "The SM90 CUTLASS fused-MoE runner exposes the exact pre-reduce buffer when finalization is deferred, while preserving finalized output compatibility.",
        "communication_risk": "fused-compute/collective buffer ABI and ownership",
        "execution_tier": "exact C++/Python data-flow contract plus SM90 output-equivalence probe",
        "required_hardware": "SM90 CUDA GPU for full execution; source-level ABI checks remain diagnostic",
        "questions": [
            "When do_finalize is false, is the returned tensor the buffer immediately before reduction/finalization?",
            "Are shape, dtype, stride, device, aliasing, and lifetime suitable for the fused all-reduce consumer?",
            "When do_finalize is true, are legacy return type and numeric output unchanged?",
            "Does a direct test distinguish pre-reduce from finalized output and prevent double finalization?",
        ],
        "decision_rule": "Accept if both deferred and compatibility paths satisfy the buffer ABI and output oracle. One local metadata/lifetime omission with a closure test is check; a wrong buffer, dangling alias, double finalization, or numeric regression is reject. Missing SM90 execution is unresolved.",
    },
    "flashinfer-pr-3880": {
        "claim": "TRT-LLM block reductions are correct and synchronization-safe when blockDim.x is not a multiple of the warp size.",
        "communication_risk": "partial-warp active-mask and shared reduction semantics",
        "execution_tier": "exact CUDA compile/run matrix against an independent reduction oracle",
        "required_hardware": "CUDA GPU",
        "questions": [
            "Do block sizes 1, 7, 31, 32, 33, 47, 63, 64, 96, 127, and 128 match the oracle?",
            "Does every warp use a valid active mask and does the final warp-aggregate read only initialized lanes?",
            "Are scalar/vector and neighboring full-warp cases unchanged?",
            "Does the changed direct test contain at least one base-distinguishing partial-warp case?",
        ],
        "decision_rule": "Accept if every frozen size is correct and the direct regression distinguishes base. One bounded uncovered type with the same safe primitive is check; invalid-lane reads, divergence, hangs, or wrong sums are reject. Missing CUDA is unresolved.",
    },
    "flashinfer-pr-3879": {
        "claim": "blockReduceSumV2 returns the correct sum for non-warp-multiple block dimensions.",
        "communication_risk": "partial-warp reduction correctness",
        "execution_tier": "the same exact CUDA reduction matrix used for the neighboring follow-up",
        "required_hardware": "CUDA GPU",
        "questions": [
            "Do all frozen partial- and full-warp block sizes match the independent oracle?",
            "Are inactive lanes excluded without reading uninitialized shared values?",
            "Does the fix remain correct for more than one warp and for a one-lane tail?",
            "Is a candidate-owned direct regression present for the base-distinguishing case?",
        ],
        "decision_rule": "Accept if the full matrix passes and the change is retained by direct coverage. A correct implementation missing only one local regression is check; any wrong sum, invalid access, or synchronization hazard is reject. Missing CUDA is unresolved.",
    },
    "megatron-pr-5720": {
        "claim": "A non-colocated bridge communicator maps and transfers tensors correctly when the destination context-parallel degree differs from the source.",
        "communication_risk": "cross-mesh rank mapping, split shape, and matched send/receive order",
        "execution_tier": "exact topology/rank state-machine matrix with mocked process groups",
        "required_hardware": "none for deterministic rank-map proof; distributed execution is corroborating",
        "questions": [
            "Are source and destination CP ranks mapped bijectively for 1x1, 1x2, 2x1, 2x2, and 2x4 matrices?",
            "Do every sender and receiver agree on peer, tensor shape, tag, and operation order?",
            "Are sequence/context splits lossless for uneven and neighboring even shapes?",
            "Do colocated and equal-CP paths remain unchanged with direct tests?",
        ],
        "decision_rule": "Accept if all supported topology matrices are bijective, shape-safe, and order-matched. One explicitly unsupported bounded topology with a direct guard is check; an unmatched peer/order, silent truncation, or deadlock-capable mapping is reject.",
    },
    "sglang-pr-31311": {
        "claim": "LongCat-2.0 real expert parallelism issues the intended all-reduce exactly once and avoids the named ScMoE RoPE crash.",
        "communication_risk": "configuration-dependent collective cardinality and tensor-shape compatibility",
        "execution_tier": "exact configuration truth table with collective call tracing and RoPE shape controls",
        "required_hardware": "none for isolated call/shape contract; multi-GPU DeepEP execution is corroborating",
        "questions": [
            "Across real-EP, DeepEP, shared-expert, and fallback combinations, is each required all-reduce issued exactly once?",
            "Are configurations requiring no model-level reduction left at zero calls?",
            "Do ScMoE RoPE inputs use compatible dimensions without masking unrelated shape errors?",
            "Do neighboring non-real-EP LongCat paths retain output and call-count behavior?",
        ],
        "decision_rule": "Accept if the named real-EP path fixes both collective cardinality and RoPE shape while preserving neighbors. One remaining local configuration with a direct closure test is check; double/missing reduction, wrong shape, or a crash in the named path is reject.",
    },
    "sglang-pr-31290": {
        "claim": "XPU DP-attention reduce_scatterv and all_gatherv use correct rank-local shapes and splits for uneven workloads.",
        "communication_risk": "variable-count collective shape agreement and XCCL API semantics",
        "execution_tier": "exact fake-XCCL call capture plus uneven world/rank shape matrix",
        "required_hardware": "XPU for full numeric execution; call-shape proof is required locally",
        "questions": [
            "For world sizes 2, 3, and 4, do uneven and zero-tail split vectors agree on every rank?",
            "Are input/output sizes, dim selection, dtype, device, and synchronous/async return semantics correct?",
            "Does all_gatherv invert the supported reduce_scatterv layouts without loss or duplication?",
            "Do direct tests include an actually uneven split that distinguishes base from head?",
        ],
        "decision_rule": "Accept if call-shape proof and available XPU execution cover the title-scoped uneven layouts. One bounded zero-size or async gap with a closure test is check; rank-disagreeing counts, silent truncation, or hang-capable calls are reject. Missing XPU numeric execution alone is unresolved.",
    },
    "torchtitan-pr-3827": {
        "claim": "The three named RL fixes preserve trainer logprob gradients, route generator TP all-reduce correctly, and synchronize FusedSwiGLU weights.",
        "communication_risk": "DTensor placement, autograd, and weight-replica consistency",
        "execution_tier": "exact three-claim graph/placement matrix with independent math oracles",
        "required_hardware": "none for DTensor graph isolation; multi-rank execution is corroborating",
        "questions": [
            "Does trainer logprob retain the expected gradient path and scale?",
            "Does generator TP perform exactly one reduction on the correct mesh/group with correct placement?",
            "Are both FusedSwiGLU weight components synchronized without duplicate or missing collectives?",
            "Are non-TP and non-fused neighboring configurations unchanged?",
        ],
        "decision_rule": "Accept only if all three explicitly named fixes satisfy their independent oracle. Exactly one bounded local residual with a runnable closure test is check; wrong gradients, replica divergence, duplicate/missing collectives, or multiple residual families are reject.",
    },
    "torchtitan-pr-3821": {
        "claim": "GraphTrainer buckets replicated all-reduce operations under HSDP/DDP without changing graph dependencies or numerical reduction semantics.",
        "communication_risk": "collective bucketing, process-group identity, and graph ordering",
        "execution_tier": "exact FX graph rewrite over mixed replicate/shard and dependency controls",
        "required_hardware": "none for graph rewrite proof; distributed timing is corroborating",
        "questions": [
            "Are only compatible replicate all-reduces placed in the same bucket?",
            "Are group, dtype, shape, reduction op, and dependency boundaries respected?",
            "Does the rewritten graph contain neither duplicate nor omitted reductions and preserve consumers?",
            "Do direct tests distinguish HSDP/DDP replicate cases from sharded and mixed controls?",
        ],
        "decision_rule": "Accept if bucketing preserves exact graph and reduction semantics for every frozen control. One bounded unsupported grouping with a guard and closure test is check; cross-group coalescing, dependency reordering, or omitted/duplicate reduction is reject.",
    },
    "verl-pr-6958": {
        "claim": "The rollout path safely reuses one persistent CUDA-IPC weight-transfer bucket across compatible synchronizations.",
        "communication_risk": "IPC allocation ownership, synchronization, resizing, and teardown",
        "execution_tier": "exact lifecycle state machine with mocked CUDA events and bucket identities",
        "required_hardware": "none for ownership proof; CUDA IPC execution is corroborating",
        "questions": [
            "Do compatible repeated syncs reuse exactly one allocation after prior work is complete?",
            "Do size, dtype, device, or peer changes safely reallocate or fail before transfer?",
            "Do cancellation, exception, sleep/wake, and close release the bucket exactly once?",
            "Can any producer overwrite or free storage while a consumer still owns the IPC view?",
        ],
        "decision_rule": "Accept if reuse is compatibility-gated, synchronized, and exactly owned through teardown. One bounded lifecycle branch with a deterministic closure test is check; use-after-free, unsynchronized overwrite, wrong-size reuse, leak, or design-level ownership ambiguity is reject.",
    },
    "vllm-pr-48763": {
        "claim": "Removing the additional MoE reduce_scatter preserves DeepSeek-MTP outputs and recovers the claimed communication/performance regression.",
        "communication_risk": "redundant collective elimination and tensor-parallel mathematical equivalence",
        "execution_tier": "exact call-count/data-flow proof plus paired multi-GPU performance comparison",
        "required_hardware": "two CUDA GPUs for the performance claim; correctness call graph is required locally",
        "questions": [
            "Is the removed reduce_scatter mathematically redundant for every changed model path?",
            "Do TP ranks retain identical logical outputs, placements, and shapes without the call?",
            "Does the communication trace remove exactly the redundant collective and no required synchronization?",
            "Does a paired benchmark reproduce a positive throughput effect with spread and configuration disclosed?",
        ],
        "decision_rule": "Accept if correctness, communication count, and paired performance evidence support the scoped claim. Correct elimination with only bounded missing/noisy performance evidence is check; rank-divergent output, missing required synchronization, or a measured regression is reject. Missing required hardware is unresolved.",
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
        raise SystemExit("R12 selection lock digest mismatch")
    hidden = (
        material["review_text_visible_to_machine_judge"],
        material["merge_outcomes_visible_to_machine_judge"],
        material["ci_fields_visible_to_machine_judge"],
        material["diff_content_visible_during_selection"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R12 selection exposes hidden evidence")
    cases = [HistoricalPRCandidate.model_validate(item) for item in material["cases"]]
    if {item.case_id for item in cases} != set(CASE_PLANS):
        raise SystemExit("R12 selection and plan case sets differ")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "machine_policy_id": material["machine_policy_id"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "review_text_requested": False,
        "frozen_before_source_diff_content_inspection": True,
        "claim_scope_policy": {
            "title_scoped_acceptance_is_blocking": True,
            "unclaimed_hardening_is_blocking": False,
            "compound_title_claims_are_all_blocking": True,
        },
        "scoring_policy": {
            "kind": "ordered exact communication-contract judgment with repairability triage",
            "weighted_score_used": False,
            "forced_polarization_used": False,
            "decisions": ["accept_with_scope", "check", "reject", "unresolved"],
            "check_is_bounded_repairability_or_bounded_claim-evidence_gap": True,
            "missing_required_environment_evidence": "unresolved, never candidate fail",
        },
        "cross_case_controls": [
            "FlashInfer 3879 and 3880 use the same partial-warp matrix, so a follow-up cannot receive an easier oracle.",
            "A collective call-count improvement cannot compensate for wrong rank, shape, order, ownership, or numeric semantics.",
            "Static source evidence cannot be promoted to runtime evidence when the frozen contract requires communication hardware.",
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
