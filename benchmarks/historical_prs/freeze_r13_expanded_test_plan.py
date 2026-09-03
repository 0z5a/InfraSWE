#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze fifteen extension contracts and bind them to the original R13 plan."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

CASE_PLANS: dict[str, dict[str, Any]] = {
    "megatron-pr-5819": {
        "claim": "Full-iteration CUDA graph capture and replay preserves the intended data-iterator position and batch sequence without corruption, duplication, or skipping.",
        "training_risk": "CUDA graph capture side effects on stateful training-data iteration",
        "execution_tier": "exact iterator state-machine isolation plus A100 capture/replay sequence comparison",
        "required_hardware": "CUDA GPU",
        "questions": [
            "Do eager and full-iteration graph paths consume exactly the same logical batches across warmup, capture, and repeated replay?",
            "Are iterator reads suppressed or restored at the correct capture boundary without stale values or extra advancement?",
            "Do fresh capture, repeated capture, exhaustion, and multi-chunk controls preserve ownership?",
            "Is there a direct regression that distinguishes the former corrupting sequence from head?",
        ],
        "decision_rule": "Accept if every reachable capture/replay sequence preserves iterator position and batch identity. One bounded untested multi-chunk lifecycle with a runnable closure test is check; skipped, duplicated, stale, or extra-consumed training data is reject. Missing CUDA graph support is unresolved.",
    },
    "megatron-pr-5761": {
        "claim": "Combined 1F1B execution enters both autocast and FP8 contexts around the intended model work while preserving disabled and single-context behavior.",
        "training_risk": "mixed-precision context composition and pipeline forward/backward dtype semantics",
        "execution_tier": "exact context-manager instrumentation plus A100 dtype and gradient controls",
        "required_hardware": "CUDA GPU; Transformer Engine availability is required for the real FP8 path",
        "questions": [
            "Are autocast and FP8 contexts both active during every intended combined-1F1B compute region and exited in stack order?",
            "Do enabled/disabled combinations preserve prior BF16/FP16 behavior and exception cleanup?",
            "Are loss, gradients, and parameter updates finite and numerically consistent with the non-combined schedule?",
            "Does candidate evidence directly execute the joint-context path rather than only parse configuration?",
        ],
        "decision_rule": "Accept if joint context ownership and reachable numeric controls pass. One bounded missing real-FP8 execution with a complete instrumented closure test is check; a missing/misnested context, leaked context, wrong dtype, or gradient divergence is reject. Missing required FP8 runtime is unresolved only for the hardware-specific portion.",
    },
    "megatron-pr-5724": {
        "claim": "THD packed-sequence padding has identical shape, token-validity, output, and gradient semantics in eager and CUDA-graph training modes.",
        "training_risk": "packed-sequence padding metadata and eager/graph training parity",
        "execution_tier": "exact metadata flow plus A100 eager-versus-graph packed-sequence matrix",
        "required_hardware": "CUDA GPU",
        "questions": [
            "Do cu_seqlens, max sequence length, padded token count, and valid-token masks agree for eager and graph modes?",
            "Do zero-padding, non-aligned lengths, multiple microbatches, and boundary-sized batches avoid reading or training on padding?",
            "Do outputs and input/parameter gradients match for the same logical token set?",
            "Do changed tests distinguish the old divergence and cover both eager and graph paths?",
        ],
        "decision_rule": "Accept if metadata and numeric parity hold across reachable THD layouts. One explicitly bounded unsupported alignment with a runnable closure test is check; wrong token validity, shape mismatch, padding leakage, graph replay failure, or gradient divergence is reject. Missing CUDA graph support is unresolved.",
    },
    "megatron-pr-5714": {
        "claim": "FSDP2 checkpoints preserve SwiGLU parameters and reconstruct the live model so save/load and resumed forward behavior match an uninterrupted reference.",
        "training_risk": "fused SwiGLU state-dict mapping under FSDP2 sharding",
        "execution_tier": "exact state-key mapping plus two-rank FSDP2 save/load and resume round-trip",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Are fused and split SwiGLU keys mapped exactly once with correct gate/up ordering, shapes, and shard placement?",
            "Does a two-rank checkpoint round-trip reproduce every parameter and the next forward/backward result?",
            "Do tied/shared, expert, and ordinary unfused MLP controls remain unchanged where supported?",
            "Do direct candidate tests distinguish base from head and fail closed on malformed keys?",
        ],
        "decision_rule": "Accept if supported SwiGLU FSDP2 round-trips are exact and neighboring MLP controls pass. One bounded additional owner such as an untested expert wrapper with a runnable closure test is check; missing/wrong gate slices, silent key loss, shard disagreement, or resumed divergence is reject. Missing two-GPU FSDP2 execution is unresolved.",
    },
    "megatron-pr-5710": {
        "claim": "Frozen-parameter FSDP backward hooks remain correctly paired so mixed frozen/trainable modules complete repeated backward passes without stale hook state or gradient corruption.",
        "training_risk": "FSDP pre/post-backward hook cardinality for frozen parameters",
        "execution_tier": "exact hook state-machine proof plus two-rank repeated-backward controls",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Does every installed pre-backward hook have exactly one valid completion path when all or some owned parameters are frozen?",
            "Do mixed frozen/trainable, all-frozen, unused, no-grad, and repeated-iteration paths avoid stale state and duplicate callbacks?",
            "Are trainable gradients and updates equal to an unsharded reference while frozen parameters remain unchanged?",
            "Does the changed test distinguish the former unpaired-hook failure?",
        ],
        "decision_rule": "Accept if hook cardinality, lifecycle, and gradients are correct across reachable frozen layouts. One bounded unused-parameter permutation with a runnable closure test is check; dangling/duplicate hooks, deadlock, stale state, frozen updates, or trainable-gradient divergence is reject. Missing two-GPU execution is unresolved.",
    },
    "slime-pr-2207": {
        "claim": "Partial-rollout off-policy loss masks align with the exact generated-token span so prefixes and stale/unowned tokens never contribute to policy loss or gradients.",
        "training_risk": "partial-rollout token alignment and off-policy loss masking",
        "execution_tier": "exact sample transformation plus tokenwise loss/gradient oracle matrix",
        "required_hardware": "none for alignment proof; CUDA training tensors are corroborating",
        "questions": [
            "Does each continued rollout mark exactly its newly trainable response tokens after prefix reuse, truncation, and padding?",
            "Are left/right padding, empty continuation, multi-turn, unequal prefix, and batched controls aligned without an off-by-one shift?",
            "Do masked loss and gradients match an independently indexed token oracle?",
            "Do candidate tests distinguish the previous alignment and include boundary cases?",
        ],
        "decision_rule": "Accept if every reachable token span and gradient mask matches the independent oracle. One bounded unsupported batching layout with a direct closure test is check; any trainable-prefix leak, dropped continuation token, off-by-one mask, or gradient mismatch is reject.",
    },
    "slime-pr-2205": {
        "claim": "Vectorized REINFORCE++ discounted returns are numerically equivalent to the scalar recurrence and improve or preserve A100 performance.",
        "training_risk": "RL return recurrence, response masking, and vectorized numerical stability",
        "execution_tier": "independent recurrence oracle plus A100 shape/dtype correctness and paired timing",
        "required_hardware": "CUDA GPU for performance; correctness is architecture-independent",
        "questions": [
            "Do vectorized returns match a right-to-left scalar recurrence for variable lengths, masks, rewards, and all supported discount settings?",
            "Are empty, length-one, long-sequence, all-masked, terminal, FP32, BF16, and noncontiguous controls safe?",
            "Does the vectorization preserve output dtype/device and avoid overflow or cancellation outside declared tolerances?",
            "Does paired A100 timing show no material regression at representative response lengths?",
        ],
        "decision_rule": "Accept if all reachable numeric controls pass and performance is non-regressive. Correctness with only noisy or bounded missing timing evidence is check; wrong recurrence, mask leakage, NaN/inf, dtype regression, or a material measured slowdown is reject. Missing CUDA is unresolved only for performance.",
    },
    "slime-pr-2204": {
        "claim": "Reward normalization uses explicit sample groups and is invariant to batch ordering while preserving per-group zero-mean/scaling semantics.",
        "training_risk": "RL reward grouping and advantage normalization",
        "execution_tier": "exact grouping isolation plus independent irregular-group normalization oracle",
        "required_hardware": "none",
        "questions": [
            "Do noncontiguous, interleaved, unequal-size, singleton, filtered, and repeated prompt groups normalize only against their declared members?",
            "Is the result invariant to batch permutation and stable for zero-variance groups?",
            "Are missing, duplicate, malformed, or inconsistent group identifiers handled explicitly rather than silently regrouped?",
            "Do changed tests distinguish implicit adjacency assumptions from explicit grouping?",
        ],
        "decision_rule": "Accept if explicit groups determine the exact independent normalization for every supported layout. One bounded malformed-input validation gap with a closure test is check; cross-group leakage, order dependence, wrong scale, NaN, or silent regrouping of valid samples is reject.",
    },
    "slime-pr-2198": {
        "claim": "Clamping PPO log-ratios before exponentiation prevents overflow without changing the intended clipped objective or gradients in the valid operating range.",
        "training_risk": "PPO importance-ratio overflow and policy-gradient semantics",
        "execution_tier": "exact formula isolation plus A100 extreme-value forward/backward dtype matrix",
        "required_hardware": "CUDA GPU for production dtypes; CPU FP32 is diagnostic",
        "questions": [
            "Are ratio, clipped objective, approximate KL, clip fraction, and gradients finite for extreme positive and negative log-ratios?",
            "Does the central operating range remain bitwise or tolerance-equivalent to the unclamped formula?",
            "Are clamp bounds safe for FP16/BF16/FP32 and consistent with all exponentiation call sites?",
            "Do direct tests cover overflow, underflow, both PPO branches, masks, and gradients?",
        ],
        "decision_rule": "Accept if extremes are finite and the valid-range PPO objective/gradients are unchanged. One bounded diagnostic-metric discrepancy with a closure test is check; retained overflow, silent zeroing of valid gradients, wrong clipping branch, or central-range regression is reject. Missing CUDA is unresolved.",
    },
    "slime-pr-2152": {
        "claim": "The optimized vocab-parallel log-probability/entropy operator reduces peak memory while preserving values and backward gradients across tensor-parallel shards.",
        "training_risk": "vocabulary-parallel fused loss statistics, collectives, and activation memory",
        "execution_tier": "two-rank exact-shard oracle plus A100 memory and backward matrix",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Do log-probability, entropy, logits gradients, and masked-token semantics match an unsharded FP32 oracle for TP=1 and TP=2?",
            "Are uneven/empty local vocabulary ownership, boundary token IDs, padding, noncontiguous inputs, and BF16/FP32 handled correctly?",
            "Do collectives use the right reduction and preserve rank agreement without retaining full-vocab intermediates?",
            "Is peak allocated memory materially lower or at least non-regressive for a representative large-vocab batch?",
        ],
        "decision_rule": "Accept if two-rank values/gradients match and memory is non-regressive with the claimed optimization visible. One bounded noisy peak-memory comparison with complete correctness is check; wrong shard math, rank divergence, bad gradients, OOM, or material memory regression is reject. Missing two-GPU execution is unresolved.",
    },
    "verl-pr-7012": {
        "claim": "Megatron forward-KL top-k distillation aligns teacher sequence tensors with the student under context-parallel splitting without losing valid tokens or misaligning indices.",
        "training_risk": "teacher/student sequence alignment and CP-sharded distillation loss",
        "execution_tier": "exact alignment flow plus two-rank CP top-k loss/gradient oracle",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Do teacher values/indices and student logits cover the same logical token positions after CP splitting for unequal stored sequence lengths?",
            "Are padding, odd lengths, boundaries, multiple top-k values, and no-CP controls aligned without truncation or duplication?",
            "Do distributed loss and student gradients match an unsharded independent oracle?",
            "Is there a direct test that distinguishes the prior sequence-length mismatch?",
        ],
        "decision_rule": "Accept if token alignment and two-rank loss/gradients match for every supported layout. One bounded unsupported odd-length layout with a runnable closure test is check; misindexed teacher targets, token loss/duplication, shape failure, or gradient divergence is reject. Missing two-GPU execution is unresolved.",
    },
    "verl-pr-7005": {
        "claim": "FSDP2 weight export can bypass whole-shard staging while emitting exactly the same complete weights and reducing or preserving peak memory.",
        "training_risk": "FSDP2 sharded parameter export ownership and staging memory",
        "execution_tier": "exact export data-flow plus two-rank old-versus-new state-dict and peak-memory comparison",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Does each logical parameter export exactly once with correct name, shape, dtype, value, and ordering under FSDP2?",
            "Do uneven shards, tied parameters, offload, empty ownership, repeated export, and exception cleanup preserve state?",
            "Does the exported mapping match the staged baseline and reload into an equivalent model?",
            "Does A100 peak allocation show the removed whole-shard staging rather than moving it elsewhere?",
        ],
        "decision_rule": "Accept if export/reload is exact and peak memory is non-regressive with the staging removal demonstrated. One bounded tied-parameter lifecycle gap with a closure test is check; missing/duplicate/stale weights, rank mismatch, unsafe ownership, or material memory regression is reject. Missing two-GPU execution is unresolved.",
    },
    "verl-pr-6996": {
        "claim": "Asynchronous FSDP model loading restores the intended parameters before use without race, partial state, or sync-path regression.",
        "training_risk": "async checkpoint/load ordering and FSDP state ownership",
        "execution_tier": "exact async state-machine instrumentation plus two-rank delayed-load and forward controls",
        "required_hardware": "two CUDA GPUs",
        "questions": [
            "Is every asynchronous load awaited at the authoritative readiness boundary before forward or weight export can observe the model?",
            "Do delayed, failed, repeated, empty, offloaded, and mixed-shard loads finish or fail deterministically without partial visibility?",
            "Do loaded parameters and the next forward/backward match a synchronous reference on every rank?",
            "Does the fix leave synchronous loading and neighboring non-FSDP engines unchanged?",
        ],
        "decision_rule": "Accept if async ownership, error propagation, and two-rank numeric parity hold. One bounded reentrant-load case with a runnable closure test is check; early visibility, partial/stale state, swallowed failure, deadlock, rank disagreement, or sync regression is reject. Missing two-GPU execution is unresolved.",
    },
    "verl-pr-6963": {
        "claim": "Every changed fully-async training path fails loudly when rollout log-probabilities are requested but absent, while valid not-requested and present-empty states remain distinct.",
        "training_risk": "off-policy correction precondition enforcement across async trainers",
        "execution_tier": "exact call-path and configuration reachability matrix with sentinel batch controls",
        "required_hardware": "none",
        "questions": [
            "Do all changed consumers reject an absent requested log-probability field before computing a silently wrong objective?",
            "Are not-requested, present-empty, masked-empty, and valid-present states handled according to distinct contracts?",
            "Is the check placed at the authoritative owner without duplicate or contradictory defaults across trainer variants?",
            "Do direct tests cover every changed path and distinguish the former fallback?",
        ],
        "decision_rule": "Accept if every reachable requested/absent path fails closed and legitimate states remain valid. One bounded trainer variant with a direct closure test is check; any silent requested-data fallback, false rejection of supported batches, or inconsistent owner is reject.",
    },
    "verl-pr-6960": {
        "claim": "Fused linear cross-entropy backward returns contiguous gradient buffers wherever the training consumer requires them without altering loss or gradients.",
        "training_risk": "custom autograd gradient layout and fused loss correctness",
        "execution_tier": "exact autograd isolation plus A100 stride, dtype, shape, compile, and numeric matrix",
        "required_hardware": "CUDA GPU",
        "questions": [
            "Are all returned input/weight/bias gradient buffers contiguous at the consumer boundary that requires contiguous storage?",
            "Do values match an unfused linear-plus-cross-entropy oracle for noncontiguous inputs, reductions, ignore-index, and supported dtypes?",
            "Do repeated backward, accumulation, torch.compile, and higher-dimensional batches preserve layout and values?",
            "Is there a regression that distinguishes the prior noncontiguous buffer in a production-reachable shape?",
        ],
        "decision_rule": "Accept if reachable buffers satisfy the required layout and all numeric controls pass. One bounded untested compiler variant with a runnable closure test is check; retained noncontiguity at the production consumer, copy aliasing, wrong loss, wrong gradient, or compile failure is reject. Missing CUDA is unresolved.",
    },
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _validate_selection(payload: dict[str, Any]) -> None:
    if payload["selection_lock_sha256"] != canonical_sha256(payload["selection_material"]):
        raise SystemExit("selection lock digest mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded-selection", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--extension-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expanded = _read(args.expanded_selection)
    extension = _read(args.extension_selection)
    base_plan = _read(args.base_plan)
    _validate_selection(expanded)
    _validate_selection(extension)
    base_material = {key: value for key, value in base_plan.items() if key != "test_plan_sha256"}
    if base_plan["test_plan_sha256"] != canonical_sha256(base_material):
        raise SystemExit("base R13 test-plan digest mismatch")
    if len(base_plan["cases"]) != 14:
        raise SystemExit("base R13 plan must contain fourteen cases")
    extension_cases = extension["selection_material"]["cases"]
    if {case["case_id"] for case in extension_cases} != set(CASE_PLANS):
        raise SystemExit("extension selection and contract sets differ")
    if expanded["selection_material"]["component_selection_lock_sha256"] != [
        base_plan["selection_lock_sha256"],
        extension["selection_lock_sha256"],
    ]:
        raise SystemExit("expanded selection/component binding mismatch")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": expanded["selection_material"]["protocol_id"],
        "selection_lock_sha256": expanded["selection_lock_sha256"],
        "component_selection_lock_sha256": expanded["selection_material"][
            "component_selection_lock_sha256"
        ],
        "base_r13_test_plan_sha256": base_plan["test_plan_sha256"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "review_text_requested": False,
        "candidate_body_requested_for_extension": False,
        "extension_frozen_before_candidate_body_acquisition": True,
        "extension_frozen_before_source_diff_content_inspection": True,
        "base_contracts_preserved_byte_for_byte": True,
        "claim_scope_policy": base_plan["claim_scope_policy"],
        "candidate_evidence_policy": base_plan["candidate_evidence_policy"],
        "scoring_policy": base_plan["scoring_policy"],
        "cross_case_controls": base_plan["cross_case_controls"],
        "cases": [
            *base_plan["cases"],
            *[
                {
                    "case_id": case["case_id"],
                    "project": case["project"],
                    "repository": case["repository"],
                    "pull_number": case["pull_number"],
                    "base_sha": case["base_sha"],
                    "head_sha": case["head_sha"],
                    "changed_paths": case["paths"],
                    **CASE_PLANS[case["case_id"]],
                }
                for case in extension_cases
            ],
        ],
    }
    if len(plan_material["cases"]) != 29:
        raise SystemExit("expanded R13 plan must contain twenty-nine cases")
    payload = {**plan_material, "test_plan_sha256": canonical_sha256(plan_material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(plan_material['cases'])}")
    print(f"test_plan_sha256={payload['test_plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
