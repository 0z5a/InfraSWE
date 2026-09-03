#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze 30 case-specific R16 training contracts before source acquisition."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

CASE_CONTRACTS: dict[str, dict[str, Any]] = {
    "liger-pr-1413": {
        "claim": "The fused-MoE Blackwell/B300 path computes the correct input gradient without changing routing, expert, or weight gradients.",
        "matrix": [
            "Blackwell declared path/A100 control",
            "FP16/BF16",
            "top-k 1/2",
            "zero/uneven expert tokens",
            "forward/backward repeated",
        ],
        "runtime": "candidate target test and exact fused/reference gradient oracle; A100 may validate only the architecture-independent control",
        "closure": "on a declared target, dx and all companion gradients meet the frozen reference tolerance for every routed-token edge case",
    },
    "liger-pr-1244": {
        "claim": "The ORPO trainer remains importable and performs the same objective, batching, backward pass, and parameter update after the TRL experimental move.",
        "matrix": [
            "old/new TRL export layout",
            "import/direct construction",
            "chosen/rejected boundary lengths",
            "FP32/BF16",
            "one optimizer step",
        ],
        "runtime": "candidate tests plus a tiny ORPO batch with independent loss/gradient and optimizer-step comparison",
        "closure": "supported TRL layouts construct the trainer and one step matches the reference loss, gradients, and updated parameters",
    },
    "liger-pr-1204": {
        "claim": "Every added DPO loss type implements its declared formula and stable gradient while preserving existing loss modes.",
        "matrix": [
            "hinge/bco_pair/robust/exo_pair/discopop/existing",
            "beta boundary values",
            "positive/equal/negative margins",
            "none/mean reduction",
            "FP32/BF16",
        ],
        "runtime": "candidate tests and an independent PyTorch formula with finite-difference and autograd gradient checks",
        "closure": "each mode matches its independent value/gradient oracle and all pre-existing modes remain unchanged",
    },
    "liger-pr-1202": {
        "claim": "The TRL-Liger GRPO path produces correctly shaped per-token losses and gradients across masking and chunk boundaries.",
        "matrix": [
            "fused/unfused",
            "single/multiple chunks",
            "all/some/zero masked tokens",
            "short/long completion",
            "FP32/BF16",
        ],
        "runtime": "candidate GRPO tests plus an unfused PyTorch loss/logprob/gradient oracle on GPU",
        "closure": "losses, normalization, masks, and logits/weight gradients match the unfused reference at every chunk boundary",
    },
    "liger-pr-1253": {
        "claim": "GroupNorm FP16 backward accumulators use the intended precision and preserve numerically valid input, weight, and bias gradients.",
        "matrix": [
            "FP16/BF16/FP32",
            "affine on/off",
            "small/large groups",
            "constant/high-dynamic-range input",
            "contiguous/strided",
        ],
        "runtime": "candidate tests plus PyTorch GroupNorm forward/backward and accumulator-dtype inspection",
        "closure": "all gradients meet dtype-specific tolerance without overflow/regression and the FP16 path no longer routes through BF16 accumulation",
    },
    "liger-pr-1208": {
        "claim": "DyT autotuning selects legal forward/backward kernels without changing outputs, gradients, cache behavior, or launch validity.",
        "matrix": [
            "cold/warm autotune cache",
            "FP16/BF16/FP32",
            "short/boundary/large hidden size",
            "contiguous/strided",
            "forward/backward",
        ],
        "runtime": "candidate tests plus eager-reference numeric checks and A100 launch/timing sweeps over frozen shapes",
        "closure": "every selected configuration is in bounds, numerically matches the reference, and warm execution does not repeatedly retune",
    },
    "megatron-pr-7021": {
        "claim": "MTP-only mode trains only the declared MTP objective and parameters while preserving topology, optimizer, checkpoint, and standard-training behavior.",
        "matrix": [
            "MTP-only/standard",
            "MTP enabled/disabled",
            "TP1/TP2",
            "one/two optimizer steps",
            "save/resume",
        ],
        "runtime": "candidate tests plus reduced two-A100 train steps with parameter-delta, gradient-owner, and checkpoint-resume comparison",
        "closure": "the intended parameters alone update in MTP-only mode, loss is finite, resume is trajectory-equivalent, and standard mode is unchanged",
    },
    "megatron-pr-5145": {
        "claim": "Latent-MoE theoretical memory accounting includes each owned tensor exactly once and tracks measured allocation across supported configurations.",
        "matrix": [
            "latent/standard MoE",
            "shared experts on/off",
            "TP/EP 1/2",
            "FP16/BF16",
            "activation recompute on/off",
        ],
        "runtime": "exact formula/component inventory plus small-model A100 measured allocation trend and legacy-estimator controls",
        "closure": "the formula matches the declared tensor inventory, changes monotonically with each component, and bounds measured deltas within frozen tolerance",
    },
    "megatron-pr-5169": {
        "claim": "The MCore/MBridge configuration refactor constructs equivalent training and inference objects without dropping, duplicating, or misrouting configuration values.",
        "matrix": [
            "MCore/MBridge entry",
            "train/inference construction",
            "default/explicit values",
            "single/distributed topology",
            "legacy builder absent/present control",
        ],
        "runtime": "candidate tests plus exhaustive adapter-field projection and a reduced forward/backward training construction smoke test",
        "closure": "all declared fields reach the intended owner exactly once and a reduced training step matches the pre-refactor configuration behavior",
    },
    "megatron-pr-5134": {
        "claim": "Removing deprecated distributed-checkpoint modules leaves every supported save/load strategy functional and fails legacy imports explicitly.",
        "matrix": [
            "torch/default supported strategy",
            "single/two rank",
            "save/load",
            "sharded/common state",
            "supported/deprecated import",
        ],
        "runtime": "candidate tests plus two-rank checkpoint round-trip with parameter and optimizer-state equality",
        "closure": "supported checkpoints round-trip byte/value-equivalently, registry resolution has no stale target, and deprecated imports fail with a bounded diagnostic",
    },
    "megatron-pr-5131": {
        "claim": "TE CUDA-graph training avoids dummy attention masks only where legal and preserves captured forward/backward values and gradients.",
        "matrix": [
            "capture/eager",
            "dummy mask required/not required",
            "one/repeated replay",
            "sequence boundary shapes",
            "TE compatible/incompatible configuration",
        ],
        "runtime": "candidate tests plus A100 capture/replay against eager output and gradient references with mask-consumer tracing",
        "closure": "legal captures replay deterministically without dummy-mask allocation, required masks remain present, and values/gradients match eager execution",
    },
    "megatron-pr-5162": {
        "claim": "Moving the TE cross-entropy guard to training arguments preserves safe configuration projection and loss/gradient behavior.",
        "matrix": [
            "fusion on/off/default",
            "compatible/incompatible TE setting",
            "CLI/config object",
            "TP1/TP2",
            "forward/backward",
        ],
        "runtime": "candidate argument tests plus exact guard propagation and fused/reference loss-gradient comparison where available",
        "closure": "each configuration selects the intended loss implementation exactly once and produces reference-equivalent loss and gradients",
    },
    "slime-pr-2345": {
        "claim": "Fully asynchronous rollout sorts nested groups by the intended stable key without losing, duplicating, or cross-assigning samples.",
        "matrix": [
            "flat/nested groups",
            "already/reverse/equal-key order",
            "empty/one/many groups",
            "partial completion",
            "repeated async batches",
        ],
        "runtime": "candidate tests plus deterministic scheduler simulation tracking sample identity, group ownership, and completion order",
        "closure": "every submitted sample is returned once to its original group in stable intended order across repeated asynchronous completion patterns",
    },
    "slime-pr-2010": {
        "claim": "Per-chunk logits scaling lowers training peak memory without changing loss normalization, gradients, or optimizer updates.",
        "matrix": [
            "one/multiple chunks",
            "chunk boundary +/-1",
            "short/long token count",
            "FP32/BF16",
            "forward/backward/step",
        ],
        "runtime": "candidate tests plus A100 unchunked value/gradient/optimizer reference and peak-memory measurement",
        "closure": "loss, gradients, and updated parameters meet tolerance while measured peak is lower and scales with chunk rather than full logits",
    },
    "slime-pr-2015": {
        "claim": "Rollout generation is quiesced before offload memory is released, including failure and repeated lifecycle paths.",
        "matrix": [
            "generation active/idle",
            "offload on/off",
            "pause success/failure",
            "one/repeated release",
            "concurrent request arrival",
        ],
        "runtime": "candidate tests plus exact call/event order and accelerator allocation-lifetime probe",
        "closure": "no generation consumer accesses released storage, pause precedes every release, failure retains recoverable ownership, and restart succeeds",
    },
    "slime-pr-2014": {
        "claim": "The rollout manager applies sample filtering exactly once while preserving group cardinality, accounting, and train-batch ownership.",
        "matrix": [
            "filter off/on",
            "keep all/some/none",
            "sync/async rollout",
            "grouped/ungrouped samples",
            "repeat batches",
        ],
        "runtime": "candidate tests plus identity-and-count tracing through manager, rollout backend, and downstream training batch",
        "closure": "only accepted samples reach training once, all counters and groups agree, and empty results take the declared bounded path",
    },
    "slime-pr-1969": {
        "claim": "Raw-mode save-hf emits a complete, correctly mapped Hugging Face checkpoint without corrupting training ownership or the existing save mode.",
        "matrix": [
            "raw/non-raw",
            "single/tensor-parallel shards",
            "direct/bridge iterator",
            "one/repeated save",
            "load/forward round-trip",
        ],
        "runtime": "candidate tests plus tiny-model parameter inventory, HF reload, logits equality, and continued optimizer-step control",
        "closure": "every expected parameter is written once with correct shape/value, reload logits match, and training can continue without state mutation",
    },
    "slime-pr-2020": {
        "claim": "Node-local writers accelerate raw HF save while producing one complete, collision-free checkpoint equivalent to the serial writer.",
        "matrix": [
            "one/two nodes simulated",
            "one/multiple shards",
            "balanced/uneven ownership",
            "success/writer failure",
            "serial/node-writer output",
        ],
        "runtime": "candidate tests plus multi-process shard manifest, checksum, atomic-finalization, reload parity, and elapsed-time evidence",
        "closure": "all tensors appear exactly once, manifests/checksums and reload outputs match serial save, failures cannot publish a partial checkpoint",
    },
    "torchtitan-pr-4358": {
        "claim": "Trainer-integrated deterministic replay reproduces the same step inputs, RNG, loss, gradients, and parameters after the declared corruption event.",
        "matrix": [
            "replay off/on",
            "no fault/injected fault",
            "single/distributed reduced step",
            "dropout on/off",
            "checkpoint before/after event",
        ],
        "runtime": "candidate tests plus two-A100 fault-injected train/replay trace comparing RNG, loss, gradients, and parameter trajectory",
        "closure": "a replayed step is trajectory-identical to the clean reference and normal training incurs no semantic change",
    },
    "torchtitan-pr-3523": {
        "claim": "CPU offload retains the wait dependency for non-Tensor consumers so replay cannot read an activation before transfer completes.",
        "matrix": [
            "Tensor/non-Tensor consumer",
            "sync/delayed transfer",
            "base/view activation",
            "one/repeated backward",
            "offload on/off",
        ],
        "runtime": "candidate tests plus forced delayed-copy event trace and non-offloaded value/gradient oracle",
        "closure": "every consumer observes completed data, dependency edges remain live, and forward/backward values match the non-offloaded reference",
    },
    "torchtitan-pr-3538": {
        "claim": "An omitted cudagraph direction defaults to forward capture while explicit backward behavior and replay numerics remain unchanged.",
        "matrix": [
            "direction omitted/true/false",
            "forward/backward graph",
            "one/repeated replay",
            "static/dynamic shape",
            "safe/unsafe node",
        ],
        "runtime": "candidate tests plus graph-direction inspection and eager-equivalence output/gradient replay",
        "closure": "the omitted value selects forward exactly, explicit values retain meaning, and captured results match eager execution",
    },
    "torchtitan-pr-3534": {
        "claim": "Cudagraph capture stages non-static inputs before capture so replay observes current values without stale aliases or shape misuse.",
        "matrix": [
            "static/non-static input",
            "same/changed values",
            "same/changed legal shape",
            "one/repeated replay",
            "forward/backward",
        ],
        "runtime": "candidate tests plus storage/alias tracing and eager-equivalence output/gradient replays",
        "closure": "each replay consumes current staged values, rejects unsupported shape changes, and matches eager outputs and gradients",
    },
    "torchtitan-pr-3533": {
        "claim": "Per-node capture-safety classification captures only legal graph nodes and preserves fallback order, values, and gradients.",
        "matrix": [
            "all-safe/all-unsafe/mixed graph",
            "forward/backward node",
            "side-effect/no side-effect",
            "one/repeated replay",
            "predicate exception",
        ],
        "runtime": "candidate tests plus synthetic mixed graph with capture trace and eager-equivalence output/gradient oracle",
        "closure": "safe nodes alone are captured, unsafe nodes execute once in order, failures fall back cleanly, and numerics equal eager execution",
    },
    "torchtitan-pr-3530": {
        "claim": "Selective-activation-rematerialization duplicates receive independent custom metadata without aliasing or changing recompute gradients.",
        "matrix": [
            "single/duplicate remat",
            "shared/independent input",
            "one/repeated backward",
            "mutation/no mutation",
            "AC enabled/disabled",
        ],
        "runtime": "candidate tests plus FX metadata identity inspection and eager-versus-rematerialized gradient comparison",
        "closure": "duplicate nodes never share mutable metadata, recompute selects the intended node, and all gradients match the non-remat reference",
    },
    "verl-pr-7697": {
        "claim": "The configured consecutive-invalid-tool-call limit terminates only the offending rollout sequence and resets after a valid call.",
        "matrix": [
            "limit 0/1/many",
            "below/at/above limit",
            "valid reset/no reset",
            "single/multiple agents",
            "sync/async completion",
        ],
        "runtime": "candidate config and agent-loop tests plus an end-to-end scripted rollout tracking termination, reward, and surviving sequences",
        "closure": "counts are per sequence, valid calls reset exactly, only threshold violations terminate, and downstream training receives consistent outcomes",
    },
    "verl-pr-6558": {
        "claim": "One-step off-policy training reports dataloader exhaustion at the correct boundary without consuming extra data or mutating optimizer state.",
        "matrix": [
            "empty/one/multiple batches",
            "exhaust before/after requested step",
            "single/repeated iterator",
            "normal/worker exception",
            "resume attempt",
        ],
        "runtime": "candidate tests plus iterator-consumption and parameter/optimizer-state trace around a reduced training step",
        "closure": "the precise bounded error occurs only on exhaustion, no batch is skipped or duplicated, and failed steps leave parameters/state unchanged",
    },
    "verl-pr-6564": {
        "claim": "Remove-padding Megatron utilities keep logprob labels aligned to their original sequence tokens through sharding and reconstruction.",
        "matrix": [
            "rmpad on/off",
            "equal/uneven lengths",
            "left/right padding",
            "TP1/TP2",
            "loss/backward",
        ],
        "runtime": "candidate tests plus two-A100 pack/shard/gather identity oracle and padded-reference logprob loss/gradient comparison",
        "closure": "each label maps to its source token once, padding never contributes, and gathered loss/gradients match the padded reference",
    },
    "verl-pr-6574": {
        "claim": "Fully asynchronous helper actors reserve sufficient but non-duplicated CPU resources so legal jobs schedule and progress.",
        "matrix": [
            "default/explicit CPU",
            "one/multiple nodes",
            "one/multiple helpers",
            "tight/ample cluster",
            "actor restart",
        ],
        "runtime": "candidate tests plus local Ray placement/resource accounting and timeout-protected actor progress probe",
        "closure": "declared bundles equal actual actor reservations, no resource is double counted, legal tight-cluster jobs start, and restart releases/reacquires resources",
    },
    "verl-pr-6560": {
        "claim": "vLLM tool-calling configuration is propagated losslessly from trainer config to every rollout server and changes tool parsing only when enabled.",
        "matrix": [
            "tool calling off/on",
            "default/explicit parser",
            "single/multiple servers",
            "valid/invalid tool output",
            "sync/async rollout",
        ],
        "runtime": "candidate config tests plus exhaustive adapter projection and scripted rollout-server request/response behavior",
        "closure": "every server receives identical intended flags once, defaults preserve old behavior, and enabled parsing yields the declared training rollout records",
    },
    "verl-pr-6598": {
        "claim": "Fully asynchronous SGLang broadcasting transfers one coherent actor weight version to all rollout workers before they serve the next generation.",
        "matrix": [
            "sync/fully async",
            "one/two rollout workers",
            "single/repeated update",
            "slow/failing receiver",
            "offload on/off",
        ],
        "runtime": "candidate tests plus two-A100 versioned parameter broadcast with call-order, checksum, generation, and failure-lifetime tracing",
        "closure": "all workers activate the same complete new version exactly once before service, failures cannot expose partial weights, and repeated updates progress",
    },
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--metadata-amendment", type=Path, required=True)
    parser.add_argument("--r15-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R16 selection lock digest mismatch")
    amendment = _read(args.metadata_amendment)
    amendment_material = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    if amendment["amendment_sha256"] != canonical_sha256(amendment_material):
        raise SystemExit("R16 metadata amendment digest mismatch")
    if selection_material["metadata_amendment_sha256"] != amendment["amendment_sha256"]:
        raise SystemExit("R16 selection/amendment binding mismatch")
    iteration = _read(args.r15_iteration)
    iteration_material = {
        key: value for key, value in iteration.items() if key != "iteration_sha256"
    }
    if iteration["iteration_sha256"] != canonical_sha256(iteration_material):
        raise SystemExit("R15 iteration digest mismatch")
    if selection_material["r15_policy_iteration_sha256"] != iteration["iteration_sha256"]:
        raise SystemExit("R16 selection/R15 iteration binding mismatch")
    hidden = (
        selection_material["review_or_comment_visible"],
        selection_material["merge_outcomes_visible"],
        selection_material["ci_or_label_visible"],
        selection_material["candidate_body_visible"],
        selection_material["diff_content_visible"],
        selection_material["excluded_resolution_gray_zone_used"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R16 selection exposes hidden evidence")
    cases = selection_material["cases"]
    if len(cases) != 30 or {item["case_id"] for item in cases} != set(CASE_CONTRACTS):
        raise SystemExit("R16 selection and case-contract sets differ")
    if any(item["benchmark_domain"] != "training" for item in cases):
        raise SystemExit("R16 contains a non-training case")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": selection_material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "metadata_amendment_sha256": amendment["amendment_sha256"],
        "r15_policy_iteration_sha256": iteration["iteration_sha256"],
        "machine_policy_id": selection_material["machine_policy_id"],
        "domain_allocation": {"training": 30},
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_candidate_body_access": True,
        "frozen_before_source_diff_content_access": True,
        "review_or_comment_requested": False,
        "merge_outcome_or_state_requested": False,
        "ci_or_label_requested": False,
        "evaluation_layers": {
            "technical_contract": ["pass", "bounded-gap", "fail", "unresolved"],
            "disposition_prediction": ["accept", "check", "reject", "unresolved"],
            "governance_gap_recorded_separately": True,
            "technical_result_never_rewritten_from_history": True,
        },
        "disposition_policy": {
            "accept": "The title-scoped production contract has exact functional closure and no reachable blocker.",
            "check": "The PR is within the <=7-day hot window, changes <=8 coherent files, has exact core-path evidence, and leaves exactly one executable integration residual with no counterexample.",
            "reject": "There is an exact reachable failure, nonfunctional target-only evidence, broad training plumbing without step/gradient/checkpoint/measured-memory closure, or a mature disposition lacking accept evidence.",
            "unresolved": "Required backend, architecture, topology, or exact evidence is unavailable; technical uncertainty is recorded without fabricating failure.",
            "hot_window_days": 7,
            "check_changed_files_max": 8,
            "mature_created_at_cutoff": "2026-06-04T23:59:59Z",
            "resolution_gray_zone_excluded": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "training_closure_rule": {
            "config_or_mock_only_is_accept": False,
            "required_functional_evidence": [
                "optimizer-step",
                "gradient",
                "checkpoint-round-trip",
                "measured-memory",
                "equally exact title-scoped training behavior",
            ],
            "mechanical_adapter_exception": "A mature change may accept when it only propagates one backward-compatible value, enumerates every implementation, passes endpoint tests, and changes no backend semantics.",
        },
        "target_hardware_rule": {
            "compile_import_or_mock_is_functional_closure": False,
            "required": ["target numeric", "target progress", "target ordering", "target memory"],
            "missing_evaluator_hardware_alone_is_reject": False,
        },
        "ordered_reachability_gate": [
            "configuration is legal and title-scoped",
            "production training call site reaches the behavior",
            "loss, gradient, state, scheduling, or resource invariant is satisfied or violated",
            "an exact base/head or independent oracle distinguishes the behavior",
            "remaining remediation is bounded to the candidate direction",
        ],
        "evidence_tiers": [
            "exact candidate-owned tests",
            "exact isolated base/head contract probe",
            "two-A100 numeric, progress, checkpoint, lifecycle, or memory probe",
            "candidate-authored outcome-free body evidence acquired after this lock",
            "full target/model end-to-end execution where dependencies permit",
        ],
        "cross_case_controls": [
            "An exact candidate-owned or measured counterexample overrides historical disposition.",
            "A timeout requires rank diagnostics and a healthy control before it proves nonprogress.",
            "Configuration, import, compile, and mocked orchestration do not alone close a broad training claim.",
            "A100 skips cannot masquerade as Blackwell, ROCm, NPU, or another target execution.",
            "Candidate body remains outcome-free and separately hashed after this plan lock.",
        ],
        "cases": [
            {
                "case_id": item["case_id"],
                "project": item["project"],
                "repository": item["repository"],
                "pull_number": item["pull_number"],
                "title": item["title"],
                "created_at": item["created_at"],
                "temporal_band": item["temporal_band"],
                "benchmark_domain": item["benchmark_domain"],
                "base_sha": item["base_sha"],
                "head_sha": item["head_sha"],
                "changed_paths": item["paths"],
                "risk_family": item["risk_family"],
                **CASE_CONTRACTS[item["case_id"]],
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
