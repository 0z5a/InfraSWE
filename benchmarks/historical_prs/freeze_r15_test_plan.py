#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze 30 case-specific R15 contracts before body or diff acquisition."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

CASE_CONTRACTS: dict[str, dict[str, Any]] = {
    "flashinfer-pr-4795": {
        "risk_family": "expert-parallel-capture",
        "claim": "The NCCL-EP split backend is CUDA-graph capturable without stale routing metadata, handle lifetime errors, or replay-dependent output.",
        "matrix": [
            "EP1/EP2",
            "eager/capture/replay",
            "zero/uneven routes",
            "one/repeated graph",
            "valid/invalid local metadata",
        ],
        "runtime": "candidate mock and multi-rank tests plus two-A100 capture/replay numeric and handle-lifetime probe",
        "closure": "captured and eager dispatch/combine outputs and route ownership match across repeated replays with balanced handles",
    },
    "flashinfer-pr-3304": {
        "risk_family": "collective-sentinel-numerics",
        "claim": "MNNVL all-reduce uses bitwise sentinel identity so valid subnormal payloads are not mistaken for transport sentinels.",
        "matrix": [
            "sentinel/subnormal/normal/zero bit patterns",
            "FP16/BF16/FP32",
            "one/multiple ranks",
            "positive/negative subnormal",
        ],
        "runtime": "exact header-level predicate oracle and candidate target-hardware test; MNNVL execution where supported",
        "closure": "every sentinel bit pattern is detected and every valid subnormal is preserved through the all-reduce reference matrix",
    },
    "megatron-pr-7029": {
        "risk_family": "process-group-lifecycle",
        "claim": "Parallel-state teardown destroys every created process group and invalidates communicator caches so repeated initialization cannot reuse stale owners.",
        "matrix": [
            "init/destroy once/repeated",
            "default/subgroups",
            "CUDA graph/FSDP/fused-A2A caches",
            "normal/partial-init exception",
        ],
        "runtime": "exact owner inventory, candidate tests, and two-rank init-use-destroy-reinit collective probe",
        "closure": "all created groups are destroyed once, caches are empty, and a fresh two-rank collective succeeds after reinitialization",
    },
    "megatron-pr-5153": {
        "risk_family": "expert-parallel-dispatch",
        "claim": "The DeepEP-v2 flex dispatcher preserves token, expert, and combine ownership across supported dispatch modes.",
        "matrix": [
            "DeepEP v1/v2",
            "flex/non-flex",
            "EP1/EP2",
            "zero/uneven tokens",
            "fine-grained callable on/off",
        ],
        "runtime": "candidate dispatcher tests plus exact source-shape and available two-rank dispatch/combine oracle",
        "closure": "permutation inversion, expert counts, and combined outputs equal the independent routing oracle in every supported mode",
    },
    "megatron-pr-5135": {
        "risk_family": "expert-overlap-ordering",
        "claim": "Latent-MoE shared-expert overlap preserves shared/routed expert ordering, outputs, and gradients while introducing real overlap.",
        "matrix": [
            "overlap on/off",
            "shared expert present/absent",
            "zero/uneven routed tokens",
            "one/two ranks",
            "forward/backward",
        ],
        "runtime": "candidate shared-expert tests plus event ordering and eager-equivalence output/gradient probe",
        "closure": "overlap matches serial outputs/gradients and the trace proves no shared-expert read races dispatch or combine",
    },
    "sglang-pr-37523": {
        "risk_family": "expert-parallel-dispatch",
        "claim": "The opt-in NCCL MoE dispatcher uses local route metadata consistently and leaves the default dispatcher unchanged.",
        "matrix": [
            "dispatcher on/off",
            "EP1/EP2",
            "zero/uneven local routes",
            "prefill/decode",
            "valid/malformed metadata",
        ],
        "runtime": "candidate manual test, isolated route-metadata oracle, and two-rank NCCL dispatch/combine comparison",
        "closure": "local route counts and inverse permutation are exact on every rank; opt-out matches the prior path and malformed metadata fails closed",
    },
    "sglang-pr-27289": {
        "risk_family": "collective-layout-numerics",
        "claim": "Removing the ROCm decode FP8-scale transpose-copy preserves the scale layout, values, and communicator ownership for DSV4 attention.",
        "matrix": [
            "MHA/MLA",
            "prefill/decode",
            "one/two ranks",
            "contiguous/noncontiguous scale",
            "ROCm supported dtype",
        ],
        "runtime": "exact data-flow/layout proof plus candidate ROCm tests; target-backend numeric execution is required for full closure",
        "closure": "the no-copy scale consumed by every decode path is layout/value-equivalent to the former transpose-copy reference",
    },
    "sglang-pr-27150": {
        "risk_family": "dynamic-expert-placement",
        "claim": "Waterfill routing remains correct when EPLB dynamically remaps expert ownership between ranks.",
        "matrix": [
            "static/dynamic EPLB",
            "balanced/skewed load",
            "EP1/EP2",
            "remap before/after batch",
            "duplicate/missing placement",
        ],
        "runtime": "candidate EPLB tests plus exact waterfill assignment and two-rank remap oracle",
        "closure": "each token reaches exactly one current expert owner and waterfill scores match the independent placement-aware reference",
    },
    "sglang-pr-27211": {
        "risk_family": "expert-parallel-fused-combine",
        "claim": "The FlashInfer CuteDSL fused combine produces correct DeepEP low-latency MoE output without rank, dtype, or shape disagreement.",
        "matrix": [
            "fused/reference combine",
            "EP1/EP2",
            "zero/uneven tokens",
            "supported dtype/architecture",
            "prefill/decode",
        ],
        "runtime": "source contract and candidate tests plus target-hardware fused/reference multi-rank comparison where available",
        "closure": "fused output and token ownership equal the reference combine across the declared DeepEP low-latency matrix",
    },
    "torchtitan-pr-4399": {
        "risk_family": "pipeline-collective-cardinality",
        "claim": "Non-loss pipeline stages skip valid-token all-reduce while loss stages retain the collective and global loss normalization.",
        "matrix": [
            "PP1/PP2",
            "loss/non-loss stage",
            "first/steady/last microbatch",
            "zero/nonzero valid tokens",
        ],
        "runtime": "candidate CPU tests plus two-rank collective trace and normalized-loss oracle",
        "closure": "non-loss stages issue no redundant collective, loss stages agree on cardinality, and reported loss matches the global-token reference",
    },
    "torchtitan-pr-3447": {
        "risk_family": "dtensor-moe-sharding",
        "claim": "Full-DTensor mode works for all declared MoE models without losing expert/shard ownership or changing training numerics.",
        "matrix": [
            "DeepSeek/Qwen/GPT-OSS MoE",
            "full-DTensor on/off",
            "FSDP/TP reduced topology",
            "one-step forward/backward",
        ],
        "runtime": "feature registration and source-shard proof plus available two-rank model-fragment numeric oracle",
        "closure": "every model maps parameters and activations to legal placements and gathered loss/gradients match the non-full-DTensor reference",
    },
    "torchtitan-pr-3499": {
        "risk_family": "pipeline-p2p-progress",
        "claim": "Per-direction pipeline P2P communication avoids the named TorchComms deadlock while preserving stage payloads and schedule order.",
        "matrix": [
            "PP2",
            "forward/backward direction",
            "simultaneous send/recv",
            "one/multiple microbatches",
            "delayed peer",
        ],
        "runtime": "candidate P2P tests plus timeout-protected two-rank schedule and tensor-value trace",
        "closure": "the base-distinguishing schedule completes on head with exact peer payloads and balanced send/recv operations",
    },
    "torchtitan-pr-3430": {
        "risk_family": "context-parallel-varlen",
        "claim": "Variable-length attention remains correct under Context Parallel plus Full DTensor for supported model families.",
        "matrix": [
            "CP1/CP2",
            "equal/uneven sequence lengths",
            "full-DTensor on/off",
            "Llama/Qwen",
            "forward/backward",
        ],
        "runtime": "candidate varlen tests plus two-rank shard/gather attention and gradient oracle",
        "closure": "packed positions, masks, outputs, and gathered gradients match the non-CP variable-length reference",
    },
    "verl-pr-7631": {
        "risk_family": "weight-sync-offload-lifecycle",
        "claim": "Actor parameters are offloaded only after disaggregated weight synchronization has transferred and consumed the current weights.",
        "matrix": [
            "sync success/failure",
            "offload on/off",
            "single/repeated update",
            "slow receiver",
            "next actor step",
        ],
        "runtime": "candidate global-step tests plus exact worker call-order/ownership and two-rank weight-value probe",
        "closure": "receiver observes the new weights before offload, failures retain recoverable ownership, and the next step reloads exact parameters",
    },
    "verl-pr-6569": {
        "risk_family": "async-broadcast-progress",
        "claim": "The HCCL checkpoint engine executes asynchronous broadcast rather than silently dropping or prematurely completing the operation.",
        "matrix": [
            "async/sync",
            "rank0/peer",
            "empty/boundary/multi-bucket payload",
            "success/timeout/error",
            "repeat load",
        ],
        "runtime": "exact async-handle/control-flow probe plus target HCCL execution where available",
        "closure": "async broadcast is launched, awaited once, propagates errors, and delivers byte-identical payloads on all ranks",
    },
    "verl-pr-6507": {
        "risk_family": "checkpoint-metadata-consistency",
        "claim": "Global training step metadata reaches every checkpoint engine and worker consistently without changing payload or resume ownership.",
        "matrix": [
            "HCCL/NCCL/NIXL/Mooncake/Kimi engines",
            "step 0/1/large",
            "save/load",
            "single/repeated checkpoint",
            "missing metadata",
        ],
        "runtime": "candidate CPU tests and exact per-engine call/round-trip matrix",
        "closure": "every engine records and restores the identical global step exactly once while legacy missing metadata follows the declared fallback",
    },
    "vllm-pr-54960": {
        "risk_family": "connector-observability",
        "claim": "EC connector metrics count scheduler and worker transfer activity accurately without double counting, stale labels, or affecting scheduling.",
        "matrix": [
            "send/receive success/failure",
            "zero/one/many requests",
            "worker/scheduler aggregation",
            "reset/repeat",
            "connector disabled",
        ],
        "runtime": "candidate metric tests plus exact event-to-counter trace and scheduling no-op control",
        "closure": "counter deltas equal independent event counts, labels remain bounded, reset is exact, and outputs/scheduling match metrics-disabled control",
    },
    "vllm-pr-44495": {
        "risk_family": "shared-memory-port-lifecycle",
        "claim": "Shared-memory broadcast eliminates the ZMQ port TOCTOU window by retaining atomic ownership from allocation through bind.",
        "matrix": [
            "one/two concurrent creators",
            "port collision",
            "bind success/failure",
            "close/retry",
            "IPv4/declared endpoint",
        ],
        "runtime": "isolated socket ownership race plus repeated multiprocess bind/broadcast probe",
        "closure": "a competing binder cannot steal the selected port and every failure releases ownership so retry and broadcast complete",
    },
    "vllm-pr-44583": {
        "risk_family": "kv-transfer-region-mapping",
        "claim": "NIXL classifies and transfers mixed full-attention and MLA KV regions with correct per-region offsets and TP ownership.",
        "matrix": [
            "full-attn/MLA/mixed groups",
            "TP1/TP2",
            "one/multiple regions",
            "partial/full transfer",
            "invalid mapping",
        ],
        "runtime": "candidate unit tests plus exact region classifier/offset and two-rank reconstruction oracle",
        "closure": "every region is classified once, transferred to its intended owner, and reconstructs the original mixed KV tensors byte-for-byte",
    },
    "vllm-pr-44577": {
        "risk_family": "kv-cache-packing-ownership",
        "claim": "DeepSeek-V4 KV caches pack into contiguous per-block allocations without overlap, holes, stale views, or attention-value changes.",
        "matrix": [
            "one/multiple attention groups",
            "full-attn/MLA",
            "one/multiple blocks",
            "allocate/free/reuse",
            "eager/captured access",
        ],
        "runtime": "candidate packing tests plus exact offset/non-overlap proof and attention read/write reconstruction oracle",
        "closure": "all cache views are in-bounds, non-overlapping and contiguous per block, and round-trip attention tensors equal the unpacked reference",
    },
    "liger-pr-1405": {
        "risk_family": "cross-entropy-loss",
        "claim": "CCE loss preserves reference cross-entropy values and gradients across supported reductions, masking, dtypes, and model integration.",
        "matrix": [
            "FP32/BF16",
            "none/sum/mean",
            "ignore index present/absent",
            "small/large vocab",
            "functional/module API",
        ],
        "runtime": "candidate CCE tests and independent PyTorch forward/backward oracle with an A100 memory/timing control",
        "closure": "loss and every input/weight gradient meet frozen tolerances, ignored rows contribute zero, and peak memory does not regress its claim",
    },
    "liger-pr-1219": {
        "risk_family": "swiglu-mixed-precision",
        "claim": "Ascend SwiGLU multiplier and mixed-precision tuning preserve forward/backward semantics and legal UB/grid launch configuration.",
        "matrix": [
            "multiplier default/custom",
            "FP16/BF16/FP32 accumulation",
            "short/long hidden size",
            "contiguous/strided",
            "forward/backward",
        ],
        "runtime": "exact algebra and launch-shape proof plus target NPU reference tests where available",
        "closure": "forward and gradients match the independent SwiGLU multiplier oracle and every declared shape has an in-bounds launch",
    },
    "megatron-pr-5146": {
        "risk_family": "optimizer-sync-ownership",
        "claim": "ChainedOptimizer defers MXFP8 synchronization only when DDP-level overlap_param_gather owns the overlap, avoiding missing or duplicate sync.",
        "matrix": [
            "overlap on/off",
            "MXFP8/non-MXFP8",
            "one/chained optimizers",
            "one/multiple steps",
            "exception/normal finalize",
        ],
        "runtime": "exact branch/call-count probe and one-step parameter/state equivalence oracle",
        "closure": "each step performs exactly one required sync at the owning layer and matches the non-deferred optimizer update",
    },
    "megatron-pr-5144": {
        "risk_family": "loss-fusion-compatibility",
        "claim": "TE cross-entropy fusion is disabled for the incompatible release configurations while compatible configurations retain their prior behavior.",
        "matrix": [
            "fusion requested/default",
            "affected/unaffected model configs",
            "TP/PP/CP release layouts",
            "forward/backward loss",
        ],
        "runtime": "candidate config tests plus exact configuration projection and fused/reference loss-gradient oracle where TE permits",
        "closure": "affected configs select the safe loss path with reference-equivalent loss/gradients and unaffected configs do not change",
    },
    "slime-pr-2304": {
        "risk_family": "training-memory-observability",
        "claim": "Peak-memory reporting for log-probability and actor training measures the intended phase on the active accelerator without perturbing training.",
        "matrix": [
            "log-prob/actor-train phase",
            "CUDA/accelerator abstraction",
            "reset/repeat",
            "empty/normal batch",
            "success/exception",
        ],
        "runtime": "candidate accelerator tests plus phase-delimited allocation probe and loss/gradient no-op control",
        "closure": "reported peaks bound independently measured phase allocations, reset between phases, and training values/gradients remain exact",
    },
    "slime-pr-2011": {
        "risk_family": "ppo-loss-memory",
        "claim": "The fused log-probability plus entropy cross-entropy path reduces peak memory while preserving PPO values, gradients, and repeated-backward behavior.",
        "matrix": [
            "entropy requested/omitted",
            "FP32/BF16",
            "short/long vocabulary",
            "single/repeated backward",
            "TP1/TP2 where supported",
        ],
        "runtime": "candidate fused tests plus A100 reference value/gradient and peak-allocation comparison",
        "closure": "loss/log-prob/entropy and gradients meet tolerances, retained-graph replay is valid, and measured peak is lower than the baseline",
    },
    "torchtitan-pr-3525": {
        "risk_family": "rl-data-parallel-topology",
        "claim": "The RL trainer/generator supports DP=2 on a four-GPU dense and MoE mesh with correct sharding, progress, loss, and gradients.",
        "matrix": [
            "dense/MoE",
            "DP1/DP2",
            "four-GPU declared mesh",
            "generate/train",
            "one/multiple rollout batches",
        ],
        "runtime": "candidate integration path and reduced two-GPU ownership checks; exact four-GPU execution is required for full closure",
        "closure": "the declared four-GPU step completes and gathered parameters, losses, and gradients match the DP1 reference",
    },
    "torchtitan-pr-3522": {
        "risk_family": "activation-offload-view-lifecycle",
        "claim": "CPU activation offload replays tensor views with correct storage, shape, stride, version, and backward gradients.",
        "matrix": [
            "base/view/nested view",
            "contiguous/strided",
            "single/repeated backward",
            "offload on/off",
            "mutation/no mutation",
        ],
        "runtime": "candidate CPU-offload tests plus exact storage/view reconstruction and gradient oracle",
        "closure": "replayed views reproduce metadata and values, avoid illegal alias mutation, and match non-offloaded gradients across repeated use",
    },
    "verl-pr-6566": {
        "risk_family": "megatron-optimizer-integration",
        "claim": "The changed Megatron optimizer integration preserves parameter grouping, optimizer construction, state ownership, and one-step updates.",
        "matrix": [
            "single/chained optimizer",
            "offload on/off",
            "FP32/mixed precision",
            "save/load state",
            "one/two steps",
        ],
        "runtime": "candidate optimizer tests plus exact parameter-group/state schema and continuous-versus-resumed trajectory oracle",
        "closure": "all trainable parameters appear once, state round-trips, and resumed parameters/moments match the continuous reference",
    },
    "verl-pr-6593": {
        "risk_family": "distillation-loss-memory",
        "claim": "Chunked gather-logsumexp computes the same top-K distillation loss and gradients while bounding peak memory at long context.",
        "matrix": [
            "one/multiple chunks",
            "chunk boundary +/-1",
            "short/long context",
            "FP32/BF16",
            "top-K small/large",
        ],
        "runtime": "candidate CPU tests plus A100 unchunked reference value/gradient and peak-allocation comparison",
        "closure": "loss and student gradients meet frozen tolerances at all boundaries and measured peak scales with chunk rather than full context",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--domain-amendment", type=Path, required=True)
    parser.add_argument("--r14-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R15 selection lock digest mismatch")
    amendment = json.loads(args.domain_amendment.read_text(encoding="utf-8"))
    amendment_material = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    if amendment["amendment_sha256"] != canonical_sha256(amendment_material):
        raise SystemExit("R15 domain amendment digest mismatch")
    if selection_material["domain_amendment_sha256"] != amendment["amendment_sha256"]:
        raise SystemExit("R15 selection/amendment binding mismatch")
    iteration = json.loads(args.r14_iteration.read_text(encoding="utf-8"))
    iteration_material = {
        key: value for key, value in iteration.items() if key != "iteration_sha256"
    }
    if iteration["iteration_sha256"] != canonical_sha256(iteration_material):
        raise SystemExit("R14 iteration digest mismatch")
    if selection_material["r14_policy_iteration_sha256"] != iteration["iteration_sha256"]:
        raise SystemExit("R15 selection/R14 iteration binding mismatch")
    hidden = (
        selection_material["review_or_comment_visible"],
        selection_material["merge_outcomes_visible"],
        selection_material["ci_or_label_visible"],
        selection_material["candidate_body_visible"],
        selection_material["diff_content_visible"],
        selection_material["excluded_resolution_gray_zone_used"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R15 selection exposes hidden evidence")
    cases = selection_material["cases"]
    if len(cases) != 30 or {item["case_id"] for item in cases} != set(CASE_CONTRACTS):
        raise SystemExit("R15 selection and case-contract sets differ")
    allocation = {
        domain: sum(item["benchmark_domain"] == domain for item in cases)
        for domain in ("communication", "training")
    }
    if allocation != {"communication": 20, "training": 10}:
        raise SystemExit("R15 domain allocation changed")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": selection_material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "domain_amendment_sha256": amendment["amendment_sha256"],
        "r14_policy_iteration_sha256": iteration["iteration_sha256"],
        "machine_policy_id": selection_material["machine_policy_id"],
        "domain_allocation": allocation,
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
            "technical_bounded_gap_does_not_automatically_map_to_disposition_check": True,
        },
        "disposition_policy": {
            "accept": "title-scoped production contract has sufficient exact evidence and no reachable blocker",
            "check": "recent, <=4 changed files, primary direction demonstrated, candidate closure coverage, exactly one executable residual, and no superseding or reachable counterexample",
            "reject": "exact reachable failure, broad remediation, mature disposition lacking accept evidence, or broad hardware feature without evaluator numeric closure",
            "unresolved": "required backend, architecture, topology, or exact evidence is unavailable; recorded technically without fabricating a failure",
            "prospective_created_at_cutoff": "2026-08-04T00:00:00Z",
            "mature_created_at_cutoff": "2026-06-04T23:59:59Z",
            "resolution_gray_zone_excluded": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "target_hardware_transfer_rule": {
            "eligible": "mature bug fix changing <=3 files with candidate target-hardware coverage, locally closed source invariant, and no exact counterexample",
            "broad_new_hardware_feature_eligible": False,
            "missing_evaluator_hardware_alone_is_reject": False,
        },
        "ordered_reachability_gate": [
            "configuration is legal and title-scoped",
            "production call site reaches the alleged behavior",
            "rank/group/shape/stream/lifetime invariant is violated or satisfied",
            "exact counterexample or independent oracle distinguishes base and head",
            "remediation remains bounded to the candidate direction",
        ],
        "evidence_tiers": [
            "exact candidate-owned tests",
            "exact isolated base/head contract probe",
            "two-A100 multi-rank numeric/progress/lifecycle probe",
            "candidate-authored body evidence projected only after this plan is frozen",
            "full model/backend E2E where hardware and dependencies permit",
        ],
        "cross_case_controls": [
            "No synthetic counterexample blocks unless production reachability is demonstrated.",
            "A timeout needs bounded rank diagnostics and a healthy control before proving deadlock.",
            "Test pass or technical correctness alone does not prove historical merge disposition.",
            "A100-side skip cannot masquerade as target NPU, ROCm, MNNVL, or newer-architecture execution.",
            "Short terms such as p2p, ppo, and dpo are token matched; test paths alone cannot define the domain.",
            "Candidate body remains an outcome-free but separately hashed evidence tier.",
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
