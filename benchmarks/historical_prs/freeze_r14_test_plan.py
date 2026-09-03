#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze 30 case-specific R14 contracts before body or diff acquisition."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

CASE_CONTRACTS: dict[str, dict[str, Any]] = {
    "vllm-pr-54643": {
        "risk_family": "transport-lifecycle",
        "claim": "MooncakeStore finish-time save completes for hybrid models without crashing, losing required save work, or double-finalizing transfer state.",
        "matrix": [
            "hybrid/non-hybrid",
            "successful/failed save",
            "finish once/repeated",
            "empty/non-empty transferred blocks",
        ],
        "runtime": "candidate-owned scheduler tests plus isolated exact base/head finish state machine",
        "closure": "a hybrid save that crashes or loses completion on base and completes exactly once on head",
    },
    "vllm-pr-50775": {
        "risk_family": "rank-topology",
        "claim": "The skip-P2P-check mode validates every peer pair needed by the custom all-reduce topology rather than accepting after a partial check.",
        "matrix": [
            "2/3/4 visible peers",
            "first/middle/last incompatible pair",
            "skip flag on/off",
            "asymmetric peer capability",
        ],
        "runtime": "exact topology call trace and candidate-owned custom-all-reduce unit test; GPU peer access corroboration",
        "closure": "a non-first incompatible peer pair that base accepts and head rejects or disables custom all-reduce",
    },
    "vllm-pr-50658": {
        "risk_family": "collective-shape-numerics",
        "claim": "Kimi-K3 DSpark auxiliary states are projected to the correct width before sequence-parallel all-gather while neighboring paths retain shape and values.",
        "matrix": [
            "SP1/SP2",
            "aux state present/absent",
            "decode/prefill",
            "CUDA graph eager/captured",
        ],
        "runtime": "exact shape/data-flow proof, candidate tests, and two-rank all-gather numeric oracle",
        "closure": "base/head shape distinction plus gathered auxiliary-state equality to an independent projection oracle",
    },
    "vllm-pr-54619": {
        "risk_family": "shared-memory-lifecycle",
        "claim": "Flock-based liveness reaps only orphaned shared-memory region files across EC transfer and KV offload owners.",
        "matrix": [
            "live/orphan/stale-lock file",
            "normal/abrupt owner exit",
            "two concurrent owners",
            "repeat scan/close",
        ],
        "runtime": "candidate CPU tests plus multiprocess exact lifecycle probe",
        "closure": "orphan is reclaimed while a live locked region survives concurrent scans and closes exactly once",
    },
    "vllm-pr-50754": {
        "risk_family": "p2p-progress",
        "claim": "A failed NIXL write notifies the decode side so it stops waiting without falsely completing or leaking pending state.",
        "matrix": [
            "post success/failure",
            "single/multiple blocks",
            "failure before/after partial post",
            "retry/cancel/close",
        ],
        "runtime": "exact metadata/worker state-machine trace and candidate unit tests",
        "closure": "injected post failure causes bounded decode termination and balanced cleanup instead of an indefinite wait",
    },
    "sglang-pr-37261": {
        "risk_family": "expert-parallel-dispatch",
        "claim": "DeepEP-V2 expanded prefill dispatch routes expanded tokens and expert metadata consistently through supported MoE configurations.",
        "matrix": [
            "expand on/off",
            "prefill/decode",
            "EP1/EP2",
            "zero/uneven tokens",
            "supported model hooks",
        ],
        "runtime": "exact configuration/call-shape matrix, candidate tests, and two-rank dispatch/combine oracle where backend permits",
        "closure": "token counts, expert ownership, permutation inversion, and outputs match an independent expanded-dispatch oracle",
    },
    "sglang-pr-33029": {
        "risk_family": "collective-progress",
        "claim": "HiCache prefetch progress no longer deadlocks at the tensor-parallel all-reduce and still publishes correct progress.",
        "matrix": [
            "TP1/TP2",
            "empty/partial/full prefetch",
            "one rank delayed",
            "success/error/cancel",
        ],
        "runtime": "two-rank schedule trace with timeout, collective cardinality, and progress-value oracle",
        "closure": "a frozen interleaving that times out on base completes on head with the expected progress on both ranks",
    },
    "sglang-pr-33220": {
        "risk_family": "stream-lifecycle",
        "claim": "A process-wide CUDA graph capture stream is safely reused without cross-instance capture conflicts or lifetime corruption and actually reduces memory.",
        "matrix": [
            "one/two runtime contexts",
            "sequential/nested capture",
            "destroy/recreate",
            "same/different device",
        ],
        "runtime": "exact lifecycle probe plus paired A100 allocation and capture/replay comparison",
        "closure": "repeated multi-context capture/replay is numerically stable, leak-free, and shows a measured non-regressing peak",
    },
    "sglang-pr-33228": {
        "risk_family": "expert-parallel-counts",
        "claim": "EPLB excludes fused shared slots from DeepEP expert counts without dropping routed experts or corrupting load records.",
        "matrix": [
            "shared fusion on/off",
            "normal/low-latency DeepEP",
            "zero/one/many shared slots",
            "uneven rank ownership",
        ],
        "runtime": "exact count/slot mapping matrix and candidate recorder tests",
        "closure": "all routed experts appear exactly once and fused shared slots contribute zero to DeepEP counts",
    },
    "sglang-pr-33053": {
        "risk_family": "device-rank-initialization",
        "claim": "Each diffusion worker binds its accelerator before distributed initialization so process-group device ownership matches local rank.",
        "matrix": [
            "rank0/rank1",
            "prebound/unbound",
            "valid/invalid local rank",
            "initialization once/repeated",
        ],
        "runtime": "candidate CPU call-order test plus two-rank NCCL initialization smoke",
        "closure": "set-device precedes process-group creation on every rank and the two-rank collective completes on distinct GPUs",
    },
    "flashinfer-pr-4302": {
        "risk_family": "expert-parallel-kernel",
        "claim": "The SM12x W4A16 fused-MoE expert-parallel path preserves token routing, output numerics, and trace behavior.",
        "matrix": [
            "EP1/EP2",
            "balanced/uneven/zero-token expert",
            "eager/trace",
            "supported SM12x configuration",
        ],
        "runtime": "candidate kernel/reference tests; exact SM12x multi-rank execution is required for full closure",
        "closure": "dispatch/combine ownership and fused output match the non-EP/reference oracle across the frozen routing matrix",
    },
    "flashinfer-pr-4139": {
        "risk_family": "p2p-progress",
        "claim": "NIXL-EP combine avoids both named real-serving deadlocks without dropping tokens or violating handle ownership.",
        "matrix": [
            "two named deadlock interleavings",
            "empty/non-empty sender",
            "delayed peer",
            "failure/cancel/teardown",
        ],
        "runtime": "candidate fleet mock plus deterministic state-machine timeout and available two-rank NIXL execution",
        "closure": "both base-distinguishing interleavings terminate with exact combined token ownership and balanced handle cleanup",
    },
    "flashinfer-pr-4240": {
        "risk_family": "communicator-topology",
        "claim": "Ulysses and mixed communicators expose a coherent topology/API contract consistent with their executable implementation.",
        "matrix": [
            "Ulysses-only/mixed",
            "world2/world4 topology model",
            "valid/invalid group",
            "construct/use/teardown",
        ],
        "runtime": "exact API/import/topology probe; multi-rank numeric smoke when hardware cardinality permits",
        "closure": "documented construction and rank mapping execute without stale names, missing exports, or topology disagreement",
    },
    "flashinfer-pr-4296": {
        "risk_family": "expert-parallel-shape",
        "claim": "MoE-EP preserves valid singleton-expert TMA modes while retaining multi-expert correctness.",
        "matrix": [
            "one/two/many local experts",
            "zero/nonzero tokens",
            "swap-A/B modes",
            "supported SM100 dtype/layout",
        ],
        "runtime": "candidate kernel/reference matrix; exact SM100 execution required for full closure",
        "closure": "singleton and neighboring multi-expert outputs match the independent reference without invalid TMA configuration",
    },
    "flashinfer-pr-4174": {
        "risk_family": "kernel-collective-progress",
        "claim": "Disabling PDL on the FP4 MoE path prevents the named tensor-parallel deadlock without regressing output correctness or supported launches.",
        "matrix": [
            "TP1/TP2",
            "PDL requested/default",
            "small/large token count",
            "supported architecture",
        ],
        "runtime": "exact launch-option proof plus timeout-protected multi-rank kernel/reference comparison",
        "closure": "the base-distinguishing TP schedule completes on head and matches reference output while neighboring launches remain valid",
    },
    "megatron-pr-6955": {
        "risk_family": "communicator-initialization",
        "claim": "Resharding initializes NCCL on otherwise idle ranks so copy-service operations have a complete participant set and terminate.",
        "matrix": [
            "active/idle source and destination ranks",
            "2-rank reduced topology",
            "initialization once/repeated",
            "copy success/failure",
        ],
        "runtime": "candidate copy-service tests plus two-rank initialization/copy state-machine smoke",
        "closure": "all required ranks initialize the same communicator and a formerly idle-rank transfer completes exactly once",
    },
    "megatron-pr-6200": {
        "risk_family": "collective-numerics",
        "claim": "GTP reduce-scatter with FP32 accumulation improves accumulation accuracy while preserving shards, dtype contract, and gradients.",
        "matrix": [
            "TP2",
            "FP16/BF16 input",
            "small/large reduction depth",
            "feature on/off",
            "uneven magnitude",
        ],
        "runtime": "candidate correctness tests plus two-rank comparison with independent FP32-accumulate and legacy oracles",
        "closure": "each rank receives the correct shard and error versus FP32 reference is non-regressing across the frozen matrix",
    },
    "megatron-pr-6963": {
        "risk_family": "ddp-storage-ownership",
        "claim": "Fused expert parameter storage remains correctly owned and address-stable through DDP buffering and distributed optimizer layout.",
        "matrix": [
            "fused/non-fused experts",
            "DDP on/off",
            "one/two ranks",
            "step/checkpoint/reload",
        ],
        "runtime": "candidate layout/grouped-MLP tests plus two-rank storage/gradient/update oracle",
        "closure": "fused weights retain intended aliasing, receive exact gradients, and update identically without stale storage",
    },
    "megatron-pr-7000": {
        "risk_family": "pipeline-p2p-shape",
        "claim": "Pipeline parallelism skips shape exchange only for truly fixed packed sequences and preserves matched send/receive tensor metadata.",
        "matrix": [
            "fixed/dynamic packed shape",
            "first/steady/last microbatch",
            "PP2",
            "MHC-compatible model path",
        ],
        "runtime": "exact branch/call-count proof plus two-rank P2P send/receive numeric matrix",
        "closure": "fixed shapes remove only the redundant exchange; dynamic shapes still exchange and every peer agrees on payload metadata",
    },
    "megatron-pr-6973": {
        "risk_family": "overlap-ordering",
        "claim": "MFSDP DP-outer gradient reduction overlaps on its own stream without racing gradient production, optimizer use, or teardown.",
        "matrix": [
            "one/multiple parameter groups",
            "deferred/eager",
            "microbatch accumulation",
            "exception/normal teardown",
        ],
        "runtime": "candidate tests plus two-rank event/stream trace and eager-equivalence gradient/update oracle",
        "closure": "overlapped and eager gradients/updates match while trace proves required waits and actual overlap",
    },
    "torchtitan-pr-3953": {
        "risk_family": "collective-numerics",
        "claim": "Chunked graph-trainer gradient collectives use one consistent normalization and remain numerically equivalent to the unchunked reference.",
        "matrix": [
            "one/multiple chunks",
            "partial final chunk",
            "one/two ranks",
            "replicate/shard",
            "accumulation on/off",
        ],
        "runtime": "candidate numerics/pass tests plus two-rank independent gradient oracle",
        "closure": "every parameter gradient and one-step update matches the unchunked normalized reference",
    },
    "torchtitan-pr-4051": {
        "risk_family": "optimizer-sharding",
        "claim": "DistributedMuon supports TP-sharded matrices with correct ownership, statistics, communication, and optimizer updates.",
        "matrix": [
            "replicated/row/column shard",
            "TP2",
            "matrix/vector parameter",
            "checkpoint/resume",
        ],
        "runtime": "candidate tests plus two-rank update and state-shard comparison with a gathered single-rank oracle",
        "closure": "gathered TP update/state matches the reference and no rank silently omits or duplicates a shard",
    },
    "torchtitan-pr-3955": {
        "risk_family": "expert-parallel-overlap",
        "claim": "MinimalAsyncEP eager overlap preserves dispatch/combine ordering and numerics while creating real communication/compute overlap.",
        "matrix": [
            "EP2",
            "zero/uneven tokens",
            "one/multiple chunks",
            "eager-overlap on/off",
            "delayed peer",
        ],
        "runtime": "candidate tests plus timeout-protected two-rank event trace and non-overlap numeric oracle",
        "closure": "outputs/gradients match non-overlap, no schedule deadlocks, and trace contains an actual overlap interval",
    },
    "torchtitan-pr-4018": {
        "risk_family": "rank-topology-testability",
        "claim": "Fake process groups simulate arbitrary ranks with group-correct rank/world semantics rather than leaking the host process rank.",
        "matrix": [
            "world1/2/4",
            "every simulated rank",
            "subgroup/non-member",
            "invalid rank",
            "nested fake group",
        ],
        "runtime": "candidate CPU tests and exact fake-group operation matrix",
        "closure": "each simulated rank observes the expected global/group rank and collective shape while invalid configurations fail closed",
    },
    "torchtitan-pr-3980": {
        "risk_family": "rl-parallel-topology",
        "claim": "The RL trainer/generator supports pipeline and context parallelism without mismatched layouts, gradients, or controller progress.",
        "matrix": [
            "PP1/2",
            "CP1/2",
            "PP2+CP1 reduced hardware case",
            "train/generate",
            "odd/padded sequence",
        ],
        "runtime": "candidate unit/integration tests plus available two-rank PP or CP numeric/progress oracle",
        "closure": "supported topology completes a train/generate step with matched layouts and reference-equivalent loss/gradients",
    },
    "verl-pr-7591": {
        "risk_family": "weight-transfer-overlap",
        "claim": "Bucketed weight transfer overlaps receiver processing without consuming incomplete buckets, reordering parameters, or leaking work.",
        "matrix": [
            "one/multiple buckets",
            "slow sender/receiver",
            "partial final bucket",
            "failure/cancel/repeat sync",
        ],
        "runtime": "candidate tests plus two-rank bucket/event trace and final-weight/forward oracle",
        "closure": "overlap and serial paths produce identical weights/output; trace proves receiver waits per bucket and overlaps later work",
    },
    "verl-pr-7107": {
        "risk_family": "collective-shape",
        "claim": "NCCL checkpoint broadcast uses a bucket size that is valid for every payload boundary without truncation, overrun, or rank disagreement.",
        "matrix": [
            "0/1/exact-boundary/boundary+1 elements",
            "two ranks",
            "multiple dtype sizes",
            "single/repeated load",
        ],
        "runtime": "two-rank exact base/head checkpoint broadcast against a byte-for-byte payload oracle",
        "closure": "all boundary payloads arrive exactly on every rank with identical collective counts and no hang",
    },
    "verl-pr-7045": {
        "risk_family": "communicator-progress",
        "claim": "The first NCCL checkpoint group initialization fails fast when a participant hangs and cleans up without poisoning retry.",
        "matrix": [
            "all ranks ready/one delayed/one absent",
            "timeout boundary",
            "first attempt/retry",
            "CPU orchestration/NCCL",
        ],
        "runtime": "candidate timeout test plus two-rank injected-delay initialization/retry probe",
        "closure": "hung initialization returns a bounded diagnostic on all live owners and a later healthy initialization succeeds",
    },
    "verl-pr-7161": {
        "risk_family": "weight-conversion-ownership",
        "claim": "Moving unfuse_moe_params into the FSDP backend preserves the training-to-rollout parameter conversion and backend ownership contract.",
        "matrix": [
            "fused/unfused MoE",
            "FSDP supported backend",
            "single/repeated sync",
            "failure rollback",
            "neighbor backend",
        ],
        "runtime": "candidate CPU conversion tests plus exact parameter-key/shape/value round trip",
        "closure": "rollout receives every expected unfused parameter exactly once and repeat conversion is idempotent without changing neighboring backends",
    },
    "verl-pr-7589": {
        "risk_family": "weight-transfer-overlap",
        "claim": "The rollout/vLLM bucketed weight-transfer overlap preserves receiver processing order, final weights, and cleanup across the full integration path.",
        "matrix": [
            "one/multiple buckets",
            "slow sender/receiver",
            "partial final bucket",
            "failure/cancel/repeat sync",
        ],
        "runtime": "candidate tests plus two-rank integration trace and final-weight/forward oracle; compare directly with paired case #7591",
        "closure": "integration and serial paths are value-identical, per-bucket readiness is respected, and #7589/#7591 differences are explicitly bounded",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--domain-amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("R14 selection lock digest mismatch")
    amendment = json.loads(args.domain_amendment.read_text(encoding="utf-8"))
    amendment_material = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    if amendment["amendment_sha256"] != canonical_sha256(amendment_material):
        raise SystemExit("R14 domain amendment digest mismatch")
    if material["domain_amendment_sha256"] != amendment["amendment_sha256"]:
        raise SystemExit("R14 selection/amendment binding mismatch")
    hidden = (
        material["review_or_comment_visible"],
        material["merge_outcomes_visible"],
        material["ci_or_label_visible"],
        material["candidate_body_visible"],
        material["diff_content_visible"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R14 selection exposes hidden evidence")
    cases = [HistoricalPRCandidate.model_validate(item) for item in material["cases"]]
    if len(cases) != 30 or {item.case_id for item in cases} != set(CASE_CONTRACTS):
        raise SystemExit("R14 selection and case-contract sets differ")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "domain_amendment_sha256": amendment["amendment_sha256"],
        "machine_policy_id": material["machine_policy_id"],
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
            "accept": "title-scoped production contract is demonstrated with sufficient exact evidence and no reachable blocker",
            "check": "only a prospectively eligible recent case with one bounded closure item; never a mature-history default",
            "reject": "reachable semantic/progress/safety failure, design-scale remediation, or mature disposition prediction lacking accept evidence",
            "unresolved": "required backend, architecture, topology, or exact evidence is unavailable; never counted as candidate failure",
            "prospective_created_at_cutoff": "2026-08-03T00:00:00Z",
            "weighted_score_used": False,
            "forced_polarization_used": False,
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
            "No synthetic counterexample can block unless production reachability is demonstrated.",
            "A timeout must have bounded rank diagnostics and a healthy control before it proves deadlock.",
            "Collective optimization cannot compensate for wrong values, rank ownership, shape, order, or lifetime.",
            "The paired verl #7589/#7591 cases use the same matrix and must explain any differing result.",
            "Candidate body is outcome-free evidence but is stored and weighted separately from evaluator execution.",
        ],
        "cases": [
            {
                "case_id": item.case_id,
                "project": item.project,
                "repository": item.repository,
                "pull_number": item.pull_number,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "base_sha": item.base_sha,
                "head_sha": item.head_sha,
                "changed_paths": item.paths,
                **CASE_CONTRACTS[item.case_id],
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
