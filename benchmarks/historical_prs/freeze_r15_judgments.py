#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the 30-case R15 mixed cohort."""

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

EXPECTED_SELECTION_FILE_SHA256 = "d9b88f1cf797f551aa7b3fdd60dbb28af80f21cc98e0be68bfb0229e99fdfca0"
EXPECTED_TEST_PLAN_FILE_SHA256 = "7d572b3bb2fc7292c78b948da4057b88a00532561b121d77d4fde88361d89ba3"
EXPECTED_SELECTION_SHA256 = (
    "sha256:0ba902883b19f4a5bb0a070354d70c59ef7c7bbcaa95964dd97eafdb69f12712"
)
EXPECTED_TEST_PLAN_SHA256 = (
    "sha256:8efbb95f125b0b2e5ce18fb9f504768fba1cc10f2947f665f48b4c1feb56c15e"
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "sha256:929fc2d96c162f4705f0d7c090e29ae444a6a9210498b3ecb0576bc1a69b052f"
)
PROSPECTIVE_CUTOFF = datetime.fromisoformat("2026-08-04T00:00:00+00:00")
POLICY_ID = "mixed-contract-disposition-split-v0.1-r15"


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
    "flashinfer-pr-4795": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Twenty candidate mock/host tests pass and the persistent handle is capture-stream aware.",
        "Both real moe_ep cases skip because no NCCL EP backend was built, while the new feature spans eight paths and its claimed graph-replay round trip is not evaluator-owned.",
        residual="Build the exact NCCL EP backend and prove numeric dispatch/combine parity plus repeated CUDA-graph replay with routing rebinding.",
    ),
    "flashinfer-pr-3304": assessment(
        ACCEPT,
        "pass",
        "The two-file mature fix replaces floating equality with an exact negative-zero bit sentinel and includes negative-subnormal patterns in its target test.",
        "The invariant is local, architecture-independent at the representation boundary, and no exact counterexample was found; target MNNVL absence alone is not treated as failure.",
    ),
    "megatron-pr-7029": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "The two-rank probe destroys all 23 tracked auxiliary process groups twice, preserves the default group, and repeats model-parallel collectives exactly.",
        "The five-path patch also invalidates uneven-DTensor, fused-A2A, and CUDA-graph communicator caches; those production consumers and failure interleavings remain outside the probe.",
        residual="Exercise every invalidated cache through create/use/destroy/recreate while collectives are live, including the fused-A2A and CUDA-graph consumers.",
    ),
    "megatron-pr-5153": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source adds a DeepEP-v2 flex-dispatcher surface across model callable, dispatcher, and configuration layers.",
        "The candidate test requires a world size divisible by eight and the declared backend/topology is unavailable on two GPUs, so routing and output parity are unexecuted.",
        residual="Run the exact eight-rank DeepEP-v2 dispatcher matrix and compare dispatch/combine ownership and outputs with the reference backend.",
    ),
    "megatron-pr-5135": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "One shared-expert subtest passes and the latent projection branch is present.",
        "The title-scoped latent overlap case stops at the required TransformerEngine boundary, leaving forward/backward parity and overlap lifetime unexecuted.",
        residual="Run both candidate latent shared-expert cases with TransformerEngine on the declared topology and check gradients plus overlap ordering.",
    ),
    "sglang-pr-37523": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "The exact two-A100 torchrun passes all four dispatcher tests on both ranks, including uneven routing, empty ranks, and deterministic combine.",
        "The opt-in dispatcher spans eight integration paths; no serving/model path closes handle reuse, failure propagation, or graph interaction outside the isolated dispatcher test.",
        residual="Run an end-to-end MoE layer/serving path through repeated eager and captured steps with the NCCL dispatcher enabled.",
    ),
    "sglang-pr-27289": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source consistently removes an FP8 scale transpose-copy across ROCm DeepSeek decode paths.",
        "There is no candidate test path and no evaluator ROCm execution, so scale layout, aliasing, and decode numeric parity remain unclosed for this seven-file mature change.",
        residual="Run target ROCm decode parity and memory-copy tracing for MHA and MLA with all supported FP8 scale layouts.",
    ),
    "sglang-pr-27150": assessment(
        ACCEPT,
        "pass",
        "All four isolated Waterfill tests pass after bypassing only an incompatible optional installed CUDA extension.",
        "The three-file change subtracts fused shared slots consistently in both recorder IDs and weight shapes, with non-fused controls.",
    ),
    "sglang-pr-27211": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source connects a CuteDSL fused combine to the DeepEP low-latency path and exposes configuration controls.",
        "The mature seven-path feature has no candidate test path and requires an unavailable model/backend combination; mapping, numerics, and replay are unexecuted.",
        residual="Run Qwen3.5 DeepEP dispatch/combine reference parity, empty-route, TP/EP ownership, and graph replay on the supported backend.",
    ),
    "torchtitan-pr-4399": assessment(
        CHECK,
        "bounded-gap",
        "Four candidate tests pass with the candidate-pinned dependencies, proving non-loss stages return a sentinel and only the loss stage reduces valid tokens.",
        "The recent two-file change has one executable residual: an actual multi-stage pipeline run rather than the focused trainer harness.",
        residual="Run a two-stage PP forward/backward step and count valid-token collectives per stage while comparing loss and gradients.",
    ),
    "torchtitan-pr-3447": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source enables Full DTensor across several MoE model and FSDP implementations.",
        "There is no candidate unit test and the nine-path mature feature lacks an evaluator-owned distributed numeric/gradient closure.",
        residual="Run dense and expert parameter materialization, optimizer, checkpoint, and gradient parity across every newly enabled MoE family.",
    ),
    "torchtitan-pr-3499": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "The available two-rank NCCL mixed-P2P candidate case passes.",
        "The title-scoped per-direction ring requires at least three GPUs, TorchComms is not installed, and four additional integration paths remain unexecuted.",
        residual="Run three-rank NCCL and TorchComms ring controls showing the single communicator deadlocks and per-direction communicators complete with exact payloads.",
    ),
    "torchtitan-pr-3430": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Twelve candidate partition and validation tests pass across single/multiple documents and boundary conditions.",
        "The mature eight-path feature combines variable-length attention, context parallelism, Full DTensor, and two model families without a distributed attention/gradient oracle.",
        residual="Run world-size-four variable-length CP forward/backward parity under Full DTensor for both model integrations.",
    ),
    "verl-pr-7631": assessment(
        CHECK,
        "bounded-gap",
        "The exact method probe observes send completion before CPU offload and confirms the Megatron and offload-enabled guards.",
        "The recent two-file fix has one residual: replace the fake checkpoint engine with a real asynchronous transfer and verify no parameter lifetime race.",
        residual="Run a real disaggregated weight sync with delayed completion and assert offload starts only after the final transfer event.",
    ),
    "verl-pr-6569": assessment(
        ACCEPT,
        "pass",
        "The exact one-file method schedules ZMQ work as a task, moves blocking HCCL into an executor, binds the device in that thread, and awaits completion.",
        "The isolated branch probe passes for publisher and subscriber ordering; the mature narrow repair has no observed counterexample.",
    ),
    "verl-pr-6507": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Both candidate worker-call tests pass and global_steps is propagated through all checkpoint-engine interfaces.",
        "The nine-path mature change lacks backend-specific persistence and stale/repeated-step controls, so signature propagation alone does not close the stated checkpoint semantics.",
        residual="Execute every checkpoint backend with repeated and resumed global steps and verify the stored/received version rather than only call signatures.",
    ),
    "vllm-pr-54960": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Ten focused EC metric tests and the scheduler aggregation test pass under a fail-closed source import shim.",
        "The recent ten-path observability feature spans worker, scheduler, outputs, and Prometheus logging without an independent event-to-counter trace or scheduling no-op control.",
        residual="Replay success/failure/reset/disabled event traces end to end and compare every exported series plus scheduler outputs to metrics-disabled controls.",
    ),
    "vllm-pr-44495": assessment(
        ACCEPT,
        "pass",
        "AST inspection confirms the one-file fix removes the live get_open_port symbol, binds the actual XPUB socket to port zero, and reads LAST_ENDPOINT.",
        "Thirty-two simultaneous live sockets receive unique endpoints, closing the probe-then-bind race at the OS ownership boundary.",
    ),
    "vllm-pr-44583": assessment(
        ACCEPT,
        "pass",
        "The mixed full-attention/MLA handshake test and all six TP-mapping tests pass from exact source.",
        "Per-region flags explicitly select replicated MLA descriptors and split full-attention descriptors, while malformed scaled MLA metadata is rejected.",
    ),
    "vllm-pr-44577": assessment(
        ACCEPT,
        "pass",
        "All five contiguous-packing tests pass for layer coverage, common block stride, offsets, and independent strided views.",
        "The exact layout invariant is shared by the NIXL/offload consumers and no reachable packing counterexample was observed.",
    ),
    "liger-pr-1405": assessment(
        REJECT_FAILURE,
        "fail",
        "The candidate's own sum-reduction backward case has 31 gradient mismatches on the exact head.",
        "The same mismatches repeat three times, directly falsifying the central new CCE correctness contract.",
        residual="Correct sum-reduction gradient scaling and pass the full dtype/shape/reduction forward-backward matrix.",
    ),
    "liger-pr-1219": assessment(
        ACCEPT,
        "pass",
        "The mature one-file NPU refinement applies multiplier chain rules, preserves native multiply dtype, and gives the launch grid a nonzero floor.",
        "The candidate body supplies target-NPU correctness coverage and benchmarks; the locally closed source invariant has no exact counterexample, so missing A100 transfer is not used as rejection.",
    ),
    "megatron-pr-5146": assessment(
        ACCEPT,
        "pass",
        "The exact method passes all five branch combinations: reuse disabled, non-distributed child, all-overlap, one non-overlap, and mixed optimizers.",
        "The one-file mature fix reads the DDP-level overlap setting from each actual DistributedOptimizer instead of an unreliable outer proxy.",
    ),
    "megatron-pr-5144": assessment(
        ACCEPT,
        "pass",
        "Both exact configuration tests pass: native cross-entropy fusion remains allowed and TransformerEngine fusion is rejected.",
        "Although eleven files change, eight are matching functional configuration fixtures; the production rule itself is narrow and closed.",
    ),
    "slime-pr-2304": assessment(
        CHECK,
        "bounded-gap",
        "Three candidate tests pass, including exception-finally reporting and unsupported-device fallback; a real A100 phase records the exact 64 MiB allocation.",
        "The recent four-file feature has one residual: production actor/log-prob phases have not been run to confirm phase boundaries do not overlap or reset an outer measurement.",
        residual="Run one real actor_train plus log_probs iteration and assert exactly one non-nested report per phase with plausible peaks.",
    ),
    "slime-pr-2011": assessment(
        ACCEPT,
        "pass",
        "Seven correctness tests pass, including TP2, gradients, entropy, and no-entropy paths.",
        "At 8192x32768 BF16 with entropy backward, the exact head reduces peak delta from 2.188 GiB to 1.000 GiB, more than halving the measured peak.",
    ),
    "torchtitan-pr-3525": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source is explicitly a draft demonstration spanning seven RL generator/trainer/model paths.",
        "Its declared six-GPU test topology is unavailable and no candidate unit test closes DP2 ownership, gradients, or generator/trainer synchronization.",
        residual="Run the declared four-generator/two-trainer topology with dense and MoE controls and compare parameters, gradients, and rollout handoff.",
    ),
    "torchtitan-pr-3522": assessment(
        REJECT_FAILURE,
        "fail",
        "The candidate's disabled-replay test expects a view chain to remain unoffloaded, but the exact implementation raises ValueError instead.",
        "A candidate-owned reachable configuration therefore violates its own backward-compatibility control.",
        residual="Preserve the disabled-replay behavior or update the contract consistently, then pass all view-chain and same-consumer controls.",
    ),
    "verl-pr-6566": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "All nine candidate optimizer-configuration tests pass across FP32, FP16, BF16, overrides, and distributed mode.",
        "The mature six-path change also touches transformer implementation, worker setup, compatibility helpers, and versioning without an optimizer step/state parity test.",
        residual="Execute at least one Megatron optimizer step per precision branch and compare parameters, optimizer state dtype, checkpoint reload, and distributed behavior.",
    ),
    "verl-pr-6593": assessment(
        REJECT_FAILURE,
        "fail",
        "Sixteen correctness tests pass, but the A100 backward peak at 8192x32768 is 2.015-2.099 GiB for useful chunks versus 1.501 GiB for the baseline.",
        "With chunk 256, peak delta grows from 0.469 GiB at 2048 tokens to 3.957 GiB at 16384 tokens, showing retained autograd state scales with full context and directly contradicts the OOM claim.",
        residual="Use a backward strategy that releases per-chunk full-vocabulary state and demonstrate peak scaling with chunk size rather than total context.",
    ),
}


FACT_CHECKS: dict[str, tuple[str, Any]] = {
    "flashinfer-pr-4795": ("persistent_handle_update_present", True),
    "flashinfer-pr-3304": ("negative_zero_uses_bitwise_test", True),
    "megatron-pr-7029": ("created_process_groups_are_tracked", True),
    "megatron-pr-5153": ("deepep_v2_dispatcher_added", True),
    "megatron-pr-5135": ("latent_shared_expert_path_added", True),
    "sglang-pr-37523": ("candidate_empty_uneven_repeatability_matrix_present", True),
    "sglang-pr-27289": ("rocm_specific_path", True),
    "sglang-pr-27150": ("fused_shared_expert_slots_are_excluded", True),
    "sglang-pr-27211": ("deepep_and_cutedsl_paths_changed", True),
    "torchtitan-pr-4399": ("valid_token_reduction_is_loss_stage_gated", True),
    "torchtitan-pr-3447": ("full_dtensor_moe_change_is_cross_model", True),
    "torchtitan-pr-3499": ("ring_tests_require_three_or_more_gpus", True),
    "torchtitan-pr-3430": ("candidate_partition_edge_matrix_present", True),
    "verl-pr-7631": ("weight_send_precedes_param_offload", True),
    "verl-pr-6569": ("blocking_collective_runs_in_executor", True),
    "verl-pr-6507": ("global_steps_propagates_across_engine_interfaces", True),
    "vllm-pr-54960": ("metrics_flow_spans_worker_scheduler_and_loggers", True),
    "vllm-pr-44495": ("get_open_port_symbol_absent", True),
    "vllm-pr-44583": ("replicate_and_split_descriptor_branches_present", True),
    "vllm-pr-44577": ("candidate_layout_invariants_present", True),
    "liger-pr-1405": ("candidate_gradient_test_present", True),
    "liger-pr-1219": ("body_reports_target_npu_tests", True),
    "megatron-pr-5146": ("gate_reads_ddp_overlap_setting", True),
    "megatron-pr-5144": ("candidate_native_and_te_branch_tests_present", True),
    "slime-pr-2304": ("phase_reporter_resets_and_reads_both_peaks", True),
    "slime-pr-2011": ("candidate_tp2_and_reference_tests_present", True),
    "torchtitan-pr-3525": ("rl_stack_change_spans_seven_paths", True),
    "torchtitan-pr-3522": ("candidate_disabled_replay_test_present", True),
    "verl-pr-6566": ("optimizer_precision_branch_matrix_present", True),
    "verl-pr-6593": ("candidate_gradient_and_chunk_invariance_tests_present", True),
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
        "evidence_sha256": payload.get("evidence_sha256")
        or payload.get("evidence_manifest_sha256"),
        "artifact_sha256": canonical_sha256(payload),
    }


def record_binding(path: Path, payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [
        (index, record)
        for index, record in enumerate(payload.get("records", []))
        if record.get("case_id") == case_id
    ]
    require(len(matches) == 1, f"{path.name}: expected one {case_id} record")
    index, record = matches[0]
    return {
        "artifact": binding(path, payload),
        "record_index": index,
        "returncode": record.get("returncode"),
        "status": record.get("status"),
        "output_sha256": record.get("output_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(
        file_sha256(args.selection_lock) == EXPECTED_SELECTION_FILE_SHA256,
        "R15 selection file digest mismatch",
    )
    require(
        file_sha256(args.test_plan) == EXPECTED_TEST_PLAN_FILE_SHA256,
        "R15 test-plan file digest mismatch",
    )
    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    require(
        selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]),
        "R15 embedded selection digest mismatch",
    )
    require(
        selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
        "R15 selection identity changed",
    )
    require(
        plan["test_plan_sha256"]
        == canonical_sha256(
            {key: value for key, value in plan.items() if key != "test_plan_sha256"}
        ),
        "R15 embedded test-plan digest mismatch",
    )
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R15 test-plan identity changed")
    require(
        plan["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
        "R15 plan/selection binding mismatch",
    )
    require(
        plan["disposition_policy"]["weighted_score_used"] is False,
        "R15 unexpectedly uses weighted scoring",
    )
    require(
        plan["disposition_policy"]["forced_polarization_used"] is False,
        "R15 unexpectedly forces polarization",
    )
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
    require(all(value is False for value in blind_flags), "R15 blind boundary is not intact")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 30, "R15 cohort is not 30 cases")
    require(
        selected.keys() == planned.keys() == ASSESSMENTS.keys() == FACT_CHECKS.keys(),
        "R15 case sets differ",
    )

    evidence_paths = {
        name: args.result_root / filename
        for name, filename in {
            "manifest": "candidate-evidence-manifest.json",
            "static": "static-evidence.json",
            "contract": "contract-probes.json",
            "initial": "upstream-test-matrix.json",
            "followup": "upstream-followup-tests.json",
            "rerun": "upstream-followup-tests-rerun-v2.json",
        }.items()
    }
    evidence = {name: read(path) for name, path in evidence_paths.items()}
    validate_digest(evidence["manifest"], "evidence_manifest_sha256", "manifest")
    for name in ("static", "contract", "initial", "followup", "rerun"):
        validate_digest(evidence[name], "evidence_sha256", name)
    require(
        evidence["manifest"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256,
        "R15 source bundle identity changed",
    )
    require(
        sum(bool(case["body_sanitization"]["redactions"]) for case in evidence["manifest"]["cases"])
        == 4,
        "R15 body sanitization case count changed",
    )
    require(
        sum(len(case["body_sanitization"]["redactions"]) for case in evidence["manifest"]["cases"])
        == 4,
        "R15 body sanitization block count changed",
    )
    for name in ("static", "contract"):
        require(
            evidence[name]["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
            f"{name}: selection binding mismatch",
        )
        require(
            evidence[name]["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256,
            f"{name}: test-plan binding mismatch",
        )
        require(
            evidence[name]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256,
            f"{name}: source binding mismatch",
        )

    contract = {row["case_id"]: row["facts"] for row in evidence["contract"]["cases"]}
    require(contract.keys() == selected.keys(), "R15 contract case set differs")
    for case_id, (key, expected) in FACT_CHECKS.items():
        require(contract[case_id][key] == expected, f"{case_id}: contract fact {key} changed")

    followup_output = {
        record["case_id"]: record["output_tail"] for record in evidence["followup"]["records"]
    }
    rerun_output = {
        record["case_id"]: record["output_tail"] for record in evidence["rerun"]["records"]
    }
    require(
        followup_output["liger-pr-1405"].count("Number of mismatched elements: 31") == 3,
        "Liger repeated counterexample changed",
    )
    require(
        "peak_delta_after_logits=2.188 GiB" in followup_output["slime-pr-2011"]
        and "peak_delta_after_logits=1.000 GiB" in followup_output["slime-pr-2011"],
        "slime base/head memory evidence changed",
    )
    require(
        '"mode": "baseline"' in followup_output["verl-pr-6593"]
        and followup_output["verl-pr-6593"].count('"mode": "chunked"') == 6,
        "verl top-K memory sweep changed",
    )
    require(
        'tracked_after_destroy": 0' in followup_output["megatron-pr-7029"],
        "Megatron teardown evidence changed",
    )
    require(
        "10 passed" in followup_output["vllm-pr-54960"]
        and "1 passed" in followup_output["vllm-pr-54960"],
        "vLLM metrics evidence changed",
    )
    require(
        "1 passed" in followup_output["vllm-pr-44583"]
        and "6 passed" in followup_output["vllm-pr-44583"],
        "vLLM NIXL evidence changed",
    )
    require("R15_VERL_HCCL_ASYNC=" in rerun_output["verl-pr-6569"], "verl HCCL rerun changed")
    require('unique_endpoint_count": 32' in rerun_output["vllm-pr-44495"], "vLLM ZMQ rerun changed")

    execution_bindings: dict[str, list[dict[str, Any]]] = {
        case_id: [record_binding(evidence_paths["initial"], evidence["initial"], case_id)]
        for case_id in selected
    }
    for record in evidence["followup"]["records"]:
        execution_bindings[record["case_id"]].append(
            record_binding(evidence_paths["followup"], evidence["followup"], record["case_id"])
        )
    for record in evidence["rerun"]["records"]:
        execution_bindings[record["case_id"]].append(
            record_binding(evidence_paths["rerun"], evidence["rerun"], record["case_id"])
        )

    expected_success = {
        "sglang-pr-37523",
        "torchtitan-pr-3499",
        "torchtitan-pr-4399",
        "verl-pr-6566",
        "sglang-pr-27150",
        "liger-pr-1405",
        "megatron-pr-7029",
        "megatron-pr-5146",
        "slime-pr-2011",
        "verl-pr-6593",
        "slime-pr-2304",
        "verl-pr-7631",
        "verl-pr-6569",
        "vllm-pr-44495",
        "vllm-pr-54960",
        "vllm-pr-44583",
        "flashinfer-pr-4795",
        "torchtitan-pr-3430",
        "verl-pr-6507",
        "vllm-pr-44577",
        "megatron-pr-5144",
    }
    for case_id in expected_success:
        require(
            any(item["returncode"] == 0 for item in execution_bindings[case_id]),
            f"{case_id}: missing successful evidence record",
        )

    common_evidence = {
        name: binding(evidence_paths[name], evidence[name]) for name in evidence_paths
    }
    frozen_at = datetime.now(UTC).isoformat()
    locks: list[dict[str, Any]] = []
    for case_id, selected_case in selected.items():
        planned_case = planned[case_id]
        require(
            selected_case["base_sha"] == planned_case["base_sha"]
            and selected_case["head_sha"] == planned_case["head_sha"],
            f"{case_id}: selection/test-plan SHA mismatch",
        )
        assessed = ASSESSMENTS[case_id]
        result = classify_case_contract(assessed.triage)
        created_at = datetime.fromisoformat(selected_case["created_at"].replace("Z", "+00:00"))
        if result.decision == "check":
            require(created_at >= PROSPECTIVE_CUTOFF, f"{case_id}: mature case cannot be check")
            require(
                len(selected_case["paths"]) <= 4, f"{case_id}: check exceeds four changed paths"
            )
        legacy = "accept_with_scope" if assessed.triage.contract_satisfied else "check"
        supplemental = execution_bindings[case_id]
        lock_material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256(
                {"selection": selected_case, "test_plan": planned_case}
            ),
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
            "prospective_check_eligible": created_at >= PROSPECTIVE_CUTOFF,
            "legacy_r10_style_decision": legacy,
            "frozen_at": frozen_at,
        }
        locks.append({"material": lock_material, "lock_sha256": canonical_sha256(lock_material)})

    decision_counts = {
        decision: sum(lock["material"]["decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check", "reject", "unresolved")
    }
    legacy_counts = {
        decision: sum(lock["material"]["legacy_r10_style_decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check")
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
        "selection_lock_file_sha256": "sha256:" + EXPECTED_SELECTION_FILE_SHA256,
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "test_plan_file_sha256": "sha256:" + EXPECTED_TEST_PLAN_FILE_SHA256,
        "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "candidate_body_integrity_note": "Four SGLang pr-states blocks were deterministically removed before any raw body was stored; only raw hashes and sanitized bodies entered evidence.",
        "common_evidence_bindings": common_evidence,
        "frozen_at": frozen_at,
        "decision_counts": decision_counts,
        "legacy_r10_style_decision_counts": legacy_counts,
        "locks": locks,
    }
    output = {**output_material, "lock_set_sha256": canonical_sha256(output_material)}
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
