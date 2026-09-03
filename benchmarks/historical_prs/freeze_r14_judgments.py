#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the 30-case R14 communication cohort."""

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

EXPECTED_SELECTION_FILE_SHA256 = "3adfd8fddcf1b86dcc4e8bd222af32180818d55e5c0e069ff7565272f84779f2"
EXPECTED_TEST_PLAN_FILE_SHA256 = "8069c032f6a774f05dabb5e21ff1f360783feaba3943020695c50f4f0187b828"
EXPECTED_SELECTION_SHA256 = (
    "sha256:bcbc9038eb7facae3cdd5ea1278927227908e552f7eaadfe9b97ace920dcce76"
)
EXPECTED_TEST_PLAN_SHA256 = (
    "sha256:8c1ea5cb3abf130dd41b7771097dc5c9f75c848be9f5ef523326619eb2a20077"
)
EXPECTED_RAW_SOURCE_BUNDLE_SHA256 = (
    "sha256:463ab53517f96a54b22ec32942316506525804a2ba3b8fb099db4ddcf5fe68f2"
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "sha256:1efdbfc09c8ba05fa183c1e6e702ef423b30f54b6e18cba94abe8c87630cdcbd"
)
CHECK_CUTOFF = datetime.fromisoformat("2026-08-03T00:00:00+00:00")
POLICY_ID = "communication-contract-disposition-split-v0.2-r14"


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
REJECT_REGRESSION = CaseContractTriageEvidence(
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
    "vllm-pr-54643": assessment(
        CHECK,
        "bounded-gap",
        "Both candidate scheduler regressions pass, and the head safely drops a pending load when the selected connector cannot load it.",
        "The frozen hybrid finish contract is broader than the two candidate cases: full-width save bookkeeping and repeated finalization remain unexecuted.",
        residual="Run the frozen hybrid/non-hybrid save-failure matrix and prove exactly-once completion bookkeeping.",
    ),
    "vllm-pr-50775": assessment(
        ACCEPT,
        "pass",
        "The exact candidate tests pass and the source checks every peer for the local rank instead of accepting after one successful pair.",
        "The all-peer failure condition is local, deterministic, and preserves the existing opt-out behavior.",
    ),
    "vllm-pr-50658": assessment(
        ACCEPT,
        "pass",
        "Nine candidate tests pass, and the two-rank oracle observes zero projection error before sequence-parallel all-gather.",
        "Capability gating covers the normal, CUDA-graph, and speculative decoding paths while reducing collective payload.",
    ),
    "vllm-pr-54619": assessment(
        ACCEPT,
        "pass",
        "Twelve candidate tests pass and the two-process flock probe distinguishes a live region from an orphan exactly.",
        "Atomic publication and fd ownership close the liveness race without deleting a live producer's region.",
    ),
    "vllm-pr-50754": assessment(
        REJECT_BROAD,
        "fail",
        "Ten candidate tests demonstrate the new poison notification for the single-group path.",
        "The source and tests explicitly leave hybrid, multi-group, and no-local-block requests to lease expiry, violating the frozen bounded-termination matrix.",
        residual="Move failure ownership above the group-specific shortcut and prove every legal request terminates without waiting for lease expiry.",
    ),
    "sglang-pr-37261": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "Two local argument tests pass and the source contains the intended DeepEP-v2 expanded-prefill feature gate.",
        "The title-scoped TP16/EP16 multi-node Hopper dispatch was not executable on two A100s; body-reported hardware results are not an evaluator-owned closure.",
        residual="Execute the frozen expanded/non-expanded prefill matrix on the declared DeepEP-v2 Hopper topology.",
    ),
    "sglang-pr-33029": assessment(
        ACCEPT,
        "pass",
        "Five tests plus five subtests pass, and the two-rank model reproduces base collective-cardinality divergence.",
        "The head performs collective bulk-progress reconciliation before request-local processing and completes on both ranks.",
    ),
    "sglang-pr-33220": assessment(
        ACCEPT,
        "pass",
        "Four candidate tests pass for the keyed lazy process-wide capture-stream registry.",
        "Stream creation remains runtime-context scoped and avoids per-runner duplicate capture allocations.",
    ),
    "sglang-pr-33228": assessment(
        ACCEPT,
        "pass",
        "Twelve tests plus eleven subtests pass across shared-expert accounting.",
        "The explicit expert count subtracts fused shared slots before DeepEP recorder cardinality is constructed.",
    ),
    "sglang-pr-33053": assessment(
        ACCEPT,
        "pass",
        "All four isolated candidate tests pass after excluding unrelated optional imports.",
        "The accelerator is bound before distributed-environment initialization while CPU and MPS remain excluded.",
    ),
    "flashinfer-pr-4302": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "Fourteen host-side contract tests pass, but twenty-one kernel paths skip because the candidate targets CUDA 13 and SM12x rather than the available SM80 GPUs.",
        "The candidate body reports target-hardware validation, but no evaluator-owned W4A16 expert-parallel numeric or replay result closes the contract.",
        residual="Run the exact head on SM12x and compare full-MoE output, rank partial sums, routing, and CUDA-graph replay.",
    ),
    "flashinfer-pr-4139": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The source disables asynchronous finish and requests the receive hook in the reported deadlock path.",
        "The only candidate test skips without a CUDA-13 build, so the two real-serving NIXL-EP progress failures remain evaluator-unreproduced.",
        residual="Execute both frozen serving deadlock schedules on the declared CUDA-13/B200 backend with bounded rank diagnostics.",
    ),
    "flashinfer-pr-4240": assessment(
        ACCEPT,
        "pass",
        "AST comparison proves all three Python modules are semantically unchanged when docstrings are ignored.",
        "The title is documentation-scoped, and the revised communicator and topology descriptions agree with the unchanged implementations.",
    ),
    "flashinfer-pr-4296": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The singleton runtime-extent branch is present, but all three candidate kernel cases skip because they require SM100 or SM103.",
        "No evaluator-owned numeric or launch-shape result demonstrates preservation of singleton expert TMA modes.",
        residual="Run the candidate reference comparison for singleton and non-singleton experts on SM100/SM103.",
    ),
    "flashinfer-pr-4174": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The one-line head disables PDL by default while retaining an environment override.",
        "There is no candidate regression and the title-scoped FP4/TP deadlock requires unavailable SM100 execution, so the primary progress claim is unproven.",
        residual="Reproduce the TP deadlock with PDL enabled and prove bounded completion with the candidate default on SM100.",
    ),
    "megatron-pr-6955": assessment(
        ACCEPT,
        "pass",
        "The candidate test passes and an exact process-group probe confirms eager communicator connection before the queue-count early exit.",
        "Connection is idempotent, so idle ranks participate without adding repeated initialization side effects.",
    ),
    "megatron-pr-6200": assessment(
        ACCEPT,
        "pass",
        "Five candidate tests pass and the two-rank numeric oracle has zero candidate and power-of-two prescale error.",
        "Although the body describes a post-sum scale argument absent from the primitive, pre-scaling is algebraically equivalent for the implemented power-of-two factor; GTP intentionally bypasses axis size two.",
    ),
    "megatron-pr-6963": assessment(
        ACCEPT,
        "pass",
        "All twelve candidate tests pass for storage identity, gapless offsets, and refusal after DDP layout loss.",
        "The grouping rule preserves fused expert views only when the underlying storage contract is provable.",
    ),
    "megatron-pr-7000": assessment(
        ACCEPT,
        "pass",
        "Nine candidate tests pass and the two-rank fixed-shape P2P exchange is exact.",
        "The optimization is gated by the packing scheduler, preserves dynamic exchange by default, and rejects unsupported dynamic context parallelism.",
    ),
    "megatron-pr-6973": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The optional outer-reduction stream and optimizer wait form the intended ownership chain in source.",
        "Candidate parity tests require at least four ranks and skip under the available two-rank torchrun, leaving overlap ordering and numeric parity unexecuted.",
        residual="Run the exact four-rank parity and stream-order matrix with the outer stream enabled and disabled.",
    ),
    "torchtitan-pr-3953": assessment(
        ACCEPT,
        "pass",
        "Three tests plus eight subtests pass, and the two-rank normalization oracle has zero error.",
        "The rewrite refuses multi-user chains and requires identical collective static arguments and gradient ancestry.",
    ),
    "torchtitan-pr-4051": assessment(
        ACCEPT,
        "pass",
        "The exact-head TP2 probe matches reference parameters and momentum on both ranks with two all-to-all calls per rank.",
        "The candidate's four-GPU 2x2 test is unavailable, but the reduced 1x2 mesh preserves both Shard(0) and Shard(1) matrix placements that the patch adds.",
    ),
    "torchtitan-pr-3955": assessment(
        ACCEPT,
        "pass",
        "Candidate tests pass and the exact two-rank torchrun repeats six passing overlap tests per rank.",
        "Separate dispatch/combine pools and graph-visible buffer dependencies preserve reuse ordering under eager EP overlap.",
    ),
    "torchtitan-pr-4018": assessment(
        ACCEPT,
        "pass",
        "Both candidate tests pass: arbitrary fake rank is forwarded and out-of-range rank fails closed.",
        "The change is localized to the fake process-group construction boundary.",
    ),
    "torchtitan-pr-3980": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "Three isolated candidate unit tests pass for context-parallel additional inputs and aligned pipeline target packing.",
        "The WIP patch spans pipeline and context parallel RL training, while the claimed eight-GPU end-to-end path is unavailable; closure is broader than a single-site repair.",
        residual="Run the frozen eight-GPU RL forward/backward pipeline and context-parallel matrix with loss and gradient parity.",
    ),
    "verl-pr-7591": assessment(
        ACCEPT,
        "pass",
        "IPC candidate paths pass, and all SHM overlap cases pass for multiple buckets, mixed dtypes, empty payloads, and callback exceptions.",
        "The exception path prebinds and releases shared-memory views before cleanup while early acknowledgement is delayed until receiver ownership is safe.",
    ),
    "verl-pr-7107": assessment(
        ACCEPT,
        "pass",
        "The two-rank boundary oracle is exact for zero, one, 1024, and 1025 elements in FP32 and BF16.",
        "Sender and receiver derive the same transmitted length and slice, eliminating bucket-padding over-broadcast.",
    ),
    "verl-pr-7045": assessment(
        REJECT_REGRESSION,
        "fail",
        "Four helper tests pass, but the production checkpoint engine imports run_group_init_with_timeout while the exact head defines only wait_for_group_init.",
        "A neutral optional-dependency shim reaches the dangling import immediately, so the candidate breaks production module import before its timeout behavior can run.",
        residual="Use one helper name at definition, import, and call sites, then add a production-module import and first-group timeout regression.",
    ),
    "verl-pr-7161": assessment(
        REJECT_BROAD,
        "bounded-gap",
        "The source moves MoE unfusing into the FSDP backend and retains a packed GPT-OSS exception.",
        "The rollout version gate is removed and conversion becomes backend-wide without a focused non-vLLM/new-vLLM regression; the supplied NPU test cannot run without its private model fixture.",
        residual="Restore capability/version gating and execute old-vLLM, new-vLLM, SGLang, and non-rollout FSDP conversion controls.",
    ),
    "verl-pr-7589": assessment(
        CHECK,
        "bounded-gap",
        "IPC and SHM happy paths pass across bucket counts, dtypes, and empty payloads, and the independent callback exception probe does not reproduce a failure.",
        "Unlike the paired #7591 head, this source does not prebind and drop shared-memory views before exception cleanup, leaving one reachable lifetime edge without a candidate-owned closure.",
        residual="Port the bounded exception-path ownership fix and pass the paired candidate-owned callback-exception regression.",
    ),
}


FACT_CHECKS: dict[str, tuple[tuple[str, Any], ...]] = {
    "vllm-pr-54643": (("head_drops_pending_load_when_can_load_is_false", True),),
    "vllm-pr-50775": (("iterates_every_peer_for_local_rank", True),),
    "vllm-pr-50658": (("projection_precedes_sequence_all_gather", True),),
    "vllm-pr-54619": (("uses_nonblocking_flock", True),),
    "vllm-pr-50754": (("explicitly_excludes_hybrid_or_multi_group_cleanup", True),),
    "sglang-pr-37261": (("multi_node_hybrid_mode_present", True),),
    "sglang-pr-33029": (("collective_progress_reduction_present", True),),
    "sglang-pr-33220": (("named_stream_factory_is_keyed_and_lazy", True),),
    "sglang-pr-33228": (("fused_shared_slots_subtracted", True),),
    "sglang-pr-33053": (("set_device_precedes_distributed_environment_init", True),),
    "flashinfer-pr-4302": (("implementation_targets_sm12x", True),),
    "flashinfer-pr-4139": (("async_finish_forced_false", True),),
    "flashinfer-pr-4240": (("python_semantics_unchanged_ignoring_docstrings", True),),
    "flashinfer-pr-4296": (("singleton_runtime_extent_fix_present", True),),
    "flashinfer-pr-4174": (("pdl_default_disabled", True),),
    "megatron-pr-6955": (("eager_connect_precedes_queue_count", True),),
    "megatron-pr-6200": (("primitive_has_scale_argument", False),),
    "megatron-pr-6963": (("storage_identity_grouping_present", True),),
    "megatron-pr-7000": (("fixed_shape_requires_packing_scheduler", True),),
    "megatron-pr-6973": (("candidate_parity_tests_require_at_least_four_ranks", True),),
    "torchtitan-pr-3953": (("rewrite_refuses_multi_user_chain", True),),
    "torchtitan-pr-4051": (("candidate_numeric_test_requires_four_gpus", True),),
    "torchtitan-pr-3955": (("two_buffer_sets_when_overlap_enabled", True),),
    "torchtitan-pr-4018": (("out_of_range_rank_rejected", True),),
    "torchtitan-pr-3980": (("body_reports_eight_gpu_end_to_end", True),),
    "verl-pr-7591": (("exception_path_prebinds_and_drops_shared_views_before_cleanup", True),),
    "verl-pr-7107": (("sender_broadcasts_exact_used_slice", True),),
    "verl-pr-7045": (("engine_imports_missing_run_group_init_symbol", True),),
    "verl-pr-7161": (("rollout_version_gate_removed", True),),
    "verl-pr-7589": (("exception_path_prebinds_and_drops_shared_views_before_cleanup", False),),
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
        "evidence_sha256": payload.get("evidence_sha256"),
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
        "R14 selection file digest mismatch",
    )
    require(
        file_sha256(args.test_plan) == EXPECTED_TEST_PLAN_FILE_SHA256,
        "R14 test-plan file digest mismatch",
    )
    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    require(
        selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]),
        "R14 embedded selection digest mismatch",
    )
    require(
        selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
        "R14 selection identity changed",
    )
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    require(
        plan["test_plan_sha256"] == canonical_sha256(plan_material),
        "R14 embedded test-plan digest mismatch",
    )
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R14 test-plan identity changed")
    require(
        plan["selection_lock_sha256"] == selection["selection_lock_sha256"],
        "R14 plan/selection binding mismatch",
    )
    require(
        plan["disposition_policy"]["weighted_score_used"] is False,
        "R14 unexpectedly uses weighted scoring",
    )
    require(
        plan["disposition_policy"]["forced_polarization_used"] is False,
        "R14 unexpectedly forces polarization",
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
    require(all(value is False for value in blind_flags), "R14 blind boundary is not intact")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 30, "R14 cohort is not exactly 30 cases")
    require(
        selected.keys() == planned.keys() == ASSESSMENTS.keys() == FACT_CHECKS.keys(),
        "R14 case sets differ",
    )

    evidence_paths = {
        name: args.result_root / filename
        for name, filename in {
            "sanitization": "candidate-body-sanitization.json",
            "static": "static-evidence.json",
            "contract": "contract-probes.json",
            "dual_gpu": "dual-gpu-communication.json",
            "vllm_initial": "upstream-test-matrix.json",
            "vllm_final": "upstream-test-vllm-isolated-rerun-v4.json",
            "vllm_50775": "upstream-test-vllm-isolated.json",
            "vllm_54619": "upstream-test-vllm-isolated-rerun-v2.json",
            "sglang_final": "upstream-test-sglang-isolated-rerun-v10.json",
            "sglang_33053": "upstream-test-sglang-33053-rerun-v16.json",
            "flashinfer_final": "upstream-test-flashinfer-isolated-rerun-v4.json",
            "flashinfer_4139": "upstream-test-flashinfer-rerun-v2.json",
            "followup": "upstream-followup-tests.json",
            "followup_v2": "upstream-followup-tests-v2.json",
            "followup_v3": "upstream-followup-tests-v3.json",
            "torchtitan_4051": "torchtitan-4051-tp2.json",
            "verl_7591_shm": "verl-7591-shm.json",
            "verl_7589_shm": "verl-7589-shm.json",
        }.items()
    }
    evidence = {name: read(path) for name, path in evidence_paths.items()}
    for name, payload in evidence.items():
        if "evidence_sha256" in payload:
            validate_digest(payload, "evidence_sha256", name)
        require(
            payload.get("outcome_review_ci_fields_requested") is False
            or name in {"sanitization", "static"},
            f"{name}: blind evidence flag changed",
        )

    sanitization = evidence["sanitization"]
    require(
        sanitization["sanitized_source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256,
        "R14 sanitized source identity changed",
    )
    require(
        sanitization["redacted_case_count"] == 5 and sanitization["redacted_block_count"] == 5,
        "R14 body sanitization scope changed",
    )
    for name in ("static", "contract", "dual_gpu"):
        payload = evidence[name]
        require(
            payload["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
            f"{name}: selection binding mismatch",
        )
        require(
            payload["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256,
            f"{name}: test-plan binding mismatch",
        )
        expected_source = (
            EXPECTED_RAW_SOURCE_BUNDLE_SHA256 if name == "static" else EXPECTED_SOURCE_BUNDLE_SHA256
        )
        require(
            payload["source_bundle_sha256"] == expected_source, f"{name}: source binding mismatch"
        )

    contract_cases = {row["case_id"]: row["facts"] for row in evidence["contract"]["cases"]}
    require(contract_cases.keys() == selected.keys(), "R14 contract-probe case set differs")
    for case_id, checks in FACT_CHECKS.items():
        for key, expected in checks:
            require(
                contract_cases[case_id][key] == expected, f"{case_id}: contract fact {key} changed"
            )

    dual = evidence["dual_gpu"]["facts"]
    require(
        dual["two_rank_nccl_smoke"]
        and dual["vllm_50658_projection_max_abs"] == 0.0
        and dual["sglang_33029_base_cardinality_mismatch_detected"]
        and dual["sglang_33029_head_bulk_completed"]
        and dual["megatron_6200_candidate_max_abs"] == 0.0
        and dual["megatron_7000_fixed_shape_p2p_exact"]
        and dual["torchtitan_3953_normalization_max_abs"] == 0.0
        and dual["verl_7107_all_boundaries_exact"],
        "R14 two-GPU facts changed",
    )
    require(
        evidence["torchtitan_4051"]["facts"]["all_ranks_match_reference"],
        "TorchTitan #4051 TP2 probe changed",
    )
    require(
        evidence["verl_7591_shm"]["all_passed"] and evidence["verl_7589_shm"]["all_passed"],
        "verl SHM probe changed",
    )

    execution_bindings: dict[str, list[dict[str, Any]]] = {
        "vllm-pr-54643": [
            record_binding(evidence_paths["vllm_final"], evidence["vllm_final"], "vllm-pr-54643")
        ],
        "vllm-pr-50775": [
            record_binding(evidence_paths["vllm_50775"], evidence["vllm_50775"], "vllm-pr-50775")
        ],
        "vllm-pr-50658": [
            record_binding(evidence_paths["vllm_final"], evidence["vllm_final"], "vllm-pr-50658")
        ],
        "vllm-pr-54619": [
            record_binding(evidence_paths["vllm_54619"], evidence["vllm_54619"], "vllm-pr-54619")
        ],
        "vllm-pr-50754": [
            record_binding(evidence_paths["vllm_final"], evidence["vllm_final"], "vllm-pr-50754")
        ],
    }
    for case_id in ("sglang-pr-37261", "sglang-pr-33029", "sglang-pr-33220", "sglang-pr-33228"):
        execution_bindings[case_id] = [
            record_binding(evidence_paths["sglang_final"], evidence["sglang_final"], case_id)
        ]
    execution_bindings["sglang-pr-33053"] = [
        record_binding(evidence_paths["sglang_33053"], evidence["sglang_33053"], "sglang-pr-33053")
    ]
    for case_id in ("flashinfer-pr-4302", "flashinfer-pr-4296"):
        execution_bindings[case_id] = [
            record_binding(
                evidence_paths["flashinfer_final"], evidence["flashinfer_final"], case_id
            )
        ]
    execution_bindings["flashinfer-pr-4139"] = [
        record_binding(
            evidence_paths["flashinfer_4139"], evidence["flashinfer_4139"], "flashinfer-pr-4139"
        )
    ]
    for case_id in (
        "megatron-pr-6955",
        "megatron-pr-6200",
        "megatron-pr-6963",
        "megatron-pr-7000",
        "megatron-pr-6973",
        "torchtitan-pr-3953",
        "torchtitan-pr-4051",
        "torchtitan-pr-3955",
        "torchtitan-pr-4018",
        "torchtitan-pr-3980",
        "verl-pr-7045",
        "verl-pr-7161",
    ):
        execution_bindings[case_id] = [
            record_binding(evidence_paths["vllm_initial"], evidence["vllm_initial"], case_id)
        ]
    for case_id in (
        "megatron-pr-6973",
        "torchtitan-pr-3955",
        "torchtitan-pr-4051",
        "verl-pr-7591",
        "verl-pr-7589",
        "verl-pr-7045",
    ):
        execution_bindings.setdefault(case_id, []).append(
            record_binding(evidence_paths["followup"], evidence["followup"], case_id)
        )
    execution_bindings["torchtitan-pr-3980"].append(
        record_binding(evidence_paths["followup_v3"], evidence["followup_v3"], "torchtitan-pr-3980")
    )
    for case_id in ("flashinfer-pr-4240", "flashinfer-pr-4174", "verl-pr-7107"):
        execution_bindings.setdefault(case_id, [])

    expected_zero = {
        "vllm-pr-54643",
        "vllm-pr-50775",
        "vllm-pr-50658",
        "vllm-pr-54619",
        "vllm-pr-50754",
        "sglang-pr-37261",
        "sglang-pr-33029",
        "sglang-pr-33220",
        "sglang-pr-33228",
        "sglang-pr-33053",
        "flashinfer-pr-4302",
        "flashinfer-pr-4139",
        "flashinfer-pr-4296",
        "megatron-pr-6955",
        "megatron-pr-6200",
        "megatron-pr-6963",
        "megatron-pr-7000",
        "megatron-pr-6973",
        "torchtitan-pr-3953",
        "torchtitan-pr-3955",
        "torchtitan-pr-4018",
        "torchtitan-pr-3980",
        "verl-pr-7591",
        "verl-pr-7589",
    }
    for case_id in expected_zero:
        require(
            any(item["returncode"] == 0 for item in execution_bindings[case_id]),
            f"{case_id}: missing successful or clean-skip execution record",
        )

    frozen_at = datetime.now(UTC).isoformat()
    common_evidence = {
        name: binding(evidence_paths[name], evidence[name])
        for name in (
            "sanitization",
            "static",
            "contract",
            "dual_gpu",
            "torchtitan_4051",
            "verl_7591_shm",
            "verl_7589_shm",
        )
    }
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
            require(created_at >= CHECK_CUTOFF, f"{case_id}: mature case cannot be check")
        legacy = "accept_with_scope" if assessed.triage.contract_satisfied else "check"
        supplemental = execution_bindings[case_id]
        if case_id == "torchtitan-pr-4051":
            supplemental.append(common_evidence["torchtitan_4051"])
        if case_id == "verl-pr-7591":
            supplemental.append(common_evidence["verl_7591_shm"])
        if case_id == "verl-pr-7589":
            supplemental.append(common_evidence["verl_7589_shm"])
        material = {
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
            "prospective_check_eligible": created_at >= CHECK_CUTOFF,
            "legacy_r10_style_decision": legacy,
            "frozen_at": frozen_at,
        }
        locks.append({"material": material, "lock_sha256": canonical_sha256(material)})

    decision_counts = {
        decision: sum(lock["material"]["decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check", "reject", "unresolved")
    }
    legacy_counts = {
        decision: sum(lock["material"]["legacy_r10_style_decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check")
    }
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
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "test_plan_file_sha256": "sha256:" + EXPECTED_TEST_PLAN_FILE_SHA256,
        "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "candidate_body_integrity_note": "Five dynamic SGLang pr-states CI blocks were deterministically redacted before judgment; one block was briefly exposed and was not used in any case assessment.",
        "common_evidence_bindings": common_evidence,
        "frozen_at": frozen_at,
        "decision_counts": decision_counts,
        "legacy_r10_style_decision_counts": legacy_counts,
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
