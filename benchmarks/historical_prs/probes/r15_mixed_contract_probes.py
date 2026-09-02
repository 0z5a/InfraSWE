#!/usr/bin/env python3
# ruff: noqa: E501
"""Extract outcome-free, case-specific facts for the mixed R15 cohort."""

from __future__ import annotations

import argparse
import ast
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _text(record: dict[str, Any] | None) -> str:
    if not record or not record.get("available"):
        return ""
    return base64.b64decode(record["content_base64"]).decode("utf-8")


def _all_source(case: dict[str, Any], side: str = "head") -> str:
    return "\n".join(
        _text(file.get(side))
        for file in case["files"]
        if not file["head_path"].lower().startswith(("test/", "tests/"))
        and "/test/" not in file["head_path"].lower()
        and "/tests/" not in file["head_path"].lower()
    )


def _all_tests(case: dict[str, Any]) -> str:
    return "\n".join(
        _text(file.get("head"))
        for file in case["files"]
        if file["head_path"].lower().startswith(("test/", "tests/"))
        or "/test/" in file["head_path"].lower()
        or "/tests/" in file["head_path"].lower()
    )


def _patches(case: dict[str, Any]) -> str:
    return "\n".join(str(file.get("patch") or "") for file in case["files"])


def _ordered(source: str, first: str, second: str) -> bool:
    left = source.find(first)
    right = source.find(second, left + len(first))
    return left >= 0 and right > left


def _facts(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    source = _all_source(case)
    tests = _all_tests(case)
    patch = _patches(case)
    body = case["body_projection"].get("body") or ""
    facts: dict[str, Any] = {
        "changed_path_count": len(case["files"]),
        "changed_test_path_count": sum(
            file["head_path"].lower().startswith(("test/", "tests/"))
            or "/test/" in file["head_path"].lower()
            or "/tests/" in file["head_path"].lower()
            for file in case["files"]
        ),
        "patch_sha256": canonical_sha256(patch),
        "sanitized_body_sha256": canonical_sha256(body),
        "body_mentions_test": "test" in body.lower(),
    }

    if case_id == "flashinfer-pr-4795":
        facts |= {
            "persistent_handle_update_present": "def update(" in source,
            "capture_stream_switch_present": "is_current_stream_capturing" in source,
            "candidate_empty_and_uneven_tests_present": "empty" in tests and "uneven" in tests,
        }
    elif case_id == "flashinfer-pr-3304":
        facts |= {
            "negative_zero_uses_bitwise_test": (
                "__float_as_uint(val) == kNEGZERO_FP32" in source
                and "kNEGZERO_FP32" in source
            ),
            "subnormal_sentinel_test_present": "subnormal" in tests.lower(),
        }
    elif case_id == "megatron-pr-7029":
        facts |= {
            "created_process_groups_are_tracked": "_CREATED_PROCESS_GROUPS" in source,
            "tracked_groups_are_destroyed": "destroy_process_group" in source,
            "communicator_cache_invalidation_present": "invalidate" in source.lower() or "cache" in patch.lower(),
        }
    elif case_id == "megatron-pr-5153":
        facts |= {
            "deepep_v2_dispatcher_added": "deepep" in source.lower() and "v2" in source.lower(),
            "flex_dispatcher_surface_spans_multiple_modules": len(case["files"]) == 5,
        }
    elif case_id == "megatron-pr-5135":
        facts |= {
            "latent_shared_expert_path_added": "latent" in source.lower() and "shared" in source.lower(),
            "candidate_test_requires_transformer_engine": "TransformerEngine" in tests,
        }
    elif case_id == "sglang-pr-37523":
        facts |= {
            "nccl_dispatcher_is_opt_in": "nccl" in source.lower() and "dispatcher" in source.lower(),
            "local_route_metadata_present": "metadata" in source.lower(),
            "candidate_empty_uneven_repeatability_matrix_present": all(
                term in tests.lower() for term in ("all_ranks_empty", "uneven", "repeatability")
            ),
        }
    elif case_id == "sglang-pr-27289":
        facts |= {
            "rocm_specific_path": "rocm" in body.lower() or "hip" in source.lower(),
            "candidate_test_path_present": bool(tests),
            "fp8_scale_copy_or_transpose_changed": "scale" in patch.lower() and "transpose" in patch.lower(),
        }
    elif case_id == "sglang-pr-27150":
        facts |= {
            "fused_shared_expert_slots_are_excluded": "num_fused_shared_experts" in patch,
            "recorder_and_weight_shape_tests_present": "topk_recorder" in tests and "get_moe_weights" in tests,
        }
    elif case_id == "sglang-pr-27211":
        facts |= {
            "fused_combine_feature_spans_seven_paths": len(case["files"]) == 7,
            "candidate_test_path_present": bool(tests),
            "deepep_and_cutedsl_paths_changed": "deepep" in patch.lower() and "cutedsl" in patch.lower(),
        }
    elif case_id == "torchtitan-pr-4399":
        facts |= {
            "non_loss_stage_sentinel_present": "sentinel" in tests.lower(),
            "valid_token_reduction_is_loss_stage_gated": "valid" in patch.lower() and "loss" in patch.lower(),
        }
    elif case_id == "torchtitan-pr-3447":
        facts |= {
            "full_dtensor_moe_change_is_cross_model": len(case["files"]) == 9,
            "candidate_unit_test_path_present": bool(tests),
        }
    elif case_id == "torchtitan-pr-3499":
        facts |= {
            "per_direction_communicators_present": "direction" in source.lower(),
            "ring_tests_require_three_or_more_gpus": ">= 3" in tests or "< 3" in tests,
            "torchcomms_matrix_present": "torchcomms" in tests.lower(),
        }
    elif case_id == "torchtitan-pr-3430":
        facts |= {
            "varlen_cp_and_full_dtensor_span_models": "varlen" in source.lower() and "full" in source.lower(),
            "candidate_partition_edge_matrix_present": "multi_doc" in tests and "world_size_4" in tests,
        }
    elif case_id == "verl-pr-7631":
        facts |= {
            "weight_send_precedes_param_offload": _ordered(source, "send_weights(", ".to(") or _ordered(source, "send_weights(", "offload"),
            "megatron_and_offload_guards_present": "megatron" in source.lower() and "is_param_offload_enabled" in source,
        }
    elif case_id == "verl-pr-6569":
        facts |= {
            "broadcast_is_created_as_asyncio_task": "loop.create_task(self._run())" in source,
            "blocking_collective_runs_in_executor": "run_in_executor" in source,
            "broadcast_thread_sets_device": "torch.npu.set_device(self.device)" in source,
            "wait_awaits_task": "await self._task" in source,
        }
    elif case_id == "verl-pr-6507":
        facts |= {
            "global_steps_propagates_across_engine_interfaces": source.count("global_steps") >= 8,
            "candidate_tests_cover_send_and_receive_callers": "send" in tests and "receive" in tests,
        }
    elif case_id == "vllm-pr-54960":
        facts |= {
            "metrics_flow_spans_worker_scheduler_and_loggers": all(
                term in patch.lower() for term in ("worker", "scheduler", "logger")
            ),
            "candidate_reset_disabled_and_aggregation_tests_present": "reset" in tests.lower() and "scheduler_ec_connector_stats" in tests,
        }
    elif case_id == "vllm-pr-44495":
        tree = ast.parse(source)
        facts |= {
            "remote_socket_binds_port_zero": ".bind(f\"tcp://{connect_ip}:0\")" in source,
            "remote_endpoint_read_from_last_endpoint": "zmq.LAST_ENDPOINT" in source,
            "get_open_port_symbol_absent": not any(
                isinstance(node, ast.Name) and node.id == "get_open_port" for node in ast.walk(tree)
            ),
        }
    elif case_id == "vllm-pr-44583":
        facts |= {
            "per_region_mla_map_present": "_region_is_mla" in source,
            "replicate_and_split_descriptor_branches_present": "_fa_desc_replicated" in source and "REPLICATE" in source and "SPLIT" in source,
            "mixed_fa_mla_handshake_test_present": "test_handshake_mixed_fa_mla_hetero_tp" in tests,
        }
    elif case_id == "vllm-pr-44577":
        facts |= {
            "contiguous_per_block_layout_present": "contiguous" in source.lower() and "block" in source.lower(),
            "candidate_layout_invariants_present": all(
                term in tests for term in ("test_all_layers_accounted_for", "test_offsets_within_one_block", "test_strided_views_are_independent")
            ),
        }
    elif case_id == "liger-pr-1405":
        facts |= {
            "new_cce_surface_spans_kernel_and_transformer_api": "cce" in source.lower() and len(case["files"]) == 8,
            "candidate_gradient_test_present": "test_cce_forward_backward" in tests,
        }
    elif case_id == "liger-pr-1219":
        facts |= {
            "swiglu_multiplier_chain_rule_present": "multiplier" in source,
            "grid_has_nonzero_floor": "max(1" in source,
            "body_reports_target_npu_tests": "npu" in body.lower() and "test" in body.lower(),
        }
    elif case_id == "megatron-pr-5146":
        facts |= {
            "reuse_guard_precedes_optimizer_scan": _ordered(source, "if not self.config.reuse_grad_buf_for_mxfp8_param_ag", "for optimizer in self.chained_optimizers"),
            "gate_reads_ddp_overlap_setting": "optimizer.ddp_config.overlap_param_gather" in source,
        }
    elif case_id == "megatron-pr-5144":
        facts |= {
            "te_cross_entropy_fusion_is_rejected": "cross_entropy_loss_fusion" in source and "transformer_engine" in source.lower(),
            "candidate_native_and_te_branch_tests_present": "native_cross_entropy" in tests and "te_cross_entropy" in tests,
        }
    elif case_id == "slime-pr-2304":
        facts |= {
            "phase_reporter_resets_and_reads_both_peaks": all(
                term in source for term in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved")
            ),
            "reporter_runs_in_finally": "finally:" in source,
            "actor_phases_are_instrumented": "actor_train" in source and "log_probs" in source,
        }
    elif case_id == "slime-pr-2011":
        facts |= {
            "chunked_entropy_and_logprob_path_present": "chunk" in source.lower() and "entropy" in source.lower(),
            "candidate_tp2_and_reference_tests_present": "tp2" in tests.lower() and "naive_reference" in tests,
            "candidate_reproducer_present": any(file["head_path"] == "tools/repro_1951.py" for file in case["files"]),
        }
    elif case_id == "torchtitan-pr-3525":
        facts |= {
            "title_scoped_path_is_four_gpu_demo": "4 GPU" in body or "4-GPU" in body,
            "candidate_test_path_present": bool(tests),
            "rl_stack_change_spans_seven_paths": len(case["files"]) == 7,
        }
    elif case_id == "torchtitan-pr-3522":
        facts |= {
            "view_replay_feature_is_config_gated": "view_replay" in source,
            "candidate_disabled_replay_test_present": "not_offloaded_without_replay" in tests,
        }
    elif case_id == "verl-pr-6566":
        facts |= {
            "optimizer_precision_branch_matrix_present": "bf16" in tests and "fp16" in tests and "fp32" in tests,
            "candidate_change_spans_config_and_worker_layers": len(case["files"]) == 6,
        }
    elif case_id == "verl-pr-6593":
        facts |= {
            "chunked_topk_uses_slice_assignment": "_chunked_topk_log_probs" in source and "chunk" in source.lower(),
            "candidate_gradient_and_chunk_invariance_tests_present": "gradient" in tests and "chunk_size_invariance" in tests,
        }
    else:
        raise KeyError(case_id)
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    if selection["selection_lock_sha256"] != canonical_sha256(selection["selection_material"]):
        raise SystemExit("R15 selection digest mismatch")
    plan = _read(args.test_plan)
    if plan["test_plan_sha256"] != canonical_sha256(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R15 test-plan digest mismatch")
    source = _read(args.source_bundle)
    if source["source_bundle_sha256"] != canonical_sha256(
        {key: value for key, value in source.items() if key != "source_bundle_sha256"}
    ):
        raise SystemExit("R15 source-bundle digest mismatch")
    if source["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R15 source/selection binding mismatch")
    if source["test_plan_sha256"] != plan["test_plan_sha256"]:
        raise SystemExit("R15 source/test-plan binding mismatch")
    if len(source["cases"]) != 30:
        raise SystemExit("R15 source bundle must contain 30 cases")

    material = {
        "schema_version": "0.1",
        "protocol_id": "r15-outcome-free-mixed-contract-probes-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": source["source_bundle_sha256"],
        "generated_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_requested": False,
        "cases": [
            {"case_id": case["case_id"], "facts": _facts(case)}
            for case in source["cases"]
        ],
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(material['cases'])}")
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
