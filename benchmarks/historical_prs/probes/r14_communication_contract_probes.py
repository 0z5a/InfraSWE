#!/usr/bin/env python3
"""Extract outcome-free, case-specific R14 communication contract facts."""

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


def _source(case: dict[str, Any], suffix: str, side: str = "head") -> str:
    matches = [
        _text(file[side])
        for file in case["files"]
        if file["head_path"].endswith(suffix) and file.get(side)
    ]
    if len(matches) != 1:
        raise ValueError(f"{case['case_id']}: expected one {suffix} {side} source")
    return matches[0]


def _maybe_source(case: dict[str, Any], suffix: str, side: str = "head") -> str:
    matches = [
        _text(file[side])
        for file in case["files"]
        if file["head_path"].endswith(suffix) and file.get(side)
    ]
    return matches[0] if len(matches) == 1 else ""


def _patches(case: dict[str, Any]) -> str:
    return "\n".join(str(file.get("patch") or "") for file in case["files"])


class _WithoutDocstrings(ast.NodeTransformer):
    @staticmethod
    def _strip(node: Any) -> Any:
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip(node)


def _python_semantics_equal_ignoring_docstrings(base: str, head: str) -> bool:
    if not base or not head:
        return False
    transformer = _WithoutDocstrings()
    base_tree = transformer.visit(ast.parse(base))
    head_tree = transformer.visit(ast.parse(head))
    return ast.dump(base_tree, include_attributes=False) == ast.dump(
        head_tree, include_attributes=False
    )


def _ordered(source: str, first: str, second: str) -> bool:
    first_index = source.find(first)
    second_index = source.find(second, first_index + len(first))
    return first_index >= 0 and second_index > first_index


def _facts(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    patch = _patches(case)
    body = case["body_projection"].get("body") or ""
    common = {
        "sanitized_body_sha256": canonical_sha256(body),
        "changed_path_count": len(case["files"]),
        "changed_test_path_count": sum(
            "/test" in file["head_path"].lower() or file["head_path"].lower().startswith("test")
            for file in case["files"]
        ),
        "patch_sha256": canonical_sha256(patch),
    }

    if case_id == "vllm-pr-54643":
        scheduler = _source(case, "mooncake/store/scheduler.py")
        return common | {
            "body_claims_full_width_save_bookkeeping": "full-width" in body.lower(),
            "head_drops_pending_load_when_can_load_is_false": (
                "load_spec is None or not load_spec.can_load" in scheduler
                and "continue" in scheduler
            ),
            "head_contains_body_named_worker_save_guard": (
                "recorded" in scheduler.lower() and "save" in scheduler.lower()
            ),
        }
    if case_id == "vllm-pr-50775":
        source = _source(case, "device_communicators/custom_all_reduce.py")
        return common | {
            "iterates_every_peer_for_local_rank": "for i in range(world_size)" in source
            and "if i == rank" in source,
            "returns_false_on_any_failed_pair": (
                "if not torch.cuda.can_device_access_peer" in source
            ),
        }
    if case_id == "vllm-pr-50658":
        source = _source(case, "spec_decode/dspark/speculator.py")
        return common | {
            "projection_precedes_sequence_all_gather": _ordered(
                source, "self.model.combine_hidden_states", "_sequence_sharded_aux_gather"
            ),
            "aux_projection_is_capability_gated": "_sequence_sharded_aux_gather" in source
            and "is None" in source,
            "cuda_graph_and_speculator_paths_changed_together": (
                "cudagraph_utils.py" in " ".join(file["head_path"] for file in case["files"])
                and "dspark/speculator.py" in " ".join(file["head_path"] for file in case["files"])
            ),
        }
    if case_id == "vllm-pr-54619":
        source = _source(case, "utils/shm_utils.py")
        return common | {
            "uses_nonblocking_flock": "fcntl.LOCK_EX | fcntl.LOCK_NB" in source,
            "uses_atomic_link_publication": "os.link" in source,
            "returns_locked_fd_to_owner": "fcntl.LOCK_SH" in source and "return fd" in source,
            "candidate_includes_multiprocess_tests": "multiprocess" in patch.lower()
            or "process" in patch.lower(),
        }
    if case_id == "vllm-pr-50754":
        source = _source(case, "nixl/push_worker.py")
        return common | {
            "failure_notification_state_exists": "failed" in source.lower()
            and "notify" in source.lower(),
            "explicitly_excludes_hybrid_or_multi_group_cleanup": (
                "recovery is unsupported for hybrid or multi-group models" in source
            ),
            "pending_state_cleanup_present": "pending" in source.lower()
            and ("pop(" in source or "discard(" in source),
        }
    if case_id == "sglang-pr-37261":
        return common | {
            "prefill_expand_feature_gate_present": "PREFILL_DO_EXPAND" in patch,
            "multi_node_hybrid_mode_present": "nnodes > 1" in patch and "hybrid" in patch,
            "body_reports_tp16_ep16_hardware_validation": "tp16" in body.lower()
            and "ep16" in body.lower(),
        }
    if case_id == "sglang-pr-33029":
        source = _source(case, "scheduler.py")
        return common | {
            "bulk_progress_loop_precedes_request_processing": _ordered(
                source, "waiting_queue", "for req in"
            ),
            "collective_progress_reduction_present": "all_reduce" in source,
        }
    if case_id == "sglang-pr-33220":
        source = _source(case, "srt/runtime_context.py")
        parallel_state = _source(case, "srt/distributed/parallel_state.py")
        return common | {
            "capture_stream_uses_runtime_context_registry": (
                "get_stream(\n            _PROCESS_GRAPH_CAPTURE_STREAM" in parallel_state
            ),
            "named_stream_factory_is_keyed_and_lazy": "self.resources.streams.get(name)" in source,
        }
    if case_id == "sglang-pr-33228":
        return common | {
            "fused_shared_slots_subtracted": "num_fused_shared_experts" in patch,
            "explicit_expert_count_propagated": "num_experts" in patch,
            "candidate_recorder_tests_present": "recorder" in patch.lower(),
        }
    if case_id == "sglang-pr-33053":
        source = _source(case, "runtime/distributed/parallel_state.py")
        return common | {
            "set_device_precedes_distributed_environment_init": _ordered(
                source, ".set_device(local_rank)", "init_distributed_environment("
            ),
            "cpu_and_mps_are_excluded": "cpu" in source and "mps" in source,
        }
    if case_id == "flashinfer-pr-4302":
        return common | {
            "implementation_targets_sm12x": "sm_12" in patch.lower() or "SM12" in patch,
            "body_reports_rtx_pro_6000_validation": "rtx pro 6000" in body.lower(),
            "candidate_kernel_tests_present": "test_" in patch and "moe" in patch.lower(),
        }
    if case_id == "flashinfer-pr-4139":
        return common | {
            "async_finish_forced_false": "async_finish=False" in patch,
            "receive_hook_requested": "return_recv_hook=True" in patch,
            "body_reports_b200_validation": "B200" in body,
        }
    if case_id == "flashinfer-pr-4240":
        semantic_results = []
        for file in case["files"]:
            if not file["head_path"].endswith(".py"):
                continue
            semantic_results.append(
                {
                    "path": file["head_path"],
                    "equal_ignoring_docstrings": _python_semantics_equal_ignoring_docstrings(
                        _text(file["base"]), _text(file["head"])
                    ),
                }
            )
        return common | {
            "python_semantics_unchanged_ignoring_docstrings": all(
                item["equal_ignoring_docstrings"] for item in semantic_results
            ),
            "python_semantic_comparisons": semantic_results,
        }
    if case_id == "flashinfer-pr-4296":
        return common | {
            "singleton_runtime_extent_fix_present": "runtime" in patch.lower()
            and "expert" in patch.lower(),
            "body_reports_b200_validation": "B200" in body,
        }
    if case_id == "flashinfer-pr-4174":
        return common | {
            "pdl_default_disabled": "enable_pdl = False" in patch,
            "environment_override_retained": "os.environ" in patch,
        }
    if case_id == "megatron-pr-6955":
        source = _source(case, "nccl_copy_service.py")
        return common | {
            "eager_connect_precedes_queue_count": _ordered(
                source, "self._ensure_nccl_connected()", "total_ops ="
            ),
            "connect_is_idempotent": "if self._nccl_connected" in source,
        }
    if case_id == "megatron-pr-6200":
        primitive = _source(case, "core/distributed/reduce_scatter_with_fp32_accumulation.py")
        gtp = _source(case, "generalized_tensor_parallelism.py")
        return common | {
            "body_mentions_post_sum_scale_argument": "scale" in body and "FP32 sum" in body,
            "primitive_has_scale_argument": "scale:" in primitive or "scale=" in primitive,
            "gtp_prescales_before_collective": _ordered(
                gtp, "_prescale_wgrads_for_mean_rs(wgrads)", "_reduce_scatter_fp32_accum"
            ),
            "gtp_bypasses_axis_size_two": "self.group.size() > 2" in gtp,
            "power_of_two_equivalence_tests_present": "power_of_two_prescale" in patch,
        }
    if case_id == "megatron-pr-6963":
        layout = _source(case, "optimizer/param_layout.py")
        experts = _source(case, "experts.py")
        return common | {
            "storage_identity_grouping_present": "untyped_storage" in layout,
            "gapless_offset_check_present": "storage_offset" in layout,
            "expert_views_refused_after_ddp_layout_loss": "refusing to create detached" in experts,
        }
    if case_id == "megatron-pr-7000":
        config = _source(case, "megatron/core/model_parallel_config.py")
        p2p = _source(case, "p2p_communication.py")
        return common | {
            "fixed_shape_requires_packing_scheduler": "requires a sequence_packing_scheduler"
            in config,
            "dynamic_context_parallel_rejected": "not supported with dynamic_context_parallel"
            in config,
            "dynamic_exchange_retained_when_flag_off": "and not getattr" in p2p,
        }
    if case_id == "megatron-pr-6973":
        test_source = _source(case, "test_fully_shard.py")
        module = _source(case, "module.py")
        return common | {
            "candidate_parity_tests_require_at_least_four_ranks": "world_size < 4" in test_source,
            "optimizer_stream_wait_present": "wait_stream(self.dp_outer_reduce_stream)" in module,
            "dedicated_outer_stream_is_optional": "overlap_dp_outer_reduction" in patch,
        }
    if case_id == "torchtitan-pr-3953":
        source = _source(case, "grad_chain_pass.py")
        return common | {
            "rewrite_requires_matching_collective_static_args": "_same_static_call" in source,
            "rewrite_refuses_multi_user_chain": "set(node.users) != {user}" in source,
            "gradient_output_ancestry_guard_present": "_grad_output_ancestors" in source,
        }
    if case_id == "torchtitan-pr-4051":
        source = _source(case, "distributed_optimizers/muon.py")
        tests = _source(case, "test_distributed_muon.py")
        return common | {
            "named_2d_shard01_allowed": "set(storage_shard_dims) == {0, 1}" in source,
            "candidate_numeric_test_requires_four_gpus": "device_count() >= 4" in tests,
        }
    if case_id == "torchtitan-pr-3955":
        source = _source(case, "minimal_async_ep/api.py")
        return common | {
            "two_buffer_sets_when_overlap_enabled": "2 if overlap_enabled else 1" in source,
            "distinct_dispatch_and_combine_pools": '"dispatch"' in source and '"combine"' in source,
            "buffer_reuse_dependency_is_graph_visible": "buffer_reuse_dependency" in source,
        }
    if case_id == "torchtitan-pr-4018":
        source = _source(case, "distributed/utils.py")
        return common | {
            "rank_environment_is_forwarded": 'os.environ.get("RANK", "0")' in source
            and "rank=rank" in source,
            "out_of_range_rank_rejected": "not 0 <= rank < world_size" in source,
        }
    if case_id == "torchtitan-pr-3980":
        return common | {
            "body_reports_eight_gpu_end_to_end": "8-GPU" in body,
            "pipeline_targets_pack_four_aligned_fields": "_pack_pipeline_targets" in patch
            and "advantages" in patch,
            "context_parallel_additional_inputs_supported": "additional_sequence_input_keys"
            in patch,
        }
    if case_id in {"verl-pr-7591", "verl-pr-7589"}:
        source = _source(case, "vllm_rollout/bucketed_weight_transfer.py")
        return common | {
            "ipc_private_staging_before_early_ack": _ordered(
                source, "staging.copy_", "self.socket.send"
            ),
            "shm_device_copy_synchronized_before_early_ack": "early_ack and self.use_shm" in source
            and "synchronize()" in source,
            "oversized_raw_ipc_forces_late_ack": 'meta["handle"] is None' in source,
            "exception_path_prebinds_and_drops_shared_views_before_cleanup": (
                "weights: list[tuple[str, torch.Tensor]] = []" in source
                and "tensor = None" in source
                and "src = None" in source
                and _ordered(source, "finally:", "del weights, tensor, src")
                and _ordered(source, "del weights, tensor, src", "self._cleanup()")
            ),
        }
    if case_id == "verl-pr-7107":
        source = _source(case, "nccl_checkpoint_engine.py")
        return common | {
            "sender_broadcasts_exact_used_slice": "send_buf[:offset]" in source,
            "receiver_slices_from_transmitted_length": 'self.metadata["length"]' in source,
            "candidate_test_path_count": common["changed_test_path_count"],
        }
    if case_id == "verl-pr-7045":
        helper = _source(case, "_group_init.py")
        engine = _source(case, "nccl_checkpoint_engine.py")
        return common | {
            "helper_defines_wait_for_group_init": "def wait_for_group_init" in helper,
            "helper_defines_run_group_init_with_timeout": "def run_group_init_with_timeout"
            in helper,
            "engine_imports_missing_run_group_init_symbol": (
                "from verl.checkpoint_engine._group_init import run_group_init_with_timeout"
                in engine
                and "def run_group_init_with_timeout" not in helper
            ),
        }
    if case_id == "verl-pr-7161":
        fsdp = _source(case, "fsdp/utils.py")
        rollout = _source(case, "vllm_rollout.py")
        return common | {
            "fsdp_backend_now_owns_moe_unfuse": "def unfuse_moe_params" in fsdp,
            "rollout_version_gate_removed": "_should_expand_vllm_moe_params" not in rollout,
            "gpt_oss_packed_exception_present": 'model_type == "gpt_oss"' in fsdp,
        }
    raise KeyError(case_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("R14 selection digest mismatch")
    plan = _read(args.test_plan)
    if plan["test_plan_sha256"] != canonical_sha256(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R14 test-plan digest mismatch")
    source = _read(args.source_bundle)
    if source["source_bundle_sha256"] != canonical_sha256(
        {key: value for key, value in source.items() if key != "source_bundle_sha256"}
    ):
        raise SystemExit("R14 source-bundle digest mismatch")
    if source["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R14 source/selection binding mismatch")
    if source["test_plan_sha256"] != plan["test_plan_sha256"]:
        raise SystemExit("R14 source/test-plan binding mismatch")
    cases = source["cases"]
    if len(cases) != 30:
        raise SystemExit("R14 source bundle must contain exactly 30 cases")

    output_material = {
        "schema_version": "0.1",
        "protocol_id": "r14-outcome-free-communication-contract-probes-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": source["source_bundle_sha256"],
        "generated_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_requested": False,
        "cases": [{"case_id": case["case_id"], "facts": _facts(case)} for case in cases],
    }
    atomic_write_json(
        args.output,
        {**output_material, "evidence_sha256": canonical_sha256(output_material)},
    )
    print(f"case_count={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
