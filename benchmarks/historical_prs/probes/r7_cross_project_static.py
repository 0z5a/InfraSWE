#!/usr/bin/env python3
"""Bind outcome-free static evidence for the six cross-project R7 cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

REPO_DIRS = {
    "cutlass-cute": "cutlass",
    "liger-kernel": "liger",
    "deepgemm": "deepgemm",
    "megatron-core": "megatron",
    "torchtitan": "torchtitan",
    "verl": "verl",
}


def _git(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with {completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _source(repo: Path, revision: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def _verify_locks(selection: dict[str, Any], plan: dict[str, Any]) -> tuple[str, str]:
    selection_sha = selection["selection_lock_sha256"]
    if canonical_sha256(selection["selection_material"]) != selection_sha:
        raise ValueError("selection lock digest mismatch")
    plan_material = dict(plan)
    plan_sha = plan_material.pop("test_plan_sha256")
    if canonical_sha256(plan_material) != plan_sha:
        raise ValueError("test-plan digest mismatch")
    if plan["selection_lock_sha256"] != selection_sha:
        raise ValueError("test plan is not bound to the selection lock")
    return selection_sha, plan_sha


def _joined(sources: dict[str, str | None]) -> str:
    return "\n".join(source for source in sources.values() if source is not None)


def _target_facts(
    project: str,
    base_sources: dict[str, str | None],
    head_sources: dict[str, str | None],
) -> dict[str, Any]:
    base = _joined(base_sources)
    head = _joined(head_sources)
    if project == "cutlass-cute":
        arithmetic = head_sources["include/cute/numeric/arithmetic_tuple.hpp"] or ""
        sparse = (
            head_sources["include/cutlass/gemm/collective/sm120_blockscaled_sparse_mma_tma.hpp"]
            or ""
        )
        return {
            "base_forms_value_comparison_after_rank_only": (
                "if constexpr (sizeof...(Ns) == sizeof...(Ms)) {\n"
                "    return bool_constant<((Ns == Ms) && ...)>{} && t.value() == u.value();"
            )
            in base,
            "head_guards_value_comparison_by_exact_basis": (
                "if constexpr (((Ns == Ms) && ...))" in arithmetic
                and "return false_type{};" in arithmetic
            ),
            "head_metadata_k_assert_count": sparse.count(
                "TileShape_K must be a multiple of the metadata atom K extent"
            ),
            "base_metadata_k_assert_count": base.count(
                "TileShape_K must be a multiple of the metadata atom K extent"
            ),
        }
    if project == "liger-kernel":
        source_path = "src/liger_kernel/ops/fused_linear_cross_entropy.py"
        test_path = "test/transformers/test_fused_linear_cross_entropy.py"
        base_op = base_sources[source_path] or ""
        head_op = head_sources[source_path] or ""
        head_test = head_sources[test_path] or ""
        return {
            "base_compile_gate_count": base_op.count("not torch.compiler.is_compiling()"),
            "head_compile_gate_count": head_op.count("not torch.compiler.is_compiling()"),
            "head_dtype_out_fast_path_retained": (
                "out_dtype=torch.float32" in head_op and "out=grad_weight" in head_op
            ),
            "head_compile_fallback_uses_mm_float": (
                "grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()" in head_op
            ),
            "direct_compile_regression_test_count": head_test.count(
                "def test_torch_compile_fp32_accum_backward"
            ),
            "direct_test_uses_fullgraph": "fullgraph=True" in head_test,
        }
    if project == "deepgemm":
        bf16_path = "deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh"
        fp8_path = "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
        utils_path = "deep_gemm/include/deep_gemm/ptx/utils.cuh"
        bf16 = head_sources[bf16_path] or ""
        fp8 = head_sources[fp8_path] or ""
        utils = head_sources[utils_path] or ""
        return {
            "base_fence_call_count": base.count("fence_proxy_async_shared_cta()"),
            "head_bf16_fence_call_count": bf16.count("fence_proxy_async_shared_cta()"),
            "head_fp8_fp4_fence_call_count": fp8.count("fence_proxy_async_shared_cta()"),
            "head_helper_uses_exact_ptx": (
                'asm volatile("fence.proxy.async.shared::cta;" ::: "memory");' in utils
            ),
            "head_fences_precede_phase_flip": all(
                source.find("fence_proxy_async_shared_cta()")
                < source.find("combine_phase ^= load_stage_idx")
                for source in (bf16, fp8)
            ),
        }
    if project == "megatron-core":
        fsdp_path = "megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py"
        plan_path = "megatron/core/models/common/model_chunk_schedule_plan.py"
        combined_path = "megatron/core/pipeline_parallel/combined_1f1b.py"
        test_path = "tests/unit_tests/a2a_overlap/test_fsdp_1f1b_overlap.py"
        fsdp = head_sources[fsdp_path] or ""
        schedule = head_sources[plan_path] or ""
        combined = head_sources[combined_path] or ""
        test = head_sources[test_path] or ""
        return {
            "head_prepare_forward_method_count": fsdp.count("def prepare_forward_module("),
            "head_prepare_clears_only_pre_backward": (
                "if submodule._training_state == TrainingState.PRE_BACKWARD:" in fsdp
                and "submodule._training_state = TrainingState.IDLE" in fsdp
            ),
            "head_hook_is_wired_from_wrapper": (
                "forward_fsdp_wrapper.prepare_forward_module" in combined
            ),
            "head_same_underlying_layer_guard": (
                "same_underlying_layer" in schedule and "f_layer.layer is b_layer.layer" in schedule
            ),
            "direct_state_transition_test_count": test.count(
                "def test_fsdp_1f1b_genuine_forward_state"
            ),
            "direct_test_microbatch_count": (2 if "num_microbatches = 2" in test else None),
        }
    if project == "torchtitan":
        helper_path = "torchtitan/experiments/graph_trainer/hw_queues.py"
        test_path = "torchtitan/experiments/graph_trainer/tests/test_hw_queues.py"
        helper = head_sources[helper_path] or ""
        tests = head_sources[test_path] or ""
        return {
            "helper_is_new_file": base_sources[helper_path] is None,
            "main_unconditionally_prints_export": 'print(f"export GPU_MAX_HW_QUEUES={queues}")'
            in helper,
            "main_assigns_environment": 'os.environ["GPU_MAX_HW_QUEUES"] =' in helper,
            "main_detects_rocm": any(
                token in helper for token in ("torch.version.hip", "is_rocm", "ROCM_HOME")
            ),
            "existing_value_only_emits_note": (
                'if "GPU_MAX_HW_QUEUES" in os.environ:' in helper and "already set to" in helper
            ),
            "direct_test_method_count": tests.count("    def test_"),
            "direct_tests_cover_existing_override": "GPU_MAX_HW_QUEUES" in tests,
            "direct_tests_cover_backend_noop": any(
                token in tests.lower()
                for token in ("backend", "is_rocm", "torch.version.hip", "cpu path")
            ),
            "direct_tests_cover_main": any(
                token in tests for token in ("hw_queues.main", "from …hw_queues import main")
            ),
        }
    if project == "verl":
        main_path = "verl/trainer/main_ppo_v0.py"
        v1_path = "verl/trainer/ppo/v1/trainer_base.py"
        main = head_sources[main_path] or ""
        v1 = head_sources[v1_path] or ""
        return {
            "base_copy_to_local_call_count": base.count("copy_to_local("),
            "base_hf_tokenizer_call_count": base.count("hf_tokenizer("),
            "head_model_config_conversion_count": head.count("omega_conf_to_dataclass("),
            "head_hf_model_config_annotation_count": head.count("HFModelConfig ="),
            "head_tokenizer_property_count": head.count("model_config.tokenizer"),
            "head_processor_property_count": head.count("model_config.processor"),
            "main_entry_uses_model_config": (
                "omega_conf_to_dataclass(config.actor_rollout_ref.model)" in main
            ),
            "v1_entry_uses_model_config": (
                "omega_conf_to_dataclass(self.config.actor_rollout_ref.model)" in v1
            ),
        }
    raise ValueError(f"unsupported project {project}")


def _existing_test_matches(repo: Path, revision: str, project: str) -> list[str]:
    queries = {
        "cutlass-cute": (
            "ScaledBasis|TileShape_K.*metadata|metadata atom K extent",
            ["test/**"],
        ),
        "liger-kernel": (
            "torch_compile_fp32_accum_backward|dtype_out|torch.compile.*fused_linear",
            ["test/**", "tests/**"],
        ),
        "deepgemm": (
            "fence.proxy.async.shared|mega.?moe.*race|combine.*reuse",
            ["test/**", "tests/**"],
        ),
        "megatron-core": (
            "genuine_forward_state|prepare_forward_module|combined.?1f1b",
            ["tests/**"],
        ),
        "torchtitan": (
            "GPU_MAX_HW_QUEUES|_stream_lanes|recommend_gpu_max_hw_queues",
            ["tests/**", "torchtitan/**/tests/**"],
        ),
        "verl": (
            "main_ppo_v0|_init_tokenizer|HFModelConfig.*tokenizer|multimodal.*processor",
            ["tests/**"],
        ),
    }
    pattern, pathspecs = queries[project]
    output = _git(
        repo,
        "grep",
        "-n",
        "-i",
        "-E",
        pattern,
        revision,
        "--",
        *pathspecs,
        check=False,
    )
    return output.splitlines()[:200]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    plan = json.loads(args.test_plan.read_text(encoding="utf-8"))
    selection_sha, plan_sha = _verify_locks(selection, plan)
    plan_cases = {case["case_id"]: case for case in plan["cases"]}
    output_digests: dict[str, str] = {}

    for case in selection["selection_material"]["cases"]:
        case_started = time.perf_counter()
        case_id = case["case_id"]
        project = case["project"]
        repo = args.checkout_root / REPO_DIRS[project]
        if not (repo / ".git").exists():
            raise FileNotFoundError(f"missing checkout {repo}")
        if plan_cases[case_id]["base_sha"] != case["base_sha"]:
            raise ValueError(f"base SHA mismatch for {case_id}")
        if plan_cases[case_id]["head_sha"] != case["head_sha"]:
            raise ValueError(f"head SHA mismatch for {case_id}")

        actual_paths = _git(
            repo, "diff", "--name-only", case["base_sha"], case["head_sha"]
        ).splitlines()
        if actual_paths != case["paths"]:
            raise ValueError(
                f"path parity failed for {case_id}: {actual_paths!r} != {case['paths']!r}"
            )
        base_sources = {path: _source(repo, case["base_sha"], path) for path in actual_paths}
        head_sources = {path: _source(repo, case["head_sha"], path) for path in actual_paths}
        if any(source is None for source in head_sources.values()):
            raise ValueError(f"head source missing for {case_id}")
        unified_diff = _git(
            repo,
            "diff",
            "--find-renames",
            case["base_sha"],
            case["head_sha"],
            "--",
            *actual_paths,
        )
        changed_test_paths = [
            path
            for path in actual_paths
            if path.startswith(("test/", "tests/")) or "/tests/" in path
        ]
        material = {
            "schema_version": "0.5",
            "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
            "probe": "r7-cross-project-static-v1",
            "case_id": case_id,
            "project": project,
            "status": "pass",
            "failure_codes": [],
            "facts": {
                "path_parity": True,
                "changed_paths": actual_paths,
                "changed_test_paths": changed_test_paths,
                "changed_test_file_count": len(changed_test_paths),
                "existing_test_search_matches": _existing_test_matches(
                    repo, case["head_sha"], project
                ),
                "target": _target_facts(project, base_sources, head_sources),
                "compilation_path": "not-required",
                "steady_state_compile_seconds": 0.0,
            },
            "source_identity": {
                "base_sha": case["base_sha"],
                "head_sha": case["head_sha"],
                "base_source_exists": {
                    path: source is not None for path, source in base_sources.items()
                },
                "base_source_sha256": {
                    path: canonical_sha256(source)
                    for path, source in base_sources.items()
                    if source is not None
                },
                "head_source_sha256": {
                    path: canonical_sha256(source)
                    for path, source in head_sources.items()
                    if source is not None
                },
                "unified_diff_sha256": canonical_sha256(unified_diff),
                "selection_lock_sha256": selection_sha,
                "test_plan_sha256": plan_sha,
            },
            "duration_seconds": time.perf_counter() - case_started,
            "created_at": datetime.now(UTC).isoformat(),
        }
        payload = {**material, "evidence_sha256": canonical_sha256(material)}
        output_path = args.output_root / f"{case_id}.json"
        atomic_write_json(output_path, payload)
        output_digests[str(output_path)] = payload["evidence_sha256"]

    summary_material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "probe": "r7-cross-project-static-summary-v1",
        "selection_lock_sha256": selection_sha,
        "test_plan_sha256": plan_sha,
        "case_count": len(output_digests),
        "outputs": output_digests,
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    summary = {**summary_material, "evidence_sha256": canonical_sha256(summary_material)}
    atomic_write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
