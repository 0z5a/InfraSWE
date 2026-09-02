#!/usr/bin/env python3
"""Bind outcome-free static evidence for the six cross-project r6 cases."""

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


def _source(repo: Path, revision: str, path: str) -> str:
    return _git(repo, "show", f"{revision}:{path}")


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


def _target_facts(
    project: str, base_sources: dict[str, str], head_sources: dict[str, str]
) -> dict[str, Any]:
    base = "\n".join(base_sources.values())
    head = "\n".join(head_sources.values())
    if project == "cutlass-cute":
        return {
            "base_final_tile_guard_count": base.count("if (k_tile_count == 1)"),
            "head_final_tile_guard_count": head.count("if (k_tile_count == 1)"),
            "changed_collective_is_sm90_fp8_blockwise": all(
                token in next(iter(head_sources)) for token in ("sm90", "fp8", "blockwise_scaling")
            ),
        }
    if project == "liger-kernel":
        return {
            "base_program_id_int64_cast_count": base.count("tl.program_id(0).to(tl.int64)"),
            "head_program_id_int64_cast_count": head.count("tl.program_id(0).to(tl.int64)"),
            "head_rms_norm_int64_cast_count": head_sources[
                "src/liger_kernel/ops/rms_norm.py"
            ].count("tl.program_id(0).to(tl.int64)"),
            "head_rope_int64_cast_count": head_sources["src/liger_kernel/ops/rope.py"].count(
                "tl.program_id(0).to(tl.int64)"
            ),
        }
    if project == "deepgemm":
        return {
            "base_uses_tensor_isinstance": "isinstance(value, torch.Tensor)" in base,
            "head_uses_tensor_isinstance": "isinstance(value, torch.Tensor)" in head,
            "head_uses_data_ptr_capability": "hasattr(value, 'data_ptr')" in head,
            "head_uses_cuda_stream_capability": "hasattr(value, 'cuda_stream')" in head,
            "head_has_explicit_float16_pointer_branch": "value.dtype == torch.float16" in head,
            "head_unknown_tensor_dtype_returns_void_pointer": (
                "else:\n            return ctypes.c_void_p(value.data_ptr())" in head
            ),
        }
    if project == "megatron-core":
        return {
            "base_batch_size_constexpr_count": sum(
                base.count(name)
                for name in (
                    "INPUT_BATCH_SIZE: tl.constexpr",
                    "OUTPUT_BATCH_SIZE: tl.constexpr",
                    "TENSOR_B_BATCH_SIZE: tl.constexpr",
                )
            ),
            "head_batch_size_constexpr_count": sum(
                head.count(name)
                for name in (
                    "INPUT_BATCH_SIZE: tl.constexpr",
                    "OUTPUT_BATCH_SIZE: tl.constexpr",
                    "TENSOR_B_BATCH_SIZE: tl.constexpr",
                )
            ),
            "row_and_block_sizes_remain_constexpr": all(
                token in head
                for token in (
                    "ROW_SIZE: tl.constexpr",
                    "BLOCK_SIZE: tl.constexpr",
                )
            ),
        }
    if project == "torchtitan":
        return {
            "head_router_kernel_count": head.count("def _apply_router_scores_"),
            "head_router_autotune_count": head.count("@triton.autotune("),
            "head_has_custom_autograd": "class _ApplyRouterScoresFunction" in head,
            "head_output_cast_is_hardcoded_bfloat16": "acc.to(tl.bfloat16)" in head,
            "head_inv_perm_dtype_is_hardcoded_int32": (
                "dtype=torch.int32" in head and "inv_perm" in head
            ),
            "head_score_gradient_accumulates_float32": (
                "tl.zeros([BLOCK_D], dtype=tl.float32)" in head
            ),
        }
    if project == "verl":
        return {
            "base_temperature_keyword_call_count": base.count("temperature=temperature"),
            "head_temperature_keyword_call_count": head.count("temperature=temperature"),
            "head_fused_temperature_gate_count": head.count(
                "if self.use_fused_kernels:\n"
                '            extra_forward_kwargs["temperature"] = temperature'
            ),
            "head_extra_kwargs_call_count": head.count("**extra_forward_kwargs"),
        }
    raise ValueError(f"unsupported project {project}")


def _existing_test_matches(repo: Path, revision: str, project: str) -> list[str]:
    queries = {
        "cutlass-cute": (
            "K.?128|128.*K|256x512x128",
            [
                "test/**",
                "examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/**",
            ],
        ),
        "liger-kernel": ("214748|2\\*\\*31|large.offset|long.context", ["test/**", "tests/**"]),
        "deepgemm": ("cuda.?graph|map_ctype|capture|replay", ["test/**", "tests/**"]),
        "megatron-core": ("tensor_get_slice_after|tensor_merge", ["tests/**"]),
        "torchtitan": ("apply_router_scores|inv_perm", ["tests/**"]),
        "verl": ("use_fused_kernels.*temperature|temperature.*use_fused_kernels", ["tests/**"]),
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
            repo,
            "diff",
            "--name-only",
            case["base_sha"],
            case["head_sha"],
        ).splitlines()
        expected_paths = case["paths"]
        if actual_paths != expected_paths:
            raise ValueError(
                f"path parity failed for {case_id}: {actual_paths!r} != {expected_paths!r}"
            )
        base_sources = {path: _source(repo, case["base_sha"], path) for path in actual_paths}
        head_sources = {path: _source(repo, case["head_sha"], path) for path in actual_paths}
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
            path for path in actual_paths if path.startswith(("test/", "tests/")) or "/test" in path
        ]
        material = {
            "schema_version": "0.5",
            "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
            "probe": "r6-cross-project-static-v1",
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
                "base_source_sha256": {
                    path: canonical_sha256(source) for path, source in base_sources.items()
                },
                "head_source_sha256": {
                    path: canonical_sha256(source) for path, source in head_sources.items()
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
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "r6-cross-project-static-summary-v1",
        "selection_lock_sha256": selection_sha,
        "test_plan_sha256": plan_sha,
        "case_count": len(output_digests),
        "outputs": output_digests,
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    summary = {
        **summary_material,
        "evidence_sha256": canonical_sha256(summary_material),
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
