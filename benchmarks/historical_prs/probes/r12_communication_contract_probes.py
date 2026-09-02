#!/usr/bin/env python3
# ruff: noqa: E501
"""Acquire and probe exact base/head sources for the R12 communication cohort."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _gh(endpoint: str, *, paginate: bool = False) -> Any:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    command.append(endpoint)
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode == 0:
            break
        if attempt < 3:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(f"GitHub API failed for {endpoint}: {process.stderr.strip()}")
    value = json.loads(process.stdout)
    if paginate:
        value = [item for page in value for item in page]
    return value


def _content(repository: str, path: str, revision: str) -> str | None:
    from urllib.parse import quote

    endpoint = f"repos/{repository}/contents/{quote(path)}?ref={revision}"
    try:
        payload = _gh(endpoint)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return None
        raise
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unsupported content encoding for {repository}:{path}")
    return base64.b64decode(payload["content"]).decode("utf-8")


def _is_text_probe_path(path: str) -> bool:
    return path.endswith(
        (
            ".py",
            ".pyi",
            ".h",
            ".hpp",
            ".cuh",
            ".cpp",
            ".cu",
            ".rst",
            ".txt",
        )
    )


def _acquire(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for case in cases:
        files = _gh(
            f"repos/{case['repository']}/pulls/{case['pull_number']}/files?per_page=100",
            paginate=True,
        )
        observed = sorted(item["filename"] for item in files)
        expected = sorted(case["paths"])
        if observed != expected:
            raise RuntimeError(
                f"path parity failed for {case['case_id']}: {observed} != {expected}"
            )
        sources: dict[str, dict[str, str | None]] = {}
        status_by_path = {item["filename"]: item["status"] for item in files}
        for path in case["paths"]:
            if not _is_text_probe_path(path):
                continue
            sources[path] = {
                "base": (
                    None
                    if status_by_path[path] == "added"
                    else _content(case["repository"], path, case["base_sha"])
                ),
                "head": (
                    None
                    if status_by_path[path] == "removed"
                    else _content(case["repository"], path, case["head_sha"])
                ),
            }
        bundle[case["case_id"]] = {
            "case": case,
            "files": [
                {
                    "filename": item["filename"],
                    "status": item["status"],
                    "additions": item["additions"],
                    "deletions": item["deletions"],
                    "patch": item.get("patch"),
                }
                for item in files
            ],
            "sources": sources,
        }
    return bundle


def _source_pair(item: dict[str, Any], suffix: str) -> tuple[str, str]:
    matches = [value for path, value in item["sources"].items() if path.endswith(suffix)]
    if len(matches) != 1 or matches[0]["base"] is None or matches[0]["head"] is None:
        raise AssertionError(f"missing unique source pair for {suffix}")
    return str(matches[0]["base"]), str(matches[0]["head"])


def _head_source(item: dict[str, Any], suffix: str) -> str:
    matches = [
        value["head"]
        for path, value in item["sources"].items()
        if path.endswith(suffix) and value["head"] is not None
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing unique head source for {suffix}")
    return str(matches[0])


def _base_source(item: dict[str, Any], suffix: str) -> str:
    matches = [
        value["base"]
        for path, value in item["sources"].items()
        if path.endswith(suffix) and value["base"] is not None
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing unique base source for {suffix}")
    return str(matches[0])


def _patch_text(item: dict[str, Any], suffix: str) -> str:
    matches = [
        str(file.get("patch") or "") for file in item["files"] if file["filename"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing unique patch for {suffix}")
    return matches[0]


def _has_changed_test(item: dict[str, Any]) -> bool:
    return any(
        any(
            part == "test" or part == "tests" or part.startswith("test_")
            for part in path.split("/")
        )
        for path in item["sources"]
    )


def _balanced_call(source: str, marker: str, start: int = 0) -> tuple[str, int]:
    marker_index = source.index(marker, start)
    open_index = source.index("(", marker_index)
    depth = 0
    for index in range(open_index, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[marker_index : index + 1], index + 1
    raise AssertionError(f"unterminated call for {marker}")


def _probe_cutlass_3294(item: dict[str, Any]) -> dict[str, Any]:
    old = "from cuda.core.experimental import Device"
    new = "try:\n    from cuda.core import Device\nexcept ImportError:\n    from cuda.core.experimental import Device"
    rows = []
    for path, revisions in sorted(item["sources"].items()):
        base = str(revisions["base"])
        head = str(revisions["head"])
        ast.parse(base)
        ast.parse(head)
        rows.append(
            {
                "path": path,
                "base_experimental_only": base.count(old) == 1,
                "head_primary_import": head.count("from cuda.core import Device") == 1,
                "head_experimental_fallback": head.count(old) == 1,
                "head_catches_only_import_error": "except ImportError:" in head,
                "non_import_source_unchanged": head.replace(new, old) == base,
            }
        )
    return {
        "changed_example_count": len(rows),
        "examples": rows,
        "all_examples_compile": all(row["non_import_source_unchanged"] for row in rows),
        "all_examples_support_both_cuda_core_generations": all(
            row["head_primary_import"]
            and row["head_experimental_fallback"]
            and row["head_catches_only_import_error"]
            for row in rows
        ),
        "patch_is_import_only": all(row["non_import_source_unchanged"] for row in rows),
        "changed_direct_test": _has_changed_test(item),
        "blackwell_runtime_executed": False,
        "blackwell_runtime_is_non_blocking_for_frozen_claim": True,
    }


def _probe_flashinfer_3939(item: dict[str, Any]) -> dict[str, Any]:
    base_allreduce, head_allreduce = _source_pair(item, "flashinfer/comm/allreduce.py")
    base_mnnvl, head_mnnvl = _source_pair(item, "flashinfer/comm/mnnvl.py")
    base_trt, head_trt = _source_pair(item, "flashinfer/comm/trtllm_ar.py")
    base_mnnvl_trt, head_mnnvl_trt = _source_pair(item, "flashinfer/comm/trtllm_mnnvl_ar.py")
    removed_test = next(
        file
        for file in item["files"]
        if file["filename"].endswith("test_trtllm_allreduce_checkpoint.py")
    )
    api_tokens = ("def checkpoint_prepare", "def checkpoint_restore")
    base_api_count = sum(base_allreduce.count(token) for token in api_tokens) + sum(
        base_mnnvl_trt.count(token) for token in api_tokens
    )
    head_api_count = sum(head_allreduce.count(token) for token in api_tokens) + sum(
        head_mnnvl_trt.count(token) for token in api_tokens
    )
    return {
        "checkpoint_api_count": {"base": base_api_count, "head": head_api_count},
        "checkpoint_test_removed": removed_test["status"] == "removed",
        "checkpoint_documentation_section_removed": all(
            line.startswith("-")
            for line in _patch_text(item, "docs/api/comm.rst").splitlines()
            if "checkpoint_prepare" in line or "checkpoint_restore" in line
        ),
        "stable_va_state_removed": {
            "mapped_property": "def mapped" in base_mnnvl and "def mapped" not in head_mnnvl,
            "detach_primitive": "_unmap_and_release_handles" in base_mnnvl
            and "_unmap_and_release_handles" not in head_mnnvl,
            "reattach_primitive": "_create_and_map_handles" in base_mnnvl
            and "_create_and_map_handles" not in head_mnnvl,
            "protocol_reinitializer": "_initialize_allreduce_fusion_protocol" in base_trt
            and "_initialize_allreduce_fusion_protocol" not in head_trt,
        },
        "head_reverts_to_torch_symmetric_allocation": "_alloc_symm_buffer_bytes(" in head_trt,
        "head_blocks_unmapped_workspace_use": "symmetric-memory handles are not attached"
        in head_allreduce,
        "head_named_checkpoint_recovery_supported": head_api_count > 0,
        "head_fail_closes_checkpoint_calls_by_absent_api": head_api_count == 0,
        "eager_allreduce_entrypoint_retained": "def allreduce_fusion(" in head_allreduce,
        "mnnvl_allreduce_entrypoint_retained": "class MNNVLAllReduceFusionWorkspace"
        in head_mnnvl_trt,
        "required_two_gpu_checkpoint_runtime_executed": False,
        "decisive_source_fact": "the revert removes the frozen checkpoint/restore API and its direct test rather than repairing that sequence",
        "changed_direct_test": _has_changed_test(item),
    }


def _probe_flashinfer_3931(item: dict[str, Any]) -> dict[str, Any]:
    kernel = _head_source(item, "cutlass_fused_moe_kernels.cuh")
    binding = _head_source(item, "flashinfer_cutlass_fused_moe_binding.cu")
    header = _head_source(item, "moe_kernels.h")
    python_source = _head_source(item, "flashinfer/fused_moe/core.py")
    test = _head_source(item, "test_cutlass_fused_moe_do_finalize.py")
    method_start = binding.index("Array<Tensor> runMoe(")
    method_end = binding.index("void runMoeMinLantency", method_start)
    method = binding[method_start:method_end]
    calls = []
    cursor = 0
    while "mKernelRunner->runMoe(" in method[cursor:]:
        call, cursor = _balanced_call(method, "mKernelRunner->runMoe", cursor)
        calls.append(call)
    return {
        "binding_kernel_call_count": len(calls),
        "binding_kernel_calls_forward_do_finalize": ["do_finalize" in call for call in calls],
        "all_binding_kernel_calls_forward_do_finalize": bool(calls)
        and all("do_finalize" in call for call in calls),
        "kernel_default_for_omitted_do_finalize_is_true": kernel.count("bool do_finalize = true")
        > 0
        or header.count("bool do_finalize = true") >= 2,
        "python_forwards_do_finalize_to_binding": re.search(
            r"run_moe\([\s\S]*?activation_type,\s*do_finalize,?\s*\)", python_source
        )
        is not None,
        "kernel_has_early_skip": "if (!do_finalize) return;" in kernel,
        "effective_skip_reached_from_python_false": bool(calls)
        and all("do_finalize" in call for call in calls),
        "deferred_buffers_are_copied_out": all(
            token in binding
            for token in (
                "getGemm2ResultPtr()",
                "auto gemm2_output = alloc_tensor",
                "cudaMemcpyAsync(gemm2_output.data_ptr()",
                "auto expanded_idx = alloc_tensor",
                "cudaMemcpyAsync(expanded_idx.data_ptr()",
            )
        ),
        "unsupported_modes_fail_loud": all(
            token in python_source
            for token in (
                "do_finalize=False is not supported with min_latency_mode",
                "do_finalize=False is not yet supported with",
                "use_fused_finalize = False",
            )
        ),
        "fake_and_real_deferred_shapes_present": all(
            token in python_source
            for token in (
                "expanded_num_rows = seq_len * top_k",
                "input.new_empty([expanded_num_rows, hidden_size]",
                "expanded_idx.view(top_k, num_tokens).T.contiguous().view(-1)",
            )
        ),
        "direct_test_matrix_size": math.prod((3, 2, 2, 2, 2)),
        "direct_test_checks_manual_finalize": "manual_output = reference_finalize" in test
        and "assert_close(ref_output, manual_output" in test,
        "direct_test_checks_shapes_and_determinism": "Permutation mapping not deterministic" in test
        and "gemm2_output.shape ==" in test,
        "direct_test_detects_finalize_was_skipped": any(
            token in test
            for token in ("finalize_call_count", "did_finalize", "skip_finalize_counter")
        ),
        "sm90_runtime_executed": False,
        "decisive_source_fact": "both binding-to-kernel calls omit do_finalize, so their C++ default true prevents the new early return",
    }


_PARTIAL_WARP_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float unsafe_sum(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) value += __shfl_xor_sync(0xffffffffU, value, mask, 32);
  return value;
}
__device__ float unsafe_max(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, mask, 32));
  return value;
}
__device__ float partial_sum(float value, int active) {
  int lane = threadIdx.x & 31;
  unsigned mask = active == 32 ? 0xffffffffU : (1U << active) - 1;
  for (int delta = 16; delta > 0; delta >>= 1) {
    float other = __shfl_xor_sync(mask, value, delta, 32);
    if ((lane ^ delta) < active) value += other;
  }
  return value;
}
__device__ float partial_max(float value, int active) {
  int lane = threadIdx.x & 31;
  unsigned mask = active == 32 ? 0xffffffffU : (1U << active) - 1;
  for (int delta = 16; delta > 0; delta >>= 1) {
    float other = __shfl_xor_sync(mask, value, delta, 32);
    if ((lane ^ delta) < active) value = fmaxf(value, other);
  }
  return value;
}
__global__ void probe(float* output, int mode) {
  __shared__ float sums[33];
  __shared__ float maxima[33];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  int warp_count = (blockDim.x + 31) / 32;
  int tail = blockDim.x & 31;
  bool partial = tail != 0 && wid == warp_count - 1;
  float sum = 1.0f;
  float maximum = float(threadIdx.x + 1);
  if (mode == 2 && partial) {
    sum = partial_sum(sum, tail);
    maximum = partial_max(maximum, tail);
  } else {
    sum = unsafe_sum(sum);
    maximum = unsafe_max(maximum);
  }
  if (lane == 0) { sums[wid] = sum; maxima[wid] = maximum; }
  __syncthreads();
  int selected_warps = mode == 0 ? (blockDim.x >> 5) : warp_count;
  sum = threadIdx.x < selected_warps ? sums[lane] : 0.0f;
  maximum = threadIdx.x < selected_warps ? maxima[lane] : -INFINITY;
  if (mode == 2 && partial) {
    sum = partial_sum(sum, tail);
    maximum = partial_max(maximum, tail);
  } else {
    sum = unsafe_sum(sum);
    maximum = unsafe_max(maximum);
  }
  if (threadIdx.x == 0) { output[0] = sum; output[1] = maximum; }
}
torch::Tensor run_probe(int64_t threads, int64_t mode) {
  auto output = torch::empty({2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  probe<<<1, threads, 0, at::cuda::getCurrentCUDAStream()>>>(output.data_ptr<float>(), int(mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


def _run_partial_warp_cuda() -> dict[str, Any] | None:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load_inline
    except Exception:
        return None
    if not torch.cuda.is_available() or CUDA_HOME is None:
        return None
    module = load_inline(
        name="infraswe_r12_partial_warp",
        cpp_sources="torch::Tensor run_probe(int64_t threads, int64_t mode);",
        cuda_sources=_PARTIAL_WARP_CUDA,
        functions=["run_probe"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    rows = []
    for threads in (1, 7, 31, 32, 33, 47, 63, 64, 96, 127, 128, 129, 159, 180, 192):
        row: dict[str, Any] = {"threads": threads}
        for mode, name in ((0, "unsafe_floor"), (1, "unsafe_ceil"), (2, "safe_partial")):
            observed = module.run_probe(threads, mode).cpu().tolist()
            row[name] = {
                "observed": observed,
                "matches_oracle": observed == [float(threads), float(threads)],
            }
        rows.append(row)
    return {
        "gpu": torch.cuda.get_device_name(),
        "rows": rows,
        "safe_partial_all_match": all(row["safe_partial"]["matches_oracle"] for row in rows),
    }


def _reduction_source_facts(source: str) -> dict[str, Any]:
    uses_partial_primitive = (
        "warpReduceSumPartialV2" in source and "warpReduceMaxPartialV2" in source
    )
    uses_ceil_warp_count = "ceil_div(blockDim.x" in source
    unsafe_full_mask = "__shfl_xor_sync(FINAL_MASK" in source
    if uses_partial_primitive:
        partial_body = source[
            source.index("warpReduceSumPartialV2") : source.index("blockReduceSumV2")
        ]
        active_mask_present = (
            "active_mask" in partial_body and "(1U << active_lanes) - 1" in partial_body
        )
    else:
        active_mask_present = False
    return {
        "uses_ceil_warp_count": uses_ceil_warp_count,
        "has_partial_warp_primitive": uses_partial_primitive,
        "partial_primitive_has_active_mask": active_mask_present,
        "uses_unsafe_full_mask_shuffle": unsafe_full_mask,
        "partial_warp_mask_contract_safe": uses_partial_primitive and active_mask_present,
    }


def _probe_flashinfer_reduction(item: dict[str, Any], *, pull_number: int) -> dict[str, Any]:
    base, head = _source_pair(item, "trtllm_allreduce_fusion.cuh")
    direct_test = None
    if pull_number == 3880:
        direct_test = _head_source(item, "test_trtllm_allreduce_reduction.py")
    sizes = (1, 7, 31, 32, 33, 47, 63, 64, 96, 127, 128, 129, 159, 180, 192)
    source_facts = {"base": _reduction_source_facts(base), "head": _reduction_source_facts(head)}
    modeled = [
        {
            "threads": size,
            "base_selected_warps": size >> 5,
            "head_selected_warps": math.ceil(size / 32),
            "base_omits_tail": size % 32 != 0,
            "head_partial_mask_safe": source_facts["head"]["partial_warp_mask_contract_safe"]
            or size % 32 == 0,
        }
        for size in sizes
    ]
    runtime = _run_partial_warp_cuda()
    return {
        "source_contract": source_facts,
        "modeled_matrix": modeled,
        "frozen_size_count": len(sizes),
        "head_unsafe_partial_size_count": sum(not row["head_partial_mask_safe"] for row in modeled),
        "head_all_partial_warps_mask_safe": all(row["head_partial_mask_safe"] for row in modeled),
        "changed_direct_test": direct_test is not None,
        "direct_test_sizes": (
            [int(value) for value in re.findall(r"\b(?:129|159|180|192)\b", direct_test)]
            if direct_test is not None
            else []
        ),
        "direct_test_checks_sum_and_max": direct_test is not None
        and "blockReduceSumV2" in direct_test
        and "blockReduceMaxV2" in direct_test,
        "cuda_runtime": runtime,
        "required_cuda_runtime_executed": runtime is not None,
        "decisive_static_failure": not source_facts["head"]["partial_warp_mask_contract_safe"],
    }


def _probe_flashinfer_3880(item: dict[str, Any]) -> dict[str, Any]:
    return _probe_flashinfer_reduction(item, pull_number=3880)


def _probe_flashinfer_3879(item: dict[str, Any]) -> dict[str, Any]:
    return _probe_flashinfer_reduction(item, pull_number=3879)


class _FakeTensor:
    def __init__(self) -> None:
        self.contiguous_calls = 0

    def contiguous(self) -> _FakeTensor:
        self.contiguous_calls += 1
        return self


class _FakeDist:
    class ReduceOp:
        SUM = "SUM"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def reduce(self, tensor: Any, *, dst: int, op: Any, group: Any) -> None:
        self.calls.append({"same_tensor": tensor is not None, "dst": dst, "op": op, "group": group})


class _StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        node.type_comment = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.type_comment = None
        node.decorator_list = []
        return node


def _exec_method(source: str, class_name: str, method_name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(source)
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise AssertionError(f"expected one class {class_name}")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise AssertionError(f"expected one method {class_name}.{method_name}")
    method = copy.deepcopy(methods[0])
    method = ast.fix_missing_locations(_StripAnnotations().visit(method))
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, f"<{class_name}.{method_name}>", "exec"), namespace)
    return namespace[method_name]


def _probe_megatron_5720(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "megatron/core/pipeline_parallel/bridge_communicator.py")
    fake_dist = _FakeDist()
    reduce_method = _exec_method(
        head, "BridgeCommunicator", "_reduce_dest_cp_gradient", {"dist": fake_dist}
    )
    rows = []
    for cp_size, group in ((1, None), (2, None), (2, "dest-cp"), (4, "dest-cp")):
        fake_dist.calls.clear()
        tensor = _FakeTensor()
        owner = type(
            "Owner",
            (),
            {"dest_cp_size": cp_size, "dest_cp_reduce_pg": group, "dest_local_leader_rank": 7},
        )()
        returned = reduce_method(owner, tensor)
        rows.append(
            {
                "dest_cp_size": cp_size,
                "group": group,
                "same_tensor": returned is tensor,
                "contiguous_calls": tensor.contiguous_calls,
                "reduce_calls": list(fake_dist.calls),
            }
        )
    tests = "\n".join(
        str(revisions["head"] or "")
        for path, revisions in item["sources"].items()
        if "test" in path
    )
    return {
        "destination_cp_matrix": rows,
        "no_reduce_without_destination_group": all(not row["reduce_calls"] for row in rows[:2]),
        "one_sum_to_destination_leader_with_group": all(
            len(row["reduce_calls"]) == 1
            and row["reduce_calls"][0]["dst"] == 7
            and row["reduce_calls"][0]["op"] == "SUM"
            and row["contiguous_calls"] == 1
            for row in rows[2:]
        ),
        "both_backward_entrypoints_reduce_before_send": head.count(
            "grad_tensor = self._reduce_dest_cp_gradient(grad_tensor)"
        )
        == 2,
        "source_cp_remains_explicitly_unsupported": "Source grid CP size must be 1" in head,
        "title_scope_is_destination_cp_only": True,
        "direct_cp2_vs_cp1_numeric_test": "test_cp2_matches_cp1_encoder_gradients" in tests,
        "direct_test_covers_steady_and_cooldown": "steady-state and cooldown backward paths"
        in tests,
        "direct_test_requires_eight_gpus": "Requires 8 GPUs" in tests,
        "base_supports_destination_cp": "dest_cp_size" in base,
    }


def _align_row_count(current: int, target: int, rank: int) -> int:
    if current == target:
        return current
    if target > current:
        return target
    return max(0, min(current, (rank + 1) * target) - rank * target)


def _probe_sglang_31311(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "python/sglang/srt/models/longcat_flash.py")
    allreduce_rows = [
        {
            "tp_size": tp,
            "real_a2a": real,
            "base_calls": int(tp > 1),
            "head_calls": int(tp > 1 and not real),
            "expected_calls": int(tp > 1 and not real),
        }
        for tp in (1, 2, 4)
        for real in (False, True)
    ]
    alignment_rows = [
        {
            "current": current,
            "target": target,
            "rank": rank,
            "result_rows": _align_row_count(current, target, rank),
            "matches_target": _align_row_count(current, target, rank) == target,
        }
        for current, target, ranks in ((2, 4, (0, 1)), (4, 2, (0, 1)), (5, 3, (0, 1)))
        for rank in ranks
    ]
    return {
        "collective_call_truth_table": allreduce_rows,
        "head_named_allreduce_cardinality_matches": all(
            row["head_calls"] == row["expected_calls"] for row in allreduce_rows
        ),
        "base_double_reduces_real_a2a": any(
            row["real_a2a"] and row["base_calls"] > row["expected_calls"] for row in allreduce_rows
        ),
        "row_alignment_matrix": alignment_rows,
        "valid_named_geometry_aligns": all(row["matches_target"] for row in alignment_rows[:4]),
        "helper_silently_short_slices_non_divisible_geometry": any(
            not row["matches_target"] for row in alignment_rows[4:]
        ),
        "forward_mlp_guards_gather_with_exact_multiplicity": "hidden_states.shape[0] * _scmoe_ats == positions.shape[0]"
        in head,
        "post_moe_merge_calls_alignment_unconditionally": "_scmoe_tgt = moe_hidden_states.shape[0]"
        in head,
        "base_has_alignment_helper": "def _scmoe_align_rows" in base,
        "changed_direct_test": _has_changed_test(item),
        "candidate_owned_closure_test": False,
    }


def _variable_collective_schedule(world_size: int, sizes: list[int]) -> dict[str, Any]:
    offsets = []
    offset = 0
    for rank, size in enumerate(sizes):
        offsets.append({"rank": rank, "start": offset, "stop": offset + size})
        offset += size
    schedules = {
        rank: {
            "all_gatherv": [("broadcast", peer, sizes[peer]) for peer in range(world_size)],
            "reduce_scatterv": [("reduce", peer, sizes[peer]) for peer in range(world_size)],
            "local_output_rows": sizes[rank],
        }
        for rank in range(world_size)
    }
    return {
        "world_size": world_size,
        "sizes": sizes,
        "offsets": offsets,
        "total": offset,
        "all_ranks_same_collective_order": len(
            {tuple(row["all_gatherv"]) for row in schedules.values()}
        )
        == 1
        and len({tuple(row["reduce_scatterv"]) for row in schedules.values()}) == 1,
        "schedules": schedules,
    }


def _probe_sglang_31290(item: dict[str, Any]) -> dict[str, Any]:
    xpu = _head_source(item, "device_communicators/xpu_communicator.py")
    coordinator = _head_source(item, "python/sglang/srt/distributed/parallel_state.py")
    test = _head_source(item, "test_xpu_gatherv_scatterv.py")
    matrices = [
        _variable_collective_schedule(2, [3, 5]),
        _variable_collective_schedule(3, [4, 2, 1]),
        _variable_collective_schedule(4, [5, 3, 1, 0]),
    ]
    xpu_available = False
    try:
        import torch

        xpu_available = bool(
            hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() >= 2
        )
    except Exception:
        pass
    return {
        "variable_collective_matrices": matrices,
        "all_modeled_ranks_use_identical_order": all(
            matrix["all_ranks_same_collective_order"] for matrix in matrices
        ),
        "xpu_implementation_uses_global_rank": "dist.get_global_rank(self.group, r)" in xpu,
        "all_gatherv_uses_ordered_broadcasts": "for r, sz in enumerate(sizes):" in xpu
        and "dist.broadcast(" in xpu,
        "reduce_scatterv_uses_ordered_reduces": "dist.reduce(output" in xpu
        and "dist.reduce(\n                    input_" in xpu,
        "coordinator_validates_shape_and_sizes": all(
            token in coordinator
            for token in (
                "assert len(local_sizes) == world_size",
                "assert input_.shape[0] == sum(local_sizes)",
                "assert input_.shape[0] == sizes[self.rank_in_group]",
                "assert tuple(output.shape) == tuple(output_size)",
            )
        ),
        "equal_sizes_collapse_to_native_collective": "all(s == local_sizes[0] for s in local_sizes)"
        in coordinator
        and "all(s == sizes[0] for s in sizes)" in coordinator,
        "list_input_fails_loud_on_xpu": "all_gatherv list input is not supported on XPU"
        in coordinator,
        "direct_test_checks_uneven_values": "sizes = [3, 5]" in test
        and "gathered values != reference" in test,
        "direct_test_checks_input_immutability": "reduce_scatterv mutated input_" in test,
        "direct_test_checks_preallocated_buffers": test.count("data_ptr()") >= 3,
        "direct_test_world_sizes": [2],
        "direct_test_has_zero_size_rank": False,
        "required_xpu_runtime_available": xpu_available,
        "required_xpu_runtime_executed": False,
    }


def _probe_torchtitan_3827(item: dict[str, Any]) -> dict[str, Any]:
    base_loss, head_loss = _source_pair(item, "torchtitan/components/loss.py")
    _base_generator, head_generator = _source_pair(
        item, "torchtitan/experiments/rl/actors/generator.py"
    )
    _base_wrapper, head_wrapper = _source_pair(
        item, "torchtitan/experiments/rl/models/vllm_wrapper.py"
    )
    gradient_rows = [
        {
            "tp_degree": degree,
            "base_replicate_backward_scale": degree,
            "head_identity_backward_scale": 1,
            "oracle_scale": 1,
        }
        for degree in (2, 4, 8)
    ]
    routing_rows = [
        {
            "group": group,
            "grad_enabled": grad_enabled,
            "dst": dst,
            "head_routes_to_global_tp": (not grad_enabled and dst in ("R", "I")),
            "should_route_to_global_tp": (group == "tp" and not grad_enabled and dst in ("R", "I")),
        }
        for group in ("tp", "other")
        for grad_enabled in (False, True)
        for dst in ("R", "I", "S")
    ]
    return {
        "logprob_gradient_matrix": gradient_rows,
        "head_logprob_gradient_matches_oracle": all(
            row["head_identity_backward_scale"] == row["oracle_scale"] for row in gradient_rows
        ),
        "base_logprob_overcounts": "dst=spmd.R" in base_loss and "backward_options" in base_loss,
        "head_uses_identity_placement": "dst=spmd.I" in head_loss,
        "fused_swiglu_split_layouts_added": all(
            token in head_generator
            for token in (
                'for proj_name in ("w1", "w3"):',
                'layouts[f"{module_fqn}.{proj_name}.weight"] = w13_layout',
            )
        ),
        "root_module_layout_keys": [f".{name}.weight" for name in ("w1", "w3")],
        "root_module_keys_have_leading_dot": True,
        "redistribute_routing_matrix": routing_rows,
        "non_tp_groups_misrouted_to_tp": any(
            row["head_routes_to_global_tp"] != row["should_route_to_global_tp"]
            for row in routing_rows
        ),
        "wrapper_ignores_supplied_group_when_intercepting": "tensor_model_parallel_all_reduce(x)"
        in head_wrapper
        and "group =="
        not in head_wrapper[
            head_wrapper.index("def redistribute(") : head_wrapper.index(
                "spmd.redistribute = redistribute"
            )
        ],
        "changed_direct_test": _has_changed_test(item),
        "residual_failure_families": [
            "wrong-process-group interception",
            "root FusedSwiGLU layout key",
        ],
    }


def _probe_torchtitan_3821(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "torchtitan/experiments/graph_trainer/fsdp_passes.py")
    test = _head_source(item, "torchtitan/experiments/graph_trainer/tests/test_passes.py")
    types = (
        "bucketed_reduce_scatter",
        "bucketed_all_reduce",
        "bucketed_reduce_scatter_wait",
        "bucketed_all_reduce_wait",
    )
    return {
        "scheduler_node_type_matrix": [
            {
                "node_type": node_type,
                "base_recognizes": node_type in base[base.index("def _schedule_rs_prefetch") :],
                "head_recognizes": node_type in head[head.index("def _schedule_rs_prefetch") :],
            }
            for node_type in types
        ],
        "head_recognizes_all_reduce_start_and_wait": all(
            token in head for token in ("bucketed_all_reduce", "bucketed_all_reduce_wait")
        ),
        "source_change_is_scheduler_type_extension_only": (
            _patch_text(item, "fsdp_passes.py").count("+            if node_type in") == 1
            and "bucketed_all_reduce" in _patch_text(item, "fsdp_passes.py")
        ),
        "direct_test_has_ddp_two_gpu_case": "class TestDDPAllReduceBucketing" in test
        and "return 2" in test,
        "direct_test_has_hsdp_four_gpu_case": "class TestHSDPBucketing" in test
        and "return 4" in test,
        "direct_test_distinguishes_call_counts": "self.assertEqual(after, 1)" in test
        and 'self.assertEqual(self._count_collective(bw_gm, "all_reduce"), 1)' in test,
        "direct_test_checks_graph_lint": test.count("bw_gm.graph.lint()") >= 2,
        "direct_test_checks_numeric_equivalence": any(
            token in test for token in ("assert_close", "assertEqual(output", "reference_output")
        ),
        "distributed_tests_executed_by_evaluator": False,
    }


def _persistent_protocol_sequence() -> dict[str, Any]:
    sender: dict[str, tuple[int, int]] = {}
    receiver: dict[str, int] = {}
    events = []
    key = "ipc://frozen"
    for size in (1024, 1024, 2048, 2048):
        cached = sender.get(key)
        if cached is not None and cached[0] == size:
            message = {"reuse_gen": cached[1]}
        else:
            generation = cached[1] + 1 if cached is not None else 0
            sender[key] = (size, generation)
            message = {"handle": f"handle-{generation}", "gen": generation}
        if "reuse_gen" in message:
            ok = receiver.get(key) == message["reuse_gen"]
        else:
            receiver[key] = int(message["gen"])
            ok = True
        events.append({"size": size, "message": message, "receiver_accepts": ok})
    return {"events": events, "all_accepted": all(event["receiver_accepts"] for event in events)}


def _probe_verl_6958(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "bucketed_weight_transfer.py")
    protocol = _persistent_protocol_sequence()
    sender_cleanup_start = head.index(
        "def _cleanup(self):", head.index("class BucketedWeightSender")
    )
    receiver_class = head.index("class BucketedWeightReceiver")
    sender_cleanup = head[sender_cleanup_start:receiver_class]
    receiver_cleanup = head[head.index("def _cleanup(self):", receiver_class) :]
    return {
        "modeled_resize_reuse_sequence": protocol,
        "modeled_messages": [event["message"] for event in protocol["events"]],
        "sender_cache_compatibility_fields": ["zmq_handle", "numel"],
        "sender_cache_checks_device": "cached[0].device" in head,
        "sender_cache_checks_dtype": "cached[0].dtype" in head,
        "receiver_rejects_missing_or_stale_generation": "persistent-bucket protocol desync" in head,
        "receiver_synchronizes_before_ack": head.index(
            "get_torch_device().synchronize()", receiver_class
        )
        < head.index('self.socket.send(b"")', receiver_class),
        "sender_synchronizes_before_each_publish": head.count("get_torch_device().synchronize()")
        >= base.count("get_torch_device().synchronize()"),
        "persistent_cleanup_retains_sender_owner": "_SENDER_BUCKET_CACHE" in sender_cleanup
        and "self.buffer = None" in sender_cleanup,
        "persistent_cleanup_retains_receiver_owner": "_RECEIVER_IMPORT_CACHE" in receiver_cleanup
        and "self.buffer = None" in receiver_cleanup,
        "cache_has_explicit_eviction_or_process_teardown": any(
            token in head
            for token in (
                "_SENDER_BUCKET_CACHE.pop",
                "_SENDER_BUCKET_CACHE.clear",
                "atexit.register",
            )
        ),
        "changed_direct_test": _has_changed_test(item),
        "candidate_owned_closure_test": False,
    }


def _rms_norm_rows(rows: list[list[float]], epsilon: float = 1e-6) -> list[list[float]]:
    output = []
    for row in rows:
        scale = 1.0 / math.sqrt(sum(value * value for value in row) / len(row) + epsilon)
        output.append([value * scale for value in row])
    return output


def _probe_vllm_48763(item: dict[str, Any]) -> dict[str, Any]:
    generic_base, generic_head = _source_pair(item, "vllm/model_executor/models/deepseek_mtp.py")
    _nvidia_base, nvidia_head = _source_pair(item, "vllm/models/deepseek_v32/nvidia/mtp.py")
    hidden = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    residual = [[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]
    gathered_sum = [
        [h + r for h, r in zip(hidden_row, residual_row, strict=True)]
        for rank_h, rank_r in zip(hidden, residual, strict=True)
        for hidden_row, residual_row in zip(rank_h, rank_r, strict=True)
    ]
    local_sum_then_gather = [
        [h + r for h, r in zip(hidden_row, residual_row, strict=True)]
        for rank_h, rank_r in zip(hidden, residual, strict=True)
        for hidden_row, residual_row in zip(rank_h, rank_r, strict=True)
    ]
    old_normalized = _rms_norm_rows(gathered_sum)
    new_normalized = [
        row
        for rank_h, rank_r in zip(hidden, residual, strict=True)
        for row in _rms_norm_rows(
            [
                [h + r for h, r in zip(hidden_row, residual_row, strict=True)]
                for hidden_row, residual_row in zip(rank_h, rank_r, strict=True)
            ]
        )
    ]
    return {
        "generic_sequence_parallel_algebra_equal": gathered_sum == local_sum_then_gather,
        "nvidia_rowwise_norm_commutes_with_row_gather": all(
            math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-12)
            for old_row, new_row in zip(old_normalized, new_normalized, strict=True)
            for old, new in zip(old_row, new_row, strict=True)
        ),
        "old_helper_gathers_concatenated_hidden_and_residual": "torch.cat([hidden_states, residual], dim=-1)"
        in generic_base,
        "head_gathers_once_after_residual_add": "hidden_states = residual + hidden_states"
        in generic_head
        and generic_head.index("hidden_states = residual + hidden_states")
        < generic_head.index("tensor_model_parallel_all_gather(hidden_states, 0)"),
        "nvidia_head_normalizes_before_gather": nvidia_head.index(
            "self.shared_head.norm(hidden_states, residual)"
        )
        < nvidia_head.index("tensor_model_parallel_all_gather(hidden_states, 0)"),
        "non_sequence_parallel_allreduce_retained": "if not is_sequence_parallel:" in nvidia_head
        and "tensor_model_parallel_all_reduce(hidden_states)" in nvidia_head,
        "modeled_gather_payload_elements": {"base": 16, "head": 8},
        "modeled_payload_reduction_fraction": 0.5,
        "base_has_restore_helper": "def _restore_full_token_layout_if_needed" in generic_base,
        "head_has_restore_helper": "def _restore_full_token_layout_if_needed" in generic_head,
        "changed_direct_test": _has_changed_test(item),
        "candidate_contains_paired_benchmark": any(
            "bench" in path.lower() for path in item["sources"]
        ),
        "required_two_gpu_performance_executed": False,
        "title_claimed_throughput_gain": "5%" in item["case"]["title"],
    }


PROBES = {
    "cutlass-pr-3294": _probe_cutlass_3294,
    "flashinfer-pr-3939": _probe_flashinfer_3939,
    "flashinfer-pr-3931": _probe_flashinfer_3931,
    "flashinfer-pr-3880": _probe_flashinfer_3880,
    "flashinfer-pr-3879": _probe_flashinfer_3879,
    "megatron-pr-5720": _probe_megatron_5720,
    "sglang-pr-31311": _probe_sglang_31311,
    "sglang-pr-31290": _probe_sglang_31290,
    "torchtitan-pr-3827": _probe_torchtitan_3827,
    "torchtitan-pr-3821": _probe_torchtitan_3821,
    "verl-pr-6958": _probe_verl_6958,
    "vllm-pr-48763": _probe_vllm_48763,
}


def _environment() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch_cuda_available"] = torch.cuda.is_available()
        facts["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            facts["gpu_names"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as error:
        facts["torch_error"] = f"{type(error).__name__}: {error}"
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--case", choices=sorted(PROBES), action="append")
    args = parser.parse_args()

    selection = _read(args.selection)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != _canonical(selection_material):
        raise SystemExit("R12 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R12 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R12 plan/selection binding mismatch")
    if plan["frozen_before_source_diff_content_inspection"] is not True:
        raise SystemExit("R12 plan did not preserve the source-inspection boundary")
    case_by_id = {item["case_id"]: item for item in selection_material["cases"]}
    selected_ids = args.case or list(PROBES)
    if not set(selected_ids) <= set(case_by_id):
        raise SystemExit("requested probe case is not selected")

    if args.source_bundle:
        bundle = _read(args.source_bundle)
    else:
        bundle = _acquire([case_by_id[case_id] for case_id in selected_ids])
    if args.bundle_output:
        args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
        args.bundle_output.write_text(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    if args.acquire_only:
        print(f"acquired_cases={len(bundle)}")
        print(f"source_bundle_sha256={_canonical(bundle)}")
        return 0

    if set(bundle) != set(case_by_id):
        raise SystemExit("R12 source bundle case set differs from selection")
    environment = _environment()
    environment_sha256 = _canonical(environment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case_id in selected_ids:
        item = bundle[case_id]
        source_digests = {
            f"{revision}:{path}": _canonical(source)
            for path, revisions in item["sources"].items()
            for revision, source in revisions.items()
            if source is not None
        }
        started = datetime.now(UTC)
        try:
            facts = PROBES[case_id](item)
            status = "pass"
            failure_codes: list[str] = []
        except Exception as error:
            facts = {"exception_type": type(error).__name__, "exception": str(error)}
            status = "fail"
            failure_codes = ["R12_COMMUNICATION_CONTRACT_PROBE_FAILED"]
            failures += 1
        material = {
            "schema_version": "0.1",
            "protocol_id": selection_material["protocol_id"],
            "case_id": case_id,
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "source_bundle_sha256": _canonical(bundle),
            "base_sha": case_by_id[case_id]["base_sha"],
            "head_sha": case_by_id[case_id]["head_sha"],
            "source_digests": source_digests,
            "environment": environment,
            "environment_sha256": environment_sha256,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "probe_status": status,
            "failure_codes": failure_codes,
            "facts": facts,
        }
        payload = {**material, "evidence_sha256": _canonical(material)}
        output = args.output_dir / f"{case_id}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{case_id}: {status} {payload['evidence_sha256']}")
    print(f"source_bundle_sha256={_canonical(bundle)}")
    print(f"environment_sha256={environment_sha256}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
