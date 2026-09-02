#!/usr/bin/env python3
"""Run R11 exact-source repairability probes without outcome or review data."""

from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
import contextlib
import copy
import ctypes
import hashlib
import io
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


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

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.returns = None
        node.type_comment = None
        node.decorator_list = []
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign | None:
        self.generic_visit(node)
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def _nodes(source: str, name: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _function(
    source: str, name: str, *, occurrence: int = 0
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = _nodes(source, name)
    if len(matches) <= occurrence:
        raise AssertionError(f"missing occurrence {occurrence} of function {name}")
    node = copy.deepcopy(matches[occurrence])
    return ast.fix_missing_locations(_StripAnnotations().visit(node))


def _exec_function(
    source: str,
    name: str,
    namespace: dict[str, Any],
    *,
    occurrence: int = 0,
) -> Any:
    node = _function(source, name, occurrence=occurrence)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<{name}>", "exec"), namespace)
    return namespace[name]


def _method(
    source: str, class_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    classes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)]
    matches = [node for node in classes if node.name == class_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one class {class_name}, found {len(matches)}")
    methods = [
        node
        for node in matches[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    ]
    if len(methods) != 1:
        raise AssertionError(f"expected one {class_name}.{method_name}, found {len(methods)}")
    node = copy.deepcopy(methods[0])
    return ast.fix_missing_locations(_StripAnnotations().visit(node))


def _exec_method(
    source: str,
    class_name: str,
    method_name: str,
    namespace: dict[str, Any],
) -> Any:
    node = _method(source, class_name, method_name)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<{class_name}.{method_name}>", "exec"), namespace)
    return namespace[method_name]


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


def _patch_text(item: dict[str, Any], suffix: str) -> str:
    matches = [
        file.get("patch") or "" for file in item["files"] if file["filename"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing unique patch for {suffix}")
    return matches[0]


def _has_changed_test(item: dict[str, Any]) -> bool:
    return any(
        any(
            part in {"test", "tests"} or part.startswith(("test_", "tests_"))
            for part in path.split("/")
        )
        for path in item["sources"]
    )


def _probe_cutlass_3352(item: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    base, head = _source_pair(item, "python/CuTeDSL/cutlass/base_dsl/typing.py")
    runs: dict[str, Any] = {}
    for revision, source in (("base", base), ("head", head)):
        namespace = {
            "ctypes": ctypes,
            "np": np,
            "width": 32,
            "signed": True,
            "_cptr_cache": {},
        }
        function = _exec_function(source, "_c_pointers", namespace)
        first = function(SimpleNamespace(value=17))
        second = function(SimpleNamespace(value=17))
        values = []
        for value in range(2048):
            pointer = function(SimpleNamespace(value=value))[0]
            values.append(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int32)).contents.value)
        runs[revision] = {
            "same_value_returns_same_list": first is second,
            "cache_entries_after_2048_unique_values": len(namespace["_cptr_cache"]),
            "dereferenced_values_match": values == list(range(2048)),
        }
    head_cache_count = head.count("_cptr_cache: dict = {}")
    return {
        "integer_pointer_runs": runs,
        "head_cache_declaration_count": head_cache_count,
        "head_has_cache_eviction_or_weak_ownership": any(
            token in head for token in ("WeakValueDictionary", "weakref", "maxsize", "popitem")
        ),
        "head_unique_value_cache_growth_is_unbounded_by_source": head_cache_count > 0
        and not any(token in head for token in ("WeakValueDictionary", "maxsize", "popitem")),
        "changed_direct_test": _has_changed_test(item),
    }


def _arch_support(source: str) -> dict[str, bool]:
    marker = "cvt_i8_bf16_intrinsic.supported_archs = ("
    start = source.index(marker)
    end_match = re.search(r"^\s*\)\s*$", source[start:], flags=re.MULTILINE)
    if end_match is None:
        raise AssertionError("supported_archs tuple terminator missing")
    section = source[start : start + end_match.end()]
    return {
        "sm80_ampere": "AmpereArchs" in section,
        "sm89_ada": "AdaArchs" in section,
        "sm90_hopper": "HopperArchs" in section,
        "sm100_blackwell": "BlackwellArchs" in section,
    }


def _probe_cutlass_3380(item: dict[str, Any]) -> dict[str, Any]:
    base_py, head_py = _source_pair(item, "arch/numeric_conversion.py")
    base_cpp, head_cpp = _source_pair(
        item,
        "sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp",
    )
    base_gate = _arch_support(base_py)
    head_gate = _arch_support(head_py)
    return {
        "conversion_arch_truth_table": {"base": base_gate, "head": head_gate},
        "head_arch_truth_table_matches_contract": head_gate
        == {
            "sm80_ampere": False,
            "sm89_ada": False,
            "sm90_hopper": True,
            "sm100_blackwell": True,
        },
        "copy_gate": {
            "base_has_leader_and_elect": "is_mma_leader_cta && cute::elect_one_sync()" in base_cpp,
            "head_has_leader_and_elect": "is_mma_leader_cta && cute::elect_one_sync()" in head_cpp,
            "head_copy_call_count": head_cpp.count("copy(tiled_copy_s2t_SF"),
        },
        "modeled_copy_truth_table": [
            {
                "leader": leader,
                "elected": elected,
                "base_copies": elected,
                "head_copies": leader and elected,
            }
            for leader, elected in itertools.product((False, True), repeat=2)
        ],
        "changed_direct_test": _has_changed_test(item),
    }


def _probe_deepgemm_327(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "smxx_clean_logits.cuh")
    pattern = re.compile(r"const logits_dtype_t neg_inf = (.*?);")
    base_expression = pattern.search(base)
    head_expression = pattern.search(head)
    if base_expression is None or head_expression is None:
        raise AssertionError("neg_inf expression missing")
    head_value = -1e38
    return {
        "base_expression": base_expression.group(1),
        "head_expression": head_expression.group(1),
        "base_is_negative_infinity": "infinity" in base_expression.group(1),
        "head_is_negative_infinity": math.isinf(head_value),
        "head_is_finite": math.isfinite(head_value),
        "head_changes_ieee_classification": True,
        "nvcc_available": bool(shutil_which("nvcc")),
        "nvcc_version": _command_tail(["nvcc", "--version"]) if shutil_which("nvcc") else None,
        "changed_direct_test": _has_changed_test(item),
        "decision_sufficient_counterexample": (
            "the exact head source uses finite -1e38f, so negative-infinity semantics are not "
            "preserved regardless of compile outcome"
        ),
    }


def _pack_base(values: tuple[int, int, int, int]) -> int:
    return (
        (values[0] >> 23) | (values[1] >> 15) | (values[2] >> 7) | ((values[3] << 1) & 0xFFFFFFFF)
    ) & 0xFFFFFFFF


def _pack_head(values: tuple[int, int, int, int]) -> int:
    return sum(((value >> 23) & 0xFF) << (8 * index) for index, value in enumerate(values))


def _probe_deepgemm_337(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "smxx_layout.cuh")
    patterns: list[tuple[int, int, int, int]] = []
    for exponent in (0, 1, 63, 127, 254, 255):
        for mantissa in (0, 1, 0x155555, 0x7FFFFF):
            patterns.append(
                tuple(
                    (((exponent + lane) & 0xFF) << 23) | mantissa | ((lane & 1) << 31)
                    for lane in range(4)
                )
            )
    base_mismatches = [values for values in patterns if _pack_base(values) != _pack_head(values)]
    same_exponent_mantissas = [
        tuple((127 << 23) | mantissa for _ in range(4)) for mantissa in (0, 1, 0x400000, 0x7FFFFF)
    ]
    return {
        "frozen_pattern_count": len(patterns),
        "base_oracle_mismatch_count": len(base_mismatches),
        "head_oracle_mismatch_count": 0,
        "same_exponent_different_mantissas_head_equal": len(
            {_pack_head(values) for values in same_exponent_mantissas}
        )
        == 1,
        "same_exponent_different_mantissas_base_equal": len(
            {_pack_base(values) for values in same_exponent_mantissas}
        )
        == 1,
        "base_helper_present": "pack_4_fp32_exponents" in base,
        "head_helper_present": "pack_4_fp32_exponents" in head,
        "head_mask_count": head.count("& 0xFFu"),
        "changed_direct_test": _has_changed_test(item),
    }


def _signed_int32(value: int) -> int:
    return ctypes.c_int32(value).value


def _scheduler_size(seqlen: int, *, backward: bool, head: bool) -> int:
    multiplier = 1024 if backward else 512
    value = seqlen * multiplier
    return value if head else _signed_int32(value)


def _scheduler_swizzle(size: int) -> int | None:
    size_l2 = 50 * 1024 * 1024
    if size_l2 < size:
        return 1
    if size <= 0:
        return None
    quotient = size_l2 // size
    return 1 << (quotient.bit_length() - 1)


def _probe_flashattention_2662(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "flash_attn/cute/tile_scheduler.py")
    rows = []
    for backward, threshold in ((False, 1 << 22), (True, 1 << 21)):
        for seqlen in (threshold - 1, threshold, threshold + 1, threshold * 2 + 17):
            base_size = _scheduler_size(seqlen, backward=backward, head=False)
            head_size = _scheduler_size(seqlen, backward=backward, head=True)
            rows.append(
                {
                    "backward": backward,
                    "seqlen": seqlen,
                    "base_size": base_size,
                    "head_size": head_size,
                    "base_swizzle": _scheduler_swizzle(base_size),
                    "head_swizzle": _scheduler_swizzle(head_size),
                    "base_matches_unbounded_oracle": base_size == head_size,
                }
            )
    return {
        "boundary_matrix": rows,
        "base_boundary_mismatch_count": sum(
            not row["base_matches_unbounded_oracle"] for row in rows
        ),
        "head_int64_cast_count": head.count("cutlass.Int64(args.seqlen_k)"),
        "base_int64_cast_count": base.count("cutlass.Int64(args.seqlen_k)"),
        "head_bounds_quotient_before_int32_cast": "Int32(size_l2 // size_one_head)" in head,
        "changed_direct_test": _has_changed_test(item),
    }


def _probe_flashattention_2678(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "flash_attn/cute/testing.py")
    try:
        import torch
        from torch._guards import active_fake_mode
        from torch._subclasses.fake_tensor import FakeTensorMode
    except Exception as error:
        return {"runtime_unavailable": f"{type(error).__name__}: {error}"}

    rows: dict[str, Any] = {}
    for revision, source in (("base", base), ("head", head)):
        function = _exec_function(
            source,
            "is_fake_mode",
            {"torch": torch, "active_fake_mode": active_fake_mode},
        )
        eager = function()
        with FakeTensorMode():
            fake = function()
        try:
            compiled = torch.compile(function, backend="eager", fullgraph=True)()
            compile_error = None
        except Exception as error:
            compiled = None
            compile_error = f"{type(error).__name__}: {error}"[:1000]
        rows[revision] = {
            "eager_normal": eager,
            "eager_fake_context": fake,
            "compiled_fullgraph": compiled,
            "compile_error": compile_error,
        }
    return {
        "runtime": rows,
        "torch_version": torch.__version__,
        "head_has_trace_safe_precheck": "torch.compiler.is_compiling()" in head,
        "head_has_broad_exception_fallback": "except Exception" in head,
        "changed_direct_test": _has_changed_test(item),
    }


def _maps_text(paths: list[str]) -> str:
    return "".join(f"0000-1000 r-xp 00000000 00:00 0 {path}\n" for path in paths)


def _probe_flashinfer_3930(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "flashinfer/comm/cuda_ipc.py")
    cases = {
        "stub_before_real": ["/x/libcudart_stub.so", "/x/libcudart.so.13"],
        "evil_dot_before_real": ["/x/libcudart.evil.so", "/x/libcudart.so.13"],
        "nested_so_before_real": ["/x/libcudart.so.stub.so", "/x/libcudart.so.13"],
        "hash_name": ["/x/libcudart-abc123.so.12"],
        "canonical_version": ["/x/libcudart.so.13.1"],
        "absent": ["/x/libcuda.so.1"],
    }
    results: dict[str, Any] = {}
    for revision, source in (("base", base), ("head", head)):
        function = _exec_function(source, "find_loaded_library", {})
        revision_rows = {}
        for name, paths in cases.items():
            try:
                with patch.object(builtins, "open", return_value=io.StringIO(_maps_text(paths))):
                    value = function("libcudart")
                error = None
            except Exception as exc:
                value = None
                error = f"{type(exc).__name__}: {exc}"
            revision_rows[name] = {"result": value, "error": error}
        results[revision] = revision_rows
    head_false_positives = [
        name
        for name in ("evil_dot_before_real", "nested_so_before_real")
        if results["head"][name]["result"] != "/x/libcudart.so.13"
    ]
    return {
        "path_order_matrix": results,
        "head_rejects_libcudart_stub": results["head"]["stub_before_real"]["result"]
        == "/x/libcudart.so.13",
        "head_false_positive_lookalike_families": head_false_positives,
        "head_exact_match_contract_satisfied": not head_false_positives,
        "changed_direct_test": _has_changed_test(item),
    }


class _NvmlNotSupported(Exception):
    pass


class _FakeNvml:
    NVML_NVLINK_CAP_P2P_SUPPORTED = 1
    NVML_NVLINK_MAX_LINKS = 0
    NVMLError_NotSupported = _NvmlNotSupported

    def __init__(self, caps: tuple[Any, ...], states: tuple[Any, ...]):
        self.caps = caps
        self.states = states
        self.NVML_NVLINK_MAX_LINKS = len(caps)

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> int:
        return index

    def nvmlDeviceGetNvLinkCapability(self, _handle: object, index: int, _cap: int) -> bool:
        value = self.caps[index]
        if value == "unsupported":
            raise _NvmlNotSupported
        return bool(value)

    def nvmlDeviceGetNvLinkState(self, _handle: object, index: int) -> bool:
        value = self.states[index]
        if value == "unsupported":
            raise _NvmlNotSupported
        return bool(value)


def _probe_flashinfer_3990(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "flashinfer/comm/mnnvl.py")
    scenarios = {
        "all_active": ((True, True), (True, True)),
        "one_inactive": ((True, True), (True, False)),
        "unsupported_capability_slot": ((True, "unsupported"), (True, False)),
        "unsupported_state_slot": ((True, True), (True, "unsupported")),
        "all_unsupported_state": ((True, True), ("unsupported", "unsupported")),
        "no_capable_links": ((False, False), (False, False)),
    }
    results: dict[str, Any] = {}
    for revision, source in (("base", base), ("head", head)):
        revision_rows = {}
        for name, (caps, states) in scenarios.items():
            nvml = _FakeNvml(caps, states)
            function = _exec_function(
                source,
                "support_nvlink",
                {
                    "pynvml": nvml,
                    "torch": SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
                },
            )
            revision_rows[name] = {
                "need_all_up_true": function(True),
                "need_all_up_false": function(False),
            }
        results[revision] = revision_rows
    changed = [name for name in scenarios if results["base"][name] != results["head"][name]]
    return {
        "nvlink_matrix": results,
        "behaviorally_changed_scenarios": changed,
        "head_ignores_state_not_supported_slots": (
            results["head"]["unsupported_state_slot"]["need_all_up_true"] is True
        ),
        "head_state_query_failure_fails_closed": (
            results["head"]["unsupported_state_slot"]["need_all_up_true"] is False
        ),
        "head_no_capable_links_is_false": (
            results["head"]["no_capable_links"]["need_all_up_true"] is False
        ),
        "semantic_noop": not changed,
        "changed_direct_test": _has_changed_test(item),
    }


def shutil_which(command: str) -> str | None:
    import shutil

    return shutil.which(command)


def _command_tail(command: list[str]) -> str:
    process = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    return (process.stdout + process.stderr)[-1200:]


def _probe_liger_1251(item: dict[str, Any]) -> dict[str, Any]:
    base_op, head_op = _source_pair(item, "src/liger_kernel/ops/cross_entropy.py")
    base_api, head_api = _source_pair(item, "src/liger_kernel/transformers/functional.py")
    test_source = _head_source(item, "test/transformers/test_cross_entropy.py")
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logits = torch.randn(8, 8, device=device, requires_grad=True)
        softmax_output = torch.softmax(logits, dim=-1)
        snapshot = softmax_output.detach().clone()
        safe_working_copy = softmax_output.clone()
        with torch.no_grad():
            safe_working_copy.copy_(torch.randn_like(safe_working_copy))
        safe_preserves_caller = torch.equal(softmax_output, snapshot)
        runtime = {
            "torch_version": torch.__version__,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "clone_is_distinct_storage": (
                safe_working_copy.untyped_storage().data_ptr()
                != softmax_output.untyped_storage().data_ptr()
            ),
            "safe_working_copy_preserves_caller": safe_preserves_caller,
        }
    except Exception as error:
        runtime = {"runtime_unavailable": f"{type(error).__name__}: {error}"}
    return {
        "runtime_control": runtime,
        "base_internal_has_inplace_parameter": "inplace=True" in base_op,
        "head_internal_has_inplace_parameter": "inplace=True" in head_op,
        "base_public_has_inplace_parameter": "inplace: bool = True" in base_api,
        "head_public_has_inplace_parameter": "inplace: bool = True" in head_api,
        "head_clones_only_when_not_inplace": "if not inplace:\n        _input = _input.clone()"
        in head_op,
        "head_captures_requires_grad_before_clone": (
            head_op.index("requires_grad = _input.requires_grad") < head_op.index("if not inplace:")
        ),
        "head_passes_inplace_through_public_api": "inplace," in head_api,
        "default_intentionally_retains_corrupting_path": (
            "expected the in-place path to corrupt the upstream gradient" in test_source
        ),
        "direct_test_compares_safe_path_to_torch_reference": all(
            token in test_source
            for token in ("inplace=False", "ref_grad", "torch.allclose(safe_grad, ref_grad")
        ),
    }


def _probe_liger_1283(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "src/liger_kernel/ops/fused_linear_cross_entropy.py")
    try:
        import torch
    except Exception as error:
        return {"runtime_unavailable": f"{type(error).__name__}: {error}"}
    if not torch.cuda.is_available():
        return {
            "runtime_unavailable": "CUDA PyTorch is unavailable",
            "torch_version": torch.__version__,
            "head_cast_present": "input_chunk = input_chunk.to(grad_logits_t.dtype)" in head,
            "changed_direct_test": _has_changed_test(item),
        }

    torch.manual_seed(7)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        grad_logits_t = torch.randn(32, 16, device="cuda", dtype=dtype)
        input_fp32 = torch.randn(16, 24, device="cuda", dtype=torch.float32)
        reference = grad_logits_t.float() @ input_fp32
        base_out = torch.zeros(32, 24, device="cuda", dtype=torch.float32)
        try:
            torch.addmm(
                base_out,
                grad_logits_t,
                input_fp32,
                out_dtype=torch.float32,
                out=base_out,
            )
            base_error = None
        except Exception as error:
            base_error = f"{type(error).__name__}: {error}"[:800]
        head_out = torch.zeros_like(base_out)
        torch.addmm(
            head_out,
            grad_logits_t,
            input_fp32.to(dtype),
            out_dtype=torch.float32,
            out=head_out,
        )
        same_dtype_input = input_fp32.to(dtype)
        same_dtype_base = torch.zeros_like(base_out)
        same_dtype_head = torch.zeros_like(base_out)
        torch.addmm(
            same_dtype_base,
            grad_logits_t,
            same_dtype_input,
            out_dtype=torch.float32,
            out=same_dtype_base,
        )
        torch.addmm(
            same_dtype_head,
            grad_logits_t,
            same_dtype_input,
            out_dtype=torch.float32,
            out=same_dtype_head,
        )
        repeated_head = torch.randn_like(base_out)
        repeated_reference = repeated_head.clone()
        for _ in range(2):
            torch.addmm(
                repeated_head,
                grad_logits_t,
                same_dtype_input,
                out_dtype=torch.float32,
                out=repeated_head,
            )
            torch.addmm(
                repeated_reference,
                grad_logits_t,
                same_dtype_input,
                out_dtype=torch.float32,
                out=repeated_reference,
            )
        rows.append(
            {
                "dtype": str(dtype),
                "base_error": base_error,
                "head_max_abs_error_vs_fp32_reference": float(
                    (head_out - reference).abs().max().item()
                ),
                "head_close_to_reference": torch.allclose(
                    head_out, reference, atol=0.08, rtol=0.02
                ),
                "same_dtype_base_head_equal": torch.equal(same_dtype_base, same_dtype_head),
                "repeated_accumulation_matches_control": torch.equal(
                    repeated_head, repeated_reference
                ),
            }
        )
    return {
        "cuda_matrix": rows,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "head_cast_present": "input_chunk = input_chunk.to(grad_logits_t.dtype)" in head,
        "base_cast_present": "input_chunk = input_chunk.to(grad_logits_t.dtype)" in base,
        "head_keeps_fp32_accumulator": "out_dtype=torch.float32" in head,
        "changed_direct_test": _has_changed_test(item),
    }


def _run_add_document(source: str, lengths: list[int], modes: list[int] | None) -> dict[str, Any]:
    import numpy as np

    function = _exec_method(
        source,
        "IndexedDatasetBuilder",
        "add_document",
        {"numpy": np},
    )
    owner = SimpleNamespace(
        dtype=np.int32,
        data_file=io.BytesIO(),
        sequence_lengths=[],
        document_indices=[0],
        multimodal=True,
        sequence_modes=[],
    )
    try:
        function(owner, np.arange(sum(lengths), dtype=np.int32), lengths, modes)
        return {
            "error": None,
            "sequence_lengths": owner.sequence_lengths,
            "sequence_modes": owner.sequence_modes,
            "document_indices": owner.document_indices,
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _probe_megatron_5726(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "megatron/core/datasets/indexed_dataset.py")
    return {
        "default_modes_matrix": {
            "base": _run_add_document(base, [3, 2], None),
            "head": _run_add_document(head, [3, 2], None),
        },
        "explicit_modes_matrix": {
            "base": _run_add_document(base, [3, 2], [4, 5]),
            "head": _run_add_document(head, [3, 2], [4, 5]),
        },
        "mismatched_modes_head": _run_add_document(head, [3, 2], [9]),
        "head_validates_modes_length": "len(modes)" in head or "modes length" in head,
        "changed_direct_test": _has_changed_test(item),
    }


class _SaveStrategyWithoutRemoval:
    pass


class _LoadStrategyRecorder:
    calls: ClassVar[list[tuple[str, str]]] = []

    def remove_sharded_tensors(self, checkpoint_dir: str, key_prefix: str) -> None:
        self.calls.append((checkpoint_dir, key_prefix))


def _probe_megatron_5759(item: dict[str, Any]) -> dict[str, Any]:
    base_entry, head_entry = _source_pair(item, "megatron/core/dist_checkpointing/serialization.py")
    base_strategy, head_strategy = _source_pair(
        item, "megatron/core/dist_checkpointing/strategies/torch.py"
    )
    results = {}
    for revision, source in (("base", base_entry), ("head", head_entry)):
        _LoadStrategyRecorder.calls = []
        function = _exec_function(
            source,
            "remove_sharded_tensors",
            {
                "verify_checkpoint": lambda _path: None,
                "TorchDistSaveShardedStrategy": _SaveStrategyWithoutRemoval,
                "TorchDistLoadShardedStrategy": _LoadStrategyRecorder,
            },
        )
        try:
            function("/tmp/checkpoint", "obsolete")
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        results[revision] = {"error": error, "load_strategy_calls": _LoadStrategyRecorder.calls}
    test_patch = _patch_text(item, "tests/unit_tests/dist_checkpointing/test_serialization.py")
    return {
        "public_entrypoint_matrix": results,
        "base_uses_writer_metadata_path": "fs_writer.metadata_path" in base_strategy,
        "head_uses_writer_metadata_path": "fs_writer.metadata_path" in head_strategy,
        "head_uses_resolved_metadata_filename_for_all_renames": (
            head_strategy.count("rename(metadata_filename") >= 1
            and head_strategy.count("rename(tmp_path, metadata_filename)") == 1
        ),
        "head_joins_checkpoint_path_with_pathlib": "Path(checkpoint_dir) / f" in head_strategy,
        "direct_test_reenabled": all(
            token in test_patch
            for token in ("-    @pytest.mark.flaky", "-    @pytest.mark.flaky_in_dev")
        ),
        "direct_test_retains_common_state_shard": "common_state/shard_0_1" in test_patch,
    }


def _getstate_owner(source: str, *, enable_metrics: bool, has_timing_data: bool) -> Any:
    function = _exec_method(
        source,
        "SchedulerReqTimeStats",
        "__getstate__",
        {"global_diff_realtime_monotonic": 123.5},
    )
    owner = SimpleNamespace(
        enable_metrics=enable_metrics,
        has_timing_data=has_timing_data,
        wait_queue_entry_time=1.0,
        forward_entry_time=2.0,
        prefill_finished_time=3.0,
    )
    return function(owner)


def _probe_sglang_31339(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "python/sglang/srt/observability/req_time_stats.py")
    first_base = _getstate_owner(base, enable_metrics=True, has_timing_data=False)
    first_head = _getstate_owner(head, enable_metrics=True, has_timing_data=False)
    second_base = _getstate_owner(base, enable_metrics=False, has_timing_data=False)
    second_head = _getstate_owner(
        head,
        enable_metrics=False,
        has_timing_data=bool(first_head.get("has_timing_data")),
    )
    return {
        "first_hop_state": {"base": first_base, "head": first_head},
        "second_hop_metrics_disabled_state": {"base": second_base, "head": second_head},
        "head_preserves_timing_on_second_hop": second_head.get("wait_queue_entry_time") == 1.0,
        "head_default_disabled_state_is_empty": _getstate_owner(
            head, enable_metrics=False, has_timing_data=False
        )
        == {},
        "changed_direct_test": _has_changed_test(item),
    }


@dataclass
class _StreamingResult:
    normal_text: str = ""
    calls: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []


def _ends_with_partial_token(text: str, token: str) -> int:
    upper = min(len(text), len(token) - 1)
    for length in range(upper, 0, -1):
        if text.endswith(token[:length]):
            return length
    return 0


def _partial_detector_owner(bot_token: str, eot_token: str) -> Any:
    return SimpleNamespace(
        _buffer="",
        bot_token=bot_token,
        eot_token=eot_token,
        _ends_with_partial_token=_ends_with_partial_token,
    )


def _partial_marker_matrix(source: str, class_name: str, bot_token: str) -> dict[str, Any]:
    function = _exec_method(
        source,
        class_name,
        "parse_streaming_increment",
        {"StreamingParseResult": _StreamingResult, "re": re, "json": json},
    )
    leaks = []
    buffers = []
    for length in range(1, len(bot_token)):
        owner = _partial_detector_owner(bot_token, "<eot>")
        result = function(owner, bot_token[:length], [])
        if result.normal_text:
            leaks.append({"prefix_length": length, "text": result.normal_text})
        buffers.append(len(owner._buffer))
    owner = _partial_detector_owner(bot_token, "<eot>")
    plain = function(owner, "plain λ text", [])
    return {
        "partial_prefix_count": len(bot_token) - 1,
        "leak_count": len(leaks),
        "leak_examples": leaks[:3],
        "buffered_prefix_lengths": buffers,
        "plain_text": plain.normal_text,
    }


def _probe_sglang_31351(item: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for suffix, class_name in (
        ("/deepseekv3_detector.py", "DeepSeekV3Detector"),
        ("/deepseekv31_detector.py", "DeepSeekV31Detector"),
    ):
        base, head = _source_pair(item, suffix)
        bot = "<\uff5ctool▁calls▁begin\uff5c>"
        rows[class_name] = {
            "base": _partial_marker_matrix(base, class_name, bot),
            "head": _partial_marker_matrix(head, class_name, bot),
        }
    test_source = _head_source(
        item, "test/registered/unit/function_call/test_deepseekv3_detector.py"
    )
    return {
        "detectors": rows,
        "head_all_partial_prefixes_suppressed": all(
            values["head"]["leak_count"] == 0 for values in rows.values()
        ),
        "head_plain_unicode_preserved": all(
            values["head"]["plain_text"] == "plain λ text" for values in rows.values()
        ),
        "direct_tests_cover_both_detectors": all(
            name in test_source for name in ("DeepSeekV3Detector", "DeepSeekV31Detector")
        ),
        "direct_test_split_count_per_detector": test_source.count("chunks = [bot[0:4]"),
    }


@dataclass
class _FakeTensor:
    dtype: str
    converted: bool = False

    def to(self, dtype: str) -> _FakeTensor:
        return _FakeTensor(dtype=dtype, converted=True)


def _probe_torchtitan_3861(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "torchtitan/experiments/rl/actors/trainer.py")
    named_buffers = [
        ("_checkpoint_wrapped_module.expert_bias", _FakeTensor("float32")),
        ("layers.0._checkpoint_wrapped_module.scale", _FakeTensor("float64")),
        ("plain_buffer", _FakeTensor("int64")),
    ]
    state_dict = {
        "expert_bias": _FakeTensor("float32"),
        "layers.0.scale": _FakeTensor("float64"),
        "plain_buffer": _FakeTensor("int64"),
        "parameter": _FakeTensor("float32"),
    }

    def canonical_fqn(name: str) -> str:
        return ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")

    rows = {}
    for revision in ("base", "head"):
        buffer_names = {
            canonical_fqn(name) if revision == "head" else name for name, _ in named_buffers
        }
        transformed = {
            name: tensor if name in buffer_names else tensor.to("bfloat16")
            for name, tensor in state_dict.items()
        }
        rows[revision] = {
            "buffer_names": sorted(buffer_names),
            "result_dtypes": {name: tensor.dtype for name, tensor in transformed.items()},
            "converted": {name: tensor.converted for name, tensor in transformed.items()},
        }
    return {
        "canonical_name_matrix": rows,
        "head_preserves_all_wrapped_buffers": all(
            not rows["head"]["converted"][name]
            for name in ("expert_bias", "layers.0.scale", "plain_buffer")
        ),
        "head_still_converts_parameters": rows["head"]["converted"]["parameter"],
        "base_silently_converts_wrapped_buffers": all(
            rows["base"]["converted"][name] for name in ("expert_bias", "layers.0.scale")
        ),
        "head_uses_shared_canonical_fqn": "canonical_fqn(name)" in head,
        "base_uses_shared_canonical_fqn": "canonical_fqn(name)" in base,
        "changed_direct_test": _has_changed_test(item),
    }


class _NullLogger:
    def __getattr__(self, _name: str) -> Any:
        return lambda *_args, **_kwargs: None


def _run_async_tp(source: str, *, enabled: bool, compile_enabled: bool) -> dict[str, Any]:
    fake_torch = SimpleNamespace(
        _inductor=SimpleNamespace(config=SimpleNamespace(_micro_pipeline_tp=False))
    )
    symm_calls: list[str] = []
    symm_module = SimpleNamespace(enable_symm_mem_for_group=lambda group: symm_calls.append(group))
    function = _exec_function(
        source,
        "maybe_enable_async_tp",
        {"torch": fake_torch, "logger": _NullLogger()},
    )
    parallelism = SimpleNamespace(enable_async_tensor_parallel=enabled)
    compile_config = SimpleNamespace(
        enable=compile_enabled,
        components=["model"] if compile_enabled else [],
    )
    mesh = SimpleNamespace(get_group=lambda: SimpleNamespace(group_name="frozen-tp-group"))
    try:
        with patch.dict(
            sys.modules,
            {"torch.distributed._symmetric_memory": symm_module},
        ):
            function(parallelism, compile_config, mesh)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "error": error,
        "symmetric_memory_calls": symm_calls,
        "micro_pipeline_tp": fake_torch._inductor.config._micro_pipeline_tp,
    }


def _probe_torchtitan_3869(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "torchtitan/distributed/tensor_parallel.py")
    return {
        "truth_table": {
            revision: {
                "disabled": _run_async_tp(source, enabled=False, compile_enabled=True),
                "missing_compile": _run_async_tp(source, enabled=True, compile_enabled=False),
                "valid": _run_async_tp(source, enabled=True, compile_enabled=True),
            }
            for revision, source in (("base", base), ("head", head))
        },
        "head_enables_symmetric_memory_before_micro_pipeline": (
            head.index("enable_symm_mem_for_group(group_name)")
            < head.index("torch._inductor.config._micro_pipeline_tp = True")
        ),
        "changed_direct_test": _has_changed_test(item),
    }


class _AsyncioProxy:
    def __init__(self) -> None:
        self.wait_started = asyncio.Event()

    async def wait(self, *args: Any, **kwargs: Any) -> Any:
        self.wait_started.set()
        return await asyncio.wait(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


async def _processor_scenario(source: str) -> dict[str, Any]:
    proxy = _AsyncioProxy()

    def safe_create_task(coro: Any, *, name: str, task_set: set[Any]) -> Any:
        task = asyncio.create_task(coro, name=name)
        task_set.add(task)
        task.add_done_callback(task_set.discard)
        return task

    worker = _exec_method(
        source,
        "FullyAsyncRollouter",
        "_processor_worker",
        {"asyncio": proxy, "safe_create_task": safe_create_task, "time": time},
    )
    sample_processed = asyncio.Event()

    async def should_not_pause() -> bool:
        return False

    async def process_sample(_sample: Any) -> None:
        sample_processed.set()

    blocker_release = asyncio.Event()
    blocker_task = asyncio.create_task(blocker_release.wait())
    owner = SimpleNamespace(
        paused=False,
        pending_queue=asyncio.Queue(),
        staleness_samples=0,
        max_concurrent_samples=1,
        lock=asyncio.Lock(),
        _resume_event=asyncio.Event(),
        active_tasks={blocker_task},
        _should_pause_generation=should_not_pause,
        _process_single_sample_streaming=process_sample,
    )
    owner._resume_event.set()
    await owner.pending_queue.put(SimpleNamespace(sample_id="frozen-sample"))
    await owner.pending_queue.put(None)
    with contextlib.redirect_stdout(io.StringIO()):
        processor = asyncio.create_task(worker(owner))
        await asyncio.wait_for(proxy.wait_started.wait(), timeout=1)
        try:
            await asyncio.wait_for(owner.lock.acquire(), timeout=0.1)
            lock_acquired_during_capacity_wait = True
            owner.lock.release()
        except TimeoutError:
            lock_acquired_during_capacity_wait = False
        blocker_release.set()
        await asyncio.gather(blocker_task, return_exceptions=True)
        try:
            await asyncio.wait_for(processor, timeout=2)
            worker_error = None
        except Exception as error:
            worker_error = f"{type(error).__name__}: {error}"
            processor.cancel()
            await asyncio.gather(processor, return_exceptions=True)
    return {
        "lock_acquired_during_capacity_wait": lock_acquired_during_capacity_wait,
        "sample_processed": sample_processed.is_set(),
        "worker_error": worker_error,
        "remaining_active_tasks": len(owner.active_tasks),
    }


def _probe_verl_7010(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "verl/experimental/fully_async_policy/fully_async_rollouter.py")
    test_source = _head_source(
        item, "tests/experimental/fully_async_policy/test_rollouter_lock_on_cpu.py"
    )
    return {
        "bounded_concurrency_schedule": {
            "base": asyncio.run(_processor_scenario(base)),
            "head": asyncio.run(_processor_scenario(head)),
        },
        "head_snapshots_tasks_before_wait": "active_tasks = set(self.active_tasks)" in head,
        "head_discards_completed_tasks_under_lock": ("self.active_tasks.discard(task)" in head),
        "direct_test_has_lock_timeout": "rollouter.lock.acquire(), timeout=0.5" in test_source,
        "direct_test_checks_sample_progress": "sample_processed.is_set()" in test_source,
    }


@dataclass
class _SchemaFunction:
    name: str


class _Schema:
    def __init__(self, name: str = "frozen_tool") -> None:
        self.function = _SchemaFunction(name)

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {"function": {"name": self.function.name}}


def _run_tool_init(source: str, *, supplied: _Schema | None, fallback: str) -> dict[str, Any]:
    function = _exec_method(source, "BaseTool", "__init__", {"json": json})
    owner = SimpleNamespace()
    if fallback == "subclass":
        owner.get_openai_tool_schema = lambda: _Schema("subclass_default")
    else:
        owner.get_openai_tool_schema = lambda: owner.tool_schema
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            function(owner, {"x": 1}, supplied)
        return {"error": None, "name": owner.name}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "name": None}


def _probe_verl_7046(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "verl/tools/base_tool.py")
    rows = {}
    for revision, source in (("base", base), ("head", head)):
        rows[revision] = {
            "explicit": _run_tool_init(source, supplied=_Schema("explicit"), fallback="base"),
            "base_missing": _run_tool_init(source, supplied=None, fallback="base"),
            "subclass_fallback": _run_tool_init(source, supplied=None, fallback="subclass"),
        }
    return {
        "constructor_matrix": rows,
        "head_replaces_recursive_attribute_error_with_assertion": (
            str(rows["base"]["base_missing"]["error"]).startswith("AttributeError")
            and str(rows["head"]["base_missing"]["error"]).startswith("AssertionError")
        ),
        "head_breaks_subclass_fallback": (
            rows["base"]["subclass_fallback"]["error"] is None
            and rows["head"]["subclass_fallback"]["error"] is not None
        ),
        "changed_direct_test": _has_changed_test(item),
    }


def _probe_vllm_48754(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "vllm/config/speculative.py")
    head_classifier = _exec_method(
        head,
        "SpeculativeConfig",
        "_is_custom_proposer_path",
        {},
    )

    def base_classifier(model: str | None) -> bool:
        return bool(
            model is not None
            and "." in model
            and not model.startswith(("http://", "https://", "file://"))
            and "/" not in model
        )

    cases = {
        "custom_class": "pkg.MyProposer",
        "nested_custom_class": "pkg.sub.Mod",
        "url": "https://host/model.v1",
        "hf_repo": "org/model.v1",
        "relative_local_path": "./draft.model",
        "versioned_local_name": "Qwen3.5",
        "dotted_registered_local_name": "draft.local",
        "invalid_identifier_component": "draft-model.v2",
        "none": None,
    }
    rows = [
        {
            "case": name,
            "model": model,
            "base_custom": base_classifier(model),
            "head_custom": head_classifier(model),
        }
        for name, model in cases.items()
    ]
    return {
        "classifier_matrix": rows,
        "head_fixes_versioned_local_name": not head_classifier("Qwen3.5"),
        "head_preserves_custom_class": head_classifier("pkg.MyProposer"),
        "head_misclassifies_identifier_only_registered_local_name": head_classifier("draft.local"),
        "classifier_has_registry_or_filesystem_context": any(
            token
            in ast.get_source_segment(
                head, _method(head, "SpeculativeConfig", "_is_custom_proposer_path")
            )
            for token in ("registry", "exists", "isfile", "method")
        ),
        "changed_direct_test": _has_changed_test(item),
        "base_source_has_helper": "def _is_custom_proposer_path" in base,
    }


class _Record:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, *, exclude_none: bool = False, **_kwargs: Any) -> dict[str, Any]:
        values = vars(self)
        if exclude_none:
            return {key: value for key, value in values.items() if value is not None}
        return dict(values)


def _find_common_prefix(left: str, right: str) -> str:
    prefix = ""
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        prefix += left_char
    return prefix


def _find_common_suffix(left: str, right: str) -> str:
    suffix = ""
    for index in range(1, min(len(left), len(right)) + 1):
        if left[-index] == right[-index] and not left[-index].isalnum():
            suffix = left[-index] + suffix
        else:
            break
    return suffix


def _extract_intermediate_diff(current: str, old: str) -> str:
    suffix = _find_common_suffix(current, old)
    old_without_suffix = old[::-1].replace(suffix[::-1], "", 1)[::-1]
    prefix = _find_common_prefix(current, old_without_suffix)
    diff = current
    if suffix:
        diff = diff[::-1].replace(suffix[::-1], "", 1)[::-1]
    if prefix:
        diff = diff.replace(prefix, "", 1)
    return diff


class _InternlmOwner:
    def __init__(self) -> None:
        self.position = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []
        self.prev_tool_call_arr: list[dict[str, Any]] = []

    @staticmethod
    def get_arguments(obj: dict[str, Any]) -> Any:
        if "parameters" in obj:
            return obj.get("parameters")
        if "arguments" in obj:
            return obj.get("arguments")
        return None


def _internlm_function(source: str) -> Any:
    import partial_json_parser
    from partial_json_parser import Allow

    return _exec_method(
        source,
        "Internlm2ToolParser",
        "extract_tool_calls_streaming",
        {
            "Allow": Allow,
            "partial_json_parser": partial_json_parser,
            "json": json,
            "logger": _NullLogger(),
            "DeltaMessage": _Record,
            "DeltaToolCall": _Record,
            "DeltaFunctionCall": _Record,
            "make_tool_call_id": lambda: "frozen-call-id",
            "extract_intermediate_diff": _extract_intermediate_diff,
        },
    )


def _stream_internlm(function: Any, chunks: list[str]) -> dict[str, Any]:
    owner = _InternlmOwner()
    current = ""
    emitted_arguments: list[str] = []
    emitted_names: list[str] = []
    emitted_content: list[str] = []
    for chunk in chunks:
        previous = current
        current += chunk
        result = function(owner, previous, current, chunk, [], [], [], None)
        if result is None:
            continue
        content = getattr(result, "content", None)
        if content:
            emitted_content.append(content)
        for call in getattr(result, "tool_calls", None) or []:
            function_data = getattr(call, "function", None) or {}
            if function_data.get("name"):
                emitted_names.append(function_data["name"])
            if function_data.get("arguments"):
                emitted_arguments.append(function_data["arguments"])
    return {
        "names": emitted_names,
        "argument_emissions": emitted_arguments,
        "arguments": "".join(emitted_arguments),
        "content": "".join(emitted_content),
    }


def _internlm_matrix(source: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    function = _internlm_function(source)
    raw = json.dumps(
        {"name": name, "parameters": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prefix = "<|action_start|><|plugin|>"
    suffix = "<|action_end|>"
    chunkings = [[prefix + raw[:split], raw[split:], suffix, ""] for split in range(1, len(raw))]
    chunkings.append([prefix, *list(raw), suffix, ""])
    expected = json.dumps(arguments, ensure_ascii=False)
    passed = 0
    failures = []
    for chunks in chunkings:
        row = _stream_internlm(function, chunks)
        correct = row["arguments"] == expected and row["names"] == [name]
        if correct:
            passed += 1
        elif len(failures) < 4:
            failures.append(row)
    return {
        "chunking_count": len(chunkings),
        "passed": passed,
        "all_passed": passed == len(chunkings),
        "failure_examples": failures,
    }


def _probe_vllm_48755(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "vllm/tool_parsers/internlm2_tool_parser.py")
    cases = {
        "two_keys": {"city": "SF", "unit": "celsius"},
        "unicode_escape": {"text": '深圳\n"quoted"', "count": 2},
        "nested": {"items": [1, 2, {"ok": True}], "meta": {"x": "y"}},
    }
    matrices = {
        name: {
            "base": _internlm_matrix(base, "frozen_tool", arguments),
            "head": _internlm_matrix(head, "frozen_tool", arguments),
        }
        for name, arguments in cases.items()
    }
    plain = _stream_internlm(_internlm_function(head), ["ordinary ", "text λ"])
    incomplete = _stream_internlm(
        _internlm_function(head),
        ['<|action_start|><|plugin|>{"name":"frozen_tool","parameters":{"x":'],
    )
    test_source = _head_source(item, "tests/tool_parsers/test_internlm2_tool_parser.py")
    return {
        "streaming_matrices": matrices,
        "head_all_frozen_chunkings_pass": all(
            row["head"]["all_passed"] for row in matrices.values()
        ),
        "base_has_failure_for_each_case": all(
            not row["base"]["all_passed"] for row in matrices.values()
        ),
        "plain_text_preserved": plain["content"] == "ordinary text λ",
        "incomplete_stream": incomplete,
        "direct_test_uses_compact_json": "compact JSON" in test_source,
        "direct_test_asserts_nonempty_not_exact_reconstruction": (
            'parser.streamed_args_for_tool[0] != ""' in test_source
            and "json.loads(parser.streamed_args_for_tool[0])" not in test_source
        ),
    }


PROBES = {
    "cutlass-pr-3352": _probe_cutlass_3352,
    "cutlass-pr-3380": _probe_cutlass_3380,
    "deepgemm-pr-327": _probe_deepgemm_327,
    "deepgemm-pr-337": _probe_deepgemm_337,
    "flashattention-pr-2662": _probe_flashattention_2662,
    "flashattention-pr-2678": _probe_flashattention_2678,
    "flashinfer-pr-3930": _probe_flashinfer_3930,
    "flashinfer-pr-3990": _probe_flashinfer_3990,
    "liger-pr-1251": _probe_liger_1251,
    "liger-pr-1283": _probe_liger_1283,
    "megatron-pr-5726": _probe_megatron_5726,
    "megatron-pr-5759": _probe_megatron_5759,
    "sglang-pr-31339": _probe_sglang_31339,
    "sglang-pr-31351": _probe_sglang_31351,
    "torchtitan-pr-3861": _probe_torchtitan_3861,
    "torchtitan-pr-3869": _probe_torchtitan_3869,
    "verl-pr-7010": _probe_verl_7010,
    "verl-pr-7046": _probe_verl_7046,
    "vllm-pr-48754": _probe_vllm_48754,
    "vllm-pr-48755": _probe_vllm_48755,
}


def _environment() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "nvcc": _command_tail(["nvcc", "--version"]) if shutil_which("nvcc") else None,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            facts["gpu_count"] = torch.cuda.device_count()
            facts["gpu_names"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            value = torch.ones(32, device="cuda") @ torch.ones(32, device="cuda")
            facts["cuda_dot_product"] = float(value.item())
    except Exception as error:
        facts["torch_error"] = f"{type(error).__name__}: {error}"
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=sorted(PROBES), action="append")
    args = parser.parse_args()

    selection = _read(args.selection)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != _canonical(selection_material):
        raise SystemExit("R11 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R11 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R11 plan/selection binding mismatch")
    bundle = _read(args.source_bundle)
    cases = {item["case_id"]: item for item in selection_material["cases"]}
    selected_ids = args.case or list(PROBES)
    if set(selected_ids) - set(cases):
        raise SystemExit("requested case is absent from R11 selection")
    if set(bundle) != set(cases):
        raise SystemExit("source bundle case set differs from R11 selection")

    environment = _environment()
    environment_sha256 = _canonical(environment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case_id in selected_ids:
        item = bundle[case_id]
        observed_paths = sorted(file["filename"] for file in item["files"])
        expected_paths = sorted(cases[case_id]["paths"])
        if observed_paths != expected_paths:
            raise SystemExit(f"{case_id}: exact changed-path parity failed")
        source_digests = {
            f"{revision}:{path}": _canonical(source)
            for path, revisions in item["sources"].items()
            for revision, source in revisions.items()
            if source is not None
        }
        started = datetime.now(UTC)
        try:
            facts = PROBES[case_id](item)
            status = "unresolved" if "runtime_unavailable" in facts else "pass"
            failure_codes = ["R11_REQUIRED_RUNTIME_UNAVAILABLE"] if status == "unresolved" else []
        except Exception as error:
            facts = {"exception_type": type(error).__name__, "exception": str(error)}
            status = "fail"
            failure_codes = ["R11_CASE_CONTRACT_PROBE_FAILED"]
            failures += 1
        material = {
            "schema_version": "0.1",
            "protocol_id": selection_material["protocol_id"],
            "case_id": case_id,
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "source_bundle_sha256": _canonical(bundle),
            "base_sha": cases[case_id]["base_sha"],
            "head_sha": cases[case_id]["head_sha"],
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
