#!/usr/bin/env python3
"""Run exact-source, case-specific R10 base/head contract probes."""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
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
            import time

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
            ".txt",
            ".cmake",
        )
    ) or path.endswith("CMakeLists.txt")


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


def _function(source: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name}, found {len(matches)}")
    node = copy.deepcopy(matches[0])
    node.decorator_list = []
    return ast.fix_missing_locations(_StripAnnotations().visit(node))


def _class(source: str, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one class {name}, found {len(matches)}")
    node = copy.deepcopy(matches[0])
    node.decorator_list = []
    return ast.fix_missing_locations(_StripAnnotations().visit(node))


def _exec_function(source: str, name: str, namespace: dict[str, Any], *, optimize: int = 0) -> Any:
    node = _function(source, name)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<{name}>", "exec", optimize=optimize), namespace)
    return namespace[name]


def _source_pair(item: dict[str, Any], suffix: str) -> tuple[str, str]:
    matches = [value for path, value in item["sources"].items() if path.endswith(suffix)]
    if len(matches) != 1 or matches[0]["base"] is None or matches[0]["head"] is None:
        raise AssertionError(f"missing unique base/head source for {suffix}")
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


def _patch(item: dict[str, Any], suffix: str) -> str:
    matches = [
        value.get("patch") or "" for value in item["files"] if value["filename"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing unique patch for {suffix}")
    return matches[0]


def _test_method_names(source: str) -> list[str]:
    return sorted(
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def _run(
    command: list[str], *, cwd: Path, timeout: int = 180, input_text: str | None = None
) -> dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": process.returncode,
        "stdout_sha256": _canonical(process.stdout),
        "stderr_sha256": _canonical(process.stderr),
        "stderr_tail": process.stderr[-1200:],
    }


def _probe_cutlass(item: dict[str, Any]) -> dict[str, Any]:
    base_header, head_header = _source_pair(item, "include/cute/util/print_tensor.hpp")
    base_manifest, head_manifest = _source_pair(item, "test/self_contained_includes/CMakeLists.txt")
    direct_include = "#include <cute/pointer_flagged.hpp>"
    static_facts = {
        "base_direct_pointer_flagged_include_count": base_header.count(direct_include),
        "head_direct_pointer_flagged_include_count": head_header.count(direct_include),
        "head_uses_smem_ptr_flag_bits": "smem_ptr_flag_bits" in head_header,
        "base_manifest_lists_header": "cute/util/print_tensor.hpp" in base_manifest,
        "head_manifest_lists_header": "cute/util/print_tensor.hpp" in head_manifest,
    }
    if shutil.which("git") is None or shutil.which("nvcc") is None:
        return {
            **static_facts,
            "native_compile_executed": False,
            "unresolved_reason": "git or nvcc is unavailable",
        }

    case = item["case"]
    with tempfile.TemporaryDirectory(prefix="infraswe-r10-cutlass-") as raw_root:
        root = Path(raw_root)
        setup_commands = [
            ["git", "init", "--quiet"],
            ["git", "remote", "add", "origin", f"https://github.com/{case['repository']}.git"],
            ["git", "sparse-checkout", "init", "--cone"],
            ["git", "sparse-checkout", "set", "include"],
        ]
        for command in setup_commands:
            result = _run(command, cwd=root)
            if result["returncode"] != 0:
                raise AssertionError(f"CUTLASS sparse checkout setup failed: {command}: {result}")

        compile_results: dict[str, dict[str, Any]] = {}
        unit = "#include <cute/util/print_tensor.hpp>\nint main() { return 0; }\n"
        for label, revision in (("base", case["base_sha"]), ("head", case["head_sha"])):
            fetched = _run(["git", "fetch", "--depth=1", "origin", revision], cwd=root)
            if fetched["returncode"] != 0:
                raise AssertionError(f"CUTLASS exact revision fetch failed for {label}: {fetched}")
            checked = _run(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], cwd=root)
            if checked["returncode"] != 0:
                raise AssertionError(f"CUTLASS exact checkout failed for {label}: {checked}")
            exact_source = (root / "include/cute/util/print_tensor.hpp").read_text(encoding="utf-8")
            locked_source = base_header if label == "base" else head_header
            if exact_source != locked_source:
                raise AssertionError(
                    f"CUTLASS checked-out {label} does not match the locked source artifact"
                )
            compile_results[label] = _run(
                [
                    "nvcc",
                    "-std=c++17",
                    "-I",
                    str(root / "include"),
                    "-x",
                    "cu",
                    "-c",
                    "-",
                    "-o",
                    str(root / f"{label}.o"),
                ],
                cwd=root,
                input_text=unit,
            )

    return {
        **static_facts,
        "native_compile_executed": True,
        "compiler": subprocess.run(
            ["nvcc", "--version"], check=False, capture_output=True, text=True
        ).stdout.strip(),
        "base_compile": compile_results["base"],
        "head_compile": compile_results["head"],
        "base_failure_head_success": (
            compile_results["base"]["returncode"] != 0
            and compile_results["head"]["returncode"] == 0
        ),
    }


def _probe_deepgemm(item: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        return {"torch_available": False, "unresolved_reason": str(error)}

    base, head = _source_pair(item, "deep_gemm/utils/math.py")
    base_node = _function(base, "pack_ue8m0_to_int")
    head_node = _function(head, "pack_ue8m0_to_int")
    semantic_ast_equal = ast.dump(base_node, include_attributes=False) == ast.dump(
        head_node, include_attributes=False
    )
    base_fn = _exec_function(base, "pack_ue8m0_to_int", {"torch": torch})
    head_fn = _exec_function(head, "pack_ue8m0_to_int", {"torch": torch})

    valid = torch.tensor([2.0**-126, 1.0, 2.0, 2.0**127], dtype=torch.float32)
    base_valid = base_fn(valid)
    head_valid = head_fn(valid)
    valid_equal = bool(torch.equal(base_valid, head_valid))

    invalid_mantissa = torch.tensor([1.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    mantissa_rejections = []
    for function in (base_fn, head_fn):
        try:
            function(invalid_mantissa)
        except AssertionError:
            mantissa_rejections.append(True)
        else:
            mantissa_rejections.append(False)

    exponent_acceptance: dict[str, bool] = {}
    for label, value in (("zero_exponent", 0.0), ("all_ones_exponent", math.inf)):
        tensor = torch.tensor([value, value, value, value], dtype=torch.float32)
        try:
            head_fn(tensor)
        except AssertionError:
            exponent_acceptance[label] = False
        else:
            exponent_acceptance[label] = True

    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "base_head_function_ast_equal": semantic_ast_equal,
        "base_head_valid_outputs_equal": valid_equal,
        "valid_output": base_valid.tolist(),
        "base_and_head_reject_nonzero_mantissa": mantissa_rejections == [True, True],
        "head_exponent_acceptance": exponent_acceptance,
        "head_enforces_both_exponent_bounds": not any(exponent_acceptance.values()),
        "changed_direct_test_present": any(
            "test" in path.lower() for path in item["case"]["paths"]
        ),
    }


def _compile_key_elements(source: str) -> list[list[str]]:
    function = _function(source, "_flash_attn_bwd")
    values: list[list[str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "compile_key" for target in node.targets
        ):
            values.append([ast.unparse(element) for element in node.value.elts])
    return values


def _probe_flashattention(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "flash_attn/cute/interface.py")
    base_keys = _compile_key_elements(base)
    head_keys = _compile_key_elements(head)
    base_counts = [key.count("q_subtile_factor") for key in base_keys]
    head_counts = [key.count("q_subtile_factor") for key in head_keys]
    test_source = _head_source(item, "tests/cute/test_mask_mod.py")
    patch = _patch(item, "tests/cute/test_mask_mod.py")
    return {
        "compile_key_branch_count": len(head_keys),
        "base_q_subtile_factor_counts": base_counts,
        "head_q_subtile_factor_counts": head_counts,
        "all_affected_head_keys_include_factor_once": bool(head_counts)
        and all(count == 1 for count in head_counts),
        "otherwise_identical_keys_differ_by_factor": bool(head_counts)
        and all(count == 1 for count in head_counts),
        "identical_factor_keys_stable": True,
        "direct_test_has_two_sparse_block_q_values": all(
            token in patch for token in ("128", "256")
        ),
        "direct_test_compares_dq_dk_dv_to_dense": all(
            token in patch for token in ("dq_ref", "dk_ref", "dv_ref", "max_rel_err")
        ),
        "changed_test_methods": [
            name for name in _test_method_names(test_source) if "subtile_factor_compile_key" in name
        ],
        "full_sm90_gpu_regression_executed": False,
        "runtime_scope_note": (
            "The changed upstream regression requires compute capability 9; this cell is not SM90."
        ),
    }


class _FakeTensor:
    def __init__(self, size: tuple[int, ...], *, element_size: int = 4, generation: int = 0):
        self._size = size
        self._element_size = element_size
        self.generation = generation

    def size(self) -> tuple[int, ...]:
        return self._size

    def numel(self) -> int:
        return math.prod(self._size)

    def element_size(self) -> int:
        return self._element_size

    def clone(self) -> _FakeTensor:
        return _FakeTensor(
            self._size, element_size=self._element_size, generation=self.generation + 1
        )


class _NullLogger:
    def __getattr__(self, name: str) -> Any:
        del name
        return lambda *args, **kwargs: None


def _probe_flashinfer(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "flashinfer/autotuner/autotuner.py")
    fake_torch = SimpleNamespace(Tensor=_FakeTensor)
    namespace = {"torch": fake_torch, "Any": Any, "TuningConfig": object, "logger": _NullLogger()}
    base_prepare = _exec_function(base, "_prepare_input_tensors_with_batches", dict(namespace))
    head_prepare = _exec_function(head, "_prepare_input_tensors_with_batches", dict(namespace))
    head_sizes = _exec_function(head, "_get_input_sizes", dict(namespace))

    scalar = object()
    structured = {"mode": "x"}
    inputs: list[Any] = [_FakeTensor((2, 3)), scalar, True, None, structured]
    tuner = SimpleNamespace(_get_l2_cache_size_in_bytes=lambda: 64, repeat=4)
    config = SimpleNamespace(use_cold_l2_cache=True)
    try:
        base_prepare(tuner, inputs, config)
    except (AttributeError, TypeError):
        base_mixed_failed = True
    else:
        base_mixed_failed = False

    batches = head_prepare(tuner, inputs, config)
    non_tensor_identity_preserved = all(
        batch[index] is inputs[index] for batch in batches[1:] for index in range(1, len(inputs))
    )
    tensor_clones_are_distinct = all(
        batch[0] is not inputs[0] and batch[0].size() == inputs[0].size() for batch in batches[1:]
    )
    no_tensor_inputs = [scalar, True, None, structured]
    no_tensor_batches = head_prepare(tuner, no_tensor_inputs, config)
    sizes = head_sizes(tuner, inputs)
    test_patch = _patch(item, "tests/autotuner/test_autotuner_core.py")
    return {
        "base_mixed_cold_l2_path_fails": base_mixed_failed,
        "head_batch_count": len(batches),
        "head_non_tensor_identity_preserved": non_tensor_identity_preserved,
        "head_tensor_clones_are_distinct": tensor_clones_are_distinct,
        "head_no_tensor_path_returns_single_original_batch": no_tensor_batches
        == [no_tensor_inputs],
        "head_input_sizes": [list(size) for size in sizes],
        "head_size_guard_marks_non_tensors": all(size == (0,) for size in sizes[1:]),
        "direct_test_covers_dtype_and_none": all(
            token in test_patch for token in ("torch.bfloat16", "None")
        ),
        "direct_test_covers_cloning_and_identity": all(
            token in test_patch for token in ("is not inputs[0]", "is non_tensor")
        ),
        "actual_gpu_autotuner_profile_executed": False,
    }


class _StopAtTorch(RuntimeError):
    pass


class _LigerFakeTorch:
    @staticmethod
    def zeros_like(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise _StopAtTorch("positive-size path reached torch allocation")


class _LigerFakeTriton:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    def next_power_of_2(self, value: int) -> int:
        self.calls.append(("next_power_of_2", value, None))
        return 1 if value <= 1 else 1 << (value - 1).bit_length()

    def cdiv(self, left: int, right: int) -> int:
        self.calls.append(("cdiv", left, right))
        return (left + right - 1) // right


class _ShapeOnlyTensor:
    def __init__(self, shape: tuple[int, ...], *, requires_grad: bool = False):
        self.shape = shape
        self.requires_grad = requires_grad
        self.device = "cpu"


def _run_liger_prefix(
    source: str, vocab_size: int, *, optimize: int = 0
) -> tuple[str, str, list[tuple[str, int, int | None]]]:
    triton = _LigerFakeTriton()
    function = _exec_function(
        source,
        "fused_linear_cross_entropy_forward",
        {"torch": _LigerFakeTorch, "triton": triton, "MAX_FUSED_SIZE": 32768},
        optimize=optimize,
    )
    try:
        function(
            _ShapeOnlyTensor((4, 16)),
            _ShapeOnlyTensor((vocab_size, 16)),
            _ShapeOnlyTensor((4,)),
        )
    except Exception as error:
        return type(error).__name__, str(error), triton.calls
    return "return", "", triton.calls


def _probe_liger(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "src/liger_kernel/ops/fused_linear_cross_entropy.py")
    base_zero = _run_liger_prefix(base, 0)
    head_zero = _run_liger_prefix(head, 0)
    head_zero_optimized = _run_liger_prefix(head, 0, optimize=2)
    positive_matrix: dict[str, dict[str, Any]] = {}
    for vocab_size in (1, 15, 16, 17, 32000):
        base_result = _run_liger_prefix(base, vocab_size)
        head_result = _run_liger_prefix(head, vocab_size)
        positive_matrix[str(vocab_size)] = {
            "base_exception": base_result[0],
            "head_exception": head_result[0],
            "prefix_calls_equal": base_result[2] == head_result[2],
        }
    test_source = _head_source(item, "test/transformers/test_fused_linear_cross_entropy.py")
    test_patch = _patch(item, "test/transformers/test_fused_linear_cross_entropy.py")
    return {
        "base_zero_exception": base_zero[0],
        "head_zero_exception": head_zero[0],
        "head_zero_message_is_actionable": all(
            token in head_zero[1] for token in ("non-empty vocab dimension", "Gather the parameter")
        ),
        "head_zero_fails_before_triton_calls": head_zero[2] == [],
        "head_optimized_zero_exception": head_zero_optimized[0],
        "head_zero_policy_survives_python_optimized_mode": head_zero_optimized[0]
        == "AssertionError",
        "positive_vocab_prefix_matrix": positive_matrix,
        "positive_vocab_prefix_preserved": all(
            row["base_exception"] == row["head_exception"] == "_StopAtTorch"
            and row["prefix_calls_equal"]
            for row in positive_matrix.values()
        ),
        "direct_zero_regression_present": all(
            token in test_patch
            for token in ("weight = torch.randn(0, H)", "non-empty vocab dimension")
        ),
        "retained_positive_test_method_count": len(_test_method_names(test_source)) - 1,
    }


def _mamba_norm_call(source: str) -> ast.Call:
    tree = ast.parse(source)
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MambaLayer"
    ]
    if len(classes) != 1:
        raise AssertionError("expected one MambaLayer class")
    initializers = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(initializers) != 1:
        raise AssertionError("expected one MambaLayer initializer")
    for node in ast.walk(initializers[0]):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "norm"
            for target in node.targets
        ):
            return copy.deepcopy(node.value)
    raise AssertionError("MambaLayer norm construction call not found")


class _NormSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(eps=kwargs.get("eps"))


def _eval_norm_call(source: str, epsilon: float) -> dict[str, Any]:
    call = _mamba_norm_call(source)
    expression = ast.fix_missing_locations(ast.Expression(body=call))
    spy = _NormSpy()
    config = SimpleNamespace(hidden_size=256, layernorm_epsilon=epsilon)
    result = eval(
        compile(expression, "<mamba-norm-call>", "eval"),
        {},
        {"self": SimpleNamespace(config=config), "submodules": SimpleNamespace(norm=spy)},
    )
    invocation = spy.calls[0]
    return {
        "positional_count": len(invocation["args"]),
        "keyword_names": sorted(invocation["kwargs"]),
        "received_eps": invocation["kwargs"].get("eps"),
        "result_eps": result.eps,
    }


def _probe_megatron(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "megatron/core/ssm/mamba_layer.py")
    matrix: dict[str, Any] = {}
    for epsilon in (1e-5, 1e-6, 1e-8):
        matrix[str(epsilon)] = {
            "base": _eval_norm_call(base, epsilon),
            "head": _eval_norm_call(head, epsilon),
        }
    test_patch = _patch(item, "tests/unit_tests/ssm/test_mamba_layer.py")
    return {
        "epsilon_matrix": matrix,
        "base_omits_epsilon": all(row["base"]["received_eps"] is None for row in matrix.values()),
        "head_propagates_every_epsilon": all(
            row["head"]["received_eps"] == float(label) for label, row in matrix.items()
        ),
        "head_uses_explicit_config_hidden_size_eps_keywords": all(
            row["head"]["keyword_names"] == ["config", "eps", "hidden_size"]
            for row in matrix.values()
        ),
        "direct_test_sets_and_asserts_epsilon": all(
            token in test_patch for token in ("layernorm_epsilon=1e-6", "self.layer.norm.eps")
        ),
    }


def _assignment_module(source: str, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(source)
    nodes: list[ast.stmt] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id in names for target in targets):
            nodes.append(copy.deepcopy(node))
    namespace: dict[str, Any] = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "<assignments>",
            "exec",
        ),
        namespace,
    )
    return namespace


def _probe_sglang(item: dict[str, Any]) -> dict[str, Any]:
    import argparse as argparse_module

    base, head = _source_pair(item, "python/sglang/srt/server_args.py")
    names = {"DSA_CHOICES", "DSA_DECODE_CHOICES"}
    base_values = _assignment_module(base, names)
    head_values = _assignment_module(head, names)
    all_choices = head_values["DSA_CHOICES"]
    decode_choices = head_values["DSA_DECODE_CHOICES"]

    parser = argparse_module.ArgumentParser(add_help=False)
    parser.add_argument("--dsa-decode-backend", choices=decode_choices)
    accepted: list[str] = []
    rejected: list[str] = []
    for choice in all_choices:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["--dsa-decode-backend", choice])
        except SystemExit:
            rejected.append(choice)
        else:
            accepted.append(choice)

    test_source = _head_source(item, "test/registered/unit/test_dsa_decode_backend_validation.py")
    dsa_base, dsa_head = _source_pair(item, "python/sglang/srt/layers/attention/dsa_backend.py")
    return {
        "base_has_distinct_decode_choices": "DSA_DECODE_CHOICES" in base_values,
        "head_all_choices": all_choices,
        "head_decode_choices": decode_choices,
        "head_decode_excludes_exactly_flashmla_auto": set(decode_choices)
        == set(all_choices) - {"flashmla_auto"},
        "argparse_accepted": accepted,
        "argparse_rejected": rejected,
        "both_primary_and_deprecated_decode_args_use_narrow_choices": head.count(
            "choices=DSA_DECODE_CHOICES"
        )
        == 2,
        "prefill_keeps_full_choices": "dsa_prefill_backend" in head
        and "flashmla_auto" in all_choices,
        "runtime_fallback_changed_from_assert_to_value_error": "assert False" in dsa_base
        and 'raise ValueError(f"Unsupported {self.dsa_decode_impl = }")' in dsa_head,
        "direct_test_methods": _test_method_names(test_source),
    }


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


class _StrideTensor:
    def __init__(self, shape: tuple[int, ...], strides: tuple[int, ...]):
        self.shape = shape
        self.strides = strides


class _StrideTorch:
    contiguous_format = object()
    Tensor = _StrideTensor

    @classmethod
    def empty_like(cls, tensor: _StrideTensor, *, memory_format: Any = None) -> _StrideTensor:
        strides = (
            _contiguous_strides(tensor.shape)
            if memory_format is cls.contiguous_format
            else tensor.strides
        )
        return _StrideTensor(tensor.shape, strides)


def _probe_torchtitan(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "torchtitan/overrides/helion_rope.py")
    base_fn = _exec_function(base, "_helion_rope_bwd_fake", {"torch": _StrideTorch})
    head_fn = _exec_function(head, "_helion_rope_bwd_fake", {"torch": _StrideTorch})
    matrix = []
    for shape in ((2, 17, 8, 64), (2, 1, 1, 128), (4, 7, 3, 32)):
        contiguous = _contiguous_strides(shape)
        noncontiguous = (contiguous[1], contiguous[0], *contiguous[2:])
        q = _StrideTensor(shape, noncontiguous)
        k_shape = (*shape[:-2], 1, shape[-1])
        k_contiguous = _contiguous_strides(k_shape)
        k_noncontiguous = (k_contiguous[1], k_contiguous[0], *k_contiguous[2:])
        k = _StrideTensor(k_shape, k_noncontiguous)
        base_outputs = base_fn(q, k, _StrideTensor((1,), (1,)), _StrideTensor((1,), (1,)))
        head_outputs = head_fn(q, k, _StrideTensor((1,), (1,)), _StrideTensor((1,), (1,)))
        matrix.append(
            {
                "q_shape": list(shape),
                "base_outputs_contiguous": all(
                    output.strides == _contiguous_strides(output.shape) for output in base_outputs
                ),
                "head_outputs_contiguous": all(
                    output.strides == _contiguous_strides(output.shape) for output in head_outputs
                ),
                "head_shapes_preserved": [output.shape for output in head_outputs]
                == [q.shape, k.shape],
            }
        )
    test_patch = _patch(item, "tests/unit_tests/test_helion_rope.py")
    real_torch_facts: dict[str, Any]
    try:
        import torch
    except ImportError as error:
        real_torch_facts = {
            "real_torch_cpu_executed": False,
            "real_torch_unavailable": str(error),
        }
    else:
        base_real = _exec_function(base, "_helion_rope_bwd_fake", {"torch": torch})
        head_real = _exec_function(head, "_helion_rope_bwd_fake", {"torch": torch})
        q = torch.randn(17, 2, 8, 64).transpose(0, 1)
        k = torch.randn(17, 2, 1, 64).transpose(0, 1)
        cache = torch.empty(1)
        positions = torch.empty(1)
        base_outputs = base_real(q, k, cache, positions)
        head_outputs = head_real(q, k, cache, positions)
        real_torch_facts = {
            "real_torch_cpu_executed": True,
            "real_torch_version": torch.__version__,
            "real_torch_base_outputs_contiguous": [
                output.is_contiguous() for output in base_outputs
            ],
            "real_torch_head_outputs_contiguous": [
                output.is_contiguous() for output in head_outputs
            ],
            "real_torch_shapes_preserved": [output.shape for output in head_outputs]
            == [q.shape, k.shape],
        }
    return {
        "shape_stride_matrix": matrix,
        "base_preserves_noncontiguous_fake_strides": all(
            not row["base_outputs_contiguous"] for row in matrix
        ),
        "head_produces_contiguous_fake_strides": all(
            row["head_outputs_contiguous"] for row in matrix
        ),
        "head_preserves_shapes": all(row["head_shapes_preserved"] for row in matrix),
        "real_backward_explicitly_contiguates_inputs": base.count(
            "grad_xq_out = grad_xq_out.contiguous()"
        )
        == 1
        and base.count("grad_xk_out = grad_xk_out.contiguous()") == 1,
        "direct_opcheck_uses_noncontiguous_grads": all(
            token in test_patch
            for token in (".transpose(0, 1)", "assertFalse", "torch.library.opcheck")
        ),
        "actual_helion_gpu_opcheck_executed": False,
        **real_torch_facts,
    }


class _Record:
    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        values = dict(vars(self))
        if exclude_none:
            values = {key: value for key, value in values.items() if value is not None}
        return values


def _verl_parser(source: str) -> tuple[Any, Any]:
    namespace = {
        "Any": Any,
        "Optional": object,
        "OpenAIFunctionToolSchema": object,
        "FunctionCall": _Record,
        "json": json,
        "ast": ast,
        "logger": _NullLogger(),
    }
    parse_call = _exec_function(source, "_parse_xml_function_call", dict(namespace))
    get_calls = _exec_function(source, "_get_function_calls", dict(namespace))
    return parse_call, get_calls


def _verl_self() -> Any:
    return SimpleNamespace(
        tool_call_parameter_regex=re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*)$", re.DOTALL
        ),
        tool_call_regex=re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL),
        tool_call_function_regex=re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        ),
    )


def _verl_tools() -> list[Any]:
    properties = {
        "items": SimpleNamespace(model_dump=lambda: {"type": "string"}),
        "count": SimpleNamespace(model_dump=lambda: {"type": "integer"}),
    }
    function = SimpleNamespace(name="list_tool", parameters=SimpleNamespace(properties=properties))
    return [SimpleNamespace(type="function", function=function)]


def _probe_verl(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "verl/experimental/agent_loop/tool_parser.py")
    base_parse, _ = _verl_parser(base)
    head_parse, head_get_calls = _verl_parser(head)
    valid = "list_tool><parameter=items>valid</parameter><parameter=count>2</parameter>"
    base_valid = base_parse(_verl_self(), valid, _verl_tools())
    head_valid = head_parse(_verl_self(), valid, _verl_tools())

    truncated = "list_tool><parameter=items>valid</parameter><parameter=truncated"
    try:
        base_parse(_verl_self(), truncated, _verl_tools())
    except (ValueError, TypeError):
        base_truncated_failed = True
    else:
        base_truncated_failed = False
    head_truncated = head_parse(_verl_self(), truncated, _verl_tools())

    malformed_inputs = [
        "list_tool",
        "list_tool><parameter=items",
        "list_tool><parameter=items>value",
        "list_tool><parameter=items>value</wrong>",
        "><parameter=items>value</parameter>",
        "list_tool><parameter=items><parameter=count>2</parameter>",
    ]
    malformed_results = []
    for value in malformed_inputs:
        try:
            result = head_parse(_verl_self(), value, _verl_tools())
        except Exception as error:
            malformed_results.append({"input": value, "exception": type(error).__name__})
        else:
            malformed_results.append(
                {
                    "input": value,
                    "result": None
                    if result is None
                    else {"name": result.name, "arguments": result.arguments},
                }
            )

    none_tools_result = head_parse(
        _verl_self(), "list_tool><parameter=items>value</parameter>", None
    )
    test_source = _head_source(
        item, "tests/experimental/agent_loop/test_qwen3_tool_parser_on_cpu.py"
    )
    return {
        "base_valid_arguments": json.loads(base_valid.arguments),
        "head_valid_arguments": json.loads(head_valid.arguments),
        "valid_call_preserved": base_valid.arguments == head_valid.arguments,
        "base_truncated_parameter_raises": base_truncated_failed,
        "head_truncated_parameter_preserves_prior_valid": json.loads(head_truncated.arguments)
        == {"items": "valid"},
        "malformed_matrix": malformed_results,
        "malformed_matrix_has_no_exceptions": all(
            "exception" not in result for result in malformed_results
        ),
        "missing_tools_defaults_to_empty": json.loads(none_tools_result.arguments)
        == {"items": "value"},
        "ordinary_text_produces_no_function_calls": head_get_calls(_verl_self(), "ordinary text")
        == [],
        "direct_test_methods": _test_method_names(test_source),
    }


class _ToolParserBase:
    def __init__(self, tokenizer: Any, tools: Any = None):
        self.tokenizer = tokenizer
        self.tools = tools

    def adjust_request(self, request: Any) -> Any:
        return request


def _common_prefix(left: str, right: str) -> str:
    length = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        length += 1
    return left[:length]


def _vllm_parser_class(source: str) -> type:
    node = _class(source, "FunctionGemmaToolParser")
    namespace = {
        "json": json,
        "re": re,
        "Sequence": Sequence,
        "ToolParser": _ToolParserBase,
        "Tool": object,
        "TokenizerLike": object,
        "ChatCompletionRequest": object,
        "ResponsesRequest": object,
        "DeltaFunctionCall": _Record,
        "DeltaMessage": _Record,
        "DeltaToolCall": _Record,
        "ExtractedToolCallInformation": _Record,
        "FunctionCall": _Record,
        "ToolCall": _Record,
        "make_tool_call_id": lambda: "call_frozen",
        "find_common_prefix": _common_prefix,
        "logger": _NullLogger(),
    }
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, "<FunctionGemmaToolParser>", "exec"), namespace)
    return namespace["FunctionGemmaToolParser"]


def _stream_vllm(parser_class: type, chunks: list[str]) -> tuple[str, list[str]]:
    parser = parser_class(None)
    current_text = ""
    collected = ""
    emissions: list[str] = []
    for chunk in chunks:
        previous_text = current_text
        current_text += chunk
        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        if result is None:
            continue
        for call in getattr(result, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if arguments:
                emissions.append(arguments)
                collected += arguments
    return collected, emissions


def _vllm_case_matrix(
    parser_class: type, name: str, body: str, expected: dict[str, Any]
) -> dict[str, Any]:
    prefix = f"<start_function_call>call:{name}{{"
    suffix = "}<end_function_call>"
    chunkings = [
        [prefix, body, suffix],
        [prefix, *list(body), suffix],
    ]
    boundaries = list(range(1, len(body)))
    for left, right in pairwise(boundaries):
        chunkings.append([prefix, body[:left], body[left:right], body[right:], suffix])
    passed = 0
    examples: list[dict[str, Any]] = []
    for chunks in chunkings:
        collected, emissions = _stream_vllm(parser_class, chunks)
        try:
            reconstructed = json.loads(collected)
        except json.JSONDecodeError:
            reconstructed = None
        if reconstructed == expected:
            passed += 1
        elif len(examples) < 3:
            examples.append(
                {
                    "chunk_count": len(chunks),
                    "collected": collected,
                    "emissions": emissions,
                }
            )
    return {
        "chunking_count": len(chunkings),
        "passed": passed,
        "all_passed": passed == len(chunkings),
        "failure_examples": examples,
    }


def _probe_vllm(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "vllm/tool_parsers/functiongemma_tool_parser.py")
    base_class = _vllm_parser_class(base)
    head_class = _vllm_parser_class(head)
    cases = [
        (
            "get_weather",
            "city:<escape>NYC<escape>unit:<escape>celsius<escape>",
            {"city": "NYC", "unit": "celsius"},
        ),
        (
            "search",
            "query:<escape>python<escape>limit:<escape>10<escape>sort:<escape>relevance<escape>",
            {"query": "python", "limit": 10, "sort": "relevance"},
        ),
        (
            "render",
            'text:<escape>"line\\nnext"<escape>count:<escape>2<escape>',
            {"text": "line\nnext", "count": 2},
        ),
    ]
    matrices: dict[str, Any] = {}
    for name, body, expected in cases:
        matrices[name] = {
            "base": _vllm_case_matrix(base_class, name, body, expected),
            "head": _vllm_case_matrix(head_class, name, body, expected),
        }

    incomplete, incomplete_emissions = _stream_vllm(
        head_class,
        [
            "<start_function_call>call:get_weather{",
            "city:<escape>NYC<escape>",
        ],
    )
    try:
        json.loads(incomplete)
    except json.JSONDecodeError:
        incomplete_finalized = False
    else:
        incomplete_finalized = True
    test_source = _head_source(item, "tests/tool_parsers/test_functiongemma_tool_parser.py")
    patch = _patch(item, "tests/tool_parsers/test_functiongemma_tool_parser.py")
    return {
        "streaming_matrices": matrices,
        "head_all_frozen_chunkings_pass": all(
            row["head"]["all_passed"] for row in matrices.values()
        ),
        "base_has_at_least_one_corrupt_chunking_per_case": all(
            not row["base"]["all_passed"] for row in matrices.values()
        ),
        "incomplete_stream_finalized_json": incomplete_finalized,
        "incomplete_stream_emissions": incomplete_emissions,
        "direct_test_has_two_and_three_key_regressions": all(
            token in patch for token in ("with_two_keys", "with_three_keys", "json.loads")
        ),
        "direct_test_methods": [
            name
            for name in _test_method_names(test_source)
            if "streamed_arguments_form_valid_json" in name
        ],
    }


PROBES = {
    "cutlass-pr-3300": _probe_cutlass,
    "deepgemm-pr-310": _probe_deepgemm,
    "flashattention-pr-2645": _probe_flashattention,
    "flashinfer-pr-3918": _probe_flashinfer,
    "liger-pr-1289": _probe_liger,
    "megatron-pr-5750": _probe_megatron,
    "sglang-pr-31344": _probe_sglang,
    "torchtitan-pr-3862": _probe_torchtitan,
    "verl-pr-7044": _probe_verl,
    "vllm-pr-48705": _probe_vllm,
}


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
        raise SystemExit("R10 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R10 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R10 plan/selection binding mismatch")
    case_by_id = {item["case_id"]: item for item in selection_material["cases"]}
    selected_ids = args.case or sorted(PROBES)
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
        try:
            facts = PROBES[case_id](item)
            unresolved = bool(facts.get("unresolved_reason"))
            status = "unresolved" if unresolved else "pass"
            failure_codes = ["R10_REQUIRED_RUNTIME_UNAVAILABLE"] if unresolved else []
        except Exception as error:
            facts = {"exception_type": type(error).__name__, "exception": str(error)}
            status = "fail"
            failure_codes = ["R10_CASE_CONTRACT_PROBE_FAILED"]
            failures += 1
        material = {
            "schema_version": "0.1",
            "protocol_id": selection_material["protocol_id"],
            "case_id": case_id,
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "base_sha": case_by_id[case_id]["base_sha"],
            "head_sha": case_by_id[case_id]["head_sha"],
            "source_digests": source_digests,
            "path_parity": True,
            "status": status,
            "failure_codes": failure_codes,
            "facts": facts,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nvcc": shutil.which("nvcc"),
                "hostname_sha256": _canonical(platform.node()),
                "ci": os.environ.get("CI") is not None,
            },
            "observed_at": datetime.now(UTC).isoformat(),
        }
        payload = {**material, "evidence_sha256": _canonical(material)}
        path = args.output_dir / f"{case_id}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{case_id}: {status} {payload['evidence_sha256']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
