#!/usr/bin/env python3
"""Compile-free capability and layout probe for vLLM PR 13996."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"missing function {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    source_path = Path("vllm/model_executor/layers/quantization/utils/fp8_utils.py")
    source = (args.head_root / source_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = _function(tree, "apply_w8a8_block_fp8_linear")
    deepgemm_branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and "VLLM_USE_DEEPGEMM" in ast.unparse(node.test)
    ]
    if len(deepgemm_branches) != 1:
        raise SystemExit("expected one DeepGEMM selection branch")
    branch = deepgemm_branches[0]
    condition = ast.unparse(branch.test)
    condition_lower = condition.lower()
    architecture_gate_present = any(
        marker in condition_lower
        for marker in ("capability", "is_hopper", "sm90", "sm_90", "device")
    )
    availability_gate_present = any(
        marker in condition_lower for marker in ("is_available", "find_spec", "has_deep_gemm")
    )
    imports_deepgemm_directly = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name == "deep_gemm" for alias in node.names)
            if isinstance(node, ast.Import)
            else node.module == "deep_gemm"
        )
        for statement in branch.body
        for node in ast.walk(statement)
    )
    quant_calls = [
        node
        for statement in branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "per_token_group_quant_fp8"
    ]
    column_major_requested = any(
        keyword.arg == "column_major_scales"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for call in quant_calls
        for keyword in call.keywords
    )
    precompile_contract_present = any(
        marker in source.lower()
        for marker in ("deepgemm_precompile", "deep_gemm_precompile", "warmup_deepgemm")
    )

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    oracle_material = dict(oracle)
    oracle_sha = oracle_material.pop("oracle_sha256")
    if canonical_sha256(oracle_material) != oracle_sha:
        raise SystemExit("DeepGEMM oracle digest mismatch")
    device_capability = list(torch.cuda.get_device_capability(0))
    device_is_supported = device_capability[0] >= 9
    deepgemm_installed = importlib.util.find_spec("deep_gemm") is not None

    failure_codes: list[str] = []
    if not architecture_gate_present:
        failure_codes.append("DEEPGEMM_HOPPER_CAPABILITY_GATE_MISSING")
    if not availability_gate_present and imports_deepgemm_directly:
        failure_codes.append("DEEPGEMM_OPTIONAL_DEPENDENCY_FALLBACK_MISSING")
    if not column_major_requested:
        failure_codes.append("DEEPGEMM_LHS_SCALE_LAYOUT_CONTRACT_VIOLATED")
    if not precompile_contract_present:
        failure_codes.append("DEEPGEMM_RUNTIME_JIT_PRECOMPILE_CONTRACT_MISSING")

    material = {
        "schema_version": "0.5",
        "probe": "vllm-deepgemm-capability-layout-v1",
        "case_id": "vllm-pr-13996",
        "status": "fail" if failure_codes else "pass",
        "failure_codes": failure_codes,
        "facts": {
            "selection_condition": condition,
            "architecture_gate_present": architecture_gate_present,
            "optional_dependency_availability_gate_present": availability_gate_present,
            "imports_deepgemm_directly": imports_deepgemm_directly,
            "column_major_tma_scale_requested": column_major_requested,
            "runtime_jit_precompile_contract_present": precompile_contract_present,
            "device_compute_capability": device_capability,
            "device_supported_by_historical_deepgemm": device_is_supported,
            "deepgemm_installed": deepgemm_installed,
            "early_exit_before_backend_import_or_jit": True,
            "compilation_path": "not-required-after-capability-failure",
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": {
            "head_fp8_utils_sha256": canonical_sha256(source),
            "oracle_sha256": oracle_sha,
            "oracle_commit_sha": oracle["commit_sha"],
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
