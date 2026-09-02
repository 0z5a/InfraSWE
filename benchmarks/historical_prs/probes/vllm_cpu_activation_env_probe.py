#!/usr/bin/env python3
"""Compile-free source-contract probe for vLLM PR 12111."""

from __future__ import annotations

import argparse
import ast
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"missing class {name}")


def _method(owner: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in owner.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"missing method {owner.name}.{name}")


def _loads_self_op(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Attribute)
        and item.attr == "op"
        and isinstance(item.value, ast.Name)
        and item.value.id == "self"
        and isinstance(item.ctx, ast.Load)
        for item in ast.walk(node)
    )


def _assigns_self_op_for_cpu(node: ast.AST) -> bool:
    for branch in ast.walk(node):
        if not isinstance(branch, ast.If):
            continue
        condition = ast.unparse(branch.test)
        if "current_platform.is_cpu()" not in condition:
            continue
        if any(
            isinstance(item, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "op"
                for target in (item.targets if isinstance(item, ast.Assign) else [item.target])
            )
            for statement in branch.body
            for item in ast.walk(statement)
        ):
            return True
    return False


def _forward_cpu_delegates_to_cuda(custom_op_tree: ast.Module) -> bool:
    method = _method(_class(custom_op_tree, "CustomOp"), "forward_cpu")
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and isinstance(item.func.value, ast.Name)
        and item.func.value.id == "self"
        and item.func.attr == "forward_cuda"
        for item in ast.walk(method)
    )


def _environment_deletions_without_restore(tree: ast.Module) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        source = ast.unparse(node)
        deletes_visibility = (
            "del os.environ['CUDA_VISIBLE_DEVICES']" in source
            or 'del os.environ["CUDA_VISIBLE_DEVICES"]' in source
        )
        scoped_restore = "monkeypatch" in source or ("try:" in source and "finally:" in source)
        if deletes_visibility and not scoped_restore:
            findings.append(f"{node.name}:{node.lineno}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    activation_path = Path("vllm/model_executor/layers/activation.py")
    custom_op_path = Path("vllm/model_executor/custom_op.py")
    test_path = Path("tests/lora/test_long_context.py")
    base_activation_source = (args.base_root / activation_path).read_text(encoding="utf-8")
    head_activation_source = (args.head_root / activation_path).read_text(encoding="utf-8")
    custom_op_source = (args.head_root / custom_op_path).read_text(encoding="utf-8")
    head_test_source = (args.head_root / test_path).read_text(encoding="utf-8")
    base_tree = ast.parse(base_activation_source)
    head_tree = ast.parse(head_activation_source)
    custom_op_tree = ast.parse(custom_op_source)
    test_tree = ast.parse(head_test_source)

    delegates = _forward_cpu_delegates_to_cuda(custom_op_tree)
    classes: dict[str, dict[str, bool]] = {}
    for class_name in ("FatreluAndMul", "MulAndSilu"):
        base_class = _class(base_tree, class_name)
        head_class = _class(head_tree, class_name)
        classes[class_name] = {
            "base_assigns_self_op_for_cpu": _assigns_self_op_for_cpu(
                _method(base_class, "__init__")
            ),
            "head_assigns_self_op_for_cpu": _assigns_self_op_for_cpu(
                _method(head_class, "__init__")
            ),
            "head_forward_cuda_loads_self_op": _loads_self_op(_method(head_class, "forward_cuda")),
        }

    uninitialized = [
        name
        for name, facts in classes.items()
        if delegates
        and facts["base_assigns_self_op_for_cpu"]
        and not facts["head_assigns_self_op_for_cpu"]
        and facts["head_forward_cuda_loads_self_op"]
    ]
    environment_leaks = _environment_deletions_without_restore(test_tree)
    failure_codes: list[str] = []
    if uninitialized:
        failure_codes.append("CPU_CUSTOM_OP_ATTRIBUTE_UNINITIALIZED")
    if environment_leaks:
        failure_codes.append("PROCESS_ENV_MUTATION_NOT_RESTORED")

    material = {
        "schema_version": "0.5",
        "probe": "vllm-cpu-activation-env-contract-v1",
        "case_id": "vllm-pr-12111",
        "status": "fail" if failure_codes else "pass",
        "failure_codes": failure_codes,
        "facts": {
            "custom_op_forward_cpu_delegates_to_forward_cuda": delegates,
            "activation_classes": classes,
            "classes_with_uninitialized_cpu_op": uninitialized,
            "environment_deletions_without_restore": environment_leaks,
            "compilation_path": "not-required",
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": {
            "base_activation_sha256": canonical_sha256(base_activation_source),
            "head_activation_sha256": canonical_sha256(head_activation_source),
            "custom_op_sha256": canonical_sha256(custom_op_source),
            "head_test_sha256": canonical_sha256(head_test_source),
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
