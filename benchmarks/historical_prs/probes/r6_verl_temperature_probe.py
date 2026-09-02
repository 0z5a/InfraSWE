#!/usr/bin/env python3
"""Execute the exact verl temperature-gating AST without importing verl."""

from __future__ import annotations

import argparse
import ast
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _method(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "DataParallelPPOActor":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "_forward_micro_batch":
                return child
    raise ValueError("missing DataParallelPPOActor._forward_micro_batch")


def _actor_calls(method: ast.FunctionDef) -> list[ast.Call]:
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "actor_module"
    ]
    return sorted(calls, key=lambda node: node.lineno)


def _keyword_shape(call: ast.Call) -> list[str]:
    return [
        f"**{ast.unparse(item.value)}" if item.arg is None else item.arg for item in call.keywords
    ]


def _compile_head_gate(method: ast.FunctionDef):
    gate_nodes: list[ast.stmt] = []
    for statement in method.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "extra_forward_kwargs"
            for target in statement.targets
        ):
            gate_nodes.append(statement)
        elif (
            gate_nodes
            and isinstance(statement, ast.If)
            and "self.use_fused_kernels" in ast.unparse(statement.test)
        ):
            gate_nodes.append(statement)
            break
    if len(gate_nodes) != 2:
        raise ValueError("could not isolate exact extra_forward_kwargs gate")
    function = ast.FunctionDef(
        name="exact_gate",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="temperature")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *gate_nodes,
            ast.Return(value=ast.Name(id="extra_forward_kwargs", ctx=ast.Load())),
        ],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, "<exact-verl-head-gate>", "exec"), namespace)
    return namespace["exact_gate"], ast.unparse(module)


def _accepts_nonfused(**kwargs: Any) -> dict[str, Any]:
    allowed = {"input_ids", "attention_mask", "position_ids", "use_cache"}
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise TypeError(f"unexpected keyword arguments: {sorted(unexpected)}")
    return kwargs


def _accepts_fused(*, temperature: float, **kwargs: Any) -> dict[str, Any]:
    return {**kwargs, "temperature": temperature}


def _exercise_branch(extra: dict[str, Any], fused: bool) -> dict[str, Any]:
    fixed = {
        "input_ids": "ids",
        "attention_mask": "mask",
        "position_ids": "positions",
        "use_cache": False,
    }
    try:
        result = (_accepts_fused if fused else _accepts_nonfused)(**fixed, **extra)
    except TypeError as exc:
        return {"status": "fail", "error": str(exc)}
    return {"status": "pass", "keywords": sorted(result)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--head-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    base_source = args.base_source.read_text(encoding="utf-8")
    head_source = args.head_source.read_text(encoding="utf-8")
    base_method = _method(ast.parse(base_source))
    head_method = _method(ast.parse(head_source))
    base_calls = _actor_calls(base_method)
    head_calls = _actor_calls(head_method)
    if len(base_calls) != 2 or len(head_calls) != 2:
        raise ValueError("expected remove-padding and padded actor-module calls")

    exact_gate, exact_gate_source = _compile_head_gate(head_method)
    temperature = 0.37
    fused_kwargs = exact_gate(SimpleNamespace(use_fused_kernels=True), temperature)
    nonfused_kwargs = exact_gate(SimpleNamespace(use_fused_kernels=False), temperature)
    base_always_kwargs = {"temperature": temperature}
    branch_results = {
        "base_fused": _exercise_branch(base_always_kwargs, fused=True),
        "base_nonfused": _exercise_branch(base_always_kwargs, fused=False),
        "head_fused": _exercise_branch(fused_kwargs, fused=True),
        "head_nonfused": _exercise_branch(nonfused_kwargs, fused=False),
    }
    head_other_keywords_preserved = all(
        [key for key in _keyword_shape(base) if key != "temperature"]
        == [key for key in _keyword_shape(head) if key != "**extra_forward_kwargs"]
        for base, head in zip(base_calls, head_calls, strict=True)
    )
    gate_is_per_call_local = any(
        isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "extra_forward_kwargs"
            for target in statement.targets
        )
        for statement in head_method.body
    )
    passes = (
        branch_results["head_fused"]["status"] == "pass"
        and branch_results["head_nonfused"]["status"] == "pass"
        and branch_results["base_nonfused"]["status"] == "fail"
        and fused_kwargs == {"temperature": temperature}
        and nonfused_kwargs == {}
        and head_other_keywords_preserved
        and gate_is_per_call_local
    )
    failure_codes = [] if passes else ["VERL_TEMPERATURE_GATE_SEMANTICS_FAILED"]
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "verl-temperature-exact-ast-v1",
        "case_id": "verl-pr-1688",
        "status": "pass" if passes else "fail",
        "failure_codes": failure_codes,
        "facts": {
            "branch_results": branch_results,
            "fused_kwargs": fused_kwargs,
            "nonfused_kwargs": nonfused_kwargs,
            "two_head_call_sites_expand_gate_kwargs": all(
                "**extra_forward_kwargs" in _keyword_shape(call) for call in head_calls
            ),
            "other_keywords_preserved": head_other_keywords_preserved,
            "gate_is_per_call_local": gate_is_per_call_local,
            "exact_gate_source": exact_gate_source,
            "compilation_path": "python-ast-only",
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": {
            "base_source_sha256": canonical_sha256(base_source),
            "head_source_sha256": canonical_sha256(head_source),
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
