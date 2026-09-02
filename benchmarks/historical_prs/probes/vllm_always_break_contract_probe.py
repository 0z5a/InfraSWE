#!/usr/bin/env python3
"""Execute extracted decorator logic and audit integration coverage for vLLM #52205."""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTensor:
    pass


class FakeTorch:
    Tensor = FakeTensor


class Mode:
    FULL = "FULL"
    PIECEWISE = "PIECEWISE"


class Capture:
    active: Capture | None = None

    def __init__(self) -> None:
        self._capturing = True
        self.eager_calls = 0

    @classmethod
    def current(cls) -> Capture | None:
        return cls.active

    def add_eager(self, callback: Callable[[], Any]) -> Any:
        self.eager_calls += 1
        return callback()


def _extract(path: Path) -> Callable[..., Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "eager_break_during_capture"
    ]
    function = candidates[-1]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    context = SimpleNamespace(cudagraph_runtime_mode=Mode.FULL)
    namespace: dict[str, Any] = {
        "functools": functools,
        "is_breakable_cudagraph_enabled": lambda: True,
        "BreakableCUDAGraphCapture": Capture,
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: context,
        "CUDAGraphMode": Mode,
        "torch": FakeTorch,
        "weak_ref_tensor": lambda value: value,
    }
    exec(compile(module, str(path), "exec"), namespace)
    result = namespace["eager_break_during_capture"]
    result._probe_context = context
    return result


def _run(decorator: Callable[..., Any], mode: str, *, always_break: bool | None) -> dict[str, Any]:
    context = decorator._probe_context
    context.cudagraph_runtime_mode = mode
    capture = Capture()
    Capture.active = capture
    calls = 0

    def operation(value: int) -> int:
        nonlocal calls
        calls += 1
        return value + 1

    wrapped = (
        decorator(operation)
        if always_break is None
        else decorator(always_break=always_break)(operation)
    )
    try:
        value = wrapped(4)
    finally:
        Capture.active = None
    return {"result": value, "operation_calls": calls, "eager_calls": capture.eager_calls}


def _changed_files(base_root: Path, head_root: Path) -> list[str]:
    base_sha = subprocess.run(
        ["git", "-C", str(base_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = subprocess.run(
        ["git", "-C", str(head_root), "diff", "--name-only", base_sha, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return [line for line in result.stdout.splitlines() if line]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    relative = Path("vllm/compilation/breakable_cudagraph.py")
    base_decorator = _extract(args.base_root / relative)
    head_decorator = _extract(args.head_root / relative)

    behavior = {
        "base_full_legacy": _run(base_decorator, Mode.FULL, always_break=None),
        "base_piecewise_legacy": _run(base_decorator, Mode.PIECEWISE, always_break=None),
        "head_full_legacy": _run(head_decorator, Mode.FULL, always_break=None),
        "head_piecewise_legacy": _run(head_decorator, Mode.PIECEWISE, always_break=None),
        "head_full_always_break": _run(head_decorator, Mode.FULL, always_break=True),
    }
    changed = _changed_files(args.base_root, args.head_root)
    tests_with_always_break = []
    for path in (args.head_root / "tests").rglob("*.py"):
        if "always_break" in path.read_text(encoding="utf-8", errors="replace"):
            tests_with_always_break.append(str(path.relative_to(args.head_root)))
    amd_callsite = args.head_root / "vllm/models/kimi_k3/amd/kda.py"
    nvidia_callsite = args.head_root / "vllm/models/kimi_k3/nvidia/kda.py"
    payload = {
        "schema_version": "0.5",
        "probe": "vllm-always-break-contract-r1",
        "behavior": behavior,
        "behavior_findings": {
            "legacy_full_behavior_preserved": behavior["head_full_legacy"]["eager_calls"] == 0,
            "legacy_piecewise_still_breaks": behavior["head_piecewise_legacy"]["eager_calls"] == 1,
            "always_break_routes_full_to_eager": behavior["head_full_always_break"]["eager_calls"]
            == 1,
        },
        "integration": {
            "changed_files": changed,
            "changed_test_files": [path for path in changed if path.startswith("tests/")],
            "tests_referencing_always_break": tests_with_always_break,
            "amd_kda_opts_in": "@eager_break_during_capture(always_break=True)"
            in amd_callsite.read_text(encoding="utf-8"),
            "nvidia_kda_opts_in": "@eager_break_during_capture(always_break=True)"
            in nvidia_callsite.read_text(encoding="utf-8"),
            "runtime_benchmark_added": any("bench" in path.lower() for path in changed),
        },
        "source_sha256": {
            "base_decorator": _sha256(args.base_root / relative),
            "head_decorator": _sha256(args.head_root / relative),
            "head_amd_kda": _sha256(amd_callsite),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
