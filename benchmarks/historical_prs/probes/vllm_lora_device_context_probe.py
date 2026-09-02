#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from functools import lru_cache
from pathlib import Path

import torch

LAUNCHER_FILES = [
    "bgmv_expand.py",
    "bgmv_expand_slice.py",
    "bgmv_shrink.py",
    "sgmv_expand.py",
    "sgmv_shrink.py",
]


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    root = options.worktree / "vllm" / "lora" / "ops" / "triton_ops"
    utils_tree = ast.parse((root / "utils.py").read_text(encoding="utf-8"))
    helper = find_function(utils_tree, "_set_cuda_device")

    launcher_results: dict[str, dict[str, object]] = {}
    for filename in LAUNCHER_FILES:
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        setter_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_set_cuda_device"
        ]
        kernel_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
        ]
        launcher_results[filename] = {
            "setter_lines": sorted(setter_lines),
            "kernel_launch_lines": sorted(kernel_lines),
            "setter_precedes_kernel": bool(setter_lines and kernel_lines)
            and min(setter_lines) < min(kernel_lines),
        }

    helper_contract = False
    exercised_devices: list[int] = []
    runtime_failure: str | None = None
    if helper is not None:
        helper_calls = [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_device"
        ]
        helper_contract = bool(helper_calls)
        if helper_contract:
            try:
                isolated = ast.Module(body=[helper], type_ignores=[])
                ast.fix_missing_locations(isolated)
                namespace = {"lru_cache": lru_cache, "torch": torch}
                exec(compile(isolated, str(root / "utils.py"), "exec"), namespace)
                original_device = torch.cuda.current_device()
                target_devices = list(range(min(torch.cuda.device_count(), 2)))
                for device_index in reversed(target_devices):
                    namespace["_set_cuda_device"](torch.device(f"cuda:{device_index}"))
                    if torch.cuda.current_device() != device_index:
                        raise RuntimeError(f"device context did not switch to {device_index}")
                    exercised_devices.append(device_index)
                torch.cuda.set_device(original_device)
            except Exception as error:  # pragma: no cover - evidence path
                runtime_failure = f"{type(error).__name__}:{error}"

    all_launchers_guarded = all(
        bool(result["setter_precedes_kernel"]) for result in launcher_results.values()
    )
    two_gpu_runtime = len(set(exercised_devices)) == 2
    passed = helper_contract and all_launchers_guarded and two_gpu_runtime and not runtime_failure
    failure_codes: list[str] = []
    if not helper_contract or not all_launchers_guarded:
        failure_codes.append("LORA_TRITON_DEVICE_CONTEXT_GUARD_MISSING")
    if not two_gpu_runtime or runtime_failure:
        failure_codes.append("TWO_GPU_DEVICE_CONTEXT_RUNTIME_FAILED")

    payload = {
        "schema_version": "0.5",
        "probe_id": "vllm-lora-two-gpu-device-context-v0.5-r1",
        "status": "pass" if passed else "fail",
        "worktree_revision": options.worktree.name,
        "helper_contract": helper_contract,
        "launcher_results": launcher_results,
        "cuda_device_count": torch.cuda.device_count(),
        "exercised_devices": exercised_devices,
        "runtime_failure": runtime_failure,
        "failure_codes": failure_codes,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
