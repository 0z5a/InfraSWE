#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
import types
from pathlib import Path

import yaml
from packaging.version import parse


def extract_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        raise RuntimeError(f"setup.py is missing {name}")
    return function


def wheel_name_for_cuda(setup_path: Path, cuda_version: str) -> str:
    tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    function = extract_function(tree, "get_wheel_url")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    torch_stub = types.SimpleNamespace(
        __version__="2.5.1",
        version=types.SimpleNamespace(cuda=cuda_version),
        _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
    )
    namespace = {
        "BASE_WHEEL_URL": (
            "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
        ),
        "IS_ROCM": False,
        "PACKAGE_NAME": "flash_attn",
        "get_package_version": lambda: "2.7.4",
        "get_platform": lambda: "linux_x86_64",
        "parse": parse,
        "sys": sys,
        "torch": torch_stub,
    }
    exec(compile(module, str(setup_path), "exec"), namespace)
    _, filename = namespace["get_wheel_url"]()
    return filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    workflow_path = options.worktree / ".github" / "workflows" / "publish.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["build_wheels"]["strategy"]["matrix"]
    cuda_matrix = list(matrix["cuda-version"])
    filenames = {
        version: wheel_name_for_cuda(options.worktree / "setup.py", version)
        for version in ("11.8", "12.4", "12.5")
    }
    expected_markers = {"11.8": "+cu118", "12.4": "+cu124", "12.5": "+cu124"}
    filename_contract = all(
        marker in filenames[version] for version, marker in expected_markers.items()
    )
    matrix_contract = "12.4.1" in cuda_matrix and "12.3.2" not in cuda_matrix
    passed = filename_contract and matrix_contract
    payload = {
        "schema_version": "0.5",
        "probe_id": "flashattention-cuda-wheel-matrix-consistency-v0.5-r1",
        "status": "pass" if passed else "fail",
        "worktree_revision": options.worktree.name,
        "cuda_matrix": cuda_matrix,
        "wheel_filenames": filenames,
        "filename_contract": filename_contract,
        "matrix_contract": matrix_contract,
        "failure_codes": [] if passed else ["CUDA_WHEEL_MATRIX_SETUP_MISMATCH"],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
