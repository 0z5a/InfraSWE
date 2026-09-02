#!/usr/bin/env python3
"""Zero-import structural probe for SGLang #3044 test relocation and suite registration."""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any


def _inventory(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = [item.name for item in ast.walk(tree) if isinstance(item, ast.ClassDef)]
    tests = sorted(
        item.name
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name.startswith("test")
    )
    return {"classes": classes, "tests": tests, "parse_status": "pass"}


def _suite_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        item.value
        for item in ast.walk(tree)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    pairs = [
        ("python/sglang/test/test_activation.py", "test/srt/test_activation.py"),
        ("python/sglang/test/test_block_fp8.py", "test/srt/test_block_fp8.py"),
        ("python/sglang/test/test_layernorm.py", "test/srt/test_layernorm.py"),
    ]
    moved: list[dict[str, Any]] = []
    failure_codes: list[str] = []
    suite_strings = _suite_strings(args.head_root / "test/srt/run_suite.py")
    for before_path, after_path in pairs:
        before = _inventory(args.base_root / before_path)
        after = _inventory(args.head_root / after_path)
        filename = Path(after_path).name
        registered_count = suite_strings.count(filename)
        inventory_preserved = before == after
        status = "pass" if inventory_preserved and registered_count == 1 else "fail"
        if not inventory_preserved:
            failure_codes.append("MOVED_TEST_INVENTORY_CHANGED")
        if registered_count != 1:
            failure_codes.append("MOVED_TEST_SUITE_REGISTRATION_INVALID")
        moved.append(
            {
                "before": before_path,
                "after": after_path,
                "before_inventory": before,
                "after_inventory": after,
                "inventory_preserved": inventory_preserved,
                "run_suite_registration_count": registered_count,
                "status": status,
            }
        )
    payload = {
        "schema_version": "0.5",
        "probe": "sglang-test-move-contract-v1",
        "case_id": "sglang-pr-3044",
        "compilation_path": "not-required",
        "compile_seconds": 0.0,
        "steady_state_compile_seconds": 0.0,
        "duration_seconds": time.perf_counter() - started,
        "moved_modules": moved,
        "candidate_failure_codes": sorted(set(failure_codes)),
        "status": "fail" if failure_codes else "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
