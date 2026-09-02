#!/usr/bin/env python3
"""Compile-free ancestry and test-contract probe for SGLang PR 3136."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _value_error_messages(source: str) -> list[str]:
    tree = ast.parse(source)
    messages: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        if not isinstance(node.exc.func, ast.Name) or node.exc.func.id != "ValueError":
            continue
        if node.exc.args and isinstance(node.exc.args[0], ast.Constant):
            messages.append(str(node.exc.args[0].value))
    return messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--submodule-repo", type=Path, required=True)
    parser.add_argument("--old-submodule-sha", required=True)
    parser.add_argument("--new-submodule-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    old_is_ancestor = (
        _git(
            args.submodule_repo,
            "merge-base",
            "--is-ancestor",
            args.old_submodule_sha,
            args.new_submodule_sha,
            check=False,
        ).returncode
        == 0
    )
    new_is_ancestor = (
        _git(
            args.submodule_repo,
            "merge-base",
            "--is-ancestor",
            args.new_submodule_sha,
            args.old_submodule_sha,
            check=False,
        ).returncode
        == 0
    )
    reverse_distance = int(
        _git(
            args.submodule_repo,
            "rev-list",
            "--count",
            f"{args.new_submodule_sha}..{args.old_submodule_sha}",
        ).stdout.strip()
    )
    diff_shortstat = _git(
        args.submodule_repo,
        "diff",
        "--shortstat",
        args.old_submodule_sha,
        args.new_submodule_sha,
    ).stdout.strip()
    changed_files = _git(
        args.submodule_repo,
        "diff",
        "--name-only",
        args.old_submodule_sha,
        args.new_submodule_sha,
    ).stdout.splitlines()

    test_path = Path("sgl-kernel/tests/test_sampling.py")
    base_source = (args.base_root / test_path).read_text(encoding="utf-8")
    head_source = (args.head_root / test_path).read_text(encoding="utf-8")
    ast.parse(base_source)
    ast.parse(head_source)
    base_messages = _value_error_messages(base_source)
    head_messages = _value_error_messages(head_source)
    test_diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            head_source.splitlines(keepends=True),
            fromfile=f"base/{test_path}",
            tofile=f"head/{test_path}",
        )
    )
    only_error_string_poke = (
        base_messages != head_messages
        and head_source.replace('"pp not recognized"', '"p not recognized"') == base_source
    )

    failure_codes: list[str] = []
    if not old_is_ancestor and new_is_ancestor:
        failure_codes.append("SUBMODULE_POINTER_REGRESSION")
    if only_error_string_poke:
        failure_codes.append("TEST_POKE_WITHOUT_COVERAGE")

    material = {
        "schema_version": "0.5",
        "probe": "sglang-submodule-poke-contract-v1",
        "case_id": "sglang-pr-3136",
        "status": "fail" if failure_codes else "pass",
        "failure_codes": failure_codes,
        "facts": {
            "old_submodule_sha": args.old_submodule_sha,
            "new_submodule_sha": args.new_submodule_sha,
            "old_is_ancestor_of_new": old_is_ancestor,
            "new_is_ancestor_of_old": new_is_ancestor,
            "reverse_commit_distance": reverse_distance,
            "submodule_changed_file_count": len(changed_files),
            "submodule_diff_shortstat": diff_shortstat,
            "test_change_only_perturbs_error_string": only_error_string_poke,
            "base_value_error_messages": base_messages,
            "head_value_error_messages": head_messages,
            "compilation_path": "not-required",
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": {
            "base_test_sha256": canonical_sha256(base_source),
            "head_test_sha256": canonical_sha256(head_source),
            "test_diff_sha256": canonical_sha256(test_diff),
            "submodule_changed_paths_sha256": canonical_sha256(changed_files),
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
