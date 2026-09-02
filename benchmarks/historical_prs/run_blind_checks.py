#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def digest_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def run(arguments: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(arguments, cwd=cwd, text=True, capture_output=True)
    return {
        "command": arguments,
        "return_code": process.returncode,
        "duration_seconds": time.monotonic() - started,
        "stdout_sha256": digest_text(process.stdout),
        "stderr_sha256": digest_text(process.stderr),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-2000:],
    }


def parse_changed_files(root: Path, paths: list[str]) -> dict[str, Any]:
    parsed: list[str] = []
    failures: list[dict[str, str]] = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
                parsed.append(relative)
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                parsed.append(relative)
            elif path.suffix in {".yaml", ".yml"}:
                yaml.safe_load(path.read_text(encoding="utf-8"))
                parsed.append(relative)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError, yaml.YAMLError) as error:
            failures.append({"path": relative, "error": f"{type(error).__name__}: {error}"})
    return {
        "status": "pass" if not failures else "fail",
        "parsed_paths": parsed,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--worktrees", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    candidates = json.loads(options.candidates.read_text(encoding="utf-8"))
    started_at = datetime.now(UTC)
    cases = []
    for candidate in candidates:
        case_id = candidate["case_id"]
        base = options.worktrees / f"{case_id}-base"
        head = options.worktrees / f"{case_id}-head"
        base_commit = run(["git", "rev-parse", "HEAD"], cwd=base)
        head_commit = run(["git", "rev-parse", "HEAD"], cwd=head)
        path_result = run(
            [
                "git",
                "diff",
                "--name-only",
                candidate["base_sha"],
                candidate["head_sha"],
            ],
            cwd=head,
        )
        actual_paths = sorted(line for line in path_result["stdout_tail"].splitlines() if line)
        expected_paths = sorted(candidate["paths"])
        diff_check = run(
            ["git", "diff", "--check", candidate["base_sha"], candidate["head_sha"]],
            cwd=head,
        )
        tests = [
            path
            for path in candidate["paths"]
            if path.startswith(("test/", "tests/")) or "/test/" in path or "/tests/" in path
        ]
        cases.append(
            {
                "case_id": case_id,
                "base_commit": base_commit,
                "head_commit": head_commit,
                "commit_identity_pass": (
                    base_commit["return_code"] == 0
                    and head_commit["return_code"] == 0
                    and base_commit["stdout_tail"].strip() == candidate["base_sha"]
                    and head_commit["stdout_tail"].strip() == candidate["head_sha"]
                ),
                "path_check": path_result,
                "path_parity_pass": actual_paths == expected_paths,
                "actual_paths": actual_paths,
                "diff_check": diff_check,
                "head_parse": parse_changed_files(head, candidate["paths"]),
                "base_parse": parse_changed_files(base, candidate["paths"]),
                "changed_test_paths": tests,
            }
        )

    payload = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-static-v0.5-r1",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "cases": cases,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = options.output.with_name(f".{options.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(options.output)
    failed = [
        case["case_id"]
        for case in cases
        if not (
            case["commit_identity_pass"]
            and case["path_parity_pass"]
            and case["diff_check"]["return_code"] == 0
            and case["head_parse"]["status"] == "pass"
        )
    ]
    print(f"blind static checks: {len(cases) - len(failed)}/{len(cases)} pass")
    if failed:
        print("failed: " + ", ".join(failed))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
