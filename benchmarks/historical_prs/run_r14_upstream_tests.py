#!/usr/bin/env python3
"""Run candidate-owned R14 tests on exact remote PR refs without outcome access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

REMOTE = "root@38.49.42.120"
PORT = 54270
IDENTITY = "~/.ssh/id_ed25519_winpc"
PROJECT_RUNTIME = {
    "vllm": {
        "worktree": "/workspace/r14-run-vllm",
        "python": "/venv/main/bin/python",
        "pythonpath": ".",
    },
    "sglang": {
        "worktree": "/workspace/r14-run-sglang",
        "python": "/venv/main/bin/python",
        "pythonpath": "/workspace/r14-shims:python:.",
    },
    "flashinfer": {
        "worktree": "/workspace/r14-run-flashinfer",
        "python": "/venv/main/bin/python",
        "pythonpath": ".",
    },
    "megatron-core": {
        "worktree": "/workspace/r13-run-megatron",
        "python": "/venv/main/bin/python",
        "pythonpath": ".",
    },
    "torchtitan": {
        "worktree": "/workspace/r13-run-torchtitan",
        "python": "/workspace/venv-tt/bin/python",
        "pythonpath": ".",
    },
    "verl": {
        "worktree": "/workspace/r13-run-verl",
        "python": "/venv/main/bin/python",
        "pythonpath": ".",
    },
}
SUMMARY_PATTERN = re.compile(
    r"(?:=+\s*)?(?:\d+\s+)?(?:passed|failed|error|errors|skipped|deselected).*$",
    re.IGNORECASE,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.endswith(".py")
        and (
            lowered.startswith(("test/", "tests/"))
            or "/test/" in lowered
            or "/tests/" in lowered
            or lowered.endswith("_test.py")
        )
    )


def _ssh(remote_command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-p",
            str(PORT),
            "-i",
            str(Path(IDENTITY).expanduser()),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            REMOTE,
            remote_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _command(
    case: dict[str, Any],
    static_case: dict[str, Any],
    *,
    test_timeout: int,
    noconftest_projects: set[str],
) -> tuple[str | None, list[str], list[str]]:
    runtime = PROJECT_RUNTIME[case["project"]]
    worktree = runtime["worktree"]
    python = runtime["python"]
    test_paths = [path for path in case["paths"] if _is_test_path(path)]
    test_names = static_case["candidate_test_functions_added"]
    if not test_paths:
        return None, [], test_names
    arguments = [
        "timeout",
        f"{test_timeout}s",
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "CUDA_VISIBLE_DEVICES=0",
        f"PYTHONPATH={runtime['pythonpath']}",
        python,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "--tb=short",
        "--maxfail=1",
    ]
    if case["project"] in noconftest_projects:
        arguments.append("--noconftest")
    arguments.extend(test_paths)
    if test_names:
        arguments.extend(["-k", " or ".join(test_names)])
    test_command = " ".join(shlex.quote(item) for item in arguments)
    number = case["pull_number"]
    expected_head = shlex.quote(case["head_sha"])
    command = (
        f"cd {shlex.quote(worktree)} && "
        f"git switch --detach refs/r14/pr-{number} >/dev/null && "
        f"test \"$(git rev-parse HEAD)\" = {expected_head} && "
        f"{test_command}"
    )
    return command, test_paths, test_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-timeout", type=int, default=120)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--noconftest-project", action="append", default=[])
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R14 selection digest mismatch")
    static = _read(args.static_evidence)
    static_material = {key: value for key, value in static.items() if key != "evidence_sha256"}
    if static["evidence_sha256"] != canonical_sha256(static_material):
        raise SystemExit("R14 static evidence digest mismatch")
    if static["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R14 static/selection binding mismatch")
    static_by_id = {item["case_id"]: item for item in static["cases"]}
    selected_cases = selection_material["cases"]
    if args.only:
        requested = set(args.only)
        selected_cases = [item for item in selected_cases if item["case_id"] in requested]
        missing = requested - {item["case_id"] for item in selected_cases}
        if missing:
            raise SystemExit(f"unknown --only case IDs: {sorted(missing)}")

    records = []
    started_at = datetime.now(UTC).isoformat()
    for index, case in enumerate(selected_cases, 1):
        command, test_paths, test_names = _command(
            case,
            static_by_id[case["case_id"]],
            test_timeout=args.test_timeout,
            noconftest_projects=set(args.noconftest_project),
        )
        if command is None:
            records.append(
                {
                    "case_id": case["case_id"],
                    "ref": f"refs/r14/pr-{case['pull_number']}",
                    "head_sha": case["head_sha"],
                    "status": "no-candidate-python-test-path",
                    "test_paths": test_paths,
                    "test_names": test_names,
                    "returncode": None,
                    "duration_seconds": 0.0,
                    "output_sha256": None,
                    "output_tail": "",
                    "summary_lines": [],
                }
            )
            print(f"[{index}/{len(selected_cases)}] {case['case_id']}: no candidate test")
            continue
        began = time.monotonic()
        try:
            process = _ssh(command, timeout=args.test_timeout + 30)
            returncode = process.returncode
            output = process.stdout + process.stderr
            status = "completed"
        except subprocess.TimeoutExpired as error:
            returncode = 124
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            output = str(stdout) + str(stderr)
            status = "ssh-timeout"
        duration = time.monotonic() - began
        summary_lines = [
            line.strip()
            for line in output.splitlines()
            if SUMMARY_PATTERN.search(line.strip())
        ][-20:]
        records.append(
            {
                "case_id": case["case_id"],
                "ref": f"refs/r14/pr-{case['pull_number']}",
                "head_sha": case["head_sha"],
                "status": status,
                "test_paths": test_paths,
                "test_names": test_names,
                "remote_command": command,
                "returncode": returncode,
                "duration_seconds": duration,
                "output_sha256": "sha256:"
                + hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "output_tail": output[-12000:],
                "summary_lines": summary_lines,
            }
        )
        print(
            f"[{index}/{len(selected_cases)}] {case['case_id']}: "
            f"rc={returncode} {duration:.1f}s"
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": "r14-exact-candidate-upstream-tests-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": static["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "remote": {"host": REMOTE, "port": PORT},
        "test_timeout_seconds": args.test_timeout,
        "noconftest_projects": sorted(set(args.noconftest_project)),
        "outcome_review_ci_fields_requested": False,
        "records": records,
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
