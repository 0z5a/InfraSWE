#!/usr/bin/env python3
"""Run candidate-owned R16 tests on exact frozen refs without outcome access."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

REMOTE = "root@38.49.42.120"
PORT = 54270
IDENTITY = "~/.ssh/id_ed25519_winpc"
PROJECT_RUNTIME = {
    "liger-kernel": ("/workspace/r13-run-liger", "/venv/main/bin/python", ".", "0"),
    "megatron-core": (
        "/workspace/r13-run-megatron",
        "/venv/main/bin/python",
        ".",
        "1",
    ),
    "slime": (
        "/workspace/r13-run-slime",
        "/venv/main/bin/python",
        ".:/workspace/r13-run-megatron",
        "1",
    ),
    "torchtitan": (
        "/workspace/r13-run-torchtitan",
        "/workspace/venv-tt/bin/python",
        ".",
        "0",
    ),
    "verl": ("/workspace/r13-run-verl", "/venv/main/bin/python", ".", "1"),
}
LANE = {
    "liger-kernel": 0,
    "torchtitan": 0,
    "megatron-core": 1,
    "slime": 1,
    "verl": 1,
}
SUMMARY_PATTERN = re.compile(
    r"(?:=+\s*)?(?:\d+\s+)?(?:passed|failed|error|errors|skipped|deselected).*$",
    re.IGNORECASE,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".py") and (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith("_test.py")
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


def _build_command(
    case: dict[str, Any],
    static_case: dict[str, Any],
    *,
    test_timeout: int,
    noconftest_projects: set[str],
) -> tuple[str | None, list[str], list[str]]:
    worktree, python, pythonpath, gpu = PROJECT_RUNTIME[case["project"]]
    test_paths = [path for path in case["paths"] if _is_test_path(path)]
    test_names = static_case["candidate_test_functions_added"]
    if not test_paths:
        return None, [], test_names
    arguments = [
        "timeout",
        f"{test_timeout}s",
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"PYTHONPATH={pythonpath}",
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
    expected_head = shlex.quote(case["head_sha"])
    command = (
        f"cd {shlex.quote(worktree)} && "
        f"git switch --detach refs/r16/pr-{case['pull_number']} >/dev/null && "
        f'test "$(git rev-parse HEAD)" = {expected_head} && '
        f"{test_command}"
    )
    return command, test_paths, test_names


def _run_case(
    index: int,
    total: int,
    case: dict[str, Any],
    static_case: dict[str, Any],
    *,
    test_timeout: int,
    noconftest_projects: set[str],
) -> dict[str, Any]:
    command, test_paths, test_names = _build_command(
        case,
        static_case,
        test_timeout=test_timeout,
        noconftest_projects=noconftest_projects,
    )
    common = {
        "case_id": case["case_id"],
        "ref": f"refs/r16/pr-{case['pull_number']}",
        "head_sha": case["head_sha"],
        "test_paths": test_paths,
        "test_names": test_names,
    }
    if command is None:
        print(f"[{index}/{total}] {case['case_id']}: no candidate Python test", flush=True)
        return {
            **common,
            "status": "no-candidate-python-test-path",
            "returncode": None,
            "duration_seconds": 0.0,
            "output_sha256": None,
            "output_tail": "",
            "summary_lines": [],
        }
    began = time.monotonic()
    try:
        process = _ssh(command, timeout=test_timeout + 30)
        returncode = process.returncode
        output = process.stdout + process.stderr
        status = "completed"
    except subprocess.TimeoutExpired as error:
        returncode = 124
        output = str(error.stdout or "") + str(error.stderr or "")
        status = "ssh-timeout"
    duration = time.monotonic() - began
    summary_lines = [
        line.strip()
        for line in output.splitlines()
        if SUMMARY_PATTERN.search(line.strip())
    ][-20:]
    print(
        f"[{index}/{total}] {case['case_id']}: rc={returncode} {duration:.1f}s",
        flush=True,
    )
    return {
        **common,
        "status": status,
        "remote_command": command,
        "returncode": returncode,
        "duration_seconds": duration,
        "output_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
        "output_tail": output[-12000:],
        "summary_lines": summary_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-timeout", type=int, default=180)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--noconftest-project", action="append", default=[])
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R16 selection digest mismatch")
    static = _read(args.static_evidence)
    static_material = {
        key: value for key, value in static.items() if key != "evidence_sha256"
    }
    if static["evidence_sha256"] != canonical_sha256(static_material):
        raise SystemExit("R16 static evidence digest mismatch")
    if static["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R16 static/selection binding mismatch")
    static_by_id = {item["case_id"]: item for item in static["cases"]}
    selected_cases = selection_material["cases"]
    if args.only:
        requested = set(args.only)
        selected_cases = [
            item for item in selected_cases if item["case_id"] in requested
        ]
        missing = requested - {item["case_id"] for item in selected_cases}
        if missing:
            raise SystemExit(f"unknown --only case IDs: {sorted(missing)}")

    indexed = list(enumerate(selected_cases, 1))
    lanes = {
        lane: [
            (index, case)
            for index, case in indexed
            if LANE[case["project"]] == lane
        ]
        for lane in (0, 1)
    }

    def run_lane(
        items: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        return [
            (
                index,
                _run_case(
                    index,
                    len(selected_cases),
                    case,
                    static_by_id[case["case_id"]],
                    test_timeout=args.test_timeout,
                    noconftest_projects=set(args.noconftest_project),
                ),
            )
            for index, case in items
        ]

    started_at = datetime.now(UTC).isoformat()
    indexed_records: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_lane, lanes[lane]) for lane in (0, 1)]
        for future in as_completed(futures):
            indexed_records.extend(future.result())
    records = [record for _, record in sorted(indexed_records)]
    material = {
        "schema_version": "0.1",
        "protocol_id": "r16-exact-candidate-upstream-tests-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": static["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "remote": {"host": REMOTE, "port": PORT, "gpu_lanes": [0, 1]},
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
