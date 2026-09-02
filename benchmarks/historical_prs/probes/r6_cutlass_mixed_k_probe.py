#!/usr/bin/env python3
"""Run the documented mixed-K CUTLASS grouped-GEMM problem set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run(binary: Path, problem_file: Path) -> dict[str, Any]:
    command = [str(binary), f"--benchmark={problem_file}", "--iterations=0"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": command,
            "return_code": None,
            "timed_out": True,
            "duration_seconds": time.perf_counter() - started,
            "disposition_passed": False,
            "stdout_tail": stdout[-8000:],
            "stderr_tail": stderr[-8000:],
        }
    output = completed.stdout + "\n" + completed.stderr
    return {
        "command": command,
        "return_code": completed.returncode,
        "timed_out": False,
        "duration_seconds": time.perf_counter() - started,
        "disposition_passed": "Disposition: Passed" in output,
        "stdout_sha256": _digest(completed.stdout),
        "stderr_sha256": _digest(completed.stderr),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-binary", type=Path, required=True)
    parser.add_argument("--head-binary", type=Path, required=True)
    parser.add_argument("--problem-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    problem_text = args.problem_file.read_text(encoding="utf-8")
    problems = [line.strip() for line in problem_text.splitlines() if line.strip()]
    expected = [
        "0 256x512x128",
        "1 256x512x512",
        "2 512x256x128",
        "3 256x256x128",
        "4 256x512x1024",
        "5 1024x512x128",
    ]
    if problems != expected:
        raise ValueError(f"unexpected documented problem set: {problems}")

    base = _run(args.base_binary, args.problem_file)
    head = _run(args.head_binary, args.problem_file)
    failure_codes: list[str] = []
    if base["disposition_passed"]:
        failure_codes.append("CUTLASS_MIXED_K_BASE_CONTROL_DID_NOT_REPRODUCE")
    if not head["disposition_passed"]:
        failure_codes.append("CUTLASS_MIXED_K_HEAD_CORRECTNESS_FAILED")

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "cutlass-documented-mixed-k-h100-v1",
        "case_id": "cutlass-pr-2275",
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "facts": {
            "problems": problems,
            "k128_problem_count": sum(item.endswith("x128") for item in problems),
            "base": base,
            "head": head,
            "aot_binaries_reused_from_primary_probe": True,
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": {"problem_file_sha256": _digest(problem_text)},
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": _digest(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
