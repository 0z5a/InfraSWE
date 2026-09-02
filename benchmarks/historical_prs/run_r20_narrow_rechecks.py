#!/usr/bin/env python3
# ruff: noqa: E501
"""Run two final outcome-blind R20 rechecks at the exact changed boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_r15_upstream_tests import _ssh

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--focused-tests", type=Path, required=True)
    parser.add_argument("--mamba-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = read(args.selection_lock)
    focused = read(args.focused_tests)
    if selection["selection_lock_sha256"] != canonical_sha256(selection["selection_material"]):
        raise SystemExit("R20 selection digest mismatch")
    if focused["evidence_sha256"] != canonical_sha256({key: value for key, value in focused.items() if key != "evidence_sha256"}):
        raise SystemExit("R20 focused evidence digest mismatch")
    if focused["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R20 focused/selection binding mismatch")
    cases = {case["case_id"]: case for case in selection["selection_material"]["cases"]}

    def switch(worktree: str, case_id: str) -> str:
        case = cases[case_id]
        lfs = "GIT_LFS_SKIP_SMUDGE=1 " if case_id.startswith("tensorrt_llm-") else ""
        return (
            f"cd {shlex.quote(worktree)} && {lfs}git switch --detach "
            f"refs/r20/pr-{int(case['pull_number'])} >/dev/null && "
            f"test \"$(git rev-parse HEAD)\" = {shlex.quote(case['head_sha'])} && "
        )

    vllm_driver = (
        "from tests.v1.streaming_input.test_scheduler_streaming import DummyRequest; "
        "from vllm.v1.core.sched.scheduler import Scheduler; "
        "from vllm.v1.request import StreamingUpdate; "
        "session=DummyRequest(request_id='session',prompt_token_ids=[1,2,3]); session.num_computed_tokens=9; "
        "incoming=DummyRequest(request_id='session',prompt_token_ids=[4,5,6]); update=StreamingUpdate.from_request(incoming); "
        "scheduler=object.__new__(Scheduler); scheduler.num_waiting_for_streaming_input=0; scheduler.log_stats=False; "
        "Scheduler._update_request_as_session(scheduler,session,update); "
        "assert session.num_computed_tokens==session.num_prompt_tokens==6 and session.num_tokens==6; "
        "print('streaming-clamp-pass',session.num_computed_tokens,session.num_prompt_tokens)"
    )
    mamba_probe_sha256 = hashlib.sha256(args.mamba_probe.read_bytes()).hexdigest()
    commands = [
        {
            "case_id": "vllm-pr-44526",
            "lane": 0,
            "timeout": 150,
            "purpose": "bypass an unrelated stale scheduler fixture and execute the exact over-computed streaming-session clamp",
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44526")
            + "timeout 120s env CUDA_VISIBLE_DEVICES=0 INFRASWE_R15_VLLM_SOURCE_IMPORT_SHIM=1 PYTHONPATH=/workspace/r17-deps/vllm:/workspace/r18-deps/common:/workspace/r15-vllm-shim:/workspace/r14-shims:. /venv/main/bin/python -c "
            + shlex.quote(vllm_driver),
        },
        {
            "case_id": "tensorrt_llm-pr-14869",
            "lane": 1,
            "timeout": 180,
            "purpose": "execute the exact Triton functions in isolation and compare their strided pool writes against a tensor reference",
            "command": switch("/workspace/r19-tensorrt-wt", "tensorrt_llm-pr-14869")
            + f"test \"$(sha256sum /workspace/r20-probes/r20_mamba_state_probe.py | cut -d' ' -f1)\" = {mamba_probe_sha256} && "
            + "timeout 150s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python /workspace/r20-probes/r20_mamba_state_probe.py --source tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py",
        },
    ]

    def execute(spec: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            process = _ssh(spec["command"], timeout=int(spec["timeout"]))
            output = process.stdout + process.stderr
            returncode = process.returncode
            status = "completed"
        except subprocess.TimeoutExpired as error:
            output = str(error.stdout or "") + str(error.stderr or "")
            returncode = 124
            status = "ssh-timeout"
        return {
            **spec,
            "head_sha": cases[spec["case_id"]]["head_sha"],
            "status": status,
            "returncode": returncode,
            "duration_seconds": time.monotonic() - started,
            "output_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
            "output_tail": output[-16000:],
        }

    started_at = datetime.now(UTC).isoformat()
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(execute, commands))
    material = {
        "schema_version": "0.1",
        "protocol_id": "r20-narrow-outcome-free-rechecks-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "focused_test_evidence_sha256": focused["evidence_sha256"],
        "mamba_probe_sha256": "sha256:" + mamba_probe_sha256,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_requested": False,
        "records": records,
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps({record["case_id"]: record["returncode"] for record in records}, indent=2, sort_keys=True))
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
