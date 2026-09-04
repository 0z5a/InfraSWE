#!/usr/bin/env python3
"""Run bounded exact-head compile and candidate-test checks for one training group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

REMOTE_REPOSITORIES = {
    "megatron-core": "/workspace/training-pr-corpus/repos/megatron-core",
    "slime": "/workspace/training-pr-corpus/repos/slime",
    "verl": "/workspace/training-pr-corpus/repos/verl",
    "verl-omni": "/workspace/training-pr-corpus/repos/verl-omni",
    "flashinfer": "/workspace/inference-pr-corpus/repos/flashinfer",
    "sglang": "/workspace/inference-pr-corpus/repos/sglang",
    "tensorrt-llm": "/workspace/inference-pr-corpus/repos/tensorrt-llm",
    "vllm": "/workspace/inference-pr-corpus/repos/vllm",
}
COMMUNICATION_REPOSITORIES = {
    "nccl": "/workspace/communication-pr-corpus/repos/nccl",
    "rccl": "/workspace/communication-pr-corpus/repos/rccl",
    "nvshmem": "/workspace/communication-pr-corpus/repos/nvshmem",
    "uccl": "/workspace/communication-pr-corpus/repos/uccl",
    "ucx": "/workspace/communication-pr-corpus/repos/ucx",
    "ucc": "/workspace/communication-pr-corpus/repos/ucc",
    "pytorch": "/workspace/communication-pr-corpus/repos/pytorch",
    "vllm": "/workspace/communication-pr-corpus/repos/vllm",
    "sglang": "/workspace/communication-pr-corpus/repos/sglang",
    "megatron-core": "/workspace/communication-pr-corpus/repos/megatron-core",
}
REMOTE_PYTHONS = {
    project: (
        f"/workspace/inference-pr-corpus/venvs/{project}/bin/python"
        if project in {"flashinfer", "sglang", "tensorrt-llm", "vllm"}
        else f"/workspace/training-pr-corpus/venvs/{project}/bin/python"
    )
    for project in REMOTE_REPOSITORIES
}
GPU_BASE_LANES = {
    "megatron-core": "0",
    "slime": "0",
    "verl": "1",
    "verl-omni": "1",
    "flashinfer": "0",
    "sglang": "1",
    "tensorrt-llm": "0",
    "vllm": "1",
}


def _activate_profile(profile: str) -> None:
    if profile != "communication":
        return
    REMOTE_REPOSITORIES.clear()
    REMOTE_REPOSITORIES.update(COMMUNICATION_REPOSITORIES)
    REMOTE_PYTHONS.clear()
    REMOTE_PYTHONS.update(
        {
            "nccl": "/workspace/infraswe/.venv/bin/python",
            "rccl": "/workspace/infraswe/.venv/bin/python",
            "nvshmem": "/workspace/infraswe/.venv/bin/python",
            "uccl": "/workspace/infraswe/.venv/bin/python",
            "ucx": "/workspace/infraswe/.venv/bin/python",
            "ucc": "/workspace/infraswe/.venv/bin/python",
            "pytorch": "/usr/bin/python3",
            "vllm": "/workspace/inference-pr-corpus/venvs/vllm/bin/python",
            "sglang": "/workspace/inference-pr-corpus/venvs/sglang/bin/python",
            "megatron-core": "/workspace/training-pr-corpus/venvs/megatron-core/bin/python",
        }
    )
    GPU_BASE_LANES.clear()
    GPU_BASE_LANES.update(
        {project: str(index % 2) for index, project in enumerate(REMOTE_REPOSITORIES)}
    )


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".py") and (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith("_test.py")
    )


def _ssh_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.identity),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={args.known_hosts}",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        args.remote,
        remote_command,
    ]


def _ssh(
    args: argparse.Namespace, remote_command: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    if args.local:
        return subprocess.run(
            ["bash", "-lc", remote_command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return subprocess.run(
        _ssh_command(args, remote_command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ref(group_index: int, pull_number: int) -> str:
    return f"refs/training-bulk/g{group_index:04d}/pr-{pull_number}"


def _repository_for_lane(project: str, lane: int) -> str:
    if lane == 0:
        return REMOTE_REPOSITORIES[project]
    corpus_root = Path(REMOTE_REPOSITORIES[project]).parent.parent
    return str(corpus_root / "worktrees" / f"{project}-lane-{lane}")


def _gpu_for_lane(project: str, lane: int) -> str:
    return str((int(GPU_BASE_LANES[project]) + lane) % 2)


def _prepare_project(
    args: argparse.Namespace,
    group_index: int,
    project: str,
    cases: list[dict[str, Any]],
) -> dict[str, str]:
    repository = REMOTE_REPOSITORIES[project]
    failures: dict[str, str] = {}

    def fetch(selected: list[dict[str, Any]], timeout: int) -> subprocess.CompletedProcess[str]:
        refspecs = [
            f"+refs/pull/{case['pull_number']}/head:{_ref(group_index, case['pull_number'])}"
            for case in selected
        ]
        command = shlex.join(
            [
                "git",
                "-C",
                repository,
                "fetch",
                "--filter=blob:none",
                "--no-tags",
                "origin",
                *refspecs,
            ]
        )
        return _ssh(args, f"timeout {timeout}s {command}", timeout=timeout + 30)

    for offset in range(0, len(cases), 100):
        chunk = cases[offset : offset + 100]
        fetched = fetch(chunk, 300)
        if fetched.returncode == 0:
            continue
        for case in chunk:
            individual = fetch([case], 90)
            if individual.returncode != 0:
                tail = (individual.stdout + individual.stderr)[-4000:]
                failures[case["case_id"]] = f"fetch-rc={individual.returncode}:{tail}"

    available = [case for case in cases if case["case_id"] not in failures]
    if not available:
        return failures

    if args.lanes_per_project > 1:
        initial_reference = _ref(group_index, available[0]["pull_number"])
        worktree_root = str(Path(_repository_for_lane(project, 1)).parent)
        steps = [f"mkdir -p {shlex.quote(worktree_root)}"]
        project_lane_count = min(args.lanes_per_project, len(available))
        for lane in range(1, project_lane_count):
            lane_repository = _repository_for_lane(project, lane)
            steps.append(
                "if test -e "
                f"{shlex.quote(lane_repository)}/.git; then "
                f"git -C {shlex.quote(lane_repository)} rev-parse --git-dir >/dev/null; "
                "else "
                f"git -C {shlex.quote(repository)} worktree add --force --detach "
                "--no-checkout "
                f"{shlex.quote(lane_repository)} {shlex.quote(initial_reference)} "
                ">/dev/null; fi"
            )
        setup = _ssh(
            args,
            f"timeout 120s bash -lc {shlex.quote(' && '.join(steps))}",
            timeout=150,
        )
        if setup.returncode != 0:
            tail = (setup.stdout + setup.stderr)[-4000:]
            return {case["case_id"]: f"worktree-rc={setup.returncode}:{tail}" for case in cases}
    return failures


def _run_case(
    args: argparse.Namespace,
    group_index: int,
    case: dict[str, Any],
    prewarm_failure: str | None,
    repository: str,
    lane: int,
    lock: threading.Lock,
) -> dict[str, Any]:
    started = time.monotonic()
    if prewarm_failure is not None:
        return {
            "case_id": case["case_id"],
            "project": case["project"],
            "pull_number": case["pull_number"],
            "head_sha": case.get("head_sha"),
            "status": "prewarm_failed",
            "returncode": None,
            "duration_seconds": time.monotonic() - started,
            "compile_paths": [],
            "test_paths": [],
            "output_tail": prewarm_failure[-4000:],
        }

    project = case["project"]
    python = REMOTE_PYTHONS[project]
    reference = _ref(group_index, case["pull_number"])
    offline_cache = str(Path(repository).parent.parent / "hf-offline-cache")
    compile_paths = [
        item["path"]
        for item in case["files"]
        if item["path"].endswith(".py") and item["change_type"] != "deleted"
    ]
    test_paths = [path for path in compile_paths if _is_test_path(path)]
    compile_source = (
        "import pathlib,sys,tokenize;"
        "paths=[pathlib.Path(p) for p in sys.argv[1:]];"
        "[compile(tokenize.open(p).read(),str(p),'exec') for p in paths if p.is_file()]"
    )
    steps = [
        "set -o pipefail",
        f"mkdir -p {shlex.quote(offline_cache)}",
        f"cd {shlex.quote(repository)}",
        "GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0 "
        "git switch --discard-changes --detach "
        f"{shlex.quote(reference)} >/dev/null",
    ]
    if case.get("head_sha"):
        steps.append(f'test "$(git rev-parse HEAD)" = {shlex.quote(case["head_sha"])}')
    if compile_paths:
        steps.append(shlex.join([python, "-c", compile_source, *compile_paths]))
    if test_paths:
        steps.append(
            shlex.join(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "-rs",
                    "--tb=short",
                    "--maxfail=1",
                    "-p",
                    "no:cacheprovider",
                    *test_paths,
                ]
            )
        )
    body = " && ".join(steps)
    test_budget = min(args.test_timeout, 20) if project == "tensorrt-llm" else args.test_timeout
    remote_command = shlex.join(
        [
            "timeout",
            "--kill-after=5s",
            f"{test_budget}s",
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            f"HF_HOME={offline_cache}",
            f"HF_HUB_CACHE={offline_cache}/hub",
            f"HUGGINGFACE_HUB_CACHE={offline_cache}/hub",
            f"TRANSFORMERS_CACHE={offline_cache}/transformers",
            f"TORCH_HOME={offline_cache}/torch",
            f"XDG_CACHE_HOME={offline_cache}/xdg",
            "HF_HUB_OFFLINE=1",
            "HF_HUB_DISABLE_XET=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "HTTP_PROXY=http://127.0.0.1:9",
            "HTTPS_PROXY=http://127.0.0.1:9",
            "ALL_PROXY=http://127.0.0.1:9",
            "NO_PROXY=localhost,127.0.0.1,::1",
            "http_proxy=http://127.0.0.1:9",
            "https_proxy=http://127.0.0.1:9",
            "all_proxy=http://127.0.0.1:9",
            "no_proxy=localhost,127.0.0.1,::1",
            f"CUDA_VISIBLE_DEVICES={_gpu_for_lane(project, lane)}",
            f"PYTHONPATH={repository}",
            "bash",
            "-lc",
            body,
        ]
    )
    try:
        with lock:
            process = _ssh(args, remote_command, timeout=test_budget + 45)
        output = process.stdout + process.stderr
        status = "timed_out" if process.returncode == 124 else "completed"
        returncode: int | None = process.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
        output = (stdout or "") + (stderr or "")
        status = "transport_timeout"
        returncode = None
    return {
        "case_id": case["case_id"],
        "project": project,
        "pull_number": case["pull_number"],
        "head_sha": case.get("head_sha"),
        "repository_lane": lane,
        "ref": reference,
        "status": status,
        "returncode": returncode,
        "duration_seconds": time.monotonic() - started,
        "compile_paths": compile_paths,
        "test_paths": test_paths,
        "test_budget_seconds": test_budget,
        "output_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
        "output_tail": output[-args.output_tail_bytes :],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote", default=os.environ.get("INFRASWE_SSH_REMOTE"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INFRASWE_SSH_PORT", "22")))
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path(os.environ.get("INFRASWE_SSH_IDENTITY", "~/.ssh/id_ed25519")).expanduser(),
    )
    parser.add_argument(
        "--known-hosts",
        type=Path,
        default=Path(
            os.environ.get("INFRASWE_SSH_KNOWN_HOSTS_FILE", "~/.ssh/known_hosts")
        ).expanduser(),
    )
    parser.add_argument("--test-timeout", type=int, default=45)
    parser.add_argument("--output-tail-bytes", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--lanes-per-project",
        type=int,
        default=int(os.environ.get("INFRASWE_TRAINING_LANES_PER_PROJECT", "1")),
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--local",
        action="store_true",
        help="execute repository commands on this host instead of over SSH",
    )
    args = parser.parse_args()
    if not args.local and not args.remote:
        raise SystemExit("remote is required")
    if args.lanes_per_project <= 0:
        raise SystemExit("lanes per project must be positive")
    if args.output_tail_bytes <= 0:
        raise SystemExit("output tail bytes must be positive")

    input_lock = _read(args.input_lock)
    material = {key: value for key, value in input_lock.items() if key != "group_input_sha256"}
    if input_lock["group_input_sha256"] != canonical_sha256(material):
        raise SystemExit("group input digest mismatch")
    _activate_profile(str(input_lock.get("profile", "training")))
    all_cases = input_lock["cases"]
    available_case_ids = {case["case_id"] for case in all_cases}
    requested_case_ids = set(args.only)
    unknown_case_ids = requested_case_ids - available_case_ids
    if unknown_case_ids:
        raise SystemExit(f"unknown case ids: {sorted(unknown_case_ids)}")
    cases = [
        case
        for case in all_cases
        if not requested_case_ids or case["case_id"] in requested_case_ids
    ]
    group_index = int(input_lock["group_index"])
    started_at = datetime.now(UTC).isoformat()

    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    checkpoint_records: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        checkpoint = _read(partial_path)
        if checkpoint.get("group_input_sha256") != input_lock[
            "group_input_sha256"
        ] or checkpoint.get("case_filter") != sorted(requested_case_ids):
            raise SystemExit("exact-head checkpoint binding mismatch")
        started_at = str(checkpoint["started_at"])
        checkpoint_records = {record["case_id"]: record for record in checkpoint.get("records", [])}
    pending_cases = [case for case in cases if case["case_id"] not in checkpoint_records]

    by_project: dict[str, list[dict[str, Any]]] = {}
    prewarm_failures: dict[str, str] = {
        case["case_id"]: (
            "metadata-acquisition-invalid:"
            f"{case.get('acquisition_failure_code', 'GITHUB_METADATA_UNAVAILABLE')}"
        )
        for case in pending_cases
        if case.get("acquisition_status", "acquired") != "acquired"
    }
    for case in pending_cases:
        if case["case_id"] not in prewarm_failures:
            by_project.setdefault(case["project"], []).append(case)
    if by_project:
        with ThreadPoolExecutor(max_workers=len(by_project)) as executor:
            futures = {
                executor.submit(
                    _prepare_project, args, group_index, project, project_cases
                ): project
                for project, project_cases in by_project.items()
            }
            for future in as_completed(futures):
                project = futures[future]
                failures = future.result()
                prewarm_failures.update(failures)
                print(
                    f"prewarm project={project} cases={len(by_project[project])} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    case_lanes: dict[str, int] = {}
    all_by_project: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        all_by_project.setdefault(case["project"], []).append(case)
    for project_cases in all_by_project.values():
        for index, case in enumerate(project_cases):
            case_lanes[case["case_id"]] = index % args.lanes_per_project
    cases_by_lane: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for case in pending_cases:
        key = (case["project"], case_lanes[case["case_id"]])
        cases_by_lane.setdefault(key, []).append(case)

    checkpoint_lock = threading.Lock()
    completed_since_checkpoint = [0]

    def checkpoint(record: dict[str, Any]) -> None:
        with checkpoint_lock:
            checkpoint_records[record["case_id"]] = record
            completed_since_checkpoint[0] += 1
            if completed_since_checkpoint[0] < 25:
                return
            material = {
                "schema_version": "0.1",
                "protocol_id": "bulk-group-exact-head-checkpoint-v0.1",
                "group_input_sha256": input_lock["group_input_sha256"],
                "case_filter": sorted(requested_case_ids),
                "started_at": started_at,
                "records": [
                    checkpoint_records[case["case_id"]]
                    for case in cases
                    if case["case_id"] in checkpoint_records
                ],
            }
            atomic_write_json(partial_path, material)
            print(
                f"exact-head checkpoint completed={len(checkpoint_records)}/{len(cases)}",
                flush=True,
            )
            completed_since_checkpoint[0] = 0

    def run_lane(key: tuple[str, int], lane_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        project, lane = key
        repository = _repository_for_lane(project, lane)
        lock = threading.Lock()
        lane_records: list[dict[str, Any]] = []
        for case in lane_cases:
            record = _run_case(
                args,
                group_index,
                case,
                prewarm_failures.get(case["case_id"]),
                repository,
                lane,
                lock,
            )
            lane_records.append(record)
            checkpoint(record)
            print(
                f"{record['case_id']}: status={record['status']} "
                f"rc={record['returncode']} {record['duration_seconds']:.1f}s",
                flush=True,
            )
        return lane_records

    records: list[dict[str, Any]] = list(checkpoint_records.values())
    if cases_by_lane:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(cases_by_lane))) as executor:
            futures = {
                executor.submit(run_lane, key, lane_cases): key
                for key, lane_cases in cases_by_lane.items()
            }
            for future in as_completed(futures):
                records.extend(future.result())
    record_by_id = {record["case_id"]: record for record in records}
    ordered_records = [record_by_id[case["case_id"]] for case in cases]
    output_material = {
        "schema_version": "0.1",
        "protocol_id": (
            f"{input_lock.get('profile', 'training')}-bulk-group-exact-head-gated-v0.1"
        ),
        "profile": input_lock.get("profile", "training"),
        "group_input_sha256": input_lock["group_input_sha256"],
        "group_index": group_index,
        "started_at": started_at,
        "remote": "local" if args.local else args.remote,
        "repository_paths": REMOTE_REPOSITORIES,
        "python_paths": REMOTE_PYTHONS,
        "gpu_base_lanes": GPU_BASE_LANES,
        "gpu_lane_strategy": "alternate-each-project-lane-across-two-gpus",
        "lanes_per_project": args.lanes_per_project,
        "test_timeout_seconds": args.test_timeout,
        "timeout_disposition": "neutral-abandon-no-retry",
        "case_filter": sorted(requested_case_ids),
        "prewarm_failure_count": sum(
            record["status"] == "prewarm_failed" for record in ordered_records
        ),
        "records": ordered_records,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    payload = {**output_material, "evidence_sha256": canonical_sha256(output_material)}
    atomic_write_json(args.output, payload)
    partial_path.unlink(missing_ok=True)
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
