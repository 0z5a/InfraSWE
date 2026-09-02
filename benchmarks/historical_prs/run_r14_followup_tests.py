#!/usr/bin/env python3
"""Run R14 follow-up tests that need two GPUs or an isolated import boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _ssh(command: str, timeout: int) -> subprocess.CompletedProcess[str]:
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
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _copy_shim(shim: Path) -> None:
    process = subprocess.run(
        [
            "scp",
            "-P",
            str(PORT),
            "-i",
            str(Path(IDENTITY).expanduser()),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            str(shim),
            f"{REMOTE}:/workspace/r14-verl-shim-sitecustomize.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip())
    setup = _ssh(
        "mkdir -p /workspace/r14-verl-shim && "
        "cp /workspace/r14-verl-shim-sitecustomize.py "
        "/workspace/r14-verl-shim/sitecustomize.py",
        30,
    )
    if setup.returncode:
        raise RuntimeError(setup.stderr.strip())


def _pytest_names(names: list[str]) -> str:
    return shlex.quote(" or ".join(names))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--sanitization-manifest", type=Path, required=True)
    parser.add_argument("--verl-shim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R14 selection digest mismatch")
    plan = _read(args.test_plan)
    if plan["test_plan_sha256"] != canonical_sha256(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R14 test-plan digest mismatch")
    static = _read(args.static_evidence)
    if static["evidence_sha256"] != canonical_sha256(
        {key: value for key, value in static.items() if key != "evidence_sha256"}
    ):
        raise SystemExit("R14 static-evidence digest mismatch")
    sanitization = _read(args.sanitization_manifest)
    if sanitization["manifest_sha256"] != canonical_sha256(
        {key: value for key, value in sanitization.items() if key != "manifest_sha256"}
    ):
        raise SystemExit("R14 sanitization-manifest digest mismatch")
    _copy_shim(args.verl_shim)

    heads = {case["case_id"]: case["head_sha"] for case in selection_material["cases"]}
    tt3955_names = [
        "test_ep_overlap_finds_ready_wait_in_production_combine_graph",
        "test_ep_overlap_keeps_transformer_batch_first_marker_wait_gated",
        "test_ep_overlap_preserves_minimal_async_ep_dispatch_buffer_lifetime",
        "test_ep_overlap_rejects_mismatched_token_exchange_count",
        "test_minimal_async_ep_requires_full_graph_recompute",
        "test_two_buffer_sets_match_reference_and_preserve_outputs",
    ]
    meg6973_names = [
        "test_hfsdp_overlap_losses_match_baseline",
        "test_hsdp_defers_dp_outer_allreduce_to_last_microbatch",
        "test_hsdp_overlap_losses_match_baseline",
    ]
    verl7591_names = [
        "test_callback_exception_not_masked_by_cleanup",
        "test_overlap_empty_weights",
        "test_overlap_large_weight",
        "test_overlap_mixed_dtypes",
        "test_overlap_multiple_buckets",
    ]
    verl7589_names = [
        "test_overlap_empty_weights",
        "test_overlap_large_weight",
        "test_overlap_mixed_dtypes",
        "test_overlap_multiple_buckets",
    ]
    records = [
        {
            "case_id": "torchtitan-pr-3955",
            "purpose": "run the candidate's real two-rank MinimalAsyncEP test under torchrun",
            "timeout": 360,
            "command": (
                "cd /workspace/r13-run-torchtitan && "
                "git switch --detach refs/r14/pr-3955 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {shlex.quote(heads["torchtitan-pr-3955"])} && '
                "timeout 300s env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. "
                "/workspace/venv-tt/bin/torchrun --standalone --nproc-per-node=2 "
                "-m pytest -q -rs --tb=short --maxfail=1 "
                "tests/unit_tests/test_minimal_async_ep_kernels.py "
                "torchtitan/experiments/graph_trainer/tests/test_passes.py "
                f"-k {_pytest_names(tt3955_names)}"
            ),
        },
        {
            "case_id": "torchtitan-pr-4051",
            "purpose": "retry the four-rank candidate test with both rented GPUs visible",
            "timeout": 180,
            "command": (
                "cd /workspace/r13-run-torchtitan && "
                "git switch --detach refs/r14/pr-4051 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {shlex.quote(heads["torchtitan-pr-4051"])} && '
                "timeout 120s env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. "
                "/workspace/venv-tt/bin/python -m pytest -q -rs --tb=short "
                "tests/unit_tests/test_distributed_muon.py "
                "-k test_shard0_shard1_bucket_matches_plain_muon"
            ),
        },
        {
            "case_id": "torchtitan-pr-3980",
            "purpose": "isolate candidate unit tests from the unrelated vLLM integration import",
            "timeout": 180,
            "command": (
                "cd /workspace/r13-run-torchtitan && "
                "git switch --detach refs/r14/pr-3980 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {shlex.quote(heads["torchtitan-pr-3980"])} && '
                "timeout 120s env CUDA_VISIBLE_DEVICES=0 "
                "INFRASWE_R14_BYPASS_TORCHTITAN_RL_INIT=1 "
                "INFRASWE_R14_TORCHTITAN_ROOT=/workspace/r13-run-torchtitan "
                "PYTHONPATH=/workspace/r14-verl-shim:. "
                "/workspace/venv-tt/bin/python -m pytest -q -rs --tb=short --maxfail=1 "
                "tests/unit_tests/test_context_parallel.py "
                "torchtitan/experiments/rl/tests/test_trainer.py"
            ),
        },
        {
            "case_id": "megatron-pr-6973",
            "purpose": (
                "confirm the candidate's topology requirement under an exact two-rank launch"
            ),
            "timeout": 300,
            "command": (
                "cd /workspace/r13-run-megatron && "
                "git switch --detach refs/r14/pr-6973 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {shlex.quote(heads["megatron-pr-6973"])} && '
                "timeout 240s env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. "
                "/venv/main/bin/torchrun --standalone --nproc-per-node=2 "
                "-m pytest -q -rs --tb=short --maxfail=1 "
                "tests/unit_tests/distributed/mfsdp_v2/test_fully_shard.py "
                f"-k {_pytest_names(meg6973_names)}"
            ),
        },
        *[
            {
                "case_id": case_id,
                "purpose": (
                    "run standalone SHM/CUDA-IPC bucket tests through a fail-closed import shim"
                ),
                "timeout": 360,
                "command": (
                    "cd /workspace/r13-run-verl && "
                    f"git switch --detach refs/r14/pr-{number} >/dev/null && "
                    f'test "$(git rev-parse HEAD)" = {shlex.quote(heads[case_id])} && '
                    "timeout 300s env CUDA_VISIBLE_DEVICES=0,1 "
                    "INFRASWE_R14_BYPASS_VERL_VLLM_ROLLOUT_INIT=1 "
                    "INFRASWE_R14_VERL_ROOT=/workspace/r13-run-verl "
                    "PYTHONPATH=/workspace/r14-verl-shim:. "
                    "/venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 "
                    "tests/utils/test_bucketed_weight_transfer.py "
                    f"-k {_pytest_names(names)}"
                ),
                "shim_sha256": "sha256:" + hashlib.sha256(args.verl_shim.read_bytes()).hexdigest(),
            }
            for case_id, number, names in (
                ("verl-pr-7591", 7591, verl7591_names),
                ("verl-pr-7589", 7589, verl7589_names),
            )
        ],
        {
            "case_id": "verl-pr-7045",
            "purpose": "import the production module that the candidate's focused tests omit",
            "timeout": 90,
            "command": (
                "cd /workspace/r13-run-verl && "
                "git switch --detach refs/r14/pr-7045 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {shlex.quote(heads["verl-pr-7045"])} && '
                "timeout 60s env PYTHONPATH=. /venv/main/bin/python -c "
                + shlex.quote(
                    "import sys,types; "
                    "cupy=types.ModuleType('cupy'); cupy.ndarray=object; "
                    "sys.modules['cupy']=cupy; "
                    "import verl.checkpoint_engine.nccl_checkpoint_engine"
                )
            ),
        },
    ]
    if args.only:
        requested = set(args.only)
        records = [record for record in records if record["case_id"] in requested]
        missing = requested - {record["case_id"] for record in records}
        if missing:
            raise SystemExit(f"unknown follow-up case IDs: {sorted(missing)}")

    started_at = datetime.now(UTC).isoformat()
    for index, record in enumerate(records, 1):
        began = time.monotonic()
        try:
            process = _ssh(record["command"], record["timeout"])
            output = process.stdout + process.stderr
            record["returncode"] = process.returncode
            record["status"] = "completed"
        except subprocess.TimeoutExpired as error:
            output = str(error.stdout or "") + str(error.stderr or "")
            record["returncode"] = 124
            record["status"] = "ssh-timeout"
        record["duration_seconds"] = time.monotonic() - began
        record["output_sha256"] = "sha256:" + hashlib.sha256(output.encode()).hexdigest()
        record["output_tail"] = output[-16000:]
        print(
            f"[{index}/{len(records)}] {record['case_id']}: "
            f"rc={record['returncode']} {record['duration_seconds']:.1f}s"
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": "r14-exact-followup-tests-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "sanitized_source_bundle_sha256": sanitization["sanitized_source_bundle_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "remote": {"host": REMOTE, "port": PORT, "gpu_count": 2},
        "outcome_review_ci_fields_requested": False,
        "records": records,
    }
    atomic_write_json(
        args.output,
        {**material, "evidence_sha256": canonical_sha256(material)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
