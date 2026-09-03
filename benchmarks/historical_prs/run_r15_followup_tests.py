#!/usr/bin/env python3
# ruff: noqa: E501
"""Run exact-head R15 follow-ups for topology, memory, and import boundaries."""

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


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


def _copy_files(paths: list[Path], destination: str) -> dict[str, str]:
    setup = _ssh(f"mkdir -p {shlex.quote(destination)}", 30)
    if setup.returncode:
        raise RuntimeError(setup.stderr.strip())
    digests: dict[str, str] = {}
    for path in paths:
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
                str(path),
                f"{REMOTE}:{destination}/{path.name}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.strip())
        digests[path.name] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _pytest_names(names: list[str]) -> str:
    return shlex.quote(" or ".join(names))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--vllm-shim-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("R15 selection digest mismatch")
    plan = _read(args.test_plan)
    if plan["test_plan_sha256"] != canonical_sha256(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R15 test-plan digest mismatch")
    static = _read(args.static_evidence)
    if static["evidence_sha256"] != canonical_sha256(
        {key: value for key, value in static.items() if key != "evidence_sha256"}
    ):
        raise SystemExit("R15 static-evidence digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R15 plan/selection binding mismatch")
    if static["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R15 static/selection binding mismatch")

    probe_names = [
        "r15_megatron_mxfp8_gate_probe.py",
        "r15_megatron_teardown_probe.py",
        "r15_slime_phase_memory_probe.py",
        "r15_topk_memory_probe.py",
        "r15_verl_hccl_async_probe.py",
        "r15_verl_weight_sync_order_probe.py",
        "r15_vllm_pytest_driver.py",
        "r15_vllm_zmq_atomic_bind_probe.py",
    ]
    probe_paths = [args.probe_dir / name for name in probe_names]
    probe_hashes = _copy_files(probe_paths, "/workspace/r15-probes")
    shim_paths = [
        args.vllm_shim_dir / "sitecustomize.py",
        args.vllm_shim_dir / "vllm-0.0.0.dist-info" / "METADATA",
        args.vllm_shim_dir / "vllm-0.0.0.dist-info" / "top_level.txt",
    ]
    shim_hashes = _copy_files(shim_paths[:1], "/workspace/r15-vllm-shim")
    shim_hashes.update(_copy_files(shim_paths[1:], "/workspace/r15-vllm-shim/vllm-0.0.0.dist-info"))

    cases = {case["case_id"]: case for case in material["cases"]}
    static_cases = {case["case_id"]: case for case in static["cases"]}

    def head(case_id: str) -> str:
        return shlex.quote(cases[case_id]["head_sha"])

    def base(case_id: str) -> str:
        return shlex.quote(cases[case_id]["base_sha"])

    sglang_27150_names = static_cases["sglang-pr-27150"]["candidate_test_functions_added"]
    tt_4399_names = static_cases["torchtitan-pr-4399"]["candidate_test_functions_added"]
    vllm_env = (
        "env CUDA_VISIBLE_DEVICES=0 INFRASWE_R15_VLLM_SOURCE_IMPORT_SHIM=1 "
        "PYTHONPATH=/workspace/r15-vllm-shim:/workspace/r14-shims:."
    )
    records: list[dict[str, Any]] = [
        {
            "case_id": "sglang-pr-37523",
            "purpose": "run the candidate NCCL dispatcher matrix on both A100 ranks",
            "timeout": 360,
            "command": (
                "cd /workspace/r14-run-sglang && git switch --detach refs/r15/pr-37523 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("sglang-pr-37523")} && '
                "timeout 300s env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/workspace/r14-shims:python:. "
                "/venv/main/bin/torchrun --standalone --nproc_per_node=2 -m pytest -q -rs --tb=short "
                "--maxfail=1 --noconftest test/manual/ep/test_nccl_dispatcher.py"
            ),
        },
        {
            "case_id": "torchtitan-pr-3499",
            "purpose": "exercise the available two-rank mixed NCCL P2P schedule",
            "timeout": 240,
            "command": (
                "cd /workspace/r13-run-torchtitan && git switch --detach refs/r15/pr-3499 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("torchtitan-pr-3499")} && '
                "timeout 180s env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /workspace/venv-tt/bin/python "
                "-m pytest -q -rs --tb=short --maxfail=1 tests/test_pp_torchcomms_p2p_deadlock.py "
                "-k mixed_p2p_completes"
            ),
        },
        {
            "case_id": "torchtitan-pr-4399",
            "purpose": "run candidate loss-stage collective tests with its pinned dependencies",
            "timeout": 240,
            "command": (
                "cd /workspace/r13-run-torchtitan && git switch --detach refs/r15/pr-4399 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("torchtitan-pr-4399")} && '
                "test -d /workspace/r15-deps/spmd-0.2.5 && test -d /workspace/r15-deps/torchtitan-pinned && "
                "timeout 180s env CUDA_VISIBLE_DEVICES=0 "
                "PYTHONPATH=/workspace/r15-deps/spmd-0.2.5:/workspace/r15-deps/torchtitan-pinned:. "
                "/workspace/venv-tt/bin/python -m pytest -q -rs --tb=short --maxfail=1 "
                f"tests/unit_tests/cpu/test_trainer.py -k {_pytest_names(tt_4399_names)}"
            ),
        },
        {
            "case_id": "verl-pr-6566",
            "purpose": "run the candidate Megatron optimizer matrix against the exact Megatron source",
            "timeout": 240,
            "command": (
                "cd /workspace/r13-run-verl && git switch --detach refs/r15/pr-6566 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("verl-pr-6566")} && '
                "timeout 180s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron "
                "/venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 "
                "tests/utils/megatron/test_optimizer.py"
            ),
        },
        {
            "case_id": "sglang-pr-27150",
            "purpose": "isolate CPU EPLB logic from an incompatible optional installed CUDA kernel",
            "timeout": 180,
            "command": (
                "cd /workspace/r14-run-sglang && git switch --detach refs/r15/pr-27150 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("sglang-pr-27150")} && '
                "timeout 120s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/r14-shims:python:. "
                "/venv/main/bin/python -c "
                + shlex.quote(
                    "from sglang.srt import utils; utils.is_cuda=lambda:False; import pytest; "
                    f"raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','--noconftest',"
                    "'test/registered/unit/eplb/test_deepep_waterfill_eplb.py','-k',"
                    f"{(' or '.join(sglang_27150_names))!r}]))"
                )
            ),
        },
        {
            "case_id": "liger-pr-1405",
            "purpose": "repeat the candidate gradient counterexample three times",
            "timeout": 240,
            "command": (
                "cd /workspace/r13-run-liger && git switch --detach refs/r15/pr-1405 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("liger-pr-1405")} && '
                "for run in 1 2 3; do echo R15_LIGER_REPEAT=$run; timeout 60s env CUDA_VISIBLE_DEVICES=0 "
                "PYTHONPATH=. /venv/main/bin/python -m pytest -q --tb=short --maxfail=1 "
                "'test/transformers/test_cce.py::test_cce_forward_backward[False-sum-dtype0-shape1]' || test $? = 1; done"
            ),
            "expected_counterexample_returncode": 1,
        },
        {
            "case_id": "megatron-pr-7029",
            "purpose": "repeat process-group create/destroy cycles on two NCCL ranks",
            "timeout": 300,
            "command": (
                "cd /workspace/r13-run-megatron && git switch --detach refs/r15/pr-7029 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("megatron-pr-7029")} && '
                "timeout 240s env CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /venv/main/bin/torchrun "
                "--standalone --nproc_per_node=2 /workspace/r15-probes/r15_megatron_teardown_probe.py"
            ),
        },
        {
            "case_id": "megatron-pr-5146",
            "purpose": "execute every MXFP8 defer-gate branch on the candidate method",
            "timeout": 120,
            "command": (
                "cd /workspace/r13-run-megatron && git switch --detach refs/r15/pr-5146 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("megatron-pr-5146")} && '
                "timeout 60s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python "
                "/workspace/r15-probes/r15_megatron_mxfp8_gate_probe.py"
            ),
        },
        {
            "case_id": "slime-pr-2011",
            "purpose": "compare exact base/head entropy-backward peak memory on A100",
            "timeout": 360,
            "command": (
                "cd /workspace/r13-run-slime && git switch --detach refs/r15/pr-2011 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("slime-pr-2011")} && '
                "cp tools/repro_1951.py /workspace/r15_slime_memory_repro.py && "
                f"git switch --detach {base('slime-pr-2011')} >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {base("slime-pr-2011")} && '
                "echo R15_SLIME_MEMORY=base && timeout 150s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "
                "/venv/main/bin/python /workspace/r15_slime_memory_repro.py --batch 8192 --vocab 32768 "
                "--dtype bfloat16 --with-entropy --backward --chunk-size 1024 && "
                "git switch --detach refs/r15/pr-2011 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("slime-pr-2011")} && '
                "echo R15_SLIME_MEMORY=head && timeout 150s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "
                "/venv/main/bin/python /workspace/r15_slime_memory_repro.py --batch 8192 --vocab 32768 "
                "--dtype bfloat16 --with-entropy --backward --chunk-size 1024"
            ),
        },
        {
            "case_id": "verl-pr-6593",
            "purpose": "measure baseline and chunked top-K backward peaks across chunk and token sweeps",
            "timeout": 600,
            "command": (
                "cd /workspace/r13-run-verl && git switch --detach refs/r15/pr-6593 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("verl-pr-6593")} && '
                "echo R15_TOPK_CASE=baseline && timeout 90s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python "
                "/workspace/r15-probes/r15_topk_memory_probe.py --mode baseline --tokens 8192 --vocab 32768 --top-k 64 && "
                "for chunk in 256 1024 8192; do echo R15_TOPK_CASE=chunk-$chunk; timeout 90s env CUDA_VISIBLE_DEVICES=1 "
                "PYTHONPATH=. /venv/main/bin/python /workspace/r15-probes/r15_topk_memory_probe.py --mode chunked "
                "--tokens 8192 --vocab 32768 --top-k 64 --chunk-size $chunk || exit $?; done && "
                "for tokens in 2048 8192 16384; do echo R15_TOPK_CASE=tokens-$tokens; timeout 90s env CUDA_VISIBLE_DEVICES=1 "
                "PYTHONPATH=. /venv/main/bin/python /workspace/r15-probes/r15_topk_memory_probe.py --mode chunked "
                "--tokens $tokens --vocab 32768 --top-k 64 --chunk-size 256 || exit $?; done"
            ),
        },
        {
            "case_id": "slime-pr-2304",
            "purpose": "observe a real 64-MiB allocation through the candidate phase reporter",
            "timeout": 120,
            "command": (
                "cd /workspace/r13-run-slime && git switch --detach refs/r15/pr-2304 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("slime-pr-2304")} && '
                "timeout 60s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:/workspace/r13-run-megatron "
                "/venv/main/bin/python /workspace/r15-probes/r15_slime_phase_memory_probe.py"
            ),
        },
        {
            "case_id": "verl-pr-7631",
            "purpose": "prove offload follows completed weight transfer and remains strategy-gated",
            "timeout": 120,
            "command": (
                "cd /workspace/r13-run-verl && git switch --detach refs/r15/pr-7631 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("verl-pr-7631")} && '
                "timeout 60s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron "
                "/venv/main/bin/python /workspace/r15-probes/r15_verl_weight_sync_order_probe.py"
            ),
        },
        {
            "case_id": "verl-pr-6569",
            "purpose": "exercise candidate asyncio task, ZMQ ordering, executor, and device binding",
            "timeout": 120,
            "command": (
                "cd /workspace/r13-run-verl && git switch --detach refs/r15/pr-6569 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("verl-pr-6569")} && '
                "timeout 60s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r14-run-vllm "
                "/venv/main/bin/python /workspace/r15-probes/r15_verl_hccl_async_probe.py"
            ),
        },
        {
            "case_id": "vllm-pr-44495",
            "purpose": "validate candidate atomic-bind source and live ZMQ endpoint uniqueness",
            "timeout": 120,
            "command": (
                "cd /workspace/r14-run-vllm && git switch --detach refs/r15/pr-44495 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("vllm-pr-44495")} && '
                "timeout 60s env PYTHONPATH=. /venv/main/bin/python /workspace/r15-probes/r15_vllm_zmq_atomic_bind_probe.py "
                "vllm/distributed/device_communicators/shm_broadcast.py"
            ),
        },
        {
            "case_id": "vllm-pr-54960",
            "purpose": "run all focused EC metric tests including scheduler aggregation",
            "timeout": 300,
            "command": (
                "cd /workspace/r14-run-vllm && git switch --detach refs/r15/pr-54960 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("vllm-pr-54960")} && '
                f"timeout 150s {vllm_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest "
                "tests/v1/ec_connector/unit/test_ec_connector_metrics.py tests/v1/ec_connector/unit/test_worker_ec_connector.py && "
                f"timeout 150s {vllm_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest "
                "tests/v1/core/test_scheduler.py::test_scheduler_ec_connector_stats"
            ),
        },
        {
            "case_id": "vllm-pr-44583",
            "purpose": "run mixed FA/MLA handshake plus the TP-mapping suite from exact source",
            "timeout": 300,
            "command": (
                "cd /workspace/r14-run-vllm && git switch --detach refs/r15/pr-44583 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("vllm-pr-44583")} && '
                f"timeout 150s {vllm_env} /venv/main/bin/python /workspace/r15-probes/r15_vllm_pytest_driver.py "
                "--stub-tests-utils -- -q -rs --tb=short --maxfail=1 "
                "tests/v1/kv_connector/unit/test_nixl_connector.py::TestNixlHandshake::test_handshake_mixed_fa_mla_hetero_tp && "
                f"timeout 120s {vllm_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest "
                "tests/v1/kv_connector/unit/test_tp_mapping.py"
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
        record["output_tail"] = output[-20000:]
        print(
            f"[{index}/{len(records)}] {record['case_id']}: "
            f"rc={record['returncode']} {record['duration_seconds']:.1f}s",
            flush=True,
        )

    evidence_material = {
        "schema_version": "0.1",
        "protocol_id": "r15-exact-followup-tests-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "probe_sha256": probe_hashes,
        "vllm_shim_sha256": shim_hashes,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "remote": {"host": REMOTE, "port": PORT, "gpu_count": 2},
        "outcome_review_ci_fields_requested": False,
        "records": records,
    }
    payload = {**evidence_material, "evidence_sha256": canonical_sha256(evidence_material)}
    atomic_write_json(args.output, payload)
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
