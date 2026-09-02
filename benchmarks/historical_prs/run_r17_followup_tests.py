#!/usr/bin/env python3
# ruff: noqa: E501
"""Run exact-head R17 follow-ups after classifying initial environment gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_r15_followup_tests import _ssh

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
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--initial-tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selection = read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R17 selection digest mismatch")
    plan = read(args.test_plan)
    if plan["test_plan_sha256"] != canonical_sha256(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R17 test-plan digest mismatch")
    static = read(args.static_evidence)
    if static["evidence_sha256"] != canonical_sha256(
        {key: value for key, value in static.items() if key != "evidence_sha256"}
    ):
        raise SystemExit("R17 static-evidence digest mismatch")
    initial = read(args.initial_tests)
    if initial["evidence_sha256"] != canonical_sha256(
        {key: value for key, value in initial.items() if key != "evidence_sha256"}
    ):
        raise SystemExit("R17 initial-test digest mismatch")
    binding = selection["selection_lock_sha256"]
    if any(item["selection_lock_sha256"] != binding for item in (plan, static, initial)):
        raise SystemExit("R17 evidence binding mismatch")

    cases = {case["case_id"]: case for case in selection_material["cases"]}
    static_cases = {case["case_id"]: case for case in static["cases"]}

    def head(case_id: str) -> str:
        return shlex.quote(cases[case_id]["head_sha"])

    def names(case_id: str) -> str:
        return shlex.quote(" or ".join(static_cases[case_id]["candidate_test_functions_added"]))

    def switch(worktree: str, case_id: str) -> str:
        number = cases[case_id]["pull_number"]
        lfs = "GIT_LFS_SKIP_SMUDGE=1 " if case_id.startswith("tensorrt_llm-") else ""
        return (
            f"cd {shlex.quote(worktree)} && {lfs}git switch --detach "
            f"refs/r17/pr-{number} >/dev/null && "
            f'test "$(git rev-parse HEAD)" = {head(case_id)} && '
        )

    vllm_env = (
        "env CUDA_VISIBLE_DEVICES=0 INFRASWE_R15_VLLM_SOURCE_IMPORT_SHIM=1 "
        "PYTHONPATH=/workspace/r17-deps/vllm:/workspace/r15-vllm-shim:/workspace/r14-shims:."
    )
    sglang_driver = (
        "import sgl_kernel,pytest; "
        "sgl_kernel.fp8_blockwise_scaled_mm=lambda *a,**k: (_ for _ in ()).throw("
        "RuntimeError('fail-closed optional kernel shim was executed')); "
    )
    records: list[dict[str, Any]] = [
        {
            "case_id": "liger-pr-1435",
            "lane": 0,
            "purpose": "run the available A100 SwiGLU MLP regression control; B200 remains explicit",
            "timeout": 360,
            "command": switch("/workspace/r13-run-liger", "liger-pr-1435")
            + "timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/transformers/test_mlp.py -k swiglu",
        },
        {
            "case_id": "liger-pr-1157",
            "lane": 0,
            "purpose": "exercise forward-only fused linear cross entropy with the corrected saved-tensor guard",
            "timeout": 300,
            "command": switch("/workspace/r13-run-liger", "liger-pr-1157")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/transformers/test_fused_linear_cross_entropy.py -k correctness_with_forward_only",
        },
        {
            "case_id": "flashinfer-pr-4861",
            "lane": 0,
            "purpose": "rerun per-request sampling seeds after restoring the package data boundary",
            "timeout": 360,
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-4861")
            + "test -L flashinfer/data && timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/utils/test_sampling.py -k "
            + names("flashinfer-pr-4861"),
        },
        {
            "case_id": "flashinfer-pr-3467",
            "lane": 0,
            "purpose": "rerun custom-mask validation after restoring the package data boundary",
            "timeout": 300,
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-3467")
            + "test -L flashinfer/data && timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/attention/test_custom_mask_validation.py -k "
            + names("flashinfer-pr-3467"),
        },
        {
            "case_id": "flashinfer-pr-3474",
            "lane": 0,
            "purpose": "allow a warm-cache A100 decode rerun beyond the initial compile timeout",
            "timeout": 660,
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-3474")
            + "test -L flashinfer/data && timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/attention/test_trtllm_gen_attention_decode.py",
        },
        {
            "case_id": "vllm-pr-44558",
            "lane": 0,
            "purpose": "run focused scheduler cadence contracts through the fail-closed source import shim",
            "timeout": 300,
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44558")
            + f"timeout 240s {vllm_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/v1/core/test_scheduler.py -k "
            + names("vllm-pr-44558"),
        },
        {
            "case_id": "vllm-pr-44544",
            "lane": 0,
            "purpose": "repeat the exact availability assertion independently of test ordering",
            "timeout": 180,
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44544")
            + f"timeout 120s {vllm_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/v1/attention/test_mla_prefill_selector.py::TestAiterAsmIsAvailable::test_unavailable_when_max_kvlen_absent",
        },
        {
            "case_id": "vllm-pr-44513",
            "lane": 0,
            "purpose": "collect the XPU online-quantization test on A100 without unrelated multimodal imports",
            "timeout": 180,
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44513")
            + f"timeout 120s {vllm_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/quantization/test_online.py -k xpu",
        },
        {
            "case_id": "tensorrt_llm-pr-18596",
            "lane": 0,
            "purpose": "run the exact FlashInfer single-token decode plan-cache test from the LFS-safe worktree",
            "timeout": 360,
            "command": switch("/workspace/r17-tensorrt-wt", "tensorrt_llm-pr-18596")
            + "timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/r17-deps/tensorrt:. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/unittest/_torch/attention/test_flashinfer_attention.py -k "
            + names("tensorrt_llm-pr-18596"),
        },
        {
            "case_id": "tensorrt_llm-pr-14945",
            "lane": 0,
            "purpose": "run candidate KV-cache-v2 beam ownership tests without model integration",
            "timeout": 300,
            "command": switch("/workspace/r17-tensorrt-wt", "tensorrt_llm-pr-14945")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/r17-deps/tensorrt:. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py -k "
            + names("tensorrt_llm-pr-14945"),
        },
        {
            "case_id": "tensorrt_llm-pr-14911",
            "lane": 0,
            "purpose": "run candidate AutoDeploy multi-pool and cyclic-window exact tests",
            "timeout": 360,
            "command": switch("/workspace/r17-tensorrt-wt", "tensorrt_llm-pr-14911")
            + "timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/r17-deps/tensorrt:. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_trtllm_attention_op.py tests/unittest/auto_deploy/singlegpu/shim/test_ad_executor_swa_eviction.py tests/unittest/auto_deploy/singlegpu/shim/test_llm_config.py tests/unittest/auto_deploy/singlegpu/transformations/library/test_kv_cache_trtllm_multipool.py tests/unittest/auto_deploy/singlegpu/transformations/library/test_kvcache_vswa_metadata.py -k "
            + names("tensorrt_llm-pr-14911"),
        },
        {
            "case_id": "tensorrt_llm-pr-14922",
            "lane": 0,
            "purpose": "run candidate FPM timing and serializer unit contracts",
            "timeout": 300,
            "command": switch("/workspace/r17-tensorrt-wt", "tensorrt_llm-pr-14922")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/r17-deps/tensorrt:. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/unittest/executor/test_stats_serializer.py tests/unittest/pyexecutor/test_iter_stats_populate.py -k "
            + names("tensorrt_llm-pr-14922"),
        },
        {
            "case_id": "megatron-pr-5047",
            "lane": 1,
            "purpose": "validate exact head syntax and enumerate the declared eight-rank TP/CP matrix",
            "timeout": 120,
            "command": switch("/workspace/r13-run-megatron", "megatron-pr-5047")
            + "timeout 60s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python -m py_compile megatron/core/transformer/moe/router.py tests/unit_tests/transformer/moe/test_aux_loss.py && grep -n \"(8, 1, 1).*2, 2, 2\" tests/unit_tests/transformer/moe/test_aux_loss.py",
        },
        {
            "case_id": "slime-pr-1959",
            "lane": 1,
            "purpose": "exercise literal, template, fallback, and evaluation path resolution",
            "timeout": 120,
            "command": switch("/workspace/r13-run-slime", "slime-pr-1959")
            + "timeout 60s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python -c "
            + shlex.quote(
                "import tempfile; from pathlib import Path; from types import SimpleNamespace as N; "
                "from slime.rollout.forge_load import _resolve_path; d=Path(tempfile.mkdtemp()); "
                "p=d/'0.pt'; p.touch(); a=N(load_forge_rollout_data=str(p)); "
                "assert _resolve_path(a,7,False)==str(p); assert _resolve_path(a,7,True) is None; "
                "t=N(load_forge_rollout_data=str(d/'{rollout_id}.pt')); "
                "assert _resolve_path(t,7,False)==str(p); assert _resolve_path(t,7,True) is None; "
                "print('forge_load_path_matrix=pass')"
            ),
        },
        {
            "case_id": "torchtitan-pr-4398",
            "lane": 1,
            "purpose": "rerun candidate valid-token contracts with the repository-pinned dependencies",
            "timeout": 300,
            "command": switch("/workspace/r13-run-torchtitan", "torchtitan-pr-4398")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r14-shims:/workspace/r15-deps/spmd-0.2.5:/workspace/r15-deps/torchtitan-pinned:. /workspace/venv-tt/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/unit_tests/cpu/components/data/test_grain_data.py tests/unit_tests/cpu/test_trainer.py -k "
            + names("torchtitan-pr-4398"),
        },
        {
            "case_id": "verl-pr-7685",
            "lane": 1,
            "purpose": "run deduplicated offload ownership tests against exact Megatron source",
            "timeout": 300,
            "command": switch("/workspace/r13-run-verl", "verl-pr-7685")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/utils/megatron/test_param_offload_deduplication.py -k "
            + names("verl-pr-7685"),
        },
        {
            "case_id": "verl-pr-6526",
            "lane": 1,
            "purpose": "run optimizer precision matrix against exact Megatron source",
            "timeout": 300,
            "command": switch("/workspace/r13-run-verl", "verl-pr-6526")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/utils/megatron/test_optimizer.py -k "
            + names("verl-pr-6526"),
        },
        {
            "case_id": "sglang-pr-27274",
            "lane": 1,
            "purpose": "cross the unrelated optional-kernel import with a fail-closed alias and run LoRA virtual-expert tests",
            "timeout": 300,
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27274")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r17-deps/opencv:/workspace/r14-shims:python:. /venv/main/bin/python -c "
            + shlex.quote(
                sglang_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','--noconftest','test/registered/lora/test_virtual_experts_kernels.py','-k',"
                + repr(" or ".join(static_cases["sglang-pr-27274"]["candidate_test_functions_added"]))
                + "]))"
            ),
        },
        {
            "case_id": "sglang-pr-27203",
            "lane": 1,
            "purpose": "cross the unrelated optional-kernel import and run causal-denoising unit tests",
            "timeout": 300,
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27203")
            + "timeout 240s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r17-deps/opencv:/workspace/r14-shims:python:. /venv/main/bin/python -c "
            + shlex.quote(
                sglang_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py','-k',"
                + repr(" or ".join(static_cases["sglang-pr-27203"]["candidate_test_functions_added"]))
                + "]))"
            ),
        },
    ]
    if args.only:
        requested = set(args.only)
        records = [record for record in records if record["case_id"] in requested]
        missing = requested - {record["case_id"] for record in records}
        if missing:
            raise SystemExit(f"unknown follow-up case IDs: {sorted(missing)}")

    def run_record(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        began = time.monotonic()
        try:
            process = _ssh(record["command"], int(record["timeout"]))
            output = process.stdout + process.stderr
            returncode = process.returncode
            status = "completed"
        except subprocess.TimeoutExpired as error:
            output = str(error.stdout or "") + str(error.stderr or "")
            returncode = 124
            status = "ssh-timeout"
        result = {
            **record,
            "returncode": returncode,
            "status": status,
            "duration_seconds": time.monotonic() - began,
            "output_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
            "output_tail": output[-20000:],
        }
        print(
            f"[{index}/{len(records)}] {record['case_id']}: "
            f"rc={returncode} {result['duration_seconds']:.1f}s",
            flush=True,
        )
        return index, result

    indexed = list(enumerate(records, 1))
    lanes = {
        lane: [(index, record) for index, record in indexed if record["lane"] == lane]
        for lane in (0, 1)
    }

    def run_lane(items: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
        return [run_record(index, record) for index, record in items]

    started_at = datetime.now(UTC).isoformat()
    completed: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_lane, lanes[lane]) for lane in (0, 1)]
        for future in as_completed(futures):
            completed.extend(future.result())
    material = {
        "schema_version": "0.1",
        "protocol_id": "r17-exact-head-followups-v0.1",
        "selection_lock_sha256": binding,
        "test_plan_sha256": plan["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "initial_test_evidence_sha256": initial["evidence_sha256"],
        "environment_repairs": [
            "FlashInfer package data symlink restored from the installed exact data package",
            "TorchTitan spmd_types==0.2.5 and pinned Grain dependency paths used",
            "verl tests receive the exact Megatron source path",
            "TensorRT-LLM runs from a clean LFS-smudge-disabled worktree",
            "vLLM source tests use the existing fail-closed compiled-extension import shim",
            "SGLang optional fp8 import alias raises if the unrelated kernel is executed",
            "Targeted mistral-common and OpenCV dependencies are isolated under /workspace/r17-deps",
        ],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "remote": {"host": "38.49.42.120", "port": 54270, "gpu_count": 2},
        "outcome_review_ci_fields_requested": False,
        "records": [record for _, record in sorted(completed)],
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
