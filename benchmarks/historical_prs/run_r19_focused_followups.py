#!/usr/bin/env python3
# ruff: noqa: E501
"""Run focused R19 probes after separating cache, import, and target gaps."""

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
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--initial-tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    static = read(args.static_evidence)
    initial = read(args.initial_tests)
    if selection["selection_lock_sha256"] != canonical_sha256(selection["selection_material"]):
        raise SystemExit("R19 selection digest mismatch")
    for payload, field, label in (
        (plan, "test_plan_sha256", "test plan"),
        (static, "evidence_sha256", "static evidence"),
        (initial, "evidence_sha256", "initial tests"),
    ):
        material = {key: value for key, value in payload.items() if key != field}
        if payload[field] != canonical_sha256(material):
            raise SystemExit(f"R19 {label} digest mismatch")
        if payload["selection_lock_sha256"] != selection["selection_lock_sha256"]:
            raise SystemExit(f"R19 {label}/selection binding mismatch")
    cases = {case["case_id"]: case for case in selection["selection_material"]["cases"]}

    def switch(worktree: str, case_id: str) -> str:
        case = cases[case_id]
        lfs = "GIT_LFS_SKIP_SMUDGE=1 " if case_id.startswith("tensorrt_llm-") else ""
        return (
            f"cd {shlex.quote(worktree)} && {lfs}git switch --detach "
            f"refs/r19/pr-{int(case['pull_number'])} >/dev/null && "
            f'test "$(git rev-parse HEAD)" = {shlex.quote(case["head_sha"])} && '
        )

    sglang_env = "env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r18-deps/common:/workspace/r17-deps/opencv:/workspace/r14-shims:python:."
    sglang_tilelang_env = "env CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/workspace/r19-deps/tilelang/z3/lib PYTHONPATH=/workspace/r19-deps/tilelang:/workspace/r18-deps/common:/workspace/r17-deps/opencv:/workspace/r14-shims:python:."
    tensor_env = "env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r18-deps/common:/workspace/r17-deps/tensorrt:."
    optional_kernel_driver = (
        "import torch,sgl_kernel,pytest; "
        "lib=torch.library.Library('sgl_kernel','FRAGMENT'); "
        "lib.define('moe_fused_gate(Tensor input_tensor, Tensor bias, int num_expert_group, int topk_group, int topk, int num_fused_shared_experts=0, float routed_scaling_factor=0, bool apply_routed_scaling_factor_on_output=False) -> (Tensor, Tensor)'); "
        "missing=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('fail-closed optional kernel shim executed')); "
        "sgl_kernel.__getattr__=lambda _name: missing; "
    )
    tensor_lora_driver = (
        "import sys,types,pytest; from pydantic import BaseModel; "
        "root='/workspace/r19-tensorrt-wt/tensorrt_llm'; "
        "paths={'tensorrt_llm':root,'tensorrt_llm._torch':root+'/_torch','tensorrt_llm._torch.modules':root+'/_torch/modules','tensorrt_llm._torch.modules.fused_moe':root+'/_torch/modules/fused_moe','tensorrt_llm._torch.peft':root+'/_torch/peft','tensorrt_llm._torch.peft.lora':root+'/_torch/peft/lora','tensorrt_llm.llmapi':root+'/llmapi'}; "
        "[sys.modules.setdefault(n,(lambda n=n,p=p:(lambda m:(setattr(m,'__path__',[p]),m)[1])(types.ModuleType(n)))()) for n,p in paths.items()]; "
        "u=types.ModuleType('tensorrt_llm.llmapi.utils'); u.StrictBaseModel=type('StrictBaseModel',(BaseModel,),{}); sys.modules[u.__name__]=u; "
        "cut=types.ModuleType('tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass'); cut.CutlassFusedMoE=type('CutlassFusedMoE',(),{}); sys.modules[cut.__name__]=cut; "
        "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','--noconftest','tests/unittest/_torch/lora/test_moe_utils.py']))"
    )
    commands = [
        {
            "case_id": "flashinfer-pr-4900",
            "lane": 0,
            "timeout": 660,
            "purpose": "rerun the exact added matrix with a unique JIT workspace to exclude cross-head module-cache contamination",
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-4900")
            + "timeout 630s env CUDA_VISIBLE_DEVICES=0 FLASHINFER_WORKSPACE_BASE=/workspace/r19-cache/flashinfer-4900-v2 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/attention/test_block_sparse.py -k "
            + shlex.quote(
                "test_a_second_plan_still_takes_a_page_size_after_auto_resolved or test_block_sparse_paged_route or test_block_sparse_paged_route_rejects_bad_geometry or test_paged_fp8_cache_sizes_its_default_scales_by_the_head_count or test_paged_route_reads_a_wide_quantized_head"
            ),
        },
        {
            "case_id": "sglang-pr-37638",
            "lane": 1,
            "timeout": 420,
            "purpose": "run the exact TileLang ragged-tail matrix with the source-pinned optional dependency",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-37638")
            + f"timeout 390s {sglang_tilelang_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/registered/kernels/test_qsa_prefill_ragged.py",
        },
        {
            "case_id": "sglang-pr-27285",
            "lane": 1,
            "timeout": 210,
            "purpose": "isolate the candidate unified-radix unit path from an unrelated installed sgl-kernel API mismatch",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27285")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                optional_kernel_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py']))"
            ),
        },
        {
            "case_id": "sglang-pr-27297",
            "lane": 1,
            "timeout": 210,
            "purpose": "run the two candidate LingBot transport/cache unit files behind fail-closed unrelated kernel imports",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27297")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                optional_kernel_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','python/sglang/multimodal_gen/test/unit/realtime/test_lingbot_causal_denoising.py','python/sglang/multimodal_gen/test/unit/realtime/test_realtime_output_transport.py']))"
            ),
        },
        {
            "case_id": "sglang-pr-27174",
            "lane": 1,
            "timeout": 210,
            "purpose": "run the load-metric unit route behind fail-closed unrelated kernel imports",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27174")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                optional_kernel_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','test/registered/unit/mem_cache/test_radix_force_miss.py']))"
            ),
        },
        {
            "case_id": "sglang-pr-27298",
            "lane": 1,
            "timeout": 210,
            "purpose": "run the serving-chat unit route behind fail-closed unrelated kernel imports",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27298")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                optional_kernel_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','test/registered/unit/entrypoints/openai/test_serving_chat.py']))"
            ),
        },
        {
            "case_id": "tensorrt_llm-pr-14764",
            "lane": 1,
            "timeout": 210,
            "purpose": "run the CPU-only MoE-LoRA contract without importing unavailable generated bindings or an unrelated backend implementation",
            "command": switch("/workspace/r19-tensorrt-wt", "tensorrt_llm-pr-14764")
            + f"timeout 180s {tensor_env} /venv/main/bin/python -c "
            + shlex.quote(tensor_lora_driver),
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
            "output_tail": output[-20000:],
        }

    started_at = datetime.now(UTC).isoformat()
    lanes = [[spec for spec in commands if spec["lane"] == lane] for lane in (0, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        lane_results = list(executor.map(lambda specs: [execute(spec) for spec in specs], lanes))
    records = [
        record
        for spec in commands
        for lane in lane_results
        for record in lane
        if record["case_id"] == spec["case_id"]
    ]
    material = {
        "schema_version": "0.1",
        "protocol_id": "r19-focused-outcome-free-followups-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "initial_test_evidence_sha256": initial["evidence_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_requested": False,
        "records": records,
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {record["case_id"]: record["returncode"] for record in records},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
