#!/usr/bin/env python3
# ruff: noqa: E501
"""Run focused R18 probes after separating evaluator import and target gaps."""

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
        raise SystemExit("R18 selection digest mismatch")
    for payload, field, label in (
        (plan, "test_plan_sha256", "test plan"),
        (static, "evidence_sha256", "static evidence"),
        (initial, "evidence_sha256", "initial tests"),
    ):
        material = {key: value for key, value in payload.items() if key != field}
        if payload[field] != canonical_sha256(material):
            raise SystemExit(f"R18 {label} digest mismatch")
        if payload["selection_lock_sha256"] != selection["selection_lock_sha256"]:
            raise SystemExit(f"R18 {label}/selection binding mismatch")
    cases = {case["case_id"]: case for case in selection["selection_material"]["cases"]}

    def switch(worktree: str, case_id: str) -> str:
        case = cases[case_id]
        lfs = "GIT_LFS_SKIP_SMUDGE=1 " if case_id.startswith("tensorrt_llm-") else ""
        return (
            f"cd {shlex.quote(worktree)} && {lfs}git switch --detach "
            f"refs/r18/pr-{int(case['pull_number'])} >/dev/null && "
            f'test "$(git rev-parse HEAD)" = {shlex.quote(case["head_sha"])} && '
        )

    sglang_env = "env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r18-deps/common:/workspace/r17-deps/opencv:/workspace/r14-shims:python:."
    vllm_env = "env CUDA_VISIBLE_DEVICES=0 INFRASWE_R15_VLLM_SOURCE_IMPORT_SHIM=1 PYTHONPATH=/workspace/r17-deps/vllm:/workspace/r18-deps/common:/workspace/r15-vllm-shim:/workspace/r14-shims:."
    tensor_env = "env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r18-deps/common:/workspace/r17-deps/tensorrt:."
    commands = [
        {
            "case_id": "sglang-pr-27181",
            "lane": 1,
            "purpose": "isolate the candidate LoRA overlap state machine from gated model integration",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27181")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/registered/unit/lora/test_lora_overlap_loader_unit.py",
        },
        {
            "case_id": "sglang-pr-27183",
            "lane": 1,
            "purpose": "run Gemma4 YOCO unit contracts behind fail-closed unrelated optional-kernel imports",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27183")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                "import torch,sgl_kernel,pytest; "
                "lib=torch.library.Library('sgl_kernel','FRAGMENT'); "
                "lib.define('moe_fused_gate(Tensor input_tensor, Tensor bias, int num_expert_group, int topk_group, int topk, int num_fused_shared_experts=0, float routed_scaling_factor=0, bool apply_routed_scaling_factor_on_output=False) -> (Tensor, Tensor)'); "
                "missing=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('fail-closed optional kernel shim executed')); "
                "sgl_kernel.__getattr__=lambda _name: missing; "
                "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','test/registered/unit/models/test_gemma4_yoco_fast_prefill.py']))"
            ),
        },
        {
            "case_id": "tensorrt_llm-pr-14765",
            "lane": 1,
            "purpose": "run the CPU-only LoRA layout parser without importing unavailable generated bindings",
            "command": switch("/workspace/r17-tensorrt-wt", "tensorrt_llm-pr-14765")
            + f"timeout 180s {tensor_env} /venv/main/bin/python -c "
            + shlex.quote(
                "import sys,types,pytest; "
                "package=types.ModuleType('tensorrt_llm'); "
                "package.__path__=['/workspace/r17-tensorrt-wt/tensorrt_llm']; "
                "sys.modules['tensorrt_llm']=package; "
                "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','--noconftest','tests/unittest/_torch/lora/test_lora_layout_sidecar.py']))"
            ),
        },
        {
            "case_id": "vllm-pr-44450",
            "lane": 0,
            "purpose": "run the two candidate model-runner LoRA tests with a minimal tests.utils boundary",
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44450")
            + f"timeout 180s {vllm_env} /venv/main/bin/python /workspace/r15-probes/r15_vllm_pytest_driver.py --stub-tests-utils -- -q -rs --tb=short --maxfail=1 tests/v1/worker/test_gpu_model_runner.py -k "
            + shlex.quote(
                "test_set_active_mm_loras_builds_tower_and_connector_mappings or test_update_states_new_request"
            ),
        },
        {
            "case_id": "vllm-pr-44518",
            "lane": 0,
            "purpose": "run the five candidate native packed-audio attention contracts with an inert unrelated asset prewarm",
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44518")
            + f"timeout 180s {vllm_env} /venv/main/bin/python /workspace/r15-probes/r15_vllm_pytest_driver.py --stub-tests-utils -- -q -rs --tb=short --maxfail=1 tests/models/multimodal/processing/test_qwen2_5_omni_audio_tower.py -k "
            + shlex.quote(
                "test_audio_attention_forwards_varlen_metadata_to_mm_encoder_attention or test_audio_encoder_forward_uses_mm_encoder_attention or test_audio_encoder_load_weights_remaps_hf_qkv_to_packed_qkv or test_audio_encoder_uses_packed_qkv_weight_structure or test_qwen2_5_omni_audio_tower_is_vllm_native"
            ),
        },
        {
            "case_id": "tensorrt_llm-pr-14849",
            "lane": 1,
            "purpose": "classify the exact cache rollback test at the generated-bindings boundary",
            "command": switch("/workspace/r17-tensorrt-wt", "tensorrt_llm-pr-14849")
            + f"timeout 180s {tensor_env} /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 --noconftest tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_stats_behavior.py -k test_waited_context_allocation_reports_pending_stats_when_scheduled",
        },
    ]

    def execute(spec: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            process = _ssh(spec["command"], timeout=210)
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
        "protocol_id": "r18-focused-outcome-free-followups-v0.1",
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
