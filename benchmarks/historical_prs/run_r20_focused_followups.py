#!/usr/bin/env python3
# ruff: noqa: E501
"""Run focused R20 probes after separating import and target capability gaps."""

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
        raise SystemExit("R20 selection digest mismatch")
    for payload, field, label in (
        (plan, "test_plan_sha256", "test plan"),
        (static, "evidence_sha256", "static evidence"),
        (initial, "evidence_sha256", "initial tests"),
    ):
        material = {key: value for key, value in payload.items() if key != field}
        if payload[field] != canonical_sha256(material):
            raise SystemExit(f"R20 {label} digest mismatch")
        if payload["selection_lock_sha256"] != selection["selection_lock_sha256"]:
            raise SystemExit(f"R20 {label}/selection binding mismatch")
    cases = {case["case_id"]: case for case in selection["selection_material"]["cases"]}

    def switch(worktree: str, case_id: str) -> str:
        case = cases[case_id]
        return (
            f"cd {shlex.quote(worktree)} && git switch --detach "
            f"refs/r20/pr-{int(case['pull_number'])} >/dev/null && "
            f'test "$(git rev-parse HEAD)" = {shlex.quote(case["head_sha"])} && '
        )

    flashinfer_env = "env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=."
    sglang_env = "env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/r18-deps/common:/workspace/r17-deps/opencv:/workspace/r14-shims:python:."
    vllm_env = "env CUDA_VISIBLE_DEVICES=0 INFRASWE_R15_VLLM_SOURCE_IMPORT_SHIM=1 PYTHONPATH=/workspace/r17-deps/vllm:/workspace/r18-deps/common:/workspace/r15-vllm-shim:/workspace/r14-shims:."
    optional_kernel_driver = (
        "import torch,sgl_kernel,pytest; "
        "lib=torch.library.Library('sgl_kernel','FRAGMENT'); "
        "lib.define('moe_fused_gate(Tensor input_tensor, Tensor bias, int num_expert_group, int topk_group, int topk, int num_fused_shared_experts=0, float routed_scaling_factor=0, bool apply_routed_scaling_factor_on_output=False) -> (Tensor, Tensor)'); "
        "missing=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('fail-closed optional kernel shim executed')); "
        "sgl_kernel.__getattr__=lambda _name: missing; "
    )
    commands = [
        {
            "case_id": "flashinfer-pr-3461",
            "lane": 0,
            "timeout": 510,
            "purpose": "exercise the scalar-k large-vocabulary fast path across the existing logits/probability alignment matrix",
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-3461")
            + f"timeout 480s {flashinfer_env} FLASHINFER_WORKSPACE_BASE=/workspace/r20-cache/flashinfer-3461 /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/utils/test_sampling.py -k top_k_top_p_sampling_from_probs_logits_alignment",
        },
        {
            "case_id": "flashinfer-pr-3465",
            "lane": 0,
            "timeout": 120,
            "purpose": "verify that the CuTe workspace view reserves exactly the TRTLLM-Gen counter slab and aliases the remaining buffer",
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-3465")
            + f"timeout 90s {flashinfer_env} /venv/main/bin/python -c "
            + shlex.quote(
                "import torch; from flashinfer.mla._core import _TRTLLM_GEN_MLA_COUNTER_REGION_BYTES as n,_cute_dsl_workspace_view as view; "
                "x=torch.zeros(n+4096,dtype=torch.uint8,device='cuda'); y=view(x); "
                "assert y.numel()==4096 and y.data_ptr()-x.data_ptr()==n; y[0]=7; assert int(x[n])==7; print('workspace-offset-pass',n,y.numel())"
            ),
        },
        {
            "case_id": "flashinfer-pr-3506",
            "lane": 0,
            "timeout": 120,
            "purpose": "execute CUDA generator state extraction and prove the rounded offset is returned as Python integers",
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-3506")
            + f"timeout 90s {flashinfer_env} /venv/main/bin/python -c "
            + shlex.quote(
                "import torch; from flashinfer.sampling import get_seed_and_offset; g=torch.Generator(device='cuda'); g.manual_seed(123); "
                "seed,offset=get_seed_and_offset(5,g,torch.device('cuda')); assert type(seed) is int and type(offset) is int and seed==123 and offset==8; "
                "seed2,offset2=get_seed_and_offset(1,g,torch.device('cuda')); assert seed2==123 and offset2==12; print('generator-state-pass',seed2,offset2)"
            ),
        },
        {
            "case_id": "flashinfer-pr-3430",
            "lane": 0,
            "timeout": 120,
            "purpose": "compile all four edited modules and verify the two deprecated public spellings are absent from their exact frozen source",
            "command": switch("/workspace/r14-run-flashinfer", "flashinfer-pr-3430")
            + "timeout 90s /venv/main/bin/python -c "
            + shlex.quote(
                "from pathlib import Path; files=['flashinfer/cute_dsl/attention/roles/correction.py','flashinfer/cute_dsl/attention/roles/loader_tma.py','flashinfer/cute_dsl/attention/roles/softmax.py','flashinfer/cute_dsl/gemm_allreduce_two_shot.py']; "
                "texts=[Path(p).read_text() for p in files]; [compile(s,p,'exec') for p,s in zip(files,texts)]; joined='\\n'.join(texts); "
                "assert 'cute.core.ThrMma' not in joined and 'cute.make_fragment(' not in joined; print('deprecated-api-replacement-pass',len(files))"
            ),
        },
        {
            "case_id": "sglang-pr-27257",
            "lane": 1,
            "timeout": 210,
            "purpose": "run the candidate request-slicing regressions behind fail-closed unrelated optional-kernel imports",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27257")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                optional_kernel_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','test/registered/unit/managers/test_io_struct.py','-k','test_getitem']))"
            ),
        },
        {
            "case_id": "sglang-pr-27290",
            "lane": 1,
            "timeout": 210,
            "purpose": "run the candidate EAGLE-v2 custom-logit-processor regression behind fail-closed unrelated optional-kernel imports",
            "command": switch("/workspace/r14-run-sglang", "sglang-pr-27290")
            + f"timeout 180s {sglang_env} /venv/main/bin/python -c "
            + shlex.quote(
                optional_kernel_driver
                + "raise SystemExit(pytest.main(['-q','-rs','--tb=short','--maxfail=1','test/registered/unit/spec/test_eagle_v2_custom_logit_processor.py']))"
            ),
        },
        {
            "case_id": "vllm-pr-44526",
            "lane": 0,
            "timeout": 240,
            "purpose": "run the existing streaming-session rebuild suite against the frozen head to exercise the newly clamped scheduler path",
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44526")
            + f"timeout 210s {vllm_env} /venv/main/bin/python /workspace/r15-probes/r15_vllm_pytest_driver.py --stub-tests-utils -- -q -rs --tb=short --maxfail=1 tests/v1/streaming_input/test_scheduler_streaming.py -k update_request_as_session",
        },
        {
            "case_id": "vllm-pr-44475",
            "lane": 0,
            "timeout": 120,
            "purpose": "execute the centralized cache-detail constructor over disabled, cold, warm, and zero-prompt boundaries",
            "command": switch("/workspace/r14-run-vllm", "vllm-pr-44475")
            + f"timeout 90s {vllm_env} /venv/main/bin/python -c "
            + shlex.quote(
                "from vllm.entrypoints.openai.engine.protocol import build_prompt_tokens_details as build; "
                "assert build(enable_prompt_tokens_details=False,enable_prefix_caching=True,num_cached_tokens=8,prompt_tokens=10) is None; "
                "assert build(enable_prompt_tokens_details=True,enable_prefix_caching=False,num_cached_tokens=8,prompt_tokens=10) is None; "
                "cold=build(enable_prompt_tokens_details=True,enable_prefix_caching=True,num_cached_tokens=None,prompt_tokens=10); "
                "warm=build(enable_prompt_tokens_details=True,enable_prefix_caching=True,num_cached_tokens=1632,prompt_tokens=1645); "
                "zero=build(enable_prompt_tokens_details=True,enable_prefix_caching=True,num_cached_tokens=0,prompt_tokens=0); "
                "assert (cold.cached_tokens,cold.cached_rate)==(0,0.0) and (warm.cached_tokens,warm.cached_rate)==(1632,0.9921) and zero.cached_rate==0.0; "
                "print('cache-details-pass',warm.cached_rate)"
            ),
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
        "protocol_id": "r20-focused-outcome-free-followups-v0.1",
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
