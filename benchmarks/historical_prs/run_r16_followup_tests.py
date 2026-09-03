#!/usr/bin/env python3
# ruff: noqa: E501
"""Run exact-head R16 follow-ups for training semantics and resource claims."""

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

from run_r15_followup_tests import _copy_files, _ssh

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R16 selection digest mismatch")
    plan = _read(args.test_plan)
    if plan["test_plan_sha256"] != canonical_sha256(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R16 test-plan digest mismatch")
    static = _read(args.static_evidence)
    if static["evidence_sha256"] != canonical_sha256(
        {key: value for key, value in static.items() if key != "evidence_sha256"}
    ):
        raise SystemExit("R16 static-evidence digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R16 plan/selection binding mismatch")
    if static["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R16 static/selection binding mismatch")

    probe_names = [
        "r16_megatron_mtp_freeze_probe.py",
        "r16_slime_sort_probe.py",
        "r16_slime_temperature_probe.py",
    ]
    probe_paths = [args.probe_dir / name for name in probe_names]
    probe_hashes = _copy_files(probe_paths, "/workspace/r16-probes")
    cases = {case["case_id"]: case for case in selection_material["cases"]}

    def head(case_id: str) -> str:
        return shlex.quote(cases[case_id]["head_sha"])

    def base(case_id: str) -> str:
        return shlex.quote(cases[case_id]["base_sha"])

    records: list[dict[str, Any]] = [
        {
            "case_id": "liger-pr-1413",
            "purpose": "A100 no-regression control for the target-only fused-MoE dx change",
            "timeout": 360,
            "command": (
                "cd /workspace/r13-run-liger && git switch --detach refs/r16/pr-1413 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("liger-pr-1413")} && '
                "timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/transformers/test_fused_moe.py"
            ),
        },
        {
            "case_id": "liger-pr-1244",
            "purpose": "import ORPO through the current TRL experimental export",
            "timeout": 120,
            "command": (
                "cd /workspace/r13-run-liger && git switch --detach refs/r16/pr-1244 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("liger-pr-1244")} && '
                "timeout 60s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /venv/main/bin/python -c "
                + shlex.quote(
                    "from trl.experimental.orpo import ORPOConfig; "
                    "from liger_kernel.transformers.trainer.orpo_trainer import LigerORPOTrainer; "
                    "assert ORPOConfig is not None and LigerORPOTrainer is not None; print('current_trl_orpo_import=pass')"
                )
            ),
        },
        {
            "case_id": "liger-pr-1253",
            "purpose": "run GroupNorm forward/backward precision matrix",
            "timeout": 300,
            "command": (
                "cd /workspace/r13-run-liger && git switch --detach refs/r16/pr-1253 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("liger-pr-1253")} && '
                "timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/transformers/test_group_norm.py"
            ),
        },
        {
            "case_id": "liger-pr-1208",
            "purpose": "run DyT autotuned forward/backward numeric matrix on A100",
            "timeout": 360,
            "command": (
                "cd /workspace/r13-run-liger && git switch --detach refs/r16/pr-1208 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("liger-pr-1208")} && '
                "timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 test/transformers/test_dyt.py"
            ),
        },
        {
            "case_id": "megatron-pr-7021",
            "purpose": "validate MTP-only arguments, parameter ownership, router bias, and pre-wrap hook",
            "timeout": 240,
            "command": (
                "cd /workspace/r13-run-megatron && git switch --detach refs/r16/pr-7021 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("megatron-pr-7021")} && '
                "timeout 120s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/unit_tests/test_argument_utils.py -k freeze_base_model_for_mtp && "
                "timeout 60s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python /workspace/r16-probes/r16_megatron_mtp_freeze_probe.py"
            ),
        },
        {
            "case_id": "megatron-pr-5134",
            "purpose": "run the candidate-updated serialization contracts at the available topology",
            "timeout": 300,
            "command": (
                "cd /workspace/r13-run-megatron && git switch --detach refs/r16/pr-5134 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("megatron-pr-5134")} && '
                "timeout 240s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/unit_tests/dist_checkpointing/test_serialization.py -k 'unexpected_keys_handling or missing_keys_raises_error_during_validation or exact_load_handling or sharded_metadata'"
            ),
        },
        {
            "case_id": "megatron-pr-5131",
            "purpose": "isolate the three candidate TE CUDA-graph attention-mask tests from unrelated setup",
            "timeout": 180,
            "command": (
                "cd /workspace/r13-run-megatron && git switch --detach refs/r16/pr-5131 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("megatron-pr-5131")} && '
                "timeout 120s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/unit_tests/transformer/test_transformer_layer.py -k 'test_te_cuda_graph_omits_absent_attention_mask or test_te_cuda_graph_keeps_configured_attention_mask or test_te_cuda_graph_warns_when_omitting_padding_attention_mask'"
            ),
        },
        {
            "case_id": "slime-pr-2345",
            "purpose": "exercise nested-list sorting through the candidate production coroutine",
            "timeout": 120,
            "command": (
                "cd /workspace/r13-run-slime && git switch --detach refs/r16/pr-2345 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("slime-pr-2345")} && '
                "timeout 60s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python /workspace/r16-probes/r16_slime_sort_probe.py"
            ),
        },
        {
            "case_id": "slime-pr-2010",
            "purpose": "compare base/head temperature semantics and first-yield CUDA peak",
            "timeout": 360,
            "command": (
                "cd /workspace/r13-run-slime && git switch --detach "
                f"{base('slime-pr-2010')} >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {base("slime-pr-2010")} && '
                "timeout 150s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python /workspace/r16-probes/r16_slime_temperature_probe.py --mode base && "
                "git switch --detach refs/r16/pr-2010 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("slime-pr-2010")} && '
                "timeout 150s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python /workspace/r16-probes/r16_slime_temperature_probe.py --mode head"
            ),
        },
        {
            "case_id": "slime-pr-2014",
            "purpose": "rerun candidate rollout filter contracts after installing the declared light dependency",
            "timeout": 180,
            "command": (
                "cd /workspace/r13-run-slime && git switch --detach refs/r16/pr-2014 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("slime-pr-2014")} && '
                "timeout 120s env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/plugin_contracts/test_plugin_rollout_contracts.py"
            ),
        },
        {
            "case_id": "torchtitan-pr-4358",
            "purpose": "run candidate CPU trainer replay and configuration contracts after restoring pinned Grain",
            "timeout": 300,
            "command": (
                "cd /workspace/r13-run-torchtitan && git switch --detach refs/r16/pr-4358 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("torchtitan-pr-4358")} && '
                "timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /workspace/venv-tt/bin/python -m pytest -q -rs --tb=short --maxfail=1 tests/unit_tests/cpu/test_trainer.py tests/unit_tests/cpu/test_config_manager.py tests/unit_tests/cpu/test_invalid_loss.py -k 'replay or sdc'"
            ),
        },
        {
            "case_id": "torchtitan-pr-3538",
            "purpose": "run exactly the call sites changed to rely on the forward-direction default",
            "timeout": 240,
            "command": (
                "cd /workspace/r13-run-torchtitan && git switch --detach refs/r16/pr-3538 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("torchtitan-pr-3538")} && '
                "timeout 180s env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /workspace/venv-tt/bin/python -m pytest -q -rs --tb=short --maxfail=1 "
                "torchtitan/experiments/graph_trainer/tests/test_precompile.py::TestCudagraphPass::test_non_graphmodule_raises "
                "torchtitan/experiments/graph_trainer/tests/test_precompile.py::TestCudagraphPass::test_graphmodule_wraps_forward "
                "torchtitan/experiments/graph_trainer/tests/test_trace_module.py::TestTraceDTensor::test_full_inductor_pass_migrates_cpu_attrs"
            ),
        },
        {
            "case_id": "verl-pr-6558",
            "purpose": "confirm the static parser counterexample on the exact head",
            "timeout": 120,
            "expected_returncode": 1,
            "command": (
                "cd /workspace/r13-run-verl && git switch --detach refs/r16/pr-6558 >/dev/null && "
                f'test "$(git rev-parse HEAD)" = {head("verl-pr-6558")} && '
                "timeout 60s /venv/main/bin/python -m py_compile verl/experimental/one_step_off_policy/ray_trainer.py"
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
            process = _ssh(record["command"], int(record["timeout"]))
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
            f"[{index}/{len(records)}] {record['case_id']}: rc={record['returncode']} {record['duration_seconds']:.1f}s",
            flush=True,
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": "r16-exact-head-training-followups-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "static_evidence_sha256": static["evidence_sha256"],
        "probe_hashes": probe_hashes,
        "environment_repairs": [
            "pylatexenc==2.11 in /venv/main",
            "grain==0.2.18 in /workspace/venv-tt",
        ],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "outcome_review_ci_fields_requested": False,
        "records": records,
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
