#!/usr/bin/env python3
# ruff: noqa: E501
"""Run exact candidate-owned R13 tests on the frozen remote worktrees.

This runner intentionally records execution rather than deciding whether a failure is
candidate-caused, an unsupported topology, or an unavailable optional dependency. That
interpretation belongs to the subsequently frozen case-contract judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


@dataclass(frozen=True, slots=True)
class UpstreamTest:
    case_id: str
    worktree: str
    ref: str
    command: str
    scope: str


TESTS = (
    UpstreamTest(
        "slime-pr-2205",
        "/workspace/r13-run-slime",
        "refs/r13/pr-2205",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/test_discounted_returns.py",
        "candidate-owned discounted-return suite",
    ),
    UpstreamTest(
        "slime-pr-2204",
        "/workspace/r13-run-slime",
        "refs/r13/pr-2204",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/test_reward_utils.py",
        "candidate-owned explicit reward-group suite",
    ),
    UpstreamTest(
        "slime-pr-2198",
        "/workspace/r13-run-slime",
        "refs/r13/pr-2198",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/test_ppo_ratio_numerics.py",
        "candidate-owned PPO extreme-ratio suite",
    ),
    UpstreamTest(
        "slime-pr-2207",
        "/workspace/r13-run-slime",
        "refs/r13/pr-2207",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/test_partial_rollout_loss_mask.py",
        "candidate-owned partial-rollout mask suite",
    ),
    UpstreamTest(
        "slime-pr-2152",
        "/workspace/r13-run-slime",
        "refs/r13/pr-2152",
        "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python -m pytest -q tests/test_ppo_logprob_entropy_gpu.py -k 'not tp2'",
        "candidate-owned single-GPU value/backward suite",
    ),
    UpstreamTest(
        "slime-pr-2152",
        "/workspace/r13-run-slime",
        "refs/r13/pr-2152",
        "CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:/workspace/r13-run-megatron /venv/main/bin/python -m pytest -q tests/test_ppo_logprob_entropy_gpu.py -k tp2",
        "candidate-owned two-GPU tensor-parallel suite",
    ),
    UpstreamTest(
        "torchtitan-pr-3841",
        "/workspace/r13-run-torchtitan",
        "refs/r13/pr-3841",
        "PYTHONPATH=. /workspace/venv-tt/bin/python -m pytest -q torchtitan/experiments/graph_trainer/tests/test_graph_pp_passes.py -k GraphPPSplitDiDwTest",
        "candidate-owned graph dI/dW split suite",
    ),
    UpstreamTest(
        "torchtitan-pr-3867",
        "/workspace/r13-run-torchtitan",
        "refs/r13/pr-3867",
        "PYTHONPATH=. /workspace/venv-tt/bin/python -m pytest -q tests/unit_tests/test_fused_qkv.py",
        "candidate-owned fused-QKV reload suite",
    ),
    UpstreamTest(
        "torchtitan-pr-3897",
        "/workspace/r13-run-torchtitan",
        "refs/r13/pr-3897",
        "/workspace/venv-tt/bin/python -m py_compile torchtitan/experiments/rl/tests/integration_tests.py",
        "exact changed-file syntax gate",
    ),
    UpstreamTest(
        "liger-pr-1274",
        "/workspace/r13-run-liger",
        "refs/r13/pr-1274",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q test/chunked_loss/test_grpo_loss.py -k sapo",
        "native SAPO eager/compile suite",
    ),
    UpstreamTest(
        "liger-pr-1268",
        "/workspace/r13-run-liger",
        "refs/r13/pr-1268",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q test/transformers/test_cross_entropy.py -k 'label_smoothing or softcap or weight'",
        "native cross-entropy option matrix",
    ),
    UpstreamTest(
        "liger-pr-1230",
        "/workspace/r13-run-liger",
        "refs/r13/pr-1230",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q test/transformers/test_orpo_trainer.py",
        "candidate-owned non-FSDP ORPO trainer regression",
    ),
    UpstreamTest(
        "megatron-pr-5808",
        "/workspace/r13-run-megatron",
        "refs/r13/pr-5808",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/unit_tests/distributed/mfsdp_v1/test_mfsdp_fully_shard.py",
        "candidate-owned root-module hook regression",
    ),
    UpstreamTest(
        "megatron-pr-5798",
        "/workspace/r13-run-megatron",
        "refs/r13/pr-5798",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/unit_tests/transformer/moe/test_aux_loss.py -k sequence",
        "candidate-owned sequence-level aux-loss regression",
    ),
    UpstreamTest(
        "megatron-pr-5742",
        "/workspace/r13-run-megatron",
        "refs/r13/pr-5742",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/unit_tests/test_lion_optimizer.py",
        "candidate-owned Lion optimizer suite",
    ),
    UpstreamTest(
        "megatron-pr-5724",
        "/workspace/r13-run-megatron",
        "refs/r13/pr-5724",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/unit_tests/transformer/test_thd_cuda_graph.py -k 'test_pad_to_max_resolves_padding_kwargs or test_eager_pad_to_max_adds_dummy_padding_sequence'",
        "candidate-owned THD padding cases",
    ),
    UpstreamTest(
        "megatron-pr-5710",
        "/workspace/r13-run-megatron",
        "refs/r13/pr-5710",
        "PYTHONPATH=. /venv/main/bin/torchrun --standalone --nproc_per_node=2 -m pytest -q tests/unit_tests/distributed/mfsdp_v2/test_annotation.py -k frozen",
        "candidate-owned two-rank frozen-hook regressions",
    ),
    UpstreamTest(
        "megatron-pr-5714",
        "/workspace/r13-run-megatron",
        "refs/r13/pr-5714",
        "PYTHONPATH=. /venv/main/bin/torchrun --standalone --nproc_per_node=2 -m pytest -q tests/unit_tests/distributed/test_torch_fully_sharded_parallel.py -k fsdp2_swiglu_sharded_tensor_factory",
        "candidate-owned test on the available TP2/DP1 topology",
    ),
    UpstreamTest(
        "flashattention-pr-2654",
        "/workspace/r13-run-flash",
        "refs/r13/pr-2654",
        "/venv/main/bin/python -c \"import flash_attn.cute.interface\"",
        "exact-source runtime-import preflight before the SM80 score-mod test",
    ),
    UpstreamTest(
        "verl-pr-7013",
        "/workspace/r13-run-verl",
        "refs/r13/pr-7013",
        "PYTHONPATH=. /venv/main/bin/python -m pytest -q tests/trainer/ppo/test_adaptive_kl_checkpoint_on_cpu.py",
        "candidate-owned adaptive-KL checkpoint/resume suite",
    ),
)


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _summary(output: str) -> list[str]:
    patterns = (
        re.compile(r"=+ .* (?:passed|failed|skipped|deselected).* =+"),
        re.compile(r"(?:^|\s)(?:\d+ passed|\d+ failed|\d+ skipped|\d+ deselected)(?:,|\s|$)"),
        re.compile(r"(?:SyntaxError|ModuleNotFoundError|ImportError|AssertionError):.*"),
        re.compile(r"FAILED .*"),
    )
    rows: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line and any(pattern.search(line) for pattern in patterns) and line not in rows:
            rows.append(line)
    return rows[-16:]


def _remote_command(test: UpstreamTest) -> str:
    return (
        f"cd {shlex.quote(test.worktree)} && "
        f"git switch --detach {shlex.quote(test.ref)} >/dev/null && "
        f"{test.command}"
    )


def _run(
    args: argparse.Namespace, test: UpstreamTest, ordinal: int, total: int
) -> dict[str, Any]:
    command = [
        "ssh",
        "-p",
        str(args.port),
        "-i",
        str(args.identity),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        args.host,
        _remote_command(test),
    ]
    started_at = datetime.now(UTC)
    start = time.monotonic()
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    duration = time.monotonic() - start
    output = process.stdout + process.stderr
    print(
        f"[{ordinal:02d}/{total}] {test.case_id}: "
        f"rc={process.returncode} duration={duration:.1f}s",
        flush=True,
    )
    return {
        "case_id": test.case_id,
        "worktree": test.worktree,
        "ref": test.ref,
        "scope": test.scope,
        "remote_command": _remote_command(test),
        "started_at": started_at.isoformat(),
        "duration_seconds": duration,
        "returncode": process.returncode,
        "output_sha256": _sha256(output),
        "summary_lines": _summary(output),
        "output_tail": output[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="root@38.49.42.120")
    parser.add_argument("--port", type=int, default=54270)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    plan = json.loads(args.test_plan.read_text(encoding="utf-8"))
    selected = {
        case["case_id"]: case for case in selection["selection_material"]["cases"]
    }
    if selection["selection_lock_sha256"] != canonical_sha256(
        selection["selection_material"]
    ):
        raise SystemExit("selection lock digest mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != canonical_sha256(plan_material):
        raise SystemExit("test-plan digest mismatch")
    for test in TESTS:
        if test.case_id not in selected:
            raise SystemExit(f"test case is not in the selection lock: {test.case_id}")
        expected_ref = f"refs/r13/pr-{selected[test.case_id]['pull_number']}"
        if test.ref != expected_ref:
            raise SystemExit(f"wrong frozen ref for {test.case_id}: {test.ref}")

    requested = set(args.only)
    tests = tuple(test for test in TESTS if not requested or test.case_id in requested)
    missing = requested - {test.case_id for test in tests}
    if missing:
        raise SystemExit(f"unknown --only case IDs: {sorted(missing)}")
    records = [
        _run(args, test, index, len(tests)) for index, test in enumerate(tests, start=1)
    ]
    material: dict[str, Any] = {
        "schema_version": "0.1",
        "protocol_id": "r13-exact-candidate-upstream-tests-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "remote": {"host": args.host, "port": args.port},
        "requested_case_ids": sorted(requested),
        "outcome_review_ci_fields_requested": False,
        "executed_at": datetime.now(UTC).isoformat(),
        "records": records,
    }
    payload = {**material, "evidence_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
