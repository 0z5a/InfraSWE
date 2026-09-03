#!/usr/bin/env python3
"""Run frozen R13 two-A100 training-contract corroboration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _DistributedLogProbEntropy(torch.autograd.Function):
    """Title-scoped copy of slime #2152's TP math for exact two-rank isolation."""

    @staticmethod
    def forward(
        ctx: Any,
        local_logits: torch.Tensor,
        targets: torch.Tensor,
        with_entropy_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank()
        local_vocab = local_logits.shape[-1]
        logits = local_logits.float()
        maximum = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        normalized = logits - maximum
        predicted = normalized.new_zeros(normalized.shape[0])
        start = rank * local_vocab
        owned = (targets >= start) & (targets < start + local_vocab)
        local_targets = (targets - start).clamp(0, local_vocab - 1)
        rows = torch.arange(targets.numel(), device=targets.device)
        predicted[owned] = normalized[rows[owned], local_targets[owned]]
        dist.all_reduce(predicted, op=dist.ReduceOp.SUM)
        softmax = normalized.exp_()
        denominator = softmax.sum(dim=-1, keepdim=True)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        softmax.div_(denominator)
        log_prob = predicted.unsqueeze(-1) - denominator.log()
        weighted_logits = (softmax * logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(weighted_logits, op=dist.ReduceOp.SUM)
        entropy = (maximum + denominator.log() - weighted_logits).squeeze(-1)
        if not with_entropy_grad:
            ctx.mark_non_differentiable(entropy)
        ctx.with_entropy_grad = with_entropy_grad
        ctx.save_for_backward(softmax, owned, local_targets, weighted_logits, logits)
        return log_prob, entropy

    @staticmethod
    def backward(
        ctx: Any,
        grad_log_prob: torch.Tensor | None,
        grad_entropy: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None]:
        softmax, owned, local_targets, weighted_logits, logits = ctx.saved_tensors
        gradient = None
        if ctx.with_entropy_grad and grad_entropy is not None:
            gradient = softmax * (weighted_logits - logits) * grad_entropy[:, None]
        if grad_log_prob is not None:
            log_gradient = -softmax
            rows = torch.arange(logits.shape[0], device=logits.device)
            log_gradient[rows[owned], local_targets[owned]] += 1
            log_gradient *= grad_log_prob.reshape(-1, 1)
            log_gradient = log_gradient.to(torch.bfloat16)
            gradient = log_gradient if gradient is None else gradient + log_gradient
        return gradient, None, None


def _slime_2152_tp2(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    full_logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, -1.0],
            [4.0, 1.0, 0.5, 2.0, 3.0, 0.0],
            [-1.0, 3.0, 2.0, 0.0, 1.0, 5.0],
            [0.2, -0.4, 1.7, -2.0, 3.3, 0.0],
        ],
        device=device,
    )
    local = full_logits.chunk(2, dim=-1)[rank].clone().requires_grad_(True)
    targets = torch.tensor([5, 0, 3, 2], device=device)
    log_weights = torch.tensor([0.25, -0.5, 1.5, -0.75], device=device)
    entropy_weights = torch.tensor([0.55, -0.2, 1.8, 0.4], device=device)
    log_prob, entropy = _DistributedLogProbEntropy.apply(local, targets, True)
    loss = (log_prob.squeeze(-1) * log_weights).sum() + (entropy * entropy_weights).sum()
    loss.backward()

    oracle = full_logits.clone().requires_grad_(True)
    oracle_log_prob = oracle.log_softmax(-1).gather(1, targets[:, None])
    oracle_entropy = -(oracle.softmax(-1) * oracle.log_softmax(-1)).sum(-1)
    oracle_loss = (oracle_log_prob.squeeze(-1) * log_weights).sum() + (
        oracle_entropy * entropy_weights
    ).sum()
    oracle_loss.backward()
    oracle_local = oracle.grad.chunk(2, dim=-1)[rank]
    return {
        "rank": rank,
        "log_prob_max_abs": float((log_prob - oracle_log_prob).abs().max().item()),
        "entropy_max_abs": float((entropy - oracle_entropy).abs().max().item()),
        "gradient_max_abs": float((local.grad.float() - oracle_local).abs().max().item()),
        "gradient_finite": bool(torch.isfinite(local.grad).all().item()),
        "target_ownership": [bool(rank * 3 <= int(target) < (rank + 1) * 3) for target in targets],
    }


def _slime_2152_memory(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    rows, local_vocab = 256, 32768

    def old_path(logits: torch.Tensor) -> torch.Tensor:
        maximum = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        normalized = logits - maximum
        exponentials = normalized.exp()
        denominator = exponentials.sum(dim=-1, keepdim=True)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        softmax = exponentials / denominator
        return softmax.sum()

    def head_path(logits: torch.Tensor) -> torch.Tensor:
        maximum = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        scratch = logits - maximum
        scratch.exp_()
        denominator = scratch.sum(dim=-1, keepdim=True)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        scratch.div_(denominator)
        return scratch.sum()

    def peak(operation: Any) -> int:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        logits = torch.randn(rows, local_vocab, device=device)
        baseline = torch.cuda.memory_allocated(device)
        result = operation(logits)
        torch.cuda.synchronize(device)
        value = torch.cuda.max_memory_allocated(device) - baseline
        if not torch.isfinite(result):
            raise RuntimeError("memory path returned non-finite result")
        del result, logits
        return int(value)

    old_peak = peak(old_path)
    dist.barrier()
    head_peak = peak(head_path)
    return {
        "rank": rank,
        "rows": rows,
        "local_vocab": local_vocab,
        "old_extra_peak_bytes": old_peak,
        "head_extra_peak_bytes": head_peak,
        "reduction_bytes": old_peak - head_peak,
        "reduction_fraction": (old_peak - head_peak) / old_peak,
    }


def _megatron_5743_deferred(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    microbatches = [
        torch.tensor([rank + 1.0, 2.0 * (rank + 1)], device=device),
        torch.tensor([3.0 * (rank + 1), 4.0 * (rank + 1)], device=device),
        torch.tensor([5.0 * (rank + 1), 6.0 * (rank + 1)], device=device),
    ]
    eager = torch.zeros_like(microbatches[0])
    for gradient in microbatches:
        reduced = gradient.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
        eager += reduced
    deferred = sum(microbatches, torch.zeros_like(microbatches[0]))
    dist.all_reduce(deferred, op=dist.ReduceOp.SUM)
    deferred /= dist.get_world_size()
    return {
        "rank": rank,
        "eager": eager.cpu().tolist(),
        "deferred": deferred.cpu().tolist(),
        "max_abs": float((eager - deferred).abs().max().item()),
        "eager_outer_collectives": len(microbatches),
        "deferred_outer_collectives": 1,
    }


def _verl_7012_alignment(rank: int) -> dict[str, Any]:
    cp_size = 2
    student_local_length = 6
    student_total = student_local_length * cp_size
    teacher_raw_length = 7

    def align(value: int) -> int:
        multiple = 2 * cp_size
        return ((value + multiple - 1) // multiple) * multiple

    def positions(total: int) -> list[int]:
        chunk = total // (2 * cp_size)
        first = list(range(rank * chunk, (rank + 1) * chunk))
        second_start = (2 * cp_size - rank - 1) * chunk
        return first + list(range(second_start, second_start + chunk))

    base_total = align(teacher_raw_length)
    head_total = align(student_total)
    return {
        "rank": rank,
        "student_local_length": student_local_length,
        "base_teacher_total": base_total,
        "base_teacher_local_length": len(positions(base_total)),
        "head_teacher_total": head_total,
        "head_teacher_local_length": len(positions(head_total)),
        "base_shape_matches_student": len(positions(base_total)) == student_local_length,
        "head_shape_matches_student": len(positions(head_total)) == student_local_length,
        "head_zigzag_positions": positions(head_total),
    }


def _verl_6996_nccl_cpu_broadcast(rank: int) -> dict[str, Any]:
    cpu = torch.tensor([rank], dtype=torch.int32, device="cpu")
    error = None
    try:
        dist.broadcast(cpu, src=0)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    cuda = torch.tensor([41 if rank == 0 else -1], dtype=torch.int32, device=f"cuda:{rank}")
    dist.broadcast(cuda, src=0)
    return {
        "rank": rank,
        "cpu_broadcast_succeeded": error is None,
        "cpu_broadcast_error": error,
        "cuda_control_value": int(cuda.item()),
        "backend": dist.get_backend(),
    }


def _worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        smoke = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
        dist.all_reduce(smoke)
        row = {
            "rank": rank,
            "device": torch.cuda.get_device_name(rank),
            "nccl_smoke_sum": float(smoke.item()),
            "megatron_5743": _megatron_5743_deferred(rank),
            "slime_2152_tp2": _slime_2152_tp2(rank),
            "slime_2152_memory": _slime_2152_memory(rank),
            "verl_7012": _verl_7012_alignment(rank),
            "verl_6996": _verl_6996_nccl_cpu_broadcast(rank),
        }
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(row, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != _canonical(selection_material):
        raise SystemExit("R13 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R13 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R13 plan/selection binding mismatch")
    source_bundle = _read(args.source_bundle)
    selected = {case["case_id"] for case in selection_material["cases"]}
    if set(source_bundle) != selected or len(selected) != 29:
        raise SystemExit("R13 source bundle is not the frozen 29-case cohort")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("R13 dual-GPU probe requires two CUDA GPUs")

    started_at = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix="infraswe-r13-training-") as directory:
        mp.spawn(
            _worker,
            args=(2, _free_port(), directory),
            nprocs=2,
            join=True,
        )
        ranks = [_read(Path(directory, f"rank-{rank}.json")) for rank in range(2)]

    facts = {
        "two_rank_nccl_smoke": all(row["nccl_smoke_sum"] == 3 for row in ranks),
        "megatron_5743_deferred_matches_eager": all(
            row["megatron_5743"]["max_abs"] == 0 for row in ranks
        ),
        "slime_2152_tp2_log_prob_max_abs": max(
            row["slime_2152_tp2"]["log_prob_max_abs"] for row in ranks
        ),
        "slime_2152_tp2_entropy_max_abs": max(
            row["slime_2152_tp2"]["entropy_max_abs"] for row in ranks
        ),
        "slime_2152_tp2_gradient_max_abs": max(
            row["slime_2152_tp2"]["gradient_max_abs"] for row in ranks
        ),
        "slime_2152_memory_reduction_on_all_ranks": all(
            row["slime_2152_memory"]["reduction_bytes"] > 0 for row in ranks
        ),
        "verl_7012_base_shape_mismatch_on_all_ranks": all(
            not row["verl_7012"]["base_shape_matches_student"] for row in ranks
        ),
        "verl_7012_head_shape_matches_on_all_ranks": all(
            row["verl_7012"]["head_shape_matches_student"] for row in ranks
        ),
        "verl_6996_cpu_broadcast_fails_on_all_nccl_ranks": all(
            not row["verl_6996"]["cpu_broadcast_succeeded"] for row in ranks
        ),
        "verl_6996_cuda_broadcast_control_passes": all(
            row["verl_6996"]["cuda_control_value"] == 41 for row in ranks
        ),
        "rank_rows": ranks,
    }
    if not facts["two_rank_nccl_smoke"]:
        raise SystemExit("R13 NCCL smoke failed")
    if not facts["megatron_5743_deferred_matches_eager"]:
        raise SystemExit("R13 deferred gradient reduction disagrees")
    if facts["slime_2152_tp2_log_prob_max_abs"] > 1e-6:
        raise SystemExit("R13 slime TP2 log-probability mismatch")
    if not facts["verl_7012_head_shape_matches_on_all_ranks"]:
        raise SystemExit("R13 verl CP alignment control failed")

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    material = {
        "schema_version": "0.1",
        "protocol_id": selection_material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": _canonical(source_bundle),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "environment": environment,
        "environment_sha256": _canonical(environment),
        "facts": facts,
    }
    payload = {**material, "evidence_sha256": _canonical(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(facts, indent=2, sort_keys=True))
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
