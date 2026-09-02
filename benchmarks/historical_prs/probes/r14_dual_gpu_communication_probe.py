#!/usr/bin/env python3
"""Run frozen R14 two-A100 communication-contract corroboration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as py_mp
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


def _vllm_50658_projection(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    aux = [
        (
            torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4)
            + rank * 20
            + index * 3
        ).to(torch.bfloat16)
        for index in range(3)
    ]
    weight = (torch.arange(60, device=device, dtype=torch.float32).reshape(5, 12) / 100).to(
        torch.bfloat16
    )
    local_projected = torch.cat(aux, dim=-1) @ weight.t()
    gathered = [torch.empty_like(local_projected) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_projected)
    head = torch.cat(gathered, dim=0)

    gathered_aux: list[list[torch.Tensor]] = []
    for tensor in aux:
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, tensor)
        gathered_aux.append(parts)
    full_aux = [torch.cat(parts, dim=0) for parts in gathered_aux]
    oracle = torch.cat(full_aux, dim=-1) @ weight.t()
    old_payload = sum(tensor.numel() * tensor.element_size() for tensor in aux)
    new_payload = local_projected.numel() * local_projected.element_size()
    return {
        "rank": rank,
        "max_abs_error": float((head.float() - oracle.float()).abs().max().item()),
        "old_collective_count": len(aux),
        "head_collective_count": 1,
        "old_payload_bytes_per_peer": old_payload,
        "head_payload_bytes_per_peer": new_payload,
    }


def _megatron_6200_fp32_reduce_scatter(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    world_size = dist.get_world_size()
    input_tensor = torch.tensor(
        [
            4096.0 + rank,
            -4096.0 + rank,
            1.0 + rank / 8,
            -1.0 + rank / 8,
            0.03125,
            -0.03125,
            17.0 + rank,
            -9.0 - rank,
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    assert input_tensor.numel() % world_size == 0
    scratch = torch.empty_like(input_tensor)
    dist.all_to_all_single(scratch, input_tensor)
    candidate = scratch.view(world_size, -1).sum(dim=0, dtype=torch.float32).to(torch.bfloat16)

    all_inputs = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(all_inputs, input_tensor)
    chunk_size = input_tensor.numel() // world_size
    oracle_fp32 = torch.stack(
        [tensor[rank * chunk_size : (rank + 1) * chunk_size].float() for tensor in all_inputs]
    ).sum(dim=0)
    oracle = oracle_fp32.to(torch.bfloat16)

    legacy = torch.empty(chunk_size, device=device, dtype=torch.bfloat16)
    dist.reduce_scatter_tensor(legacy, input_tensor, op=dist.ReduceOp.SUM)

    prescaled_input = (input_tensor * (1.0 / world_size)).to(torch.bfloat16)
    prescaled_scratch = torch.empty_like(prescaled_input)
    dist.all_to_all_single(prescaled_scratch, prescaled_input)
    prescaled_candidate = (
        prescaled_scratch.view(world_size, -1).sum(dim=0, dtype=torch.float32).to(torch.bfloat16)
    )
    postscaled_oracle = (oracle_fp32 / world_size).to(torch.bfloat16)
    return {
        "rank": rank,
        "candidate_vs_oracle_max_abs": float(
            (candidate.float() - oracle.float()).abs().max().item()
        ),
        "legacy_vs_oracle_max_abs": float((legacy.float() - oracle.float()).abs().max().item()),
        "power_of_two_prescale_vs_postscale_max_abs": float(
            (prescaled_candidate.float() - postscaled_oracle.float()).abs().max().item()
        ),
        "world2_feature_has_accuracy_gain": bool(
            (legacy.float() - oracle.float()).abs().max()
            > (candidate.float() - oracle.float()).abs().max()
        ),
    }


def _torchtitan_3953_collective_linearity(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    chunk_a = torch.tensor([rank + 1.0, 3.0 - rank], device=device)
    chunk_b = torch.tensor([2.0 * rank - 1.0, rank + 0.25], device=device)
    old_a, old_b = chunk_a.clone(), chunk_b.clone()
    dist.all_reduce(old_a)
    dist.all_reduce(old_b)
    old = old_a + old_b
    head = chunk_a + chunk_b
    dist.all_reduce(head)
    return {
        "rank": rank,
        "old_collective_count": 2,
        "head_collective_count": 1,
        "max_abs_error": float((old - head).abs().max().item()),
    }


def _verl_7107_boundary_broadcast(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for length in (0, 1, 1024, 1025):
            if rank == 0:
                tensor = torch.arange(length, device=device, dtype=torch.float32).to(dtype)
            else:
                tensor = torch.empty(length, device=device, dtype=dtype)
            dist.broadcast(tensor, src=0)
            expected = torch.arange(length, device=device, dtype=torch.float32).to(dtype)
            cases.append(
                {
                    "dtype": str(dtype),
                    "length": length,
                    "exact": bool(torch.equal(tensor, expected)),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
            )
    return {"rank": rank, "cases": cases, "all_exact": all(item["exact"] for item in cases)}


def _megatron_7000_fixed_shape_p2p(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    shape = (4, 3)
    if rank == 0:
        tensor = torch.arange(12, device=device, dtype=torch.float32).reshape(shape)
        dist.send(tensor, dst=1)
        exact = True
    else:
        tensor = torch.empty(shape, device=device)
        dist.recv(tensor, src=0)
        expected = torch.arange(12, device=device, dtype=torch.float32).reshape(shape)
        exact = bool(torch.equal(tensor, expected))
    dist.barrier()
    return {"rank": rank, "shape": list(shape), "exact": exact}


def _sglang_33029_collective_cardinality(rank: int) -> dict[str, Any]:
    # Both ranks begin with the same waiting queue, but rank-local scheduling
    # decisions would have reached different numbers of in-loop checks on base.
    waiting = ["r0", "r1", "r2"]
    base_calls = 1 if rank == 0 else 3
    base_counts = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(base_counts, base_calls)
    progress = {}
    for index, request_id in enumerate(waiting):
        value = torch.tensor([int(index <= rank + 1)], device=f"cuda:{rank}")
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        progress[request_id] = bool(value.item())
    return {
        "rank": rank,
        "base_in_loop_collective_counts": base_counts,
        "base_counts_mismatch": len(set(base_counts)) != 1,
        "head_bulk_collective_count": len(waiting),
        "head_progress": progress,
    }


def _nccl_eager_connect(rank: int) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    error = None
    try:
        group._get_backend(device).eager_connect_single_device(device)
    except Exception as exception:  # pragma: no cover - hardware dependent
        error = f"{type(exception).__name__}: {exception}"
    dist.barrier()
    return {"rank": rank, "succeeded": error is None, "error": error}


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
            "vllm_50658": _vllm_50658_projection(rank),
            "megatron_6200": _megatron_6200_fp32_reduce_scatter(rank),
            "torchtitan_3953": _torchtitan_3953_collective_linearity(rank),
            "verl_7107": _verl_7107_boundary_broadcast(rank),
            "megatron_7000": _megatron_7000_fixed_shape_p2p(rank),
            "sglang_33029": _sglang_33029_collective_cardinality(rank),
            "megatron_6955": _nccl_eager_connect(rank),
            "cuda_peer_access": [
                bool(torch.cuda.can_device_access_peer(rank, peer))
                for peer in range(world_size)
                if peer != rank
            ],
        }
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(row, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def _try_exclusive_lock(path: str, queue: Any) -> None:
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            queue.put(False)
        else:
            queue.put(True)
    finally:
        os.close(fd)


def _flock_liveness_probe() -> dict[str, Any]:
    context = py_mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="infraswe-r14-flock-") as directory:
        path = Path(directory, "region")
        path.touch()
        owner_fd = os.open(path, os.O_RDWR)
        fcntl.flock(owner_fd, fcntl.LOCK_SH)
        first_queue = context.Queue()
        first = context.Process(target=_try_exclusive_lock, args=(str(path), first_queue))
        first.start()
        first.join(15)
        while_live = first_queue.get(timeout=5)
        os.close(owner_fd)
        second_queue = context.Queue()
        second = context.Process(target=_try_exclusive_lock, args=(str(path), second_queue))
        second.start()
        second.join(15)
        after_close = second_queue.get(timeout=5)
    return {
        "exclusive_lock_while_owner_live": while_live,
        "exclusive_lock_after_owner_close": after_close,
        "live_region_survives_reaper_condition": not while_live,
        "closed_region_meets_orphan_condition": after_close,
    }


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
        raise SystemExit("R14 selection digest mismatch")
    plan = _read(args.plan)
    if plan["test_plan_sha256"] != _canonical(
        {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    ):
        raise SystemExit("R14 plan digest mismatch")
    source = _read(args.source_bundle)
    if source["source_bundle_sha256"] != _canonical(
        {key: value for key, value in source.items() if key != "source_bundle_sha256"}
    ):
        raise SystemExit("R14 source-bundle digest mismatch")
    if len(selection_material["cases"]) != 30 or len(source["cases"]) != 30:
        raise SystemExit("R14 probe requires the frozen 30-case cohort")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("R14 dual-GPU probe requires two CUDA GPUs")

    started_at = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix="infraswe-r14-communication-") as directory:
        mp.spawn(_worker, args=(2, _free_port(), directory), nprocs=2, join=True)
        ranks = [_read(Path(directory, f"rank-{rank}.json")) for rank in range(2)]

    facts = {
        "two_rank_nccl_smoke": all(row["nccl_smoke_sum"] == 3 for row in ranks),
        "vllm_50658_projection_max_abs": max(row["vllm_50658"]["max_abs_error"] for row in ranks),
        "vllm_50658_collectives_reduced": all(
            row["vllm_50658"]["old_collective_count"] > row["vllm_50658"]["head_collective_count"]
            for row in ranks
        ),
        "megatron_6200_candidate_max_abs": max(
            row["megatron_6200"]["candidate_vs_oracle_max_abs"] for row in ranks
        ),
        "megatron_6200_power_of_two_prescale_max_abs": max(
            row["megatron_6200"]["power_of_two_prescale_vs_postscale_max_abs"] for row in ranks
        ),
        "megatron_6200_world2_has_accuracy_gain": any(
            row["megatron_6200"]["world2_feature_has_accuracy_gain"] for row in ranks
        ),
        "torchtitan_3953_normalization_max_abs": max(
            row["torchtitan_3953"]["max_abs_error"] for row in ranks
        ),
        "verl_7107_all_boundaries_exact": all(row["verl_7107"]["all_exact"] for row in ranks),
        "megatron_7000_fixed_shape_p2p_exact": all(row["megatron_7000"]["exact"] for row in ranks),
        "sglang_33029_base_cardinality_mismatch_detected": all(
            row["sglang_33029"]["base_counts_mismatch"] for row in ranks
        ),
        "sglang_33029_head_bulk_completed": all(
            row["sglang_33029"]["head_bulk_collective_count"] == 3 for row in ranks
        ),
        "megatron_6955_eager_connect_succeeded": all(
            row["megatron_6955"]["succeeded"] for row in ranks
        ),
        "all_cross_gpu_peer_accessible": all(all(row["cuda_peer_access"]) for row in ranks),
        "vllm_54619_flock_liveness": _flock_liveness_probe(),
    }
    material = {
        "schema_version": "0.1",
        "protocol_id": "r14-two-a100-communication-contract-probe-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "source_bundle_sha256": source["source_bundle_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        },
        "outcome_review_ci_fields_requested": False,
        "facts": facts,
        "ranks": ranks,
    }
    atomic = {**material, "evidence_sha256": _canonical(material)}
    args.output.write_text(json.dumps(atomic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"evidence_sha256={atomic['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
