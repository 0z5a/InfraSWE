#!/usr/bin/env python3
"""Run blind, two-GPU NCCL corroboration for the R12 communication cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _timed_ms(operation: Callable[[], None], *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
        "mean_ms": mean,
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "cv": statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0,
    }


def _all_gather_pair(rank: int) -> list[dict[str, Any]]:
    dtype = torch.bfloat16
    hidden_size = 7168
    rows: list[dict[str, Any]] = []
    for tokens in (1, 16, 64, 256):
        generator = torch.Generator(device=f"cuda:{rank}").manual_seed(20260902 + rank + tokens)
        hidden = torch.randn(
            (tokens, hidden_size), dtype=dtype, device=f"cuda:{rank}", generator=generator
        )
        residual = torch.randn(
            (tokens, hidden_size), dtype=dtype, device=f"cuda:{rank}", generator=generator
        )

        def old_path(
            hidden: torch.Tensor = hidden,
            residual: torch.Tensor = residual,
            tokens: int = tokens,
        ) -> torch.Tensor:
            local = torch.cat((hidden, residual), dim=-1)
            gathered = torch.empty(
                (dist.get_world_size() * tokens, 2 * hidden_size),
                dtype=dtype,
                device=hidden.device,
            )
            dist.all_gather_into_tensor(gathered, local)
            left, right = gathered.chunk(2, dim=-1)
            return left + right

        def new_path(
            hidden: torch.Tensor = hidden,
            residual: torch.Tensor = residual,
            tokens: int = tokens,
        ) -> torch.Tensor:
            local = hidden + residual
            gathered = torch.empty(
                (dist.get_world_size() * tokens, hidden_size),
                dtype=dtype,
                device=hidden.device,
            )
            dist.all_gather_into_tensor(gathered, local)
            return gathered

        old_output = old_path()
        new_output = new_path()
        torch.cuda.synchronize()
        equivalent = torch.equal(old_output, new_output)
        max_abs_diff = float((old_output.float() - new_output.float()).abs().max().item())

        for _ in range(10):
            old_path()
            new_path()
        torch.cuda.synchronize()
        old_samples: list[float] = []
        new_samples: list[float] = []
        for batch in range(9):
            operations = ((old_path, old_samples), (new_path, new_samples))
            if batch % 2:
                operations = tuple(reversed(operations))
            for operation, samples in operations:
                dist.barrier()
                samples.append(_timed_ms(operation, iterations=20))
        old_stats = _summary(old_samples)
        new_stats = _summary(new_samples)
        old_median = old_stats["median_ms"]
        new_median = new_stats["median_ms"]
        rows.append(
            {
                "tokens_per_rank": tokens,
                "hidden_size": hidden_size,
                "dtype": str(dtype),
                "old_payload_bytes_per_rank": tokens
                * 2
                * hidden_size
                * torch.tensor([], dtype=dtype).element_size(),
                "new_payload_bytes_per_rank": tokens
                * hidden_size
                * torch.tensor([], dtype=dtype).element_size(),
                "payload_reduction_fraction": 0.5,
                "outputs_exactly_equal": equivalent,
                "max_abs_diff": max_abs_diff,
                "old_path": old_stats,
                "new_path": new_stats,
                "latency_reduction_fraction": (old_median - new_median) / old_median,
                "new_path_faster": new_median < old_median,
            }
        )
    return rows


def _bucket_equivalence(rank: int) -> dict[str, Any]:
    tensors = [
        torch.full((128, 256), rank + index / 8, dtype=torch.float32, device=f"cuda:{rank}")
        for index in range(6)
    ]
    separate = [tensor.clone() for tensor in tensors]
    for tensor in separate:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    flat = torch.cat([tensor.flatten() for tensor in tensors])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    bucketed = list(flat.split([tensor.numel() for tensor in tensors]))
    bucketed = [
        value.view_as(reference) for value, reference in zip(bucketed, tensors, strict=True)
    ]
    return {
        "tensor_count": len(tensors),
        "separate_collective_count": len(tensors),
        "bucketed_collective_count": 1,
        "dtype": str(tensors[0].dtype),
        "same_process_group": True,
        "same_reduction_op": True,
        "all_outputs_exactly_equal": all(
            torch.equal(reference, candidate)
            for reference, candidate in zip(separate, bucketed, strict=True)
        ),
        "max_abs_diff": max(
            float((reference - candidate).abs().max().item())
            for reference, candidate in zip(separate, bucketed, strict=True)
        ),
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
        scalar = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
        dist.all_reduce(scalar)
        gathered = torch.empty(world_size, device=f"cuda:{rank}")
        local = torch.tensor([float(rank)], device=f"cuda:{rank}")
        dist.all_gather_into_tensor(gathered, local)

        destination_value = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
        dist.reduce(destination_value, dst=0, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        destination_rows = [None for _ in range(world_size)]
        dist.all_gather_object(destination_rows, float(destination_value.item()))

        payload_rows = _all_gather_pair(rank)
        bucket = _bucket_equivalence(rank)
        material = {
            "rank": rank,
            "device": torch.cuda.get_device_name(rank),
            "all_reduce_sum": float(scalar.item()),
            "all_gather_values": gathered.cpu().tolist(),
            "destination_cp_reduce_values": destination_rows,
            "vllm_48763_paired_payload_rows": payload_rows,
            "torchtitan_3821_bucket_equivalence": bucket,
        }
        path = Path(output_dir) / f"rank-{rank}.json"
        path.write_text(json.dumps(material, sort_keys=True), encoding="utf-8")
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
        raise SystemExit("R12 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R12 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R12 plan/selection binding mismatch")
    source_bundle = _read(args.source_bundle)
    selected_ids = {item["case_id"] for item in selection_material["cases"]}
    if set(source_bundle) != selected_ids:
        raise SystemExit("R12 source bundle case set differs from selection")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("R12 dual-GPU probe requires at least two CUDA GPUs")

    started_at = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix="infraswe-r12-nccl-") as temporary:
        mp.spawn(
            _worker,
            args=(2, _free_port(), temporary),
            nprocs=2,
            join=True,
        )
        ranks = [_read(Path(temporary) / f"rank-{rank}.json") for rank in range(2)]

    smoke_ok = all(
        row["all_reduce_sum"] == 3.0 and row["all_gather_values"] == [0.0, 1.0] for row in ranks
    )
    destination_ok = all(row["destination_cp_reduce_values"][0] == 3.0 for row in ranks)
    paired_rows = []
    for index, rank_zero_row in enumerate(ranks[0]["vllm_48763_paired_payload_rows"]):
        rank_rows = [rank["vllm_48763_paired_payload_rows"][index] for rank in ranks]
        paired_rows.append(
            {
                **rank_zero_row,
                "all_ranks_outputs_exactly_equal": all(
                    row["outputs_exactly_equal"] for row in rank_rows
                ),
                "all_ranks_new_path_faster": all(row["new_path_faster"] for row in rank_rows),
                "rank_latency_reduction_fractions": [
                    row["latency_reduction_fraction"] for row in rank_rows
                ],
            }
        )
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
    facts = {
        "nccl_two_rank_smoke_passed": smoke_ok,
        "megatron_5720_destination_cp_sum_reaches_leader": destination_ok,
        "torchtitan_3821_bucket_equivalence_all_ranks": all(
            row["torchtitan_3821_bucket_equivalence"]["all_outputs_exactly_equal"] for row in ranks
        ),
        "torchtitan_3821_bucket_rows": [row["torchtitan_3821_bucket_equivalence"] for row in ranks],
        "vllm_48763_paired_payload_rows": paired_rows,
        "vllm_48763_all_shapes_equivalent": all(
            row["all_ranks_outputs_exactly_equal"] for row in paired_rows
        ),
        "vllm_48763_all_shapes_faster": all(
            row["all_ranks_new_path_faster"] for row in paired_rows
        ),
        "vllm_48763_scope": (
            "paired NCCL communication-path microbenchmark, not full-model E2E throughput"
        ),
        "exact_candidate_runtime_imported": False,
    }
    correctness_ok = (
        smoke_ok and destination_ok and facts["torchtitan_3821_bucket_equivalence_all_ranks"]
    )
    if not correctness_ok:
        raise SystemExit("R12 dual-GPU correctness control failed")
    if not facts["vllm_48763_all_shapes_equivalent"]:
        raise SystemExit("R12 vLLM paired payload outputs differ")
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
        "rank_evidence_sha256": [_canonical(row) for row in ranks],
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
