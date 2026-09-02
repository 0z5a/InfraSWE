from __future__ import annotations

import gc
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
plan = json.loads(Path(os.environ["INFRASWE_OVERLAP_PLAN"]).read_text(encoding="utf-8"))
workload = json.loads(Path(os.environ["INFRASWE_OVERLAP_CASES"]).read_text(encoding="utf-8"))
stages = sorted(workload["stages"], key=lambda stage: stage["sequence"])

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()
before = int(torch.cuda.memory_allocated(local_rank))
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
comm_stream = torch.cuda.Stream(device=local_rank)
tensors = [
    torch.empty(int(stage["collective_elements"]), device=local_rank, dtype=torch.float32)
    for stage in stages
]


def compute(cycles: int) -> None:
    torch.cuda._sleep(cycles)


def serial_round() -> tuple[float, bool]:
    torch.cuda.synchronize(local_rank)
    started = time.perf_counter()
    correct = True
    for stage, tensor in zip(stages, tensors, strict=True):
        tensor.fill_(rank + 1)
        compute(int(stage["compute_cycles"]))
        dist.all_reduce(tensor)
        correct = correct and bool(torch.all(tensor == world_size * (world_size + 1) / 2).item())
    torch.cuda.synchronize(local_rank)
    return (time.perf_counter() - started) * 1000, correct


def overlap_round() -> tuple[float, bool]:
    torch.cuda.synchronize(local_rank)
    started = time.perf_counter()
    works = []
    for stage, tensor in zip(stages, tensors, strict=True):
        with torch.cuda.stream(comm_stream):
            tensor.fill_(rank + 1)
            works.append(dist.all_reduce(tensor, async_op=True))
        compute(int(stage["compute_cycles"]))
        works[-1].wait()
        done = torch.cuda.Event()
        done.record(comm_stream)
        torch.cuda.current_stream(local_rank).wait_event(done)
    torch.cuda.synchronize(local_rank)
    correct = all(
        bool(torch.all(tensor == world_size * (world_size + 1) / 2).item()) for tensor in tensors
    )
    return (time.perf_counter() - started) * 1000, correct


serial_round()
overlap_round()
serial_samples: list[float] = []
overlap_samples: list[float] = []
correct = True
for _ in range(3):
    serial_ms, serial_correct = serial_round()
    overlap_ms, overlap_correct = overlap_round()
    serial_samples.append(serial_ms)
    overlap_samples.append(overlap_ms)
    correct = correct and serial_correct and overlap_correct

serial_ms = statistics.median(serial_samples)
overlap_ms = statistics.median(overlap_samples)
enabled = all(
    plan.get(key) is True
    for key in (
        "comm_stream",
        "async_collectives",
        "event_fencing",
        "overlap_next_compute",
    )
)
candidate_ms = overlap_ms if enabled else serial_ms
speedup = serial_ms / candidate_ms if candidate_ms > 0 else 0.0

dist.barrier()
dist.destroy_process_group()
comm_stream.synchronize()
tensors.clear()
comm_stream = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize(local_rank)
after = int(torch.cuda.memory_allocated(local_rank))
(evidence / f"rank-{rank}.json").write_text(
    json.dumps(
        {
            "candidate_ms": candidate_ms,
            "correct": correct,
            "memory_after_bytes": after,
            "memory_before_bytes": before,
            "overlap_enabled": enabled,
            "overlap_ms": overlap_ms,
            "overlap_samples_ms": overlap_samples,
            "rank": rank,
            "serial_ms": serial_ms,
            "serial_samples_ms": serial_samples,
            "speedup_ratio": speedup,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if correct and after <= before + 1_048_576 else 1)
