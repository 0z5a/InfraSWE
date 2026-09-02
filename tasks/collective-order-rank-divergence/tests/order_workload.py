from __future__ import annotations

import gc
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
plan = json.loads(Path(os.environ["INFRASWE_ORDER_PLAN"]).read_text(encoding="utf-8"))
cases = json.loads(Path(os.environ["INFRASWE_ORDER_CASES"]).read_text(encoding="utf-8"))
metadata = {step["id"]: step for step in cases["rank_steps"][rank]}

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()
before = int(torch.cuda.memory_allocated(local_rank))
dist.init_process_group("nccl", timeout=timedelta(seconds=30))
warmup = torch.ones(1, device=local_rank)
dist.all_reduce(warmup)
torch.cuda.synchronize(local_rank)
del warmup
operations: list[dict[str, object]] = []
correct = True
total_ms = 0.0
try:
    for identifier in plan["rank_schedules"][str(rank)]:
        step = metadata[identifier]
        tensor = torch.full(
            (int(step["elements"]),), rank + 1, device=local_rank, dtype=torch.float32
        )
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(tensor)
        stop.record()
        stop.synchronize()
        elapsed = float(start.elapsed_time(stop))
        matches = bool(torch.all(tensor == world_size * (world_size + 1) / 2).item())
        correct = correct and matches and elapsed > 0
        total_ms += elapsed
        operations.append({"collective_ms": elapsed, "id": identifier, "matches": matches})
        del tensor, start, stop
    dist.barrier()
finally:
    dist.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(local_rank)

after = int(torch.cuda.memory_allocated(local_rank))
(evidence / f"rank-{rank}.json").write_text(
    json.dumps(
        {
            "collective_ms": total_ms,
            "correct": correct,
            "memory_after_bytes": after,
            "memory_before_bytes": before,
            "operations": operations,
            "rank": rank,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if correct and after <= before + 1_048_576 else 1)
