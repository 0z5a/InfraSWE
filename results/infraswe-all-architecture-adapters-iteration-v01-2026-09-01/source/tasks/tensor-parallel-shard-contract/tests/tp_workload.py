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
plan = json.loads(Path(os.environ["INFRASWE_TP_PLAN"]).read_text(encoding="utf-8"))

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()
before = int(torch.cuda.memory_allocated(local_rank))
dist.init_process_group("nccl", timeout=timedelta(seconds=30))
warmup = torch.ones(1, device=local_rank)
dist.all_reduce(warmup)
torch.cuda.synchronize(local_rank)
del warmup
results: list[dict[str, object]] = []
correct = True
collective_ms = 0.0
try:
    for entry in plan["parameters"]:
        shape = tuple(entry["shape"])
        full = torch.arange(
            1, 1 + int(torch.tensor(shape).prod().item()), device=local_rank, dtype=torch.float32
        ).reshape(shape)
        shard = entry["shards"][rank]
        axis = int(entry["axis"])
        start = int(shard["start"])
        end = int(shard["end"])
        local = full.narrow(axis, start, end - start).contiguous()
        timer_start = torch.cuda.Event(enable_timing=True)
        timer_stop = torch.cuda.Event(enable_timing=True)
        timer_start.record()
        if entry["replicated"]:
            checksum = local.sum().reshape(1)
            dist.all_reduce(checksum)
            expected_checksum = full.sum() * world_size
            reconstructed = local
            matches = bool(torch.equal(reconstructed, full)) and bool(
                torch.equal(checksum, expected_checksum.reshape(1))
            )
        else:
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            reconstructed = torch.cat(gathered, dim=axis)
            matches = bool(torch.equal(reconstructed, full))
        timer_stop.record()
        timer_stop.synchronize()
        elapsed = float(timer_start.elapsed_time(timer_stop))
        collective_ms += elapsed
        correct = correct and matches and elapsed > 0
        results.append(
            {
                "collective_ms": elapsed,
                "matches": matches,
                "name": entry["name"],
                "shard_shape": list(local.shape),
            }
        )
        del full, local, reconstructed, timer_start, timer_stop
        if not entry["replicated"]:
            del gathered
        else:
            del checksum, expected_checksum
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
            "collective_ms": collective_ms,
            "correct": correct,
            "memory_after_bytes": after,
            "memory_before_bytes": before,
            "parameters": results,
            "rank": rank,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if correct and after <= before + 1_048_576 else 1)
