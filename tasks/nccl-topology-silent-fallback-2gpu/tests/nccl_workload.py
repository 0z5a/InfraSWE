from __future__ import annotations

import json
import os
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
torch.cuda.set_device(rank)
distributed.init_process_group("nccl", timeout=timedelta(seconds=60))

element_count = 1 << 20
tensor = torch.empty(element_count, device=f"cuda:{rank}", dtype=torch.float32)
for _ in range(5):
    tensor.fill_(rank + 1)
    distributed.all_reduce(tensor)
torch.cuda.synchronize()

latencies_ms: list[float] = []
for _ in range(20):
    tensor.fill_(rank + 1)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    distributed.all_reduce(tensor)
    stop.record()
    stop.synchronize()
    latencies_ms.append(float(start.elapsed_time(stop)))

expected = world_size * (world_size + 1) / 2
correct = bool(torch.allclose(tensor, torch.full_like(tensor, expected)))
ordered = sorted(latencies_ms)
p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
payload = {
    "rank": rank,
    "world_size": world_size,
    "correct": correct,
    "median_ms": statistics.median(latencies_ms),
    "p95_ms": ordered[p95_index],
    "bytes": tensor.numel() * tensor.element_size(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "nccl_version": list(torch.cuda.nccl.version()),
}
(evidence / f"rank-{rank}.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
distributed.barrier()
distributed.destroy_process_group()
