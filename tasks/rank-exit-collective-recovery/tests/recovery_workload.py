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
plan = json.loads(Path(os.environ["INFRASWE_RECOVERY_PLAN"]).read_text(encoding="utf-8"))

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()
before = int(torch.cuda.memory_allocated(local_rank))
dist.init_process_group("nccl", timeout=timedelta(seconds=30))
resume_step = int(plan["resume_step"])
tensor = torch.tensor([resume_step + rank], device=local_rank, dtype=torch.int64)
dist.all_reduce(tensor)
expected = world_size * resume_step + world_size * (world_size - 1) // 2
correct = bool(tensor.item() == expected)
dist.barrier()
dist.destroy_process_group()
del tensor
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize(local_rank)
after = int(torch.cuda.memory_allocated(local_rank))
(evidence / f"recovered-rank-{rank}.json").write_text(
    json.dumps(
        {
            "checkpoint_value": expected,
            "correct": correct,
            "memory_after_bytes": after,
            "memory_before_bytes": before,
            "rank": rank,
            "replayed_request_ids": plan["replay_request_ids"],
            "resume_step": resume_step,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
raise SystemExit(0 if correct and after <= before + 1_048_576 else 1)
