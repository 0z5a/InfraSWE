from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=15))
dist.barrier()
if rank == 1:
    (evidence / "injected-rank-exit.json").write_text(
        json.dumps(
            {"exit_code": 17, "failed_operation": "all_reduce", "rank": rank},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os._exit(17)

detected = False
error_text = None
try:
    time.sleep(0.5)
    tensor = torch.ones(16, device=local_rank)
    dist.all_reduce(tensor)
except Exception as error:
    detected = True
    error_text = f"{type(error).__name__}: {error}"
finally:
    (evidence / "surviving-rank.json").write_text(
        json.dumps(
            {"detected_peer_exit": detected, "error": error_text, "rank": rank},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with contextlib.suppress(Exception):
        dist.destroy_process_group()
raise SystemExit(0 if detected else 1)
