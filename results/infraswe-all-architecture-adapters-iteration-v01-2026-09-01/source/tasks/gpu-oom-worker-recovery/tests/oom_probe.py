from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

output_path = Path(os.environ["INFRASWE_OOM_PROBE_OUTPUT"])
result: dict[str, object] = {
    "caught_cuda_oom": False,
    "error_type": None,
    "message_contains_oom": False,
}
allocation = None
try:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.002, 0)
    result["memory_before_bytes"] = int(torch.cuda.memory_allocated(0))
    try:
        allocation = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda:0")
        torch.cuda.synchronize(0)
    except (torch.OutOfMemoryError, RuntimeError) as error:
        message = str(error).lower()
        result["caught_cuda_oom"] = "out of memory" in message
        result["error_type"] = type(error).__name__
        result["message_contains_oom"] = "out of memory" in message
finally:
    del allocation
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    result["memory_after_bytes"] = int(torch.cuda.memory_allocated(0))
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

raise SystemExit(0 if result["caught_cuda_oom"] else 1)
