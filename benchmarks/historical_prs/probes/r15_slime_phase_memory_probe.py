#!/usr/bin/env python3
"""Real-CUDA phase peak-memory oracle for R15 slime PR #2304."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile

import torch
import torch.distributed as dist

os.environ.setdefault("SLIME_ACCELERATOR", "cuda")

from slime.utils.memory_utils import report_peak_memory


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    logging.basicConfig(level=logging.INFO)
    with tempfile.NamedTemporaryFile(delete=False) as store:
        store_path = store.name
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=0,
            world_size=1,
        )
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        with report_peak_memory("r15_probe"):
            allocation = torch.empty(16 * 1024 * 1024, device="cuda")
            allocation.fill_(1)
            torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        if allocated < 64 * 1024 * 1024:
            raise AssertionError(f"phase allocation was not observed: {allocated}")
        print(
            "R15_SLIME_PHASE_MEMORY="
            + json.dumps(
                {
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "expected_minimum_bytes": 64 * 1024 * 1024,
                },
                sort_keys=True,
            )
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(store_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
