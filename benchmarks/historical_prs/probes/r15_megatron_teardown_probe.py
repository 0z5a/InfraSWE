#!/usr/bin/env python3
"""Two-rank lifecycle oracle for R15 Megatron PR #7029."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    cycles = []
    try:
        for generation in range(2):
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
            tracked = list(parallel_state._CREATED_PROCESS_GROUPS)
            if not tracked:
                raise AssertionError("initialize_model_parallel tracked no process groups")

            value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
            dist.all_reduce(
                value,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            if value.item() != 3.0:
                raise AssertionError(f"generation {generation}: collective sum={value.item()}")

            parallel_state.destroy_model_parallel()
            remaining_tracked = len(parallel_state._CREATED_PROCESS_GROUPS)
            still_registered = sum(
                group in dist.distributed_c10d._world.pg_map for group in tracked
            )
            if remaining_tracked or still_registered:
                raise AssertionError(
                    f"generation {generation}: remaining={remaining_tracked}, "
                    f"registered={still_registered}"
                )

            default_value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
            dist.all_reduce(default_value)
            if default_value.item() != 3.0:
                raise AssertionError(f"generation {generation}: default group was damaged")
            cycles.append(
                {
                    "generation": generation,
                    "tracked_before_destroy": len(tracked),
                    "tracked_after_destroy": remaining_tracked,
                    "registered_after_destroy": still_registered,
                    "model_parallel_sum": float(value.item()),
                    "default_group_sum_after_destroy": float(default_value.item()),
                }
            )

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, {"rank": rank, "cycles": cycles})
        if rank == 0:
            print("R15_MEGATRON_TEARDOWN=" + json.dumps(gathered, sort_keys=True))
    finally:
        if parallel_state._CREATED_PROCESS_GROUPS:
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
