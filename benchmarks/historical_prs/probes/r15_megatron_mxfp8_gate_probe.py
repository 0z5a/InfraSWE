#!/usr/bin/env python3
"""Branch-matrix oracle for R15 Megatron PR #5146."""

from __future__ import annotations

import json
from types import SimpleNamespace

from megatron.core.optimizer import distrib_optimizer, optimizer


class FakeDistributedOptimizer:
    def __init__(self, overlap_param_gather: bool) -> None:
        self.ddp_config = SimpleNamespace(overlap_param_gather=overlap_param_gather)


def evaluate(reuse: bool, overlaps: list[bool], include_other: bool) -> bool:
    chained: list[object] = [FakeDistributedOptimizer(value) for value in overlaps]
    if include_other:
        chained.append(object())
    owner = SimpleNamespace(
        config=SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=reuse),
        chained_optimizers=chained,
    )
    return optimizer.ChainedOptimizer._should_defer_mxfp8_param_sync(owner)


def main() -> int:
    original = distrib_optimizer.DistributedOptimizer
    distrib_optimizer.DistributedOptimizer = FakeDistributedOptimizer
    try:
        rows = [
            {"reuse": False, "overlaps": [False], "other": False, "expected": False},
            {"reuse": True, "overlaps": [], "other": True, "expected": False},
            {"reuse": True, "overlaps": [True], "other": False, "expected": False},
            {"reuse": True, "overlaps": [False], "other": False, "expected": True},
            {"reuse": True, "overlaps": [True, False], "other": True, "expected": True},
        ]
        for row in rows:
            row["observed"] = evaluate(row["reuse"], row["overlaps"], row["other"])
            if row["observed"] != row["expected"]:
                raise AssertionError(row)
        print("R15_MEGATRON_MXFP8_GATE=" + json.dumps(rows, sort_keys=True))
    finally:
        distrib_optimizer.DistributedOptimizer = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
