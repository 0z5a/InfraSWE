#!/usr/bin/env python3
"""Exercise R16 slime nested fully-async rollout ordering on exact candidate code."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import slime.rollout.fully_async_rollout as rollout


class FakeWorker:
    def __init__(self) -> None:
        self._drained = False

    def queue_size(self) -> int:
        return 3

    def get_completed_groups(self, limit: int):
        if self._drained:
            return []
        self._drained = True
        groups = [
            (8, [[SimpleNamespace(index=7)]]),
            (9, [[], [SimpleNamespace(index=2)]]),
            (10, [[SimpleNamespace(index=5)], SimpleNamespace(index=6)]),
        ]
        return groups[:limit]


async def main() -> None:
    worker = FakeWorker()
    rollout._get_global_worker = lambda _args, _buffer: worker
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=3)
    result = await rollout._generate_rollout_async(args, 16, object())
    indexes = []
    for group in result:
        stack = list(group)
        while stack:
            item = stack.pop(0)
            if isinstance(item, list):
                stack = item + stack
                continue
            indexes.append(item.index)
            break
    assert indexes == [2, 5, 7], indexes
    print("nested_sort_indexes=2,5,7")


if __name__ == "__main__":
    asyncio.run(main())
