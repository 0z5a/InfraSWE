#!/usr/bin/env python3
"""Call-order oracle for R15 verl PR #7631."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from verl.workers.engine_workers import ActorRolloutRefWorker


class FakeEngine:
    def __init__(self, events: list[str], enabled: bool = True) -> None:
        self.events = events
        self.is_param_offload_enabled = enabled

    def get_per_tensor_param(self):
        return iter((("weight", object()),)), None

    def to(self, device: str, **kwargs: bool) -> None:
        self.events.append(f"offload:{device}:{kwargs}")


class FakeCheckpointEngine:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def send_weights(self, weights, global_steps=None):
        self.events.append(f"send-start:{global_steps}")
        list(weights)
        await asyncio.sleep(0)
        self.events.append("send-complete")
        return {"sent": 1}


async def run_case(strategy: str, enabled: bool) -> dict[str, object]:
    events: list[str] = []
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        actor=SimpleNamespace(strategy=strategy),
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="modelexpress")
        ),
    )
    worker.actor = SimpleNamespace(engine=FakeEngine(events, enabled=enabled))
    worker.checkpoint_engine = FakeCheckpointEngine(events)
    metrics = await ActorRolloutRefWorker.update_weights.__wrapped__(
        worker,
        global_steps=17,
        mode="auto",
    )
    return {"strategy": strategy, "enabled": enabled, "events": events, "metrics": metrics}


async def async_main() -> int:
    rows = [
        await run_case("megatron", True),
        await run_case("megatron", False),
        await run_case("fsdp", True),
    ]
    expected = ["send-start:17", "send-complete"]
    if rows[0]["events"][:2] != expected or not str(rows[0]["events"][2]).startswith(
        "offload:cpu:"
    ):
        raise AssertionError(f"offload did not follow completed transfer: {rows[0]}")
    if rows[1]["events"] != expected or rows[2]["events"] != expected:
        raise AssertionError(f"offload guard failed: {rows[1:]}")
    print("R15_VERL_WEIGHT_SYNC_ORDER=" + json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
