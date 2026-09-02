#!/usr/bin/env python3
"""Async scheduling and device-binding oracle for R15 verl PR #6569."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module


def _load_broadcast_operation():
    verl = _module("verl", package=True)
    checkpoint_engine = _module("verl.checkpoint_engine", package=True)
    utils = _module("verl.utils", package=True)
    verl.checkpoint_engine = checkpoint_engine
    verl.utils = utils

    base = _module("verl.checkpoint_engine.base")

    class CheckpointEngineRegistry:
        @staticmethod
        def register(_name: str):
            return lambda value: value

    base.CheckpointEngine = object
    base.CheckpointEngineRegistry = CheckpointEngineRegistry
    base.TensorMeta = dict

    device = _module("verl.utils.device")
    device.is_torch_npu_available = lambda check_device=False: True
    distributed = _module("verl.utils.distributed")
    distributed.stateless_init_process_group = lambda *args, **kwargs: None
    net_utils = _module("verl.utils.net_utils")
    net_utils.get_free_port = lambda _ip: (0, None)
    net_utils.is_valid_ipv6_address = lambda _ip: False

    vllm = _module("vllm", package=True)
    vllm_distributed = _module("vllm.distributed", package=True)
    vllm_utils = _module("vllm.distributed.utils")
    vllm_utils.StatelessProcessGroup = object
    vllm.distributed = vllm_distributed
    vllm_distributed.utils = vllm_utils

    name = "verl.checkpoint_engine.hccl_checkpoint_engine"
    path = Path("verl/checkpoint_engine/hccl_checkpoint_engine.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[name] = candidate
    spec.loader.exec_module(candidate)
    return candidate.BroadcastOperation


BroadcastOperation = _load_broadcast_operation()


class FakeSocket:
    def __init__(self, events: list[str], metadata: dict[str, object]) -> None:
        self.events = events
        self.metadata = metadata

    async def send_string(self, topic: str, flags: int) -> None:
        self.events.append(f"send-topic:{topic}:{flags}")

    async def send_pyobj(self, metadata: dict[str, object]) -> None:
        self.events.append(f"send-meta:{metadata}")

    async def recv_string(self) -> str:
        self.events.append("recv-topic")
        return "topic"

    async def recv_pyobj(self) -> dict[str, object]:
        self.events.append("recv-meta")
        return self.metadata


class FakeGroup:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def broadcast(self, _bucket: torch.Tensor, src: int) -> None:
        self.events.append(f"broadcast:{src}:thread={threading.current_thread().name}")


async def run_rank(rank: int) -> dict[str, object]:
    events: list[str] = []
    metadata = {"rank": rank}
    operation = BroadcastOperation(
        rank=rank,
        process_group=FakeGroup(events),
        bucket=torch.zeros(1),
        metadata=metadata if rank == 0 else None,
        socket=FakeSocket(events, metadata),
        topic="r15",
        device=7,
    )
    events.append("constructor-returned")
    result = await operation.wait_for_complete()
    events.append("wait-returned")
    return {"rank": rank, "events": events, "metadata": result}


async def async_main() -> int:
    original_npu = getattr(torch, "npu", None)
    device_events: list[str] = []
    torch.npu = SimpleNamespace(set_device=lambda device: device_events.append(f"device:{device}"))
    try:
        rows = [await run_rank(0), await run_rank(1)]
    finally:
        if original_npu is None:
            del torch.npu
        else:
            torch.npu = original_npu
    for row in rows:
        events = row["events"]
        broadcast_index = next(i for i, item in enumerate(events) if item.startswith("broadcast:"))
        if events.index("constructor-returned") >= broadcast_index:
            raise AssertionError(f"broadcast was not scheduled asynchronously: {row}")
        if events.index("wait-returned") <= broadcast_index:
            raise AssertionError(f"wait returned before broadcast completion: {row}")
    if device_events != ["device:7", "device:7"]:
        raise AssertionError(f"thread device binding mismatch: {device_events}")
    print(
        "R15_VERL_HCCL_ASYNC="
        + json.dumps({"rows": rows, "device_events": device_events}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
