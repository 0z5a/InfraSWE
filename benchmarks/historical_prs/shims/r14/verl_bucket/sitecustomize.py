"""Bypass an unrelated vLLM rollout facade while testing its standalone bucket module.

The targeted bucket-transfer module explicitly has no vLLM dependency, but importing it
through Python's normal package path executes ``vllm_rollout/__init__.py`` first.  That facade
requires an installed vLLM distribution.  This shim replaces only that package facade with a
namespace pointing at the exact checked-out source; candidate module code remains untouched.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from pathlib import Path

if os.environ.get("INFRASWE_R14_BYPASS_VERL_VLLM_ROLLOUT_INIT") == "1":
    root = Path(os.environ["INFRASWE_R14_VERL_ROOT"])
    name = "verl.workers.rollout.vllm_rollout"
    package = types.ModuleType(name)
    package.__file__ = str(root / "verl/workers/rollout/vllm_rollout/__init__.py")
    package.__package__ = name
    package.__path__ = [str(root / "verl/workers/rollout/vllm_rollout")]
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package

if os.environ.get("INFRASWE_R14_BYPASS_TORCHTITAN_RL_INIT") == "1":
    root = Path(os.environ["INFRASWE_R14_TORCHTITAN_ROOT"])
    name = "torchtitan.experiments.rl"
    package = types.ModuleType(name)
    package.__file__ = str(root / "torchtitan/experiments/rl/__init__.py")
    package.__package__ = name
    package.__path__ = [str(root / "torchtitan/experiments/rl")]
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package

    # The selected CPU-only tests exercise static packing/reduction helpers.  Keep
    # the optional actor/TorchStore runtime outside their import boundary.
    torchstore = types.ModuleType("torchstore")
    sys.modules["torchstore"] = torchstore

    monarch = types.ModuleType("monarch")
    monarch_actor = types.ModuleType("monarch.actor")

    class _Actor:
        pass

    class _Rank:
        rank = 0

    def _current_rank() -> _Rank:
        return _Rank()

    def _endpoint(function: object) -> object:
        return function

    monarch_actor.Actor = _Actor
    monarch_actor.current_rank = _current_rank
    monarch_actor.endpoint = _endpoint
    monarch.actor = monarch_actor
    sys.modules["monarch"] = monarch
    sys.modules["monarch.actor"] = monarch_actor
