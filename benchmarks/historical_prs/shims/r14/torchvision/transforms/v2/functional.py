"""Fail-closed v2 functions used only to cross unrelated imports."""

from __future__ import annotations

from typing import Never


def _unsupported(*_args: object, **_kwargs: object) -> Never:
    raise RuntimeError("R14 torchvision import shim cannot transform images")


def __getattr__(_name: str) -> object:
    return _unsupported
