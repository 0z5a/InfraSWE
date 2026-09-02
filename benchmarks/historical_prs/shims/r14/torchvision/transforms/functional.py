"""Fail-closed functional transforms used only to cross unrelated imports."""

from __future__ import annotations

from typing import Never

from . import InterpolationMode


def _unsupported(*_args: object, **_kwargs: object) -> Never:
    raise RuntimeError("R14 torchvision import shim cannot transform images")


pil_to_tensor = _unsupported
to_pil_image = _unsupported

__all__ = ["InterpolationMode", "pil_to_tensor", "to_pil_image"]
