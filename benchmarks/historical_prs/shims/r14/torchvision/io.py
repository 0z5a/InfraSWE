"""Fail-closed torchvision.io surface used only to cross an unrelated import."""

from __future__ import annotations

from enum import Enum
from typing import Never


class ImageReadMode(Enum):
    """Names provided by torchvision without implementing image decoding."""

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


def _unsupported(*_args: object, **_kwargs: object) -> Never:
    raise RuntimeError("R14 torchvision import shim cannot decode images")


decode_image = _unsupported
decode_jpeg = _unsupported
