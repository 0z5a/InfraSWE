"""Fail-closed transform names needed by Transformers during test collection."""

from __future__ import annotations

from enum import Enum


class InterpolationMode(Enum):
    """Match torchvision's public interpolation member names only."""

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


__all__ = ["InterpolationMode"]
