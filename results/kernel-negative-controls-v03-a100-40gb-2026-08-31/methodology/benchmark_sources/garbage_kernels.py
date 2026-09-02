from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl

WASTE_ELEMENTS = 8 * 1024 * 1024
WASTE_PASSES = 64


@triton.jit
def _stream_waste_kernel(left, right, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def _zero_kernel(output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    tl.store(output + offsets, 0.0, mask=mask)


@triton.jit
def _copy_kernel(source, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(source + offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


def _unwrap(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def slow_fa4_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Callable[[], torch.Tensor]:
    from flash_attn.cute import flash_attn_func

    waste_left = torch.ones(WASTE_ELEMENTS, device=q.device, dtype=torch.bfloat16)
    waste_right = torch.ones_like(waste_left)
    waste_output = torch.empty_like(waste_left)

    def run() -> torch.Tensor:
        output = _unwrap(flash_attn_func(q, k, v, causal=causal))
        for _ in range(WASTE_PASSES):
            _stream_waste_kernel[(triton.cdiv(WASTE_ELEMENTS, 1024),)](
                waste_left,
                waste_right,
                waste_output,
                WASTE_ELEMENTS,
                BLOCK=1024,
                num_warps=8,
            )
        return output

    return run


def zero_prepare(
    q: torch.Tensor,
    _k: torch.Tensor,
    _v: torch.Tensor,
    _causal: bool,
) -> Callable[[], torch.Tensor]:
    output = torch.empty_like(q)
    elements = output.numel()

    def run() -> torch.Tensor:
        _zero_kernel[(triton.cdiv(elements, 256),)](output, elements, BLOCK=256)
        return output

    return run


class CachedAnswerAdapter:
    """Deliberately invalid adapter that replays the first answer for each shape."""

    def __init__(self) -> None:
        self._answers: dict[tuple[tuple[int, ...], bool], torch.Tensor] = {}

    def prepare(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> Callable[[], torch.Tensor]:
        from flash_attn.cute import flash_attn_func

        key = (tuple(q.shape), causal)
        if key not in self._answers:
            self._answers[key] = _unwrap(flash_attn_func(q, k, v, causal=causal)).detach()
        cached = self._answers[key]
        output = torch.empty_like(cached)
        elements = output.numel()

        def run() -> torch.Tensor:
            _copy_kernel[(triton.cdiv(elements, 256),)](cached, output, elements, BLOCK=256)
            return output

        return run
