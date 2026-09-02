from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x, y, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    left = tl.load(x + offsets, mask=mask)
    right = tl.load(y + offsets, mask=mask)
    tl.store(output + offsets, left + right, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    elements = x.numel()
    _vector_add_kernel[(triton.cdiv(elements, 256),)](x, y, output, elements, BLOCK=256)
    return output


@triton.jit
def _softmax_kernel(output, source, columns: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    values = tl.load(source + row * columns + offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(output + row * columns + offsets, numerator / denominator, mask=mask)


def softmax(source: torch.Tensor) -> torch.Tensor:
    rows, columns = source.shape
    output = torch.empty_like(source)
    block = triton.next_power_of_2(columns)
    warps = 8 if block >= 4096 else 4
    _softmax_kernel[(rows,)](output, source, columns=columns, BLOCK=block, num_warps=warps)
    return output


@triton.jit
def _layernorm_kernel(
    source,
    weight,
    bias,
    output,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    values = tl.load(source + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    normalized = centered * tl.rsqrt(variance + epsilon)
    scale = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + row * columns + offsets, normalized * scale + shift, mask=mask)


def layernorm(
    source: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    rows, columns = source.shape
    output = torch.empty_like(source)
    block = triton.next_power_of_2(columns)
    _layernorm_kernel[(rows,)](
        source,
        weight,
        bias,
        output,
        columns=columns,
        epsilon=epsilon,
        BLOCK=block,
        num_warps=8,
    )
    return output


@triton.jit
def _rmsnorm_kernel(
    source,
    weight,
    output,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    values = tl.load(source + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / columns
    scale = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = values * tl.rsqrt(variance + epsilon) * scale
    tl.store(output + row * columns + offsets, normalized, mask=mask)


def rmsnorm(source: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    rows, columns = source.shape
    output = torch.empty_like(source)
    block = triton.next_power_of_2(columns)
    _rmsnorm_kernel[(rows,)](
        source,
        weight,
        output,
        columns=columns,
        epsilon=epsilon,
        BLOCK=block,
        num_warps=8,
    )
    return output


@triton.jit
def _swiglu_kernel(gate, value, output, elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    gate_values = tl.load(gate + offsets, mask=mask).to(tl.float32)
    value_values = tl.load(value + offsets, mask=mask).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate_values))
    tl.store(output + offsets, gate_values * sigmoid * value_values, mask=mask)


def swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(gate)
    elements = gate.numel()
    _swiglu_kernel[(triton.cdiv(elements, 256),)](gate, value, output, elements, BLOCK=256)
    return output


@triton.jit
def _rope_kernel(
    source,
    cosine,
    sine,
    output,
    rows,
    heads: tl.constexpr,
    seqlen: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < head_dim
    half = head_dim // 2
    position = (row // heads) % seqlen
    pair_columns = columns % half
    first = tl.load(source + row * head_dim + pair_columns, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(source + row * head_dim + pair_columns + half, mask=mask, other=0.0).to(
        tl.float32
    )
    cos_values = tl.load(cosine + position * half + pair_columns, mask=mask, other=0.0).to(
        tl.float32
    )
    sin_values = tl.load(sine + position * half + pair_columns, mask=mask, other=0.0).to(tl.float32)
    rotated = tl.where(
        columns < half,
        first * cos_values - second * sin_values,
        first * sin_values + second * cos_values,
    )
    tl.store(output + row * head_dim + columns, rotated, mask=mask)


def rope(source: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    batch, seqlen, heads, head_dim = source.shape
    output = torch.empty_like(source)
    rows = batch * seqlen * heads
    _rope_kernel[(rows,)](
        source,
        cosine,
        sine,
        output,
        rows,
        heads=heads,
        seqlen=seqlen,
        head_dim=head_dim,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


@triton.jit
def _matmul_kernel(
    left,
    right,
    output,
    rows,
    columns,
    inner,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    program = tl.program_id(0)
    programs_m = tl.cdiv(rows, BLOCK_M)
    programs_n = tl.cdiv(columns, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group = program // programs_per_group
    first_program_m = group * GROUP_M
    group_size_m = min(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + ((program % programs_per_group) % group_size_m)
    program_n = (program % programs_per_group) // group_size_m
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    left_pointers = left + offsets_m[:, None] * stride_lm + offsets_k[None, :] * stride_lk
    right_pointers = right + offsets_k[:, None] * stride_rk + offsets_n[None, :] * stride_rn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, tl.cdiv(inner, BLOCK_K)):
        left_values = tl.load(
            left_pointers,
            mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] + k_offset * BLOCK_K < inner),
            other=0.0,
        )
        right_values = tl.load(
            right_pointers,
            mask=(offsets_k[:, None] + k_offset * BLOCK_K < inner) & (offsets_n[None, :] < columns),
            other=0.0,
        )
        accumulator += tl.dot(left_values, right_values)
        left_pointers += BLOCK_K * stride_lk
        right_pointers += BLOCK_K * stride_rk
    output_offsets = output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
    output_mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    tl.store(output_offsets, accumulator, mask=output_mask)


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    rows, inner = left.shape
    inner_right, columns = right.shape
    if inner != inner_right:
        raise ValueError("matmul dimensions do not align")
    output = torch.empty((rows, columns), device=left.device, dtype=left.dtype)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)
    _matmul_kernel[grid](
        left,
        right,
        output,
        rows,
        columns,
        inner,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    return output
