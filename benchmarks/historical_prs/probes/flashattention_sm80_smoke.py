#!/usr/bin/env python3
"""Blind SM80 forward/backward smoke for FlashAttention CUDA entry points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import flash_attn_2_cuda as flash_attn_gpu
import torch


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    if causal:
        query_position = torch.arange(q.shape[1], device=q.device) + k.shape[1] - q.shape[1]
        key_position = torch.arange(k.shape[1], device=q.device)
        allowed = query_position.unsqueeze(1) >= key_position.unsqueeze(0)
        scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probabilities, v.float()).to(q.dtype)


def _fixed_forward_backward(
    name: str,
    dtype: torch.dtype,
    causal: bool,
    head_dim: int,
) -> dict[str, Any]:
    torch.manual_seed(20260901 + head_dim + int(causal))
    shape_q = (2, 17, 4, head_dim)
    shape_kv = (2, 31, 4, head_dim)
    q_data = torch.randn(shape_q, dtype=dtype, device="cuda")
    k_data = torch.randn(shape_kv, dtype=dtype, device="cuda")
    v_data = torch.randn(shape_kv, dtype=dtype, device="cuda")
    q = q_data.detach().clone()
    k = k_data.detach().clone()
    v = v_data.detach().clone()
    q_ref = q_data.detach().clone().requires_grad_(True)
    k_ref = k_data.detach().clone().requires_grad_(True)
    v_ref = v_data.detach().clone().requires_grad_(True)

    softmax_scale = head_dim ** (-0.5)
    output, softmax_lse, _, rng_state = flash_attn_gpu.fwd(
        q,
        k,
        v,
        None,
        None,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        False,
        None,
    )
    output_ref = _reference(q_ref, k_ref, v_ref, causal)
    output_error = float((output.float() - output_ref.float()).abs().max().item())
    output_atol = 0.02 if dtype == torch.float16 else 0.06
    torch.testing.assert_close(output, output_ref, rtol=output_atol, atol=output_atol)

    grad_output = torch.randn_like(output)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dq, dk, dv, _ = flash_attn_gpu.bwd(
        grad_output,
        q,
        k,
        v,
        output,
        softmax_lse,
        dq,
        dk,
        dv,
        None,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        False,
        None,
        rng_state,
    )
    grads = (dq, dk, dv)
    grads_ref = torch.autograd.grad(output_ref, (q_ref, k_ref, v_ref), grad_output)
    grad_errors = [
        float((actual.float() - expected.float()).abs().max().item())
        for actual, expected in zip(grads, grads_ref, strict=True)
    ]
    grad_atol = 0.05 if dtype == torch.float16 else 0.15
    for actual, expected in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(actual, expected, rtol=grad_atol, atol=grad_atol)

    return {
        "name": name,
        "dtype": str(dtype),
        "causal": causal,
        "head_dim": head_dim,
        "max_abs_output_error": output_error,
        "max_abs_gradient_errors": grad_errors,
        "status": "pass",
    }


def _varlen_forward() -> dict[str, Any]:
    torch.manual_seed(20260902)
    dtype = torch.float16
    head_dim = 64
    num_heads = 4
    q_lens = [5, 11]
    kv_lens = [7, 13]
    q_parts = [
        torch.randn(1, length, num_heads, head_dim, dtype=dtype, device="cuda") for length in q_lens
    ]
    k_parts = [
        torch.randn(1, length, num_heads, head_dim, dtype=dtype, device="cuda")
        for length in kv_lens
    ]
    v_parts = [
        torch.randn(1, length, num_heads, head_dim, dtype=dtype, device="cuda")
        for length in kv_lens
    ]
    q = torch.cat([part.squeeze(0) for part in q_parts], dim=0)
    k = torch.cat([part.squeeze(0) for part in k_parts], dim=0)
    v = torch.cat([part.squeeze(0) for part in v_parts], dim=0)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32, device="cuda")
    output, _, _, _ = flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_q,
        cu_k,
        None,
        None,
        None,
        None,
        max(q_lens),
        max(kv_lens),
        0.0,
        head_dim ** (-0.5),
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
    )
    output_ref = torch.cat(
        [
            _reference(q_part, k_part, v_part, True).squeeze(0)
            for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts, strict=True)
        ],
        dim=0,
    )
    error = float((output.float() - output_ref.float()).abs().max().item())
    torch.testing.assert_close(output, output_ref, rtol=0.02, atol=0.02)
    return {
        "name": "varlen_causal",
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "max_abs_output_error": error,
        "status": "pass",
    }


def _kvcache_forward() -> dict[str, Any]:
    torch.manual_seed(20260903)
    dtype = torch.float16
    q = torch.randn(2, 3, 4, 64, dtype=dtype, device="cuda")
    k = torch.randn(2, 19, 4, 64, dtype=dtype, device="cuda")
    v = torch.randn(2, 19, 4, 64, dtype=dtype, device="cuda")
    cache_seqlens = torch.full((2,), 19, dtype=torch.int32, device="cuda")
    output, _ = flash_attn_gpu.fwd_kvcache(
        q,
        k,
        v,
        None,
        None,
        cache_seqlens,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        64 ** (-0.5),
        True,
        -1,
        -1,
        0.0,
        True,
        0,
    )
    output_ref = _reference(q, k, v, True)
    error = float((output.float() - output_ref.float()).abs().max().item())
    torch.testing.assert_close(output, output_ref, rtol=0.02, atol=0.02)
    return {
        "name": "kvcache_causal",
        "max_abs_output_error": error,
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        _fixed_forward_backward("fp16_noncausal_hdim64", torch.float16, False, 64),
        _fixed_forward_backward("bf16_causal_hdim128", torch.bfloat16, True, 128),
        _fixed_forward_backward("fp16_causal_hdim256", torch.float16, True, 256),
        _varlen_forward(),
        _kvcache_forward(),
    ]
    torch.cuda.synchronize()
    payload = {
        "probe": "flashattention-sm80-entrypoints-v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "status": "pass",
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
