#!/usr/bin/env python3
"""Measure slime per-chunk temperature semantics and first-yield CUDA peak."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace

import torch
from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.loss import get_responses


def correctness() -> None:
    loss_module.mpu.get_context_parallel_world_size = lambda: 1
    source = torch.randn(1, 11, 17, device="cuda", dtype=torch.float32, requires_grad=True)
    tokens = [torch.arange(6, device="cuda"), torch.arange(5, device="cuda")]
    args = Namespace(qkv_format="thd", rollout_temperature=2.5, allgather_cp=False)
    outputs = list(
        get_responses(
            source,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[6, 5],
            response_lengths=[3, 2],
        )
    )
    expected = [source.squeeze(0)[2:5] / 2.5, source.squeeze(0)[8:10] / 2.5]
    for (actual, actual_tokens), reference, expected_tokens in zip(
        outputs,
        expected,
        (tokens[0][-3:], tokens[1][-2:]),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(actual_tokens, expected_tokens)
    sum(item[0].sum() for item in outputs).backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def peak(tokens: int, vocab: int, response: int) -> int:
    loss_module.mpu.get_context_parallel_world_size = lambda: 1
    source = torch.randn(1, tokens, vocab, device="cuda", dtype=torch.float32)
    token_ids = [torch.arange(tokens, device="cuda")]
    args = Namespace(qkv_format="thd", rollout_temperature=2.5, allgather_cp=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    iterator = get_responses(
        source,
        args=args,
        unconcat_tokens=token_ids,
        total_lengths=[tokens],
        response_lengths=[response],
    )
    chunk, _ = next(iterator)
    torch.cuda.synchronize()
    assert torch.isfinite(chunk).all()
    return torch.cuda.max_memory_allocated() - baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--response", type=int, default=512)
    args = parser.parse_args()
    correctness()
    measured = peak(args.tokens, args.vocab, args.response)
    print(json.dumps({"mode": args.mode, "first_yield_peak_bytes": measured}))


if __name__ == "__main__":
    main()
