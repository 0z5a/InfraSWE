#!/usr/bin/env python3
"""Blind correctness probe for SGLang's CUDA MoE token alignment kernel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sgl_kernel import moe_align_block_size

NUM_EXPERTS = 256


def _make_ids(num_tokens: int, topk: int, mode: str) -> torch.Tensor:
    if mode == "skew":
        values = torch.full((num_tokens, topk), 7, dtype=torch.int32)
    else:
        generator = torch.Generator().manual_seed(20260901 + num_tokens + topk)
        values = torch.randint(
            0,
            NUM_EXPERTS,
            (num_tokens, topk),
            dtype=torch.int32,
            generator=generator,
        )
    return values.cuda()


def _run_case(name: str, num_tokens: int, topk: int, block_size: int, mode: str) -> dict[str, Any]:
    topk_ids = _make_ids(num_tokens, topk, mode)
    numel = topk_ids.numel()
    flat_cpu = topk_ids.cpu().flatten().tolist()
    expected_positions: dict[int, list[int]] = {expert: [] for expert in range(NUM_EXPERTS)}
    for position, expert in enumerate(flat_cpu):
        expected_positions[expert].append(position)

    max_num_tokens_padded = numel + NUM_EXPERTS * (block_size - 1)
    sorted_token_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32, device="cuda")
    expert_ids = torch.full(
        (max_num_tokens_padded // block_size,), -1, dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")
    token_cnts_buffer = torch.empty(
        (NUM_EXPERTS + 1) * NUM_EXPERTS, dtype=torch.int32, device="cuda"
    )
    cumsum_buffer = torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device="cuda")

    moe_align_block_size(
        topk_ids,
        NUM_EXPERTS,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )
    torch.cuda.synchronize()

    got_total = int(num_tokens_post_pad.item())
    expected_total = sum(
        ((len(positions) + block_size - 1) // block_size) * block_size
        for positions in expected_positions.values()
    )
    if got_total != expected_total:
        raise AssertionError(f"{name}: padded total {got_total} != {expected_total}")

    sorted_cpu = sorted_token_ids[:got_total].cpu().tolist()
    expert_cpu = expert_ids[: got_total // block_size].cpu().tolist()
    seen: list[int] = []
    offset = 0
    for expert in range(NUM_EXPERTS):
        expected = expected_positions[expert]
        padded = ((len(expected) + block_size - 1) // block_size) * block_size
        segment = sorted_cpu[offset : offset + padded]
        actual = [position for position in segment if position != numel]
        padding = [position for position in segment if position == numel]
        if Counter(actual) != Counter(expected):
            raise AssertionError(
                f"{name}: expert {expert} token positions differ; "
                f"got={sorted(actual)[:20]} expected={expected[:20]}"
            )
        if len(padding) != padded - len(expected):
            raise AssertionError(f"{name}: expert {expert} padding was overwritten")
        if any(position < 0 or position >= numel for position in actual):
            raise AssertionError(f"{name}: expert {expert} emitted an invalid token index")
        block_experts = expert_cpu[offset // block_size : (offset + padded) // block_size]
        if block_experts != [expert] * (padded // block_size):
            raise AssertionError(f"{name}: expert block labels differ for expert {expert}")
        seen.extend(actual)
        offset += padded

    if Counter(seen) != Counter(range(numel)):
        raise AssertionError(f"{name}: output is not a permutation of all token positions")

    return {
        "name": name,
        "num_tokens": num_tokens,
        "topk": topk,
        "block_size": block_size,
        "mode": mode,
        "numel": numel,
        "tokens_post_pad": got_total,
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        ("tiny", 1, 1, 32, "random"),
        ("small", 17, 4, 64, "random"),
        ("boundary", 257, 8, 128, "random"),
        ("multi_block", 4096, 8, 64, "random"),
        ("atomic_skew", 4096, 16, 32, "skew"),
    ]
    results = [_run_case(*case) for case in cases]
    payload = {
        "probe": "sglang-moe-align-correctness-v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "status": "pass",
        "cases": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
