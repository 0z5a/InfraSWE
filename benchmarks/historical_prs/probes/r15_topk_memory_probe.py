#!/usr/bin/env python3
"""A100 peak-memory probe for R15 verl PR #6593."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as functional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "chunked"), required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(6593)
    logits = torch.randn(
        1,
        args.tokens,
        args.vocab,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    topk_ids = torch.randint(
        args.vocab,
        (1, args.tokens, args.top_k),
        device="cuda",
    )
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    if args.mode == "baseline":
        result = functional.log_softmax(logits, dim=-1).gather(-1, topk_ids)
    else:
        from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs

        result = _chunked_topk_log_probs(logits, topk_ids, args.chunk_size)
    result.float().sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    row = {
        "mode": args.mode,
        "tokens": args.tokens,
        "vocab": args.vocab,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "allocated_before_bytes": allocated_before,
        "peak_bytes": peak,
        "peak_delta_bytes": peak - allocated_before,
        "finite_gradient": bool(torch.isfinite(logits.grad).all().item()),
    }
    print("R15_TOPK_MEMORY=" + json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
