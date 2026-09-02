#!/usr/bin/env python3
"""Explain the cross-warp shared-memory hazard repaired by vLLM #13140."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path


def _kernel(source: str) -> str:
    match = re.search(
        r"template <typename scalar_t>\n"
        r"__global__ void sgl_moe_align_block_size_kernel\([\s\S]+?\n}\n\n"
        r"template <typename scalar_t, int TOPK>",
        source,
    )
    if not match:
        raise RuntimeError("cannot locate exact sgl_moe_align_block_size_kernel")
    return match.group(0).rsplit("\ntemplate <typename scalar_t, int TOPK>", 1)[0]


def _depth_before_lines(source: str) -> list[int]:
    depth = 0
    depths: list[int] = []
    for line in source.splitlines():
        depths.append(depth)
        depth += line.count("{") - line.count("}")
    return depths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--head-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_full = args.base_source.read_text(encoding="utf-8")
    head_full = args.head_source.read_text(encoding="utf-8")
    base = _kernel(base_full)
    head = _kernel(head_full)
    diff = list(
        difflib.unified_diff(
            base.splitlines(),
            head.splitlines(),
            fromfile="base-kernel",
            tofile="head-kernel",
            lineterm="",
        )
    )
    additions = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    deletions = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    head_lines = head.splitlines()
    depths = _depth_before_lines(head)
    barrier_lines = [
        (index + 1, depths[index])
        for index, line in enumerate(head_lines)
        if "__syncthreads();" in line
    ]
    initialization = "shared_counts[warp_id][i] = 0;"
    cross_warp_atomic = "atomicAdd(&shared_counts[warp_idx][expert_offset], 1);"
    facts = {
        "launch_threads": 1024,
        "warps_per_block": 32,
        "shared_count_rows": 32,
        "experts_per_row": 8,
        "max_experts": 256,
        "initializer_index": "warp_id = threadIdx.x / 32",
        "consumer_index": "warp_idx = topk_ids[i] / 8",
        "base_has_initializer": initialization in base,
        "base_has_cross_warp_atomic": cross_warp_atomic in base,
        "base_barrier_count": base.count("__syncthreads();"),
        "head_barrier_count": head.count("__syncthreads();"),
        "head_barrier_lines_and_pre_line_brace_depth": barrier_lines,
        "diff_additions": additions,
        "diff_deletions": deletions,
    }
    substantive_additions = [line for line in additions if line.strip()]
    substantive_deletions = [line for line in deletions if line.strip()]
    exact_change = substantive_additions == ["  __syncthreads();"] and not substantive_deletions
    added_barrier_depth = barrier_lines[0][1] if barrier_lines else None
    unconditional = added_barrier_depth == 1
    hazard = (
        facts["base_has_initializer"]
        and facts["base_has_cross_warp_atomic"]
        and facts["base_barrier_count"] + 1 == facts["head_barrier_count"]
    )
    status = "pass" if exact_change and unconditional and hazard else "unresolved"
    payload = {
        "schema_version": "0.5",
        "probe": "vllm-moe-barrier-static-v1",
        "case_id": "vllm-pr-13140",
        "status": status,
        "facts": facts,
        "source_identity": {
            "base_source_sha256": "sha256:" + hashlib.sha256(base_full.encode()).hexdigest(),
            "head_source_sha256": "sha256:" + hashlib.sha256(head_full.encode()).hexdigest(),
            "kernel_diff_sha256": "sha256:" + hashlib.sha256("\n".join(diff).encode()).hexdigest(),
        },
        "conclusions": [
            "Each of 32 warps initializes one shared_counts row.",
            "Any token thread can atomically update any row selected by input expert_id.",
            (
                "Without a block barrier, one warp can update a row before its owning warp "
                "finishes initialization."
            ),
            (
                "The added depth-1 barrier is reached by every block thread and separates "
                "initialization from atomics."
            ),
        ],
        "failure_codes": [] if status == "pass" else ["CUDA_BARRIER_STRUCTURE_UNRESOLVED"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
