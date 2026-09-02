#!/usr/bin/env python3
"""Interleaved paired CUDA Graph timing for baseline and candidate LoRA ops."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from graph_lora_ops_benchmark import (
    CASES,
    _capture,
    _load_module,
    _make_inputs,
    _sha256,
)


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    base_output: torch.Tensor,
    base_seed: torch.Tensor,
    iterations: int,
) -> float:
    base_output.copy_(base_seed)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    baseline = _load_module(args.baseline)
    candidate = _load_module(args.candidate)
    seed = 2_026_090_200 + args.replay_index * 100_000
    results = []
    for case_index, case in enumerate(CASES):
        baseline_tensors = _make_inputs(case, seed + case_index * 1000)
        candidate_tensors = _make_inputs(case, seed + case_index * 1000)
        baseline_graph, _, baseline_seed = _capture(baseline, baseline_tensors)
        candidate_graph, _, candidate_seed = _capture(candidate, candidate_tensors)
        samples = {"baseline": [], "candidate": []}
        pairs = []
        for sample_index in range(args.samples):
            order = (
                ("baseline", "candidate")
                if (args.replay_index + case_index + sample_index) % 2 == 0
                else ("candidate", "baseline")
            )
            values = {}
            for variant in order:
                if variant == "baseline":
                    value = _time_graph(
                        baseline_graph,
                        baseline_tensors["base_output"],
                        baseline_seed,
                        args.iterations,
                    )
                else:
                    value = _time_graph(
                        candidate_graph,
                        candidate_tensors["base_output"],
                        candidate_seed,
                        args.iterations,
                    )
                values[variant] = value
                samples[variant].append(value)
            pairs.append(
                {
                    "sample_index": sample_index,
                    "order": list(order),
                    "baseline_us": values["baseline"],
                    "candidate_us": values["candidate"],
                    "candidate_to_baseline": values["candidate"] / values["baseline"],
                    "speedup": values["baseline"] / values["candidate"],
                }
            )
        results.append(
            {
                "case_id": case.case_id,
                "slots": case.slots,
                "weight": case.weight,
                "baseline_median_us": statistics.median(samples["baseline"]),
                "candidate_median_us": statistics.median(samples["candidate"]),
                "paired_ratio_median": statistics.median(
                    pair["candidate_to_baseline"] for pair in pairs
                ),
                "paired_speedup_median": statistics.median(pair["speedup"] for pair in pairs),
                "pairs": pairs,
            }
        )

    payload = {
        "schema_version": "0.5",
        "benchmark": "sglang-graph-lora-b-paired-cuda-graph-v1",
        "replay_index": args.replay_index,
        "fresh_process": True,
        "baseline_source_sha256": _sha256(args.baseline),
        "candidate_source_sha256": _sha256(args.candidate),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "samples_per_case": args.samples,
        "iterations_per_sample": args.iterations,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
