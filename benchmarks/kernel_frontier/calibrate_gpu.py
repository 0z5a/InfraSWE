from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
from bench_utils import (
    atomic_write_json,
    device_timer_name,
    event_latency_us,
    hardware_manifest,
    utc_now,
)


def samples(function, *, repetitions: int, count: int = 30) -> list[float]:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    return [event_latency_us(function, repetitions) for _ in range(count)]


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "cv": statistics.stdev(values) / statistics.fmean(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, required=True)
    parser.add_argument("--samples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP accelerator API is not available")
    torch.manual_seed(31_000 + args.replay_index)
    torch.cuda.manual_seed_all(31_000 + args.replay_index)
    torch.backends.cuda.matmul.allow_tf32 = True

    launch_tensor = torch.ones(1, device="cuda", dtype=torch.float32)

    def launch():
        return launch_tensor.add_(1.0)

    launch_samples = samples(launch, repetitions=10_000, count=args.samples)

    transfer_elements = 128 * 1024 * 1024
    source = torch.empty(transfer_elements, device="cuda", dtype=torch.uint8).random_()
    destination = torch.empty_like(source)

    def memory_copy():
        return destination.copy_(source)

    memory_samples = samples(memory_copy, repetitions=16, count=args.samples)
    bytes_per_copy = transfer_elements * 2
    bandwidth_samples = [bytes_per_copy / (latency * 1e-6) / 1e9 for latency in memory_samples]

    dimension = 8192
    left = torch.randn((dimension, dimension), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((dimension, dimension), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(left)

    def matrix_multiply():
        return torch.mm(left, right, out=output)

    compute_samples = samples(matrix_multiply, repetitions=4, count=args.samples)
    semantic_flops = 2 * dimension**3
    tflops_samples = [semantic_flops / (latency * 1e-6) / 1e12 for latency in compute_samples]

    payload = {
        "schema_version": "0.3",
        "calibration_id": "bf16-gemm-hbm-copy-launch-v1",
        "replay_index": args.replay_index,
        "generated_at": utc_now(),
        "hardware": hardware_manifest(),
        "protocol": {
            "device_timer": device_timer_name(),
            "sample_count": args.samples,
            "launch_repetitions": 10_000,
            "copy_repetitions": 16,
            "gemm_repetitions": 4,
            "copy_bytes_semantic": bytes_per_copy,
            "gemm_shape": [dimension, dimension, dimension],
            "gemm_flops_semantic": semantic_flops,
        },
        "launch_floor_us": summarize(launch_samples),
        "hbm_bandwidth_gbps": summarize(bandwidth_samples),
        "bf16_matmul_tflops": summarize(tflops_samples),
        "raw": {
            "launch_latency_us": launch_samples,
            "copy_latency_us": memory_samples,
            "hbm_bandwidth_gbps": bandwidth_samples,
            "gemm_latency_us": compute_samples,
            "bf16_matmul_tflops": tflops_samples,
        },
        "anchor_kind": "calibrated-target",
        "confidence": "medium",
        "known_omissions": [
            "copy calibration is a device-to-device streaming proxy",
            "GEMM calibration is one large square shape",
            "calibrated target is not a physical lower bound",
        ],
    }
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
