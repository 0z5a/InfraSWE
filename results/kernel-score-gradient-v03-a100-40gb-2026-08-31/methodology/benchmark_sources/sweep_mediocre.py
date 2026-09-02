from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import garbage_kernels
import torch
from attention_bench import CASES, make_qkv, reference_prepare, work_model
from bench_utils import atomic_write_json, event_latency_us, tensor_correctness, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--passes",
        type=int,
        nargs="+",
        default=(0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128),
    )
    parser.add_argument("--samples", type=int, default=7)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_calibration(root: Path) -> dict[str, float]:
    payloads = [
        json.loads(path.read_text())
        for path in sorted((root / "raw/calibration").glob("*.json"))
    ]
    if len(payloads) != 3:
        raise ValueError("pilot requires exactly three calibration replays")
    return {
        "launch_floor_us": statistics.median(
            float(item["launch_floor_us"]["median"]) for item in payloads
        ),
        "hbm_bandwidth_gbps": statistics.median(
            float(item["hbm_bandwidth_gbps"]["median"]) for item in payloads
        ),
        "bf16_matmul_tflops": statistics.median(
            float(item["bf16_matmul_tflops"]["median"]) for item in payloads
        ),
    }


def load_baselines(root: Path) -> dict[str, float]:
    paths = sorted((root / "raw/negative/garbage-slow-fa4-waste64").glob("*.json"))
    if len(paths) != 3:
        raise ValueError("pilot requires the three completed waste64 replays")
    values: dict[str, list[float]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for case in payload["cases"]:
            values.setdefault(case["case_id"], []).extend(
                float(block["reference_latency_us"]) for block in case["measurement"]["blocks"]
            )
    return {case_id: statistics.median(samples) for case_id, samples in values.items()}


def anchor_us(model: dict[str, Any], calibration: dict[str, float]) -> float:
    launch = calibration["launch_floor_us"]
    bandwidth = (
        float(model["minimum_external_bytes"])
        / (calibration["hbm_bandwidth_gbps"] * 1e9)
        * 1e6
    )
    compute = (
        float(model["semantic_flops"])
        / (calibration["bf16_matmul_tflops"] * 1e12)
        * 1e6
    )
    return max(launch, bandwidth, compute)


def estimate_artifact(cases: list[dict[str, Any]]) -> tuple[float, float, float]:
    portfolio = sum(float(case["weight"]) * float(case["anchor_score"]) for case in cases)
    public = [case for case in cases if case["group"] == "common"]
    hidden = [case for case in cases if case["group"] != "common"]

    def normalized(items: list[dict[str, Any]]) -> float:
        total = sum(float(item["weight"]) for item in items)
        return sum(float(item["weight"]) * float(item["anchor_score"]) for item in items) / total

    public_score = normalized(public)
    hidden_score = normalized(hidden)
    hidden_values = [float(case["anchor_score"]) for case in hidden]
    retention = min(1.0, hidden_score / max(public_score, 1e-12))
    tail = min(1.0, percentile(hidden_values, 0.10) / statistics.median(hidden_values))
    generalization = 0.50 * retention + 0.30 * tail + 0.20
    return 100 * (0.80 * portfolio + 0.20 * generalization), portfolio, generalization


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.evidence_root)
    baselines = load_baselines(args.evidence_root)
    variants = []
    for passes in args.passes:
        case_results = []
        for case_index, case in enumerate(CASES):
            q, k, v = make_qkv(case, 210_000 + passes * 100 + case_index)
            reference = reference_prepare(q, k, v, case["causal"])
            candidate = garbage_kernels.make_fa4_waste_prepare(passes)(q, k, v, case["causal"])
            for _ in range(5):
                candidate()
            torch.cuda.synchronize()
            samples = [event_latency_us(candidate, 1) for _ in range(args.samples)]
            reference_output = reference()
            candidate_output = candidate()
            torch.cuda.synchronize()
            correctness = tensor_correctness(reference_output, candidate_output)
            passed = (
                correctness["max_abs_error"] <= 0.05
                and correctness["relative_l2_error"] <= 0.03
                and correctness["cosine_similarity"] >= 0.999
            )
            candidate_us = statistics.median(samples)
            baseline_us = baselines[case["id"]]
            anchor = anchor_us(work_model(case), calibration)
            score = (baseline_us - anchor) / (
                (candidate_us - anchor) + (baseline_us - anchor)
            )
            case_results.append(
                {
                    "case_id": case["id"],
                    "group": case["group"],
                    "weight": case["weight"],
                    "candidate_latency_us": candidate_us,
                    "candidate_samples_us": samples,
                    "baseline_latency_us": baseline_us,
                    "anchor_latency_us": anchor,
                    "anchor_score": score,
                    "correctness_passed": passed,
                    "correctness": correctness,
                }
            )
        artifact, performance, generalization = estimate_artifact(case_results)
        variants.append(
            {
                "passes": passes,
                "backend": f"mediocre-fa4-waste{passes}",
                "estimated_artifact_100": artifact,
                "performance_component": performance,
                "generalization_component": generalization,
                "all_correct": all(case["correctness_passed"] for case in case_results),
                "cases": case_results,
            }
        )
    atomic_write_json(
        args.output,
        {
            "schema_version": "0.3",
            "evidence_kind": "controlled-degradation-pilot",
            "generated_at": utc_now(),
            "calibration": calibration,
            "baseline_source": "completed garbage-slow-fa4-waste64 formal replays",
            "sample_count": args.samples,
            "variants": variants,
        },
    )


if __name__ == "__main__":
    main()
