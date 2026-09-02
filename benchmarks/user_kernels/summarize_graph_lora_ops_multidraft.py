#!/usr/bin/env python3
"""Aggregate seven graph-LoRA multi-Draft fresh-process replays."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(statistics.mean(math.log(value) for value in values))


def _correctness_passed(result: dict[str, Any]) -> bool:
    phases = ("eager", "cuda_graph_first_replay", "cuda_graph_dynamic_replay")
    return bool(
        all(result[phase]["passed"] and result[phase]["all_finite"] for phase in phases)
        and result["eager"]["base_output_alias"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("replay-*.json"))
    if len(paths) != 7:
        raise SystemExit(f"expected 7 replays, found {len(paths)}")
    runs = [_read(path) for path in paths]
    if {run["replay_index"] for run in runs} != set(range(1, 8)):
        raise SystemExit("replay identity drift")
    if {run["benchmark"] for run in runs} != {"sglang-graph-lora-multidraft-v1"}:
        raise SystemExit("benchmark identity drift")
    if len({json.dumps(run["draft_profiles"], sort_keys=True) for run in runs}) != 1:
        raise SystemExit("Draft profile drift")
    if len({json.dumps(run["case_plan"], sort_keys=True) for run in runs}) != 1:
        raise SystemExit("case plan drift")
    if len({json.dumps(run["environment"], sort_keys=True) for run in runs}) != 1:
        raise SystemExit("environment drift")
    if any(run["samples_per_case"] != 5 for run in runs):
        raise SystemExit("sample count drift")
    if any(run["iterations_per_sample"] != 300 for run in runs):
        raise SystemExit("iteration count drift")

    case_plan = runs[0]["case_plan"]
    case_results = []
    for case_index, planned in enumerate(case_plan):
        process_ratios = []
        pooled_ratios = []
        for run in runs:
            result = run["performance"][case_index]
            ratios = [float(pair["candidate_to_baseline"]) for pair in result["pairs"]]
            if len(ratios) != 5:
                raise SystemExit("paired sample count drift")
            process_ratios.append(statistics.median(ratios))
            pooled_ratios.extend(ratios)
        ratio = statistics.median(process_ratios)
        baseline_ok = all(
            _correctness_passed(run["correctness"]["baseline"][case_index]) for run in runs
        )
        candidate_ok = all(
            _correctness_passed(run["correctness"]["candidate"][case_index]) for run in runs
        )
        case_results.append(
            {
                **planned,
                "baseline_correctness_passed": baseline_ok,
                "candidate_correctness_passed": candidate_ok,
                "process_median_ratios": process_ratios,
                "candidate_to_baseline_ratio": ratio,
                "pooled_ratio_median": statistics.median(pooled_ratios),
                "speedup_x": 1.0 / ratio,
                "latency_reduction_percent": 100 * (1.0 - ratio),
                "three_percent_retention_gate_passed": ratio <= 1.03,
            }
        )

    contract_results = {}
    for variant in ("baseline", "candidate"):
        names = [item["name"] for item in runs[0]["contracts"][variant]]
        contract_results[variant] = [
            {
                "name": name,
                "passed": all(
                    run["contracts"][variant][index]["name"] == name
                    and run["contracts"][variant][index]["passed"]
                    for run in runs
                ),
            }
            for index, name in enumerate(names)
        ]

    profile_results = []
    for profile in runs[0]["draft_profiles"]:
        cases = [result for result in case_results if result["profile_id"] == profile["profile_id"]]
        profile_results.append(
            {
                **profile,
                "case_count": len(cases),
                "all_correctness_passed": all(
                    case["baseline_correctness_passed"] and case["candidate_correctness_passed"]
                    for case in cases
                ),
                "all_retention_gates_passed": all(
                    case["three_percent_retention_gate_passed"] for case in cases
                ),
                "geomean_speedup_x": _geomean([case["speedup_x"] for case in cases]),
                "minimum_speedup_x": min(case["speedup_x"] for case in cases),
                "maximum_speedup_x": max(case["speedup_x"] for case in cases),
            }
        )

    candidate_contracts_passed = all(item["passed"] for item in contract_results["candidate"])
    overall_passed = bool(
        candidate_contracts_passed
        and all(profile["all_correctness_passed"] for profile in profile_results)
        and all(profile["all_retention_gates_passed"] for profile in profile_results)
    )
    payload = {
        "schema_version": "0.5",
        "benchmark": "sglang-graph-lora-multidraft-v1-summary",
        "authority": (
            "SGLang is the native target; other Draft profiles are contract proxies and do not "
            "constitute native cross-project ProjectFit scores."
        ),
        "environment": runs[0]["environment"],
        "baseline_source_sha256": runs[0]["baseline_source_sha256"],
        "candidate_source_sha256": runs[0]["candidate_source_sha256"],
        "fresh_process_replays": len(runs),
        "total_cases": len(case_results),
        "paired_samples_per_case": 35,
        "contract_results": contract_results,
        "profile_results": profile_results,
        "case_results": case_results,
        "overall_passed": overall_passed,
        "raw_evidence": [str(path) for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
