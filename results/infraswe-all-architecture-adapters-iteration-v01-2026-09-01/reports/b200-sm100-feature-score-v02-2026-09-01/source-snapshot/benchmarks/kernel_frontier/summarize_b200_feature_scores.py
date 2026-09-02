# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FEATURE_IDS = (
    "BW-TMEM-001",
    "BW-CLC-001",
    "BW-TMA-001",
    "BW-TMEM-003",
    "BW-TMA-002",
)
CORE_WEIGHTS = {
    "BW-TMEM-001": 0.40,
    "BW-TMA-001": 0.20,
    "BW-TMEM-003": 0.25,
    "BW-TMA-002": 0.15,
}
PERFORMANCE_WEIGHTS = {"C": 0.35, "N": 0.25, "P": 0.35, "R": 0.03, "B": 0.02}
BUGFIX_WEIGHTS = {"C": 0.55, "N": 0.15, "P": 0.10, "R": 0.15, "B": 0.05}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def clamp(value: float, low: float = 0.0, high: float = 1.2) -> float:
    return min(high, max(low, value))


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take median of an empty sequence")
    return float(statistics.median(values))


def load_replays(root: Path) -> list[dict[str, Any]]:
    replays = [
        json.loads((root / f"replay-{index}.json").read_text(encoding="utf-8"))
        for index in (1, 2, 3)
    ]
    if [int(item["replay_index"]) for item in replays] != [1, 2, 3]:
        raise ValueError("B200 score replay set must be exactly fresh processes 1, 2, 3")
    if any(item.get("suite_id") != "b200-sm100-feature-score-v0.2" for item in replays):
        raise ValueError("unexpected B200 score replay suite id")
    return replays


def correctness_fraction(feature_id: str, feature: dict[str, Any]) -> tuple[int, int]:
    correctness = feature["correctness"]
    if feature_id == "BW-TMA-001":
        total = int(correctness.get("case_count", 0))
        passed = total if correctness.get("gather4") and correctness.get("scatter4") else 0
        return passed, total
    return int(correctness.get("passed_cases", 0)), int(correctness.get("total_cases", 0))


def extract_pairs(feature_id: str, feature: dict[str, Any]) -> list[tuple[float, float, str]]:
    performance = feature["performance"]
    if feature_id == "BW-TMEM-001":
        return [
            (
                float(performance["candidate"]["latency_us_median"]),
                float(performance["portable"]["latency_us_median"]),
                "4096^3 TCGen05/TMEM vs torch.bmm",
            )
        ]
    if feature_id == "BW-TMEM-003":
        return [
            (
                float(performance["candidate_stress"]["latency_us_median"]),
                float(performance["control"]["latency_us_median"]),
                "3000-launch lifecycle stress vs short control",
            )
        ]
    if feature_id == "BW-CLC-001":
        return [
            (
                float(pair["candidate"]["latency_us_median"]),
                float(pair["baseline"]["latency_us_median"]),
                "x".join(str(value) for value in pair["shape"][:3]),
            )
            for pair in performance["makespan_pairs"]
        ]
    if feature_id == "BW-TMA-001":
        return [
            (
                float(performance["gather4_us"]),
                float(performance["scalar_gather4_us"]),
                "gather4 vs scalar gather",
            ),
            (
                float(performance["scatter4_us"]),
                float(performance["scalar_scatter4_us"]),
                "scatter4 vs scalar scatter",
            ),
        ]
    if feature_id == "BW-TMA-002":
        return [
            (
                float(performance["candidate"]["latency_us_median"]),
                float(performance["baseline"]["latency_us_median"]),
                "4096^3 2-CTA vs 1-CTA TMA",
            )
        ]
    raise ValueError(f"unknown feature: {feature_id}")


def latency_anchor_score(
    pairs: list[tuple[float, float, str]], *, allow_no_regression: bool = False
) -> dict[str, Any]:
    candidates = [candidate for candidate, _, _ in pairs]
    baselines = [baseline for _, baseline, _ in pairs]
    candidate = median(candidates)
    portable = median(baselines)
    native_reference = min(candidates)
    if portable > native_reference and portable > 0 and candidate > 0:
        denominator = math.log(native_reference) - math.log(portable)
        raw = (
            (math.log(candidate) - math.log(portable)) / denominator
            if abs(denominator) > 1e-12
            else 1.0
        )
        method = "RFC latency log interpolation: portable baseline to best valid native replay"
    elif allow_no_regression and candidate > 0:
        raw = portable / candidate
        method = "bug-fix no-regression ratio because the native/portable anchors do not separate"
    else:
        raw = 0.0
        method = "zero: candidate did not beat the portable baseline"
    score = clamp(raw)
    return {
        "score": score,
        "raw_score": raw,
        "method": method,
        "candidate_latency_us": candidate,
        "portable_latency_us": portable,
        "native_reference_latency_us": native_reference,
        "candidate_over_portable": candidate / portable,
        "speedup_over_portable": portable / candidate,
        "pairs": [
            {"candidate_us": candidate, "baseline_us": baseline, "label": label}
            for candidate, baseline, label in pairs
        ],
    }


def robustness_score(
    feature_id: str, features: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    replay_pass = sum(feature["status"] == "passed" for feature in features) / len(features)
    if feature_id == "BW-TMEM-003":
        rejection = sum(
            feature["correctness"]["invalid_alignment_rejection"]["passed"]
            for feature in features
        ) / len(features)
        liveness = sum(feature["liveness"]["passed"] for feature in features) / len(features)
        memory_stable = sum(
            feature["liveness"]["memory_free_after_bytes"]
            >= feature["liveness"]["memory_free_before_bytes"] - 256 * 2**20
            for feature in features
        ) / len(features)
        score = 0.30 * replay_pass + 0.30 * rejection + 0.30 * liveness + 0.10 * memory_stable
        return score, {
            "fresh_process_pass_fraction": replay_pass,
            "invalid_input_rejection_fraction": rejection,
            "lifecycle_watchdog_fraction": liveness,
            "memory_stability_fraction": memory_stable,
        }
    profiler = sum(feature.get("profiler", {}).get("captured", False) for feature in features)
    profiler /= len(features)
    if feature_id == "BW-CLC-001":
        workload_breadth = sum(
            len(feature["performance"]["makespan_pairs"]) >= 2 for feature in features
        ) / len(features)
        score = 0.50 * replay_pass + 0.25 * profiler + 0.25 * workload_breadth
        return score, {
            "fresh_process_pass_fraction": replay_pass,
            "profiler_capture_fraction": profiler,
            "uniform_and_tail_workload_fraction": workload_breadth,
        }
    if feature_id == "BW-TMA-001":
        case_breadth = sum(
            int(feature["correctness"].get("case_count", 0)) >= 5 for feature in features
        ) / len(features)
        score = 0.65 * replay_pass + 0.35 * case_breadth
        return score, {
            "fresh_process_pass_fraction": replay_pass,
            "five_case_irregular_row_fraction": case_breadth,
        }
    case_breadth = sum(
        int(feature["correctness"].get("total_cases", 0)) >= 2 for feature in features
    ) / len(features)
    score = 0.50 * replay_pass + 0.25 * profiler + 0.25 * case_breadth
    return score, {
        "fresh_process_pass_fraction": replay_pass,
        "profiler_capture_fraction": profiler,
        "hidden_case_breadth_fraction": case_breadth,
    }


def build_score(features: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    per_replay = []
    for feature in features:
        native = feature["native"]
        passed = bool(
            native.get("ptx_files")
            and native.get("cubin_files")
            and native.get("ptx_size_bytes", 0) > 0
            and native.get("binary_size_bytes", 0) > 0
            and all(item.get("returncode") == 0 for item in native.get("disassembly", []))
        )
        per_replay.append(passed)
    score = sum(per_replay) / len(per_replay)
    return score, {"artifact_and_disassembly_passed_per_replay": per_replay}


def task_score(feature_id: str, replays: list[dict[str, Any]]) -> dict[str, Any]:
    features = [replay["features"][feature_id] for replay in replays]
    hard_gate = all(feature["status"] == "passed" for feature in features)
    hard_gate = hard_gate and all(feature["native"]["passed"] for feature in features)
    passed_cases = 0
    total_cases = 0
    for feature in features:
        passed, total = correctness_fraction(feature_id, feature)
        passed_cases += passed
        total_cases += total
    correctness = passed_cases / total_cases if total_cases else 0.0
    native = sum(feature["native"]["passed"] for feature in features) / len(features)
    pairs = [pair for feature in features for pair in extract_pairs(feature_id, feature)]
    performance = latency_anchor_score(
        pairs,
        allow_no_regression=feature_id == "BW-TMEM-003",
    )
    robustness, robustness_evidence = robustness_score(feature_id, features)
    build, build_evidence = build_score(features)
    components = {
        "C": correctness,
        "N": native,
        "P": performance["score"],
        "R": robustness,
        "B": build,
    }
    weights = BUGFIX_WEIGHTS if feature_id == "BW-TMEM-003" else PERFORMANCE_WEIGHTS
    weighted = sum(weights[name] * value for name, value in components.items())
    gated = weighted if hard_gate else 0.0
    return {
        "feature_id": feature_id,
        "status": "scored" if hard_gate else "hard_gate_failed",
        "hard_gate": hard_gate,
        "score_100": 100.0 * min(gated, 1.0),
        "uncapped_score_100": 100.0 * gated,
        "components": components,
        "weights": weights,
        "correctness_evidence": {
            "passed_cases": passed_cases,
            "total_cases": total_cases,
        },
        "performance_evidence": performance,
        "robustness_evidence": robustness_evidence,
        "build_evidence": build_evidence,
        "replay_statuses": [feature["status"] for feature in features],
    }


def weighted_geometric_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    epsilon = 1e-9
    total_weight = sum(weights.values())
    return math.exp(
        sum(weights[key] * math.log(max(scores[key], epsilon)) for key in weights)
        / total_weight
    )


def build_summary(replays: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {feature_id: task_score(feature_id, replays) for feature_id in FEATURE_IDS}
    core_task_scores = {
        feature_id: tasks[feature_id]["score_100"] / 100.0 for feature_id in CORE_WEIGHTS
    }
    core_score = 100.0 * weighted_geometric_mean(core_task_scores, CORE_WEIGHTS)
    clc = tasks["BW-CLC-001"]
    scheduler_score = (
        0.70 * clc["score_100"] + 0.30 * 100.0 * clc["components"]["R"]
        if clc["hard_gate"]
        else 0.0
    )
    all_passed = all(task["hard_gate"] for task in tasks.values())
    return {
        "schema_version": "0.2",
        "generated_at": utc_now(),
        "suite_id": "b200-sm100-feature-score-v0.2",
        "status": "scored" if all_passed else "hard_gate_failure",
        "hardware": replays[0]["hardware"],
        "replay_count": 3,
        "tasks": tasks,
        "score_namespaces": {
            "SM100-Core": {
                "status": (
                    "scored"
                    if all(tasks[key]["hard_gate"] for key in CORE_WEIGHTS)
                    else "failed"
                ),
                "score_100": core_score,
                "aggregation": "weighted geometric mean",
                "phase1_task_weights": CORE_WEIGHTS,
            },
            "SM100-Scheduler": {
                "status": "scored" if clc["hard_gate"] else "failed",
                "score_100": scheduler_score,
                "aggregation": "70% CLC task score + 30% measured robustness/liveness",
            },
            "SM100-Fabric": {
                "status": "not_applicable",
                "score_100": None,
                "reason": "single visible B200 has no evaluator-owned multi-GPU multicast mapping",
            },
            "PTX-Preview": {
                "status": "disabled",
                "score_100": None,
                "reason": "stable lane is CUDA 13.3 / PTX 9.3",
            },
        },
        "methodology": {
            "task_formula_performance_migration": PERFORMANCE_WEIGHTS,
            "task_formula_bugfix": BUGFIX_WEIGHTS,
            "performance_normalization": (
                "RFC logarithmic latency interpolation from portable baseline to the best "
                "valid native replay; BW-TMEM-003 uses an explicit no-regression ratio when "
                "anchors do not separate"
            ),
            "public_scores_capped_at_100": True,
            "uncapped_task_scores_retained": True,
        },
        "raw_replays": [f"replay-{index}.json" for index in (1, 2, 3)],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    hardware = summary["hardware"]
    lines = [
        "# InfraSWE B200 / SM100 Phase-1 特性测试与跑分",
        "",
        f"状态：`{summary['status']}`；3 次 fresh-process replay。",
        f"GPU：`{hardware['gpu_name']}`；CC `{hardware['compute_capability']}`；"
        f"SM `{hardware['sm_count']}`；CUDA runtime `{hardware['cuda_runtime']}`。",
        "",
        "## 总分",
        "",
        "| 命名空间 | 分数 | 状态 |",
        "|---|---:|---|",
    ]
    for namespace, result in summary["score_namespaces"].items():
        score = result["score_100"]
        rendered = "N/A" if score is None else f"{score:.2f}"
        lines.append(f"| {namespace} | {rendered} | {result['status']} |")
    lines.extend(
        [
            "",
            "## Phase-1 五任务",
            "",
            "| 任务 | 得分 | C | N | P | R | B | 性能 candidate / baseline |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for feature_id in FEATURE_IDS:
        task = summary["tasks"][feature_id]
        components = task["components"]
        ratio = task["performance_evidence"]["candidate_over_portable"]
        lines.append(
            f"| {feature_id} | {task['score_100']:.2f} | "
            f"{components['C']:.3f} | {components['N']:.3f} | "
            f"{components['P']:.3f} | {components['R']:.3f} | "
            f"{components['B']:.3f} | {ratio:.4f}× |"
        )
    lines.extend(
        [
            "",
            "## 性能摘要",
            "",
        ]
    )
    for feature_id in FEATURE_IDS:
        task = summary["tasks"][feature_id]
        performance = task["performance_evidence"]
        lines.append(
            f"- `{feature_id}`：candidate `{performance['candidate_latency_us']:.3f} µs`，"
            f"baseline `{performance['portable_latency_us']:.3f} µs`，"
            f"speedup `{performance['speedup_over_portable']:.3f}×`。"
        )
    lines.extend(
        [
            "",
            "## 证据与边界",
            "",
            "- 五个任务均要求三轮正确性、watchdog/liveness 与 PTX+cubin+SASS 原生门禁同时通过；"
            "任一 hard gate 失败，该任务为 0 分。",
            "- `BW-TMEM-001` 覆盖 aligned、M/N/K tail 与非默认 leading-dimension；"
            "`BW-TMEM-003` 额外执行数千次 launch 以及非法对齐显式拒绝。",
            "- `BW-CLC-001` 比较 dynamic CLC 与 static persistent 的 uniform/tail makespan；"
            "原生证据必须同时出现 `clusterlaunchcontrol.*` 与 `UGETNEXTWORKID`。",
            "- `BW-TMA-001` 运行连续、离散、重复、逆序和边界 row case；"
            "`BW-TMA-002` 比较 2-CTA 与 1-CTA pipeline。",
            "- Fabric 在当前单卡 lease 中为 N/A，不以 0 分惩罚；PTX Preview 保持 disabled。",
            "- 分数遵循 RFC 的 C/N/P/R/B 结构；性能使用 latency 的对数 anchor 插值。",
            "",
            "参考：https://docs.nvidia.com/cuda/parallel-thread-execution/",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize three B200 feature-score replays")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--require-passed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(load_replays(args.root))
    atomic_write_json(args.json_output, summary)
    args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "scores": summary["score_namespaces"]}))
    if args.require_passed and summary["status"] != "scored":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
