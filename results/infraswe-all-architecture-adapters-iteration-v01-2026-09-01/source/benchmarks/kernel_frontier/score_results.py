# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import sys
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from infraswe.io import atomic_write_json  # noqa: E402
from infraswe.kernel.results import assemble_suite, load_json_evidence  # noqa: E402


def _number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _hardware_description(hardware: dict[str, Any]) -> str:
    vendor = str(hardware.get("accelerator_vendor") or "nvidia").upper()
    architecture = hardware.get("architecture")
    if not architecture and hardware.get("compute_capability"):
        architecture = "sm" + str(hardware["compute_capability"]).replace(".", "")
    unit_label = "CU" if vendor == "AMD" else "SM"
    unit_count = hardware.get("compute_unit_count", hardware.get("sm_count"))
    runtime = str(hardware.get("runtime") or "cuda").upper()
    runtime_version = hardware.get("runtime_version") or hardware.get("torch_cuda")
    return (
        f"GPU：{hardware.get('gpu_name')}；Vendor {vendor}；Arch {architecture}；"
        f"{unit_label} {unit_count}；显存 {int(hardware.get('total_memory_bytes') or 0) / 2**30:.1f} GiB；"
        f"PyTorch {hardware.get('torch_version')} / {runtime} {runtime_version}；"
        f"Driver {hardware.get('driver_version')}。"
    )


def markdown_report(score: dict[str, Any]) -> str:
    lines = [
        "# InfraSWE Kernel Frontier v0.3 — FA1–FA4、AOTriton 与经典 Kernel 评分",
        "",
        f"生成时间：`{score['generated_at']}`  ",
        f"Suite：`{score['suite_id']}`  ",
        f"公式：`{score['formula_version']}`；AnchorScore 来源：`{score['formula_origin']}`",
        "",
        "评分使用三次独立进程 replay、每 case 30 个 matched ABBA/BAAB blocks、"
        "evaluator-owned CUDA/HIP device events 与 block/replay 两层 bootstrap CI95。"
        "FA 库分数为 `100 × (0.80P + 0.20G)`；经典 micro kernel 各自为 `100 × AnchorScore`。",
        "同一行只在同一硬件 cell 内可比较，不同 GPU/架构 cell 的分数不得直接混排。",
        "",
        "实现与公式来源：",
        "",
        "- FA1 固定为 Dao-AILab/flash-attention "
        "[`6d48e14`](https://github.com/Dao-AILab/flash-attention/commit/6d48e14a6c2f551db96f0badc658a6279a929df3)（v1.0.9）。",
        "- FA2/FA3 固定为 Dao-AILab/flash-attention "
        "[`ce088ab`](https://github.com/Dao-AILab/flash-attention/commit/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820)。",
        "- FA4 固定为 PyPI "
        "[`flash-attn-4==4.0.0b28`](https://pypi.org/project/flash-attn-4/)。",
        "- ROCm attention 路径固定为 PyTorch Flash SDPA，并以其内置 AOTriton binary hash 与原生 trace 闭环。",
        "- AnchorScore 采用 "
        "[SOL-ExecBench](https://arxiv.org/abs/2603.19173) 等价公式。",
        "",
    ]
    for cell in score["cells"]:
        hardware = cell["hardware"]
        lines.extend(
            [
                f"## Cell: {cell['cell_id']}",
                "",
                _hardware_description(hardware),
                "",
            ]
        )
        calibration = cell.get("calibration")
        if calibration:
            lines.append(
                "校准中位数：launch floor "
                f"{calibration['launch_floor_us']:.4f} µs；HBM proxy "
                f"{calibration['hbm_bandwidth_gbps']:.1f} GB/s；BF16 GEMM "
                f"{calibration['bf16_matmul_tflops']:.1f} TFLOP/s。"
            )
            lines.append("")
        else:
            lines.extend([f"校准无效：{cell.get('calibration_error')}", ""])

        lines.extend(
            [
                "### 实现支持矩阵",
                "",
                "| 实现 | 状态 | 原因 |",
                "|---|---|---|",
            ]
        )
        for name, eligibility in cell["eligibility"].items():
            lines.append(f"| {name.upper()} | {eligibility['status']} | {eligibility['reason']} |")
        lines.extend(["", "### Attention / FA 评分", ""])
        libraries = [
            candidate
            for candidate in cell["candidates"]
            if candidate.get("benchmark_kind") == "kernel-library"
        ]
        if libraries:
            lines.extend(
                [
                    "| Backend | 版本 | KernelCert | 状态 | Artifact-100 | P | G | 几何加权 speedup |",
                    "|---|---:|---:|---|---:|---:|---:|---:|",
                ]
            )
            for candidate in libraries:
                components = candidate.get("components", {})
                lines.append(
                    f"| {candidate['backend']} | {candidate.get('backend_version') or '—'} | "
                    f"{_status(candidate['certified'])} | {candidate['artifact_status']} | "
                    f"{_number(candidate.get('artifact_100'), 2)} | "
                    f"{_number(components.get('performance_anchor_score'))} | "
                    f"{_number(components.get('generalization'))} | "
                    f"{_number(candidate.get('weighted_geometric_speedup_raw'))}× |"
                )
            lines.append("")
            for candidate in libraries:
                lines.extend(
                    [
                        f"#### {candidate['backend']} per-shape",
                        "",
                        "| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |",
                        "|---|---|---:|---:|---:|---:|---:|---|",
                    ]
                )
                for case in candidate.get("case_results", []):
                    anchor_result = case["anchor_result"]
                    ci = case["paired_speedup_ci95"]
                    lines.append(
                        f"| {case['case_id']} | {case['case_group']} | "
                        f"{case['candidate_latency_us']['median']:.3f} | "
                        f"{case['reference_latency_us']['median']:.3f} | "
                        f"{case['paired_speedup']:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}] | "
                        f"{case['anchor']['latency_us']:.3f} | "
                        f"{_number(anchor_result['anchor_score_raw'])} | {anchor_result['status']} |"
                    )
                if candidate.get("failure_codes"):
                    lines.append("")
                    lines.append("Failure codes：`" + "`, `".join(candidate["failure_codes"]) + "`")
                lines.append("")
        else:
            lines.extend(["无 attention 正式 replay 数据。", ""])

        classics = [
            candidate
            for candidate in cell["candidates"]
            if candidate.get("benchmark_kind") == "kernel-micro"
        ]
        lines.extend(["### 经典 Kernel 独立评分", ""])
        if classics:
            lines.extend(
                [
                    "| Kernel | KernelCert | 状态 | Artifact-100 | Candidate µs | Baseline µs | Speedup [CI95] | AnchorScore |",
                    "|---|---:|---|---:|---:|---:|---:|---:|",
                ]
            )
            for candidate in classics:
                micro_scores = candidate.get("micro_artifact_100", {})
                for case in candidate.get("case_results", []):
                    ci = case["paired_speedup_ci95"]
                    lines.append(
                        f"| {case['case_id']} | {_status(candidate['certified'])} | "
                        f"{case['anchor_result']['status']} | "
                        f"{_number(micro_scores.get(case['case_id']), 2)} | "
                        f"{case['candidate_latency_us']['median']:.3f} | "
                        f"{case['reference_latency_us']['median']:.3f} | "
                        f"{case['paired_speedup']:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}] | "
                        f"{_number(case['anchor_result']['anchor_score_raw'])} |"
                    )
            lines.append("")
        else:
            lines.extend(["无经典 kernel 正式 replay 数据。", ""])

    lines.extend(
        [
            "## 判读边界",
            "",
            "- functional reference 是 PyTorch math/eager；它同时作为本次 scoring baseline，不代表最佳厂商实现。",
            "- anchor 为同机 launch/HBM/BF16 GEMM 校准目标，confidence=medium，不宣称物理下界。",
            "- raw speedup、CI、AnchorScore 均保留未裁剪值；超越 anchor tolerance 的结果进入 quarantine。",
            "- FA4 按固定 b28 包的显式架构 dispatch 判定资格；纳入的硬件 cell 均需提供"
            "实测原生路径证据，未注册的架构记为 unresolved，不用其他实现代填。",
            "- ROCm AOTriton 与经典 Triton 只在精确 PyTorch/ROCm/gfx cell 内判定资格；"
            "适配可安装不等于 KernelCert，仍需三 replay、正确性和原生 trace。",
            "- 所有原始 JSON、实现摘要、报告与 SHA-256 清单均包含在 ZIP 中。",
            "",
        ]
    )
    return "\n".join(lines)


def html_report(score: dict[str, Any]) -> str:
    def table(headers: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    sections = []
    for cell in score["cells"]:
        hardware = cell["hardware"]
        calibration = cell.get("calibration") or {}
        support_rows = [
            [name.upper(), item["status"], item["reason"]]
            for name, item in cell["eligibility"].items()
        ]
        library_rows = []
        detail = []
        classic_rows = []
        for candidate in cell["candidates"]:
            if candidate.get("benchmark_kind") == "kernel-library":
                components = candidate.get("components", {})
                library_rows.append(
                    [
                        candidate["backend"],
                        str(candidate.get("backend_version") or "—"),
                        _status(candidate["certified"]),
                        candidate["artifact_status"],
                        _number(candidate.get("artifact_100"), 2),
                        _number(components.get("performance_anchor_score")),
                        _number(components.get("generalization")),
                        _number(candidate.get("weighted_geometric_speedup_raw")) + "×",
                    ]
                )
                rows = []
                for case in candidate.get("case_results", []):
                    ci = case["paired_speedup_ci95"]
                    rows.append(
                        [
                            case["case_id"],
                            case["case_group"],
                            _number(case["candidate_latency_us"]["median"]),
                            _number(case["reference_latency_us"]["median"]),
                            f"{case['paired_speedup']:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}]",
                            _number(case["anchor"]["latency_us"]),
                            _number(case["anchor_result"]["anchor_score_raw"]),
                            case["anchor_result"]["status"],
                        ]
                    )
                detail.append(
                    f"<h4>{html.escape(candidate['backend'])} per-shape</h4>"
                    + table(
                        [
                            "Case",
                            "Group",
                            "Candidate µs",
                            "Baseline µs",
                            "Speedup [CI95]",
                            "Anchor µs",
                            "AnchorScore",
                            "Status",
                        ],
                        rows,
                    )
                )
            elif candidate.get("benchmark_kind") == "kernel-micro":
                for case in candidate.get("case_results", []):
                    ci = case["paired_speedup_ci95"]
                    classic_rows.append(
                        [
                            case["case_id"],
                            _status(candidate["certified"]),
                            case["anchor_result"]["status"],
                            _number(
                                candidate.get("micro_artifact_100", {}).get(case["case_id"]), 2
                            ),
                            _number(case["candidate_latency_us"]["median"]),
                            _number(case["reference_latency_us"]["median"]),
                            f"{case['paired_speedup']:.3f}× [{ci[0]:.3f}, {ci[1]:.3f}]",
                            _number(case["anchor_result"]["anchor_score_raw"]),
                        ]
                    )
        calibration_text = (
            f"launch {calibration['launch_floor_us']:.4f} µs · HBM {calibration['hbm_bandwidth_gbps']:.1f} GB/s · "
            f"BF16 {calibration['bf16_matmul_tflops']:.1f} TFLOP/s"
            if calibration
            else html.escape(str(cell.get("calibration_error")))
        )
        sections.append(
            f"<section><h2>{html.escape(cell['cell_id'])}</h2>"
            f"<p>{html.escape(_hardware_description(hardware))}</p>"
            f"<p class='muted'>Calibration: {calibration_text}</p>"
            "<h3>Implementation support matrix</h3>"
            + table(["Implementation", "Status", "Reason"], support_rows)
            + "<h3>Attention / FA score</h3>"
            + (
                table(
                    ["Backend", "Version", "Cert", "Status", "Artifact-100", "P", "G", "Speedup"],
                    library_rows,
                )
                if library_rows
                else "<p>No formal attention replay data.</p>"
            )
            + "".join(detail)
            + "<h3>Classic kernel score</h3>"
            + (
                table(
                    [
                        "Kernel",
                        "Cert",
                        "Status",
                        "Artifact-100",
                        "Candidate µs",
                        "Baseline µs",
                        "Speedup [CI95]",
                        "AnchorScore",
                    ],
                    classic_rows,
                )
                if classic_rows
                else "<p>No formal classic-kernel replay data.</p>"
            )
            + "</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InfraSWE Kernel Frontier v0.3</title><style>
:root{{--bg:#0a1020;--panel:#121a2c;--text:#eaf0ff;--muted:#9fb0d0;--line:#293754;--accent:#68d5ff}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#071020,#10172a);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui;padding:32px}}
main{{max-width:1280px;margin:auto}}h1{{font-size:34px}}h2{{color:var(--accent)}}section{{background:rgba(18,26,44,.94);border:1px solid var(--line);border-radius:14px;padding:24px;margin:22px 0;box-shadow:0 14px 40px #0005}}
.muted{{color:var(--muted)}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;margin:10px 0 24px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:#bfefff;background:#17233a;position:sticky;top:0}}code{{color:#bfefff}}
</style></head><body><main><h1>InfraSWE Kernel Frontier v0.3</h1>
<p class="muted">FA1–FA4, AOTriton and classic kernels · generated {html.escape(score["generated_at"])}</p>
<p>Three fresh-process replays, matched ABBA/BAAB blocks, hierarchical CI95. Scores across hardware cells are not directly rankable.</p>
<p class="muted">Sources: <a href="https://github.com/Dao-AILab/flash-attention">Dao-AILab FlashAttention</a> · <a href="https://pypi.org/project/flash-attn-4/">flash-attn-4 4.0.0b28</a> · <a href="https://arxiv.org/abs/2603.19173">SOL-ExecBench</a></p>
{"".join(sections)}
<section><h2>Interpretation boundary</h2><p>The calibrated target is an engineering anchor with medium confidence, not a physical lower bound. Raw speedup and AnchorScore are not silently clipped. Every raw JSON and digest is packaged with this report.</p></section>
</main></body></html>"""


NEGATIVE_CONTROL_EXPECTATIONS = {
    "garbage-slow-fa4-waste64": {
        "defect": "正确结果后追加 64 轮无用 Triton 显存流量",
        "expected": "KernelCert 通过，但 Artifact-100 明显受罚（<80）",
        "kind": "slow",
    },
    "garbage-zero-triton": {
        "defect": "原生 Triton kernel 始终写零",
        "expected": "correctness fail，leaderboard effective score=0",
        "kind": "correctness",
    },
    "garbage-cache-copy": {
        "defect": "按 shape 缓存第一次正确答案，之后只做 Triton copy",
        "expected": "动态输入探针 fail，leaderboard effective score=0",
        "kind": "correctness",
    },
    "fa-garbage-math-fallback": {
        "defect": "伪装成 FA backend，实际调用 PyTorch SDPA math",
        "expected": "原生 Flash trace 缺失，证据无效且不评分",
        "kind": "fallback",
    },
}


def negative_control_assessment(score: dict[str, Any]) -> dict[str, Any]:
    controls = []
    for cell in score["cells"]:
        candidates = {candidate["backend"]: candidate for candidate in cell["candidates"]}
        for backend, expectation in NEGATIVE_CONTROL_EXPECTATIONS.items():
            candidate = candidates.get(backend)
            if candidate is None:
                controls.append(
                    {
                        "cell_id": cell["cell_id"],
                        "backend": backend,
                        **expectation,
                        "expectation_passed": False,
                        "failure_codes": ["EVIDENCE_NEGATIVE_CONTROL_MISSING"],
                    }
                )
                continue
            failures = candidate.get("failure_codes", [])
            if expectation["kind"] == "slow":
                passed = (
                    candidate.get("certified") is True
                    and candidate.get("artifact_status") == "scored"
                    and candidate.get("artifact_100") is not None
                    and float(candidate["artifact_100"]) < 80.0
                )
            elif expectation["kind"] == "correctness":
                passed = (
                    candidate.get("certified") is False
                    and candidate.get("verdict") == "fail"
                    and candidate.get("leaderboard_effective_artifact_100") == 0.0
                    and "CORRECTNESS_MANDATORY_FAILED" in failures
                )
            else:
                passed = (
                    candidate.get("certified") is False
                    and candidate.get("disposition") == "invalid"
                    and "FALLBACK_NATIVE_TRACE_MISSING" in failures
                    and candidate.get("artifact_100") is None
                )
            controls.append(
                {
                    "cell_id": cell["cell_id"],
                    "backend": backend,
                    **expectation,
                    "expectation_passed": passed,
                    "certified": candidate.get("certified"),
                    "verdict": candidate.get("verdict"),
                    "disposition": candidate.get("disposition"),
                    "artifact_status": candidate.get("artifact_status"),
                    "artifact_100": candidate.get("artifact_100"),
                    "leaderboard_effective_artifact_100": candidate.get(
                        "leaderboard_effective_artifact_100"
                    ),
                    "failure_codes": failures,
                    "replay_count": candidate.get("replay_count"),
                    "supplemental_profile_count": candidate.get(
                        "supplemental_profile_count"
                    ),
                    "case_results": candidate.get("case_results", []),
                }
            )
    return {
        "schema_version": "0.3",
        "report_kind": "kernel-negative-controls",
        "score_generated_at": score["generated_at"],
        "hardware_cells": [
            {
                "cell_id": cell["cell_id"],
                "hardware": cell["hardware"],
                "calibration": cell.get("calibration"),
            }
            for cell in score["cells"]
        ],
        "all_expectations_passed": bool(controls)
        and all(control["expectation_passed"] for control in controls),
        "controls": controls,
    }


def negative_control_markdown(assessment: dict[str, Any]) -> str:
    overall = "PASS" if assessment["all_expectations_passed"] else "FAIL"
    lines = [
        "# InfraSWE v0.3 垃圾 Kernel 负控报告",
        "",
        f"负控总门禁：**{overall}**  ",
        f"评分生成时间：`{assessment['score_generated_at']}`",
        "",
        "这些候选仅用于验证评分器；它们与正式 FA1–FA4 榜单隔离。每个候选仍执行 3 次独立 replay、"
        "每 case 30 个 matched blocks，以及 5 个独立 profiler 进程。",
        "",
        "| Backend | 故意缺陷 | Cert | 处置 | Artifact-100 | Effective | 预期命中 | Failure codes |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for cell in assessment["hardware_cells"]:
        hardware = cell["hardware"]
        lines.insert(
            5,
            f"硬件 cell：`{cell['cell_id']}`；{hardware['gpu_name']}；CC "
            f"{hardware['compute_capability']}；显存 "
            f"{int(hardware['total_memory_bytes']) / 2**30:.1f} GiB；Driver "
            f"{hardware['driver_version']}。",
        )
    for control in assessment["controls"]:
        lines.append(
            f"| {control['backend']} | {control['defect']} | "
            f"{_status(bool(control.get('certified')))} | {control.get('disposition', '—')} | "
            f"{_number(control.get('artifact_100'), 2)} | "
            f"{_number(control.get('leaderboard_effective_artifact_100'), 2)} | "
            f"{'PASS' if control['expectation_passed'] else 'FAIL'} | "
            f"{', '.join(control.get('failure_codes', [])) or '—'} |"
        )
    lines.extend(["", "## 预期", ""])
    for control in assessment["controls"]:
        lines.append(f"- `{control['backend']}`：{control['expected']}。")
    lines.extend(
        [
            "",
            "## 逐 case 诊断",
            "",
            "| Backend | Case | Correctness | Candidate µs | Baseline µs | Raw speedup | AnchorScore | Anchor status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for control in assessment["controls"]:
        for case in control.get("case_results", []):
            lines.append(
                f"| {control['backend']} | {case['case_id']} | "
                f"{_status(bool(case['correctness_passed']))} | "
                f"{_number(case['candidate_latency_us']['median'])} | "
                f"{_number(case['reference_latency_us']['median'])} | "
                f"{case['paired_speedup']:.3f}× | "
                f"{_number(case['anchor_result']['anchor_score_raw'])} | "
                f"{case['anchor_result']['status']} |"
            )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            "- correctness 或动态输入门禁失败属于有效失败：不发布 Artifact 分，排行榜有效分为 0。",
            "- profiler 缺少声明的原生 Flash 路径属于证据无效：保持 N/A，不把可疑 timing 当成 0 分。",
            "- 正确但浪费工作的候选仍可被评分，低 Artifact-100 用于验证性能公式确实产生惩罚。",
            "- 所有 raw block、profiler event、provenance 和 SHA-256 清单均随 ZIP 提供。",
            "",
        ]
    )
    return "\n".join(lines)


def negative_control_html(assessment: dict[str, Any]) -> str:
    rows = []
    for control in assessment["controls"]:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(control['backend'])}</code></td>"
            f"<td>{html.escape(control['defect'])}</td>"
            f"<td>{_status(bool(control.get('certified')))}</td>"
            f"<td>{html.escape(str(control.get('disposition', '—')))}</td>"
            f"<td>{_number(control.get('artifact_100'), 2)}</td>"
            f"<td>{'PASS' if control['expectation_passed'] else 'FAIL'}</td>"
            f"<td>{html.escape(', '.join(control.get('failure_codes', [])) or '—')}</td>"
            "</tr>"
        )
    overall = "PASS" if assessment["all_expectations_passed"] else "FAIL"
    hardware_text = " · ".join(
        f"{cell['hardware']['gpu_name']} / CC {cell['hardware']['compute_capability']} / "
        f"{int(cell['hardware']['total_memory_bytes']) / 2**30:.1f} GiB"
        for cell in assessment["hardware_cells"]
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>InfraSWE 垃圾 Kernel 负控</title>
<style>body{{font:15px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172033}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #ccd5e3;text-align:left}}
th{{background:#eef4fb}}code{{color:#075985}}.pass{{color:#067647;font-weight:700}}</style></head>
<body><h1>InfraSWE v0.3 垃圾 Kernel 负控</h1><p>负控总门禁：<span class="pass">{overall}</span></p>
<p>{html.escape(hardware_text)}</p>
<p>本报告与正式 FA1–FA4 排名隔离；每项均来自真实 A100 replay 与 profiler。</p>
<table><thead><tr><th>Backend</th><th>故意缺陷</th><th>Cert</th><th>处置</th><th>Artifact-100</th><th>预期</th><th>Failure codes</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""


def _waste_passes(backend: str) -> int | None:
    if backend == "garbage-slow-fa4-waste64":
        return 64
    prefix = "mediocre-fa4-waste"
    if backend.startswith(prefix):
        return int(backend.removeprefix(prefix))
    return None


def score_gradient_assessment(score: dict[str, Any], root: Path) -> dict[str, Any]:
    pilot_path = root / "evidence/a100-sm80-40gb/pilot/sweep.json"
    if not pilot_path.exists():
        matches = sorted(root.rglob("pilot/sweep.json"))
        if len(matches) != 1:
            raise ValueError("score-gradient report requires exactly one pilot/sweep.json")
        pilot_path = matches[0]
    pilot = json.loads(pilot_path.read_text())
    pilot_by_pass = {
        int(item["passes"]): float(item["estimated_artifact_100"])
        for item in pilot["variants"]
    }
    points = []
    hardware_cells = []
    for cell in score["cells"]:
        hardware_cells.append(
            {
                "cell_id": cell["cell_id"],
                "hardware": cell["hardware"],
                "calibration": cell.get("calibration"),
            }
        )
        for candidate in cell["candidates"]:
            passes = _waste_passes(candidate["backend"])
            if passes is None:
                continue
            artifact = candidate.get("artifact_100")
            pilot_score = pilot_by_pass.get(passes)
            components = candidate.get("components", {})
            points.append(
                {
                    "cell_id": cell["cell_id"],
                    "passes": passes,
                    "backend": candidate["backend"],
                    "certified": candidate.get("certified"),
                    "verdict": candidate.get("verdict"),
                    "disposition": candidate.get("disposition"),
                    "artifact_status": candidate.get("artifact_status"),
                    "artifact_100": artifact,
                    "pilot_artifact_100": pilot_score,
                    "pilot_delta": (
                        None
                        if artifact is None or pilot_score is None
                        else float(artifact) - pilot_score
                    ),
                    "performance_component": components.get("performance_anchor_score"),
                    "generalization_component": components.get("generalization"),
                    "weighted_geometric_speedup_raw": candidate.get(
                        "weighted_geometric_speedup_raw"
                    ),
                    "replay_count": candidate.get("replay_count"),
                    "supplemental_profile_count": candidate.get(
                        "supplemental_profile_count"
                    ),
                    "failure_codes": candidate.get("failure_codes", []),
                    "case_results": candidate.get("case_results", []),
                }
            )
    points.sort(key=lambda item: item["passes"])
    scores = [float(point["artifact_100"]) for point in points if point["artifact_100"] is not None]
    adjacent_gaps = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
    deltas = [abs(float(point["pilot_delta"])) for point in points if point["pilot_delta"] is not None]
    all_scored = bool(points) and all(
        point["certified"] is True
        and point["disposition"] == "valid"
        and point["artifact_status"] == "scored"
        and point["replay_count"] == 3
        and point["supplemental_profile_count"] == 5
        for point in points
    )
    monotonic = len(scores) == len(points) and all(gap > 0 for gap in adjacent_gaps)
    span = scores[0] - scores[-1] if len(scores) >= 2 else 0.0
    return {
        "schema_version": "0.3",
        "report_kind": "kernel-score-gradient",
        "score_generated_at": score["generated_at"],
        "hardware_cells": hardware_cells,
        "pilot_evidence_path": pilot_path.relative_to(root).as_posix(),
        "all_points_scored": all_scored,
        "strictly_monotonic": monotonic,
        "score_span": span,
        "minimum_adjacent_gap": min(adjacent_gaps) if adjacent_gaps else None,
        "maximum_pilot_error": max(deltas) if deltas else None,
        "all_expectations_passed": (
            all_scored
            and monotonic
            and len(points) >= 6
            and span >= 40.0
            and min(adjacent_gaps, default=0.0) >= 3.0
        ),
        "points": points,
    }


def score_gradient_markdown(assessment: dict[str, Any]) -> str:
    overall = "PASS" if assessment["all_expectations_passed"] else "FAIL"
    lines = [
        "# InfraSWE v0.3 Kernel 分数梯度报告",
        "",
        f"梯度门禁：**{overall}**  ",
        f"评分生成时间：`{assessment['score_generated_at']}`",
        "",
    ]
    for cell in assessment["hardware_cells"]:
        hardware = cell["hardware"]
        lines.append(
            f"硬件 cell：`{cell['cell_id']}`；{hardware['gpu_name']}；CC "
            f"{hardware['compute_capability']}；显存 "
            f"{int(hardware['total_memory_bytes']) / 2**30:.1f} GiB；Driver "
            f"{hardware['driver_version']}。"
        )
    lines.extend(
        [
            "",
            "候选均使用相同 FA4 正确路径，仅改变其后追加的无用 Triton streaming passes，"
            "从而构造可复现、可解释的性能退化曲线。pass=0 是未追加浪费工作的上界控制；"
            "pass=64 复用同一硬件 cell 已完成的正式负控证据。",
            "",
            f"正式分数跨度：**{assessment['score_span']:.2f}**；最小相邻间隔："
            f"**{_number(assessment['minimum_adjacent_gap'], 2)}**；最大 pilot 误差："
            f"**{_number(assessment['maximum_pilot_error'], 2)}**。",
            "",
            "| Passes | Backend | Cert | Pilot | Formal Artifact-100 | Δ | P | G | Raw speedup |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for point in assessment["points"]:
        lines.append(
            f"| {point['passes']} | {point['backend']} | "
            f"{_status(bool(point['certified']))} | "
            f"{_number(point['pilot_artifact_100'], 2)} | "
            f"{_number(point['artifact_100'], 2)} | "
            f"{_number(point['pilot_delta'], 2)} | "
            f"{_number(point['performance_component'])} | "
            f"{_number(point['generalization_component'])} | "
            f"{_number(point['weighted_geometric_speedup_raw'])}× |"
        )
    lines.extend(
        [
            "",
            "## 完整性结论",
            "",
            f"- 六级或以上梯度：{'PASS' if len(assessment['points']) >= 6 else 'FAIL'}。",
            f"- 分数随浪费工作严格递减：{'PASS' if assessment['strictly_monotonic'] else 'FAIL'}。",
            f"- 每点 3 replay + 5 profiler 且 KernelCert 通过："
            f"{'PASS' if assessment['all_points_scored'] else 'FAIL'}。",
            "- 这些是评分器校准候选，不进入正式 FA1–FA4 排名。",
            "- pilot、raw matched blocks、profiler、provenance 与 SHA-256 清单均随 ZIP 提供。",
            "",
        ]
    )
    return "\n".join(lines)


def score_gradient_html(assessment: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{point['passes']}</td><td><code>{html.escape(point['backend'])}</code></td>"
        f"<td>{_status(bool(point['certified']))}</td>"
        f"<td>{_number(point['pilot_artifact_100'], 2)}</td>"
        f"<td>{_number(point['artifact_100'], 2)}</td>"
        f"<td>{_number(point['pilot_delta'], 2)}</td>"
        f"<td>{_number(point['performance_component'])}</td>"
        f"<td>{_number(point['generalization_component'])}</td></tr>"
        for point in assessment["points"]
    )
    overall = "PASS" if assessment["all_expectations_passed"] else "FAIL"
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>InfraSWE Kernel 分数梯度</title>
<style>body{{font:15px/1.55 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#172033}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #ccd5e3;text-align:left}}
th{{background:#eef4fb}}code{{color:#075985}}.pass{{color:#067647;font-weight:700}}</style></head>
<body><h1>InfraSWE v0.3 Kernel 分数梯度</h1><p>梯度门禁：<span class="pass">{overall}</span></p>
<p>分数跨度 {assessment['score_span']:.2f}；最小相邻间隔 {_number(assessment['minimum_adjacent_gap'], 2)}。</p>
<table><thead><tr><th>Passes</th><th>Backend</th><th>Cert</th><th>Pilot</th><th>Formal</th><th>Δ</th><th>P</th><th>G</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_manifest(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "evidence-manifest.json":
            continue
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "media_type": media_type,
                "producer": "infraswe-kernel-frontier-v03",
            }
        )
    atomic_write_json(
        root / "evidence-manifest.json",
        {"schema_version": "0.3", "root": root.name, "file_count": len(files), "files": files},
    )


def write_zip(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.resolve() == destination.resolve():
                continue
            archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{sha256_file(destination).removeprefix('sha256:')}  {destination.name}\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--negative-controls", action="store_true")
    mode.add_argument("--score-gradient", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    score = assemble_suite(load_json_evidence(root))
    atomic_write_json(root / "score.json", score)
    if args.negative_controls:
        assessment = negative_control_assessment(score)
        atomic_write_json(root / "negative-controls.json", assessment)
        (root / "report.md").write_text(
            negative_control_markdown(assessment), encoding="utf-8"
        )
        (root / "index.html").write_text(
            negative_control_html(assessment), encoding="utf-8"
        )
    elif args.score_gradient:
        assessment = score_gradient_assessment(score, root)
        atomic_write_json(root / "score-gradient.json", assessment)
        (root / "report.md").write_text(
            score_gradient_markdown(assessment), encoding="utf-8"
        )
        (root / "index.html").write_text(
            score_gradient_html(assessment), encoding="utf-8"
        )
    else:
        (root / "report.md").write_text(markdown_report(score), encoding="utf-8")
        (root / "index.html").write_text(html_report(score), encoding="utf-8")
    write_manifest(root)
    if args.zip:
        write_zip(root, args.zip.resolve())


if __name__ == "__main__":
    main()
