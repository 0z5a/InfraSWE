# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import hashlib
import html
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


def markdown_report(score: dict[str, Any]) -> str:
    lines = [
        "# InfraSWE Kernel Frontier v0.3 — FA1–FA4 与经典 Kernel 评分",
        "",
        f"生成时间：`{score['generated_at']}`  ",
        f"Suite：`{score['suite_id']}`  ",
        f"公式：`{score['formula_version']}`；AnchorScore 来源：`{score['formula_origin']}`",
        "",
        "评分使用三次独立进程 replay、每 case 30 个 matched ABBA/BAAB blocks、"
        "evaluator-owned CUDA events 与 block/replay 两层 bootstrap CI95。"
        "FA 库分数为 `100 × (0.80P + 0.20G)`；经典 micro kernel 各自为 `100 × AnchorScore`。",
        "同一行只在同一硬件 cell 内可比较，A100 与 SM120 的分数不得直接混排。",
        "",
        "实现与公式来源：",
        "",
        "- FA1 固定为 Dao-AILab/flash-attention "
        "[`6d48e14`](https://github.com/Dao-AILab/flash-attention/commit/6d48e14a6c2f551db96f0badc658a6279a929df3)（v1.0.9）。",
        "- FA2/FA3 固定为 Dao-AILab/flash-attention "
        "[`ce088ab`](https://github.com/Dao-AILab/flash-attention/commit/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820)。",
        "- FA4 固定为 PyPI "
        "[`flash-attn-4==4.0.0b28`](https://pypi.org/project/flash-attn-4/)。",
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
                f"GPU：{hardware['gpu_name']}；CC {hardware['compute_capability']}；"
                f"SM {hardware['sm_count']}；显存 {int(hardware['total_memory_bytes']) / 2**30:.1f} GiB；"
                f"PyTorch {hardware['torch_version']} / CUDA {hardware['torch_cuda']}；"
                f"Driver {hardware['driver_version']}。",
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
                "### FA 支持矩阵",
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
            "- FA4 按固定 b28 包的显式架构 dispatch 判定资格；该版本包含 SM80 路径，"
            "未包含的架构明确记为 N/A，不用其他实现代填。",
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
            f"<p>{html.escape(str(hardware['gpu_name']))} · CC {html.escape(str(hardware['compute_capability']))} · "
            f"{int(hardware['total_memory_bytes']) / 2**30:.1f} GiB · PyTorch {html.escape(str(hardware['torch_version']))}</p>"
            f"<p class='muted'>Calibration: {calibration_text}</p>"
            "<h3>FA support matrix</h3>"
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
<p class="muted">FA1–FA4 and classic kernels · generated {html.escape(score["generated_at"])}</p>
<p>Three fresh-process replays, matched ABBA/BAAB blocks, hierarchical CI95. Scores across hardware cells are not directly rankable.</p>
<p class="muted">Sources: <a href="https://github.com/Dao-AILab/flash-attention">Dao-AILab FlashAttention</a> · <a href="https://pypi.org/project/flash-attn-4/">flash-attn-4 4.0.0b28</a> · <a href="https://arxiv.org/abs/2603.19173">SOL-ExecBench</a></p>
{"".join(sections)}
<section><h2>Interpretation boundary</h2><p>The calibrated target is an engineering anchor with medium confidence, not a physical lower bound. Raw speedup and AnchorScore are not silently clipped. Every raw JSON and digest is packaged with this report.</p></section>
</main></body></html>"""


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    score = assemble_suite(load_json_evidence(root))
    atomic_write_json(root / "score.json", score)
    (root / "report.md").write_text(markdown_report(score), encoding="utf-8")
    (root / "index.html").write_text(html_report(score), encoding="utf-8")
    write_manifest(root)
    if args.zip:
        write_zip(root, args.zip.resolve())


if __name__ == "__main__":
    main()
