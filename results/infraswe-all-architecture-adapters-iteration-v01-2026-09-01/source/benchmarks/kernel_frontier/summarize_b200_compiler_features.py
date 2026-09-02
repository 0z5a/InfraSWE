# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.kernel.blackwell import SCORE_NAMESPACES


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_replays(root: Path) -> list[dict[str, Any]]:
    replays = [
        json.loads((root / f"replay-{index}.json").read_text(encoding="utf-8"))
        for index in (1, 2, 3)
    ]
    indices = [int(replay["replay_index"]) for replay in replays]
    if indices != [1, 2, 3]:
        raise ValueError(f"B200 replay set must be exactly [1, 2, 3], got {indices}")
    capability_hashes = {replay["capability_manifest_sha256"] for replay in replays}
    if len(capability_hashes) != 1:
        raise ValueError("B200 capability identity drift across replays")
    capability_fingerprints = {replay["capability_fingerprint"] for replay in replays}
    if len(capability_fingerprints) != 1:
        raise ValueError("B200 stable capability fingerprint drift across replays")
    feature_sets = [tuple(item["feature_id"] for item in replay["features"]) for replay in replays]
    if len(set(feature_sets)) != 1:
        raise ValueError("B200 feature set drift across replays")
    return replays


def feature_summary(feature_id: str, replays: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        next(item for item in replay["features"] if item["feature_id"] == feature_id)
        for replay in replays
    ]
    statuses = [record["status"] for record in records]
    if all(status == "not_applicable" for status in statuses):
        status = "not_applicable"
        certified = False
    elif all(record["certified"] for record in records):
        status = "certified"
        certified = True
    elif any(value in {"failed", "blocked"} for value in statuses):
        status = "failed"
        certified = False
    else:
        status = "pending"
        certified = False
    failure_codes = sorted(
        {
            code
            for record in records
            for code in record.get("native_verification", {}).get("failure_codes", [])
        }
    )
    return {
        "feature_id": feature_id,
        "namespace": records[0]["namespace"],
        "status": status,
        "certified_all_replays": certified,
        "replay_statuses": statuses,
        "failure_codes": failure_codes,
        "reason": records[0].get("reason"),
    }


def namespace_summary(namespace: str, features: list[dict[str, Any]]) -> dict[str, Any]:
    members = [feature for feature in features if feature["namespace"] == namespace]
    applicable = [feature for feature in members if feature["status"] != "not_applicable"]
    certified = [feature for feature in applicable if feature["certified_all_replays"]]
    if namespace == "PTX-Preview":
        status = "disabled"
    elif not members:
        status = "not_run"
    elif not applicable:
        status = "not_applicable"
    elif len(certified) == len(applicable):
        status = "certified_conformance"
    elif any(feature["status"] == "failed" for feature in applicable):
        status = "failed"
    else:
        status = "pending_evidence"
    coverage = 100.0 * len(certified) / len(applicable) if applicable else None
    return {
        "status": status,
        "registered_feature_count": len(members),
        "applicable_feature_count": len(applicable),
        "certified_feature_count": len(certified),
        "certification_coverage_percent": coverage,
        "leaderboard_score_100": None,
        "note": "coverage is evidence completeness, not a performance score",
    }


def build_summary(replays: list[dict[str, Any]]) -> dict[str, Any]:
    feature_ids = [item["feature_id"] for item in replays[0]["features"]]
    features = [feature_summary(feature_id, replays) for feature_id in feature_ids]
    applicable = [feature for feature in features if feature["status"] != "not_applicable"]
    all_certified = bool(applicable) and all(
        feature["certified_all_replays"] for feature in applicable
    )
    if all_certified:
        status = "native_conformance_certified"
    elif any(feature["status"] == "failed" for feature in applicable):
        status = "failed"
    else:
        status = "evidence_pending"
    return {
        "schema_version": "0.1",
        "generated_at": utc_now(),
        "suite_id": "b200-sm100-compiler-features-v0.1",
        "status": status,
        "replay_count": 3,
        "replay_indices": [1, 2, 3],
        "capability_manifest_sha256": replays[0]["capability_manifest_sha256"],
        "capability_fingerprint": replays[0]["capability_fingerprint"],
        "hardware": replays[0]["hardware"],
        "toolchain": replays[0]["toolchain"],
        "features": features,
        "score_namespaces": {
            namespace: namespace_summary(namespace, features) for namespace in SCORE_NAMESPACES
        },
        "leaderboard_ready": False,
        "leaderboard_score_100": None,
        "leaderboard_blocker": (
            "the initial adapter certifies compiler/native/runtime evidence only; "
            "frozen evaluator performance workloads and anchors are not yet present"
        ),
        "raw_replays": [f"replays/replay-{index}.json" for index in (1, 2, 3)],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    hardware = summary.get("hardware") or {}
    toolchain = summary["toolchain"]
    detected = toolchain.get("detected_versions", {})
    lines = [
        "# B200 / SM100 编译器特性初版适配报告",
        "",
        f"状态：`{summary['status']}`；三次 fresh-process replay 已纳入协议。",
        "",
        f"GPU：`{hardware.get('name', 'unknown')}`；CC "
        f"`{hardware.get('compute_capability', 'unknown')}`；架构 "
        f"`{hardware.get('architecture', 'unknown')}`。",
        f"工具链：`{toolchain['status']}`；nvcc "
        f"`{detected.get('nvcc_version') or 'unknown'}`；ptxas "
        f"`{detected.get('ptxas_version') or 'unknown'}`。",
        "",
        "## 特性证据",
        "",
        "| 特性 | 分数命名空间 | 三次回放状态 | 原生认证 |",
        "|---|---|---|---|",
    ]
    for feature in summary["features"]:
        lines.append(
            f"| {feature['feature_id']} | {feature['namespace']} | "
            f"{', '.join(feature['replay_statuses'])} | "
            f"{'PASS' if feature['certified_all_replays'] else feature['status'].upper()} |"
        )
    lines.extend(
        [
            "",
            "## 命名空间",
            "",
            "| 命名空间 | 状态 | 证据覆盖率 | 榜单分数 |",
            "|---|---|---:|---:|",
        ]
    )
    for namespace, result in summary["score_namespaces"].items():
        coverage = result["certification_coverage_percent"]
        rendered_coverage = "N/A" if coverage is None else f"{coverage:.1f}%"
        lines.append(f"| {namespace} | {result['status']} | {rendered_coverage} | N/A |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 证据覆盖率只表示 PTX、cubin/SASS、正确性、watchdog、profiler 与 "
            "mutation 门禁完整度，不是性能分。",
            "- 当前初版不会从静态正则直接发榜：缺动态证据、能力清单绑定或"
            "三次回放时均保持 pending/failed。",
            "- `PTX-Preview` 在 CUDA 13.3 / PTX 9.3 基线中禁用；不会混入稳定榜单。",
            "- 性能榜单仍需 evaluator-owned 工作负载、基线、Anchor 与计时数据，"
            "因此本报告的榜单分数为 N/A。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize three B200 compiler-feature replays")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--require-certified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(load_replays(args.root))
    atomic_write_json(args.json_output, summary)
    args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    if args.require_certified and summary["status"] != "native_conformance_certified":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
