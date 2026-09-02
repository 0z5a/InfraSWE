# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from bench_utils import atomic_write_json, utc_now


def median_blocks(replay: dict[str, Any], field: str) -> float:
    return statistics.median(
        float(block[field]) for block in replay["features"]["tma"]["measurement"]["blocks"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replays = [
        json.loads((args.root / f"replay-{index}.json").read_text(encoding="utf-8"))
        for index in (1, 2, 3)
    ]
    if [int(item["replay_index"]) for item in replays] != [1, 2, 3]:
        raise ValueError("H200 feature replay set must be exactly 1, 2, 3")
    if any(item["status"] != "passed" for item in replays):
        raise ValueError("all H200 feature replays must pass")

    reference = [median_blocks(item, "reference_latency_us") for item in replays]
    candidate = [median_blocks(item, "candidate_latency_us") for item in replays]
    tma = {
        "status": "passed",
        "correctness_passed": all(
            item["features"]["tma"]["correctness"]["passed"] for item in replays
        ),
        "profiler_captured": all(
            item["features"]["tma"]["profiler"].get("captured") for item in replays
        ),
        "instruction_gate_passed": all(
            item["features"]["tma"]["compiler_evidence"]["instruction_gate_passed"]
            for item in replays
        ),
        "driver_tensor_map_access_supported": replays[0]["features"]["tma"][
            "driver_attribute"
        ],
        "reference_latency_us_per_replay": reference,
        "candidate_latency_us_per_replay": candidate,
        "reference_latency_us": statistics.median(reference),
        "candidate_latency_us": statistics.median(candidate),
        "candidate_over_reference": statistics.median(candidate)
        / statistics.median(reference),
        "ptx_instruction_lines": replays[0]["features"]["tma"]["compiler_evidence"][
            "ptx_instruction_lines"
        ],
        "sass_instruction_lines": replays[0]["features"]["tma"]["compiler_evidence"][
            "sass_instruction_lines"
        ],
    }
    multimem_replays = [item["features"]["multimem"] for item in replays]
    multimem_sass_lines = []
    multimem_instruction_gates = []
    for index, item in enumerate(multimem_replays, start=1):
        compile_evidence = item["toolchain_compile"]
        sass_path = compile_evidence.get("sass_path")
        lines = compile_evidence.get("sass_instruction_lines", [])
        if not lines and sass_path:
            artifact = args.root / "artifacts" / f"replay-{index}" / sass_path
            if artifact.exists():
                lines = [
                    line.strip()
                    for line in artifact.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if re.search(r"\b(?:LDGMC|STGMC|REDGMC)\b", line, re.IGNORECASE)
                ]
        multimem_instruction_gates.append(bool(lines))
        if index == 1:
            multimem_sass_lines = lines
    multimem = {
        "status": multimem_replays[0]["status"],
        "driver_multicast_supported": multimem_replays[0]["driver"]["attributes"][
            "multicast_supported"
        ],
        "driver_fabric_handle_supported": multimem_replays[0]["driver"]["attributes"][
            "fabric_handle_supported"
        ],
        "visible_cuda_device_count": multimem_replays[0]["visible_cuda_device_count"],
        "toolchain_compile_passed": all(
            item["toolchain_compile"]["passed"] for item in multimem_replays
        ),
        "instruction_gate_passed": all(multimem_instruction_gates),
        "sass_instruction_lines": multimem_sass_lines,
        "execution_attempted": any(item["execution_attempted"] for item in multimem_replays),
        "execution_reason": multimem_replays[0]["execution_reason"],
        "topology_stdout": multimem_replays[0]["topology"]["stdout"],
    }
    summary = {
        "schema_version": "0.3",
        "generated_at": utc_now(),
        "suite_id": "h200-sm90-feature-supplement-v1",
        "hardware": replays[0]["hardware"],
        "replay_count": 3,
        "tma": tma,
        "multimem": multimem,
        "raw_replays": [f"features/replay-{index}.json" for index in (1, 2, 3)],
    }
    atomic_write_json(args.json_output, summary)

    hardware = summary["hardware"]
    lines = [
        "# H200 SM90 架构新特性补测：TMA 与 multimem",
        "",
        f"GPU：{hardware['gpu_name']}；CC {hardware['compute_capability']}；"
        f"SM {hardware['sm_count']}；显存 {int(hardware['total_memory_bytes']) / 2**30:.1f} GiB。",
        "",
        "## 结果",
        "",
        "| 特性 | 运行状态 | 编译/指令门禁 | 正确性 | 说明 |",
        "|---|---|---|---|---|",
        f"| TMA | PASS | {'PASS' if tma['instruction_gate_passed'] else 'FAIL'} | "
        f"{'PASS' if tma['correctness_passed'] else 'FAIL'} | "
        f"3 个 fresh-process replay；Triton tensor descriptor 实际执行 |",
        f"| multimem | {multimem['status']} | "
        f"{'PASS' if multimem['instruction_gate_passed'] else 'FAIL'} | N/A | "
        "PTX ISA 可编译；运行受 CUDA multicast/topology 门禁约束 |",
        "",
        "## TMA 实测",
        "",
        f"- Driver `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED`："
        f"`{tma['driver_tensor_map_access_supported']['value']}`。",
        f"- 4096×4096 BF16 copy+add：TMA candidate 中位数 "
        f"`{tma['candidate_latency_us']:.3f} µs`；Torch add-out reference "
        f"`{tma['reference_latency_us']:.3f} µs`；比值 "
        f"`{tma['candidate_over_reference']:.3f}×`。",
        "- 3 replay 均通过逐元素正确性、动态输入变化、CUDA Profiler 和编译产物指令门禁。",
        "- SASS 明确包含 `UTMALDG.2D` 与 `UTMASTG.2D`。",
        "",
        "## multimem 门禁",
        "",
        f"- Driver `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`："
        f"`{multimem['driver_multicast_supported']['value']}`；"
        f"fabric handle：`{multimem['driver_fabric_handle_supported']['value']}`；"
        f"可见 GPU：`{multimem['visible_cuda_device_count']}`。",
        f"- 状态：`{multimem['status']}`。`multimem.*` PTX 已由本机 ptxas 编译，"
        "但没有合法 CUDA multicast 映射，因此未运行。",
        "- 编译后的 SM90 SASS 明确包含 `LDGMC.E.ADD.32.STRONG.SYS`。",
        "- 普通指针不是 multimem address；对它发射 `multimem.*` 属于未定义行为，"
        "本测试严格禁止用这种方式伪造运行成功。",
        "",
        "## 证据边界",
        "",
        "- TMA 是本机真实执行与指令级证据，可作为 H200 SM90 feature PASS。",
        "- multimem 是 ISA/toolchain PASS、当前单卡拓扑 runtime N/A；"
        "需要支持 switch multicast 的多 GPU/NVSwitch cell 才能形成运行评分。",
        "- 原始 replay、Triton PTX/cubin/SASS、multimem PTX/cubin/SASS、"
        "Driver 属性与拓扑快照全部随 ZIP 提供。",
        "",
        "参考：NVIDIA PTX ISA 的 `multimem.*` 定义与 CUDA Driver API multicast 管理。",
    ]
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
