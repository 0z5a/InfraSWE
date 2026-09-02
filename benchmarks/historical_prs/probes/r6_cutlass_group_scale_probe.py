#!/usr/bin/env python3
"""Build and run exact CUTLASS PR 2275 base/head group-scale examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TARGET = "68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling"
SOURCE = Path(
    "include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run(command: list[str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": command,
            "return_code": None,
            "timed_out": True,
            "duration_seconds": time.perf_counter() - started,
            "stdout_sha256": _digest(stdout),
            "stderr_sha256": _digest(stderr),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    return {
        "command": command,
        "return_code": completed.returncode,
        "timed_out": False,
        "duration_seconds": time.perf_counter() - started,
        "stdout_sha256": _digest(completed.stdout),
        "stderr_sha256": _digest(completed.stderr),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _configure_and_build(
    source_root: Path,
    build_root: Path,
    cmake: Path,
    nvcc: Path,
) -> dict[str, Any]:
    configure = _run(
        [
            str(cmake),
            "-S",
            str(source_root),
            "-B",
            str(build_root),
            "-GNinja",
            f"-DCMAKE_CUDA_COMPILER={nvcc}",
            "-DCUTLASS_NVCC_ARCHS=90a",
            "-DCUTLASS_ENABLE_TESTS=OFF",
            "-DCUTLASS_ENABLE_EXAMPLES=ON",
            "-DCUTLASS_ENABLE_LIBRARY=OFF",
        ],
        timeout=300,
    )
    build = (
        _run(
            [str(cmake), "--build", str(build_root), "--target", TARGET, "-j", "2"],
            timeout=900,
        )
        if configure["return_code"] == 0
        else None
    )
    return {"configure": configure, "build": build}


def _binary(build_root: Path) -> Path:
    return (
        build_root
        / "examples"
        / "68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling"
        / TARGET
    )


def _case(binary: Path, *, k: int, iterations: int, timeout: float) -> dict[str, Any]:
    result = _run(
        [
            str(binary),
            "--m=1024",
            "--n=1024",
            f"--k={k}",
            "--groups=10",
            f"--iterations={iterations}",
        ],
        timeout=timeout,
    )
    output = result["stdout_tail"] + "\n" + result["stderr_tail"]
    match = re.search(r"Avg runtime:\s*([0-9.]+)\s*ms", output)
    result["disposition_passed"] = "Disposition: Passed" in output
    result["average_runtime_ms"] = float(match.group(1)) if match else None
    return result


def _paired_timing(base_binary: Path, head_binary: Path) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for index in range(7):
        order = ("base", "head") if index % 2 == 0 else ("head", "base")
        values: dict[str, float] = {}
        raw: dict[str, Any] = {}
        for variant in order:
            binary = base_binary if variant == "base" else head_binary
            result = _case(binary, k=256, iterations=100, timeout=120)
            raw[variant] = result
            if not result["disposition_passed"] or result["average_runtime_ms"] is None:
                raise RuntimeError(f"timed {variant} K=256 run failed: {result}")
            values[variant] = result["average_runtime_ms"]
        pairs.append({"first": order[0], "runtime_ms": values, "raw": raw})
    base = [item["runtime_ms"]["base"] for item in pairs]
    head = [item["runtime_ms"]["head"] for item in pairs]
    return {
        "pairs": pairs,
        "base_median_ms": statistics.median(base),
        "head_median_ms": statistics.median(head),
        "head_over_base_median_ratio": statistics.median(head) / statistics.median(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--cmake", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    base_source = (args.base_root / SOURCE).read_text(encoding="utf-8")
    head_source = (args.head_root / SOURCE).read_text(encoding="utf-8")
    source_contract = {
        "base_final_tile_guard_count": base_source.count("if (k_tile_count == 1)"),
        "head_final_tile_guard_count": head_source.count("if (k_tile_count == 1)"),
    }
    if source_contract != {
        "base_final_tile_guard_count": 0,
        "head_final_tile_guard_count": 1,
    }:
        raise ValueError(f"unexpected exact-source contract: {source_contract}")

    builds = {
        "base": _configure_and_build(
            args.base_root, args.build_root / "base", args.cmake, args.nvcc
        ),
        "head": _configure_and_build(
            args.head_root, args.build_root / "head", args.cmake, args.nvcc
        ),
    }
    build_failures = [
        variant
        for variant, result in builds.items()
        if result["configure"]["return_code"] != 0
        or result["build"] is None
        or result["build"]["return_code"] != 0
    ]
    base_binary = _binary(args.build_root / "base")
    head_binary = _binary(args.build_root / "head")
    correctness: dict[str, Any] = {}
    timing: dict[str, Any] | None = None
    if not build_failures:
        correctness = {
            "base_k128": _case(base_binary, k=128, iterations=0, timeout=30),
            "head_k128": _case(head_binary, k=128, iterations=0, timeout=30),
            "base_k256": _case(base_binary, k=256, iterations=0, timeout=60),
            "head_k256": _case(head_binary, k=256, iterations=0, timeout=60),
        }
        timing = _paired_timing(base_binary, head_binary)

    failure_codes: list[str] = []
    infrastructure_codes: list[str] = []
    if build_failures:
        infrastructure_codes.append("CUTLASS_EXAMPLE_BUILD_FAILED")
    else:
        if correctness["base_k128"]["disposition_passed"]:
            failure_codes.append("CUTLASS_BASE_K128_CONTROL_DID_NOT_REPRODUCE")
        if not correctness["head_k128"]["disposition_passed"]:
            failure_codes.append("CUTLASS_HEAD_K128_CORRECTNESS_FAILED")
        if (
            not correctness["base_k256"]["disposition_passed"]
            or not correctness["head_k256"]["disposition_passed"]
        ):
            failure_codes.append("CUTLASS_K256_NEIGHBOR_CORRECTNESS_FAILED")
        if timing is not None and timing["head_over_base_median_ratio"] > 1.03:
            failure_codes.append("CUTLASS_K256_LATENCY_REGRESSION_GT_3PCT")

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "cutlass-group-scale-k128-h100-v1",
        "case_id": "cutlass-pr-2275",
        "status": (
            "unresolved" if infrastructure_codes else "pass" if not failure_codes else "fail"
        ),
        "failure_codes": failure_codes,
        "infrastructure_codes": infrastructure_codes,
        "facts": {
            "source_contract": source_contract,
            "builds": builds,
            "correctness": correctness,
            "timing": timing,
            "aot_compilation_completed_before_execution": not build_failures,
            "steady_state_compile_seconds": 0.0 if not build_failures else None,
        },
        "source_identity": {
            "base_source_sha256": _digest(base_source),
            "head_source_sha256": _digest(head_source),
        },
        "environment": {
            "cmake": str(args.cmake),
            "nvcc": str(args.nvcc),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": _digest(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 2 if infrastructure_codes else 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
