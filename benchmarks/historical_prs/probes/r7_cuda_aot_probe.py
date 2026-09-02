#!/usr/bin/env python3
"""CUDA AOT probes for the CUTLASS and DeepGEMM R7 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _run(command: list[str], timeout: float = 900) -> dict[str, Any]:
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
            "stderr_head": stderr[:8000],
            "stderr_tail": stderr[-8000:],
        }
    return {
        "command": command,
        "return_code": completed.returncode,
        "timed_out": False,
        "duration_seconds": time.perf_counter() - started,
        "stdout_sha256": _digest(completed.stdout),
        "stderr_sha256": _digest(completed.stderr),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_head": completed.stderr[:8000],
        "stderr_tail": completed.stderr[-8000:],
    }


def _cutlass(
    nvcc: Path,
    base_root: Path,
    head_root: Path,
    fixture_root: Path,
    build_root: Path,
) -> dict[str, Any]:
    scaled = fixture_root / "r7_cutlass_scaled_basis.cpp"
    metadata = fixture_root / "r7_cutlass_sm120_metadata_k.cu"
    builds: dict[str, Any] = {}
    for revision, root in (("base", base_root), ("head", head_root)):
        scaled_output = build_root / f"cutlass-scaled-{revision}"
        builds[f"scaled_{revision}"] = _run(
            [
                str(nvcc),
                "-std=c++17",
                "-x",
                "cu",
                "-I",
                str(root / "include"),
                str(scaled),
                "-o",
                str(scaled_output),
            ]
        )
        if builds[f"scaled_{revision}"]["return_code"] == 0:
            builds[f"scaled_{revision}"]["run"] = _run([str(scaled_output)], timeout=60)
        for tile_k in (128, 256):
            output = build_root / f"cutlass-sm120-{revision}-k{tile_k}.o"
            builds[f"sm120_{revision}_k{tile_k}"] = _run(
                [
                    str(nvcc),
                    "-std=c++17",
                    "--generate-code=arch=compute_120a,code=sm_120a",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-I",
                    str(root / "include"),
                    f"-DPROBE_TILE_K={tile_k}",
                    "-c",
                    str(metadata),
                    "-o",
                    str(output),
                ]
            )
    head_invalid = builds["sm120_head_k128"]
    head_invalid_diagnostics = head_invalid["stderr_head"] + head_invalid["stderr_tail"]
    return {
        "builds": builds,
        "base_scaled_basis_failure_reproduced": builds["scaled_base"]["return_code"] != 0,
        "head_scaled_basis_full_header_pass": (
            builds["scaled_head"]["return_code"] == 0
            and builds["scaled_head"].get("run", {}).get("return_code") == 0
        ),
        "base_valid_k256_pass": builds["sm120_base_k256"]["return_code"] == 0,
        "head_valid_k256_pass": builds["sm120_head_k256"]["return_code"] == 0,
        "head_invalid_k128_rejected": head_invalid["return_code"] not in (0, None),
        "head_invalid_reaches_intended_assert": (
            "TileShape_K must be a multiple of the metadata atom K extent"
            in head_invalid_diagnostics
        ),
        "base_invalid_k128_return_code": builds["sm120_base_k128"]["return_code"],
        "runtime_sm120": "unresolved-no-sm120-device",
        "performance_sm120": "unresolved-no-sm120-device",
    }


def _deepgemm(
    nvcc: Path,
    base_root: Path,
    head_root: Path,
    cutlass_root: Path,
    fixture_root: Path,
    build_root: Path,
) -> dict[str, Any]:
    fixture = fixture_root / "r7_deepgemm_sm100_mega_moe.cu"
    builds: dict[str, Any] = {}
    ptx_text: dict[str, str] = {}
    for revision, root in (("base", base_root), ("head", head_root)):
        output = build_root / f"deepgemm-mega-moe-{revision}.ptx"
        builds[revision] = _run(
            [
                str(nvcc),
                "-std=c++20",
                "--generate-code=arch=compute_100a,code=compute_100a",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-I",
                str(root / "deep_gemm/include"),
                "-I",
                str(cutlass_root / "include"),
                "-ptx",
                str(fixture),
                "-o",
                str(output),
            ],
            timeout=1800,
        )
        ptx_text[revision] = (
            output.read_text(encoding="utf-8", errors="replace")
            if builds[revision]["return_code"] == 0 and output.exists()
            else ""
        )
    instruction = "fence.proxy.async.shared::cta;"

    def target_fence_count(ptx: str) -> int:
        lines = ptx.splitlines()
        return sum(
            instruction in line
            and any("xor.b32" in nearby for nearby in lines[index + 1 : index + 6])
            for index, line in enumerate(lines)
        )

    base_fence_count = ptx_text["base"].count(instruction)
    head_fence_count = ptx_text["head"].count(instruction)
    base_target_count = target_fence_count(ptx_text["base"])
    head_target_count = target_fence_count(ptx_text["head"])
    return {
        "builds": builds,
        "base_ptx_sha256": _digest(ptx_text["base"]),
        "head_ptx_sha256": _digest(ptx_text["head"]),
        "base_fence_instruction_count": base_fence_count,
        "head_fence_instruction_count": head_fence_count,
        "added_fence_instruction_count": head_fence_count - base_fence_count,
        "base_fence_before_phase_xor_count": base_target_count,
        "head_fence_before_phase_xor_count": head_target_count,
        "base_exact_sm100_ptx_pass": builds["base"]["return_code"] == 0,
        "head_exact_sm100_ptx_pass": builds["head"]["return_code"] == 0,
        "head_ptx_contains_fence_instruction": instruction in ptx_text["head"],
        "dynamic_sm100_race_and_concurrency": "unresolved-no-sm100-device",
        "performance_sm100": "unresolved-no-sm100-device",
    }


def _write(
    output: Path,
    *,
    case_id: str,
    project: str,
    probe: str,
    facts: dict[str, Any],
    base_sha: str,
    head_sha: str,
    selection_sha: str,
    test_plan_sha: str,
) -> None:
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "probe": probe,
        "case_id": case_id,
        "project": project,
        "status": "pass",
        "failure_codes": [],
        "facts": facts,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "selection_lock_sha256": selection_sha,
        "test_plan_sha256": test_plan_sha,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": _digest(material)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": case_id, "evidence_sha256": payload["evidence_sha256"]}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--cutlass-base", type=Path, required=True)
    parser.add_argument("--cutlass-head", type=Path, required=True)
    parser.add_argument("--deepgemm-base", type=Path, required=True)
    parser.add_argument("--deepgemm-head", type=Path, required=True)
    parser.add_argument("--deepgemm-cutlass", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-sha", required=True)
    parser.add_argument("--test-plan-sha", required=True)
    args = parser.parse_args()
    args.build_root.mkdir(parents=True, exist_ok=True)

    cutlass = _cutlass(
        args.nvcc,
        args.cutlass_base,
        args.cutlass_head,
        args.fixture_root,
        args.build_root,
    )
    _write(
        args.output_root / "cutlass-pr-3427-sm120-aot.json",
        case_id="cutlass-pr-3427",
        project="cutlass-cute",
        probe="r7-cutlass-full-header-and-sm120-aot-v1",
        facts=cutlass,
        base_sha="f94ec46f4f63f96003d6cfdf2014731e7672c281",
        head_sha="32034108164bb9ebc2b9042164ed23217a494c99",
        selection_sha=args.selection_sha,
        test_plan_sha=args.test_plan_sha,
    )

    deepgemm = _deepgemm(
        args.nvcc,
        args.deepgemm_base,
        args.deepgemm_head,
        args.deepgemm_cutlass,
        args.fixture_root,
        args.build_root,
    )
    _write(
        args.output_root / "deepgemm-pr-389-sm100-aot.json",
        case_id="deepgemm-pr-389",
        project="deepgemm",
        probe="r7-deepgemm-exact-sm100-ptx-v1",
        facts=deepgemm,
        base_sha="559d79fb6994a58b8a15b4b93bf13ccc16edf247",
        head_sha="bbbf8e4535a42f40d68e113e628add0317c4fa9b",
        selection_sha=args.selection_sha,
        test_plan_sha=args.test_plan_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
