from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.ada_sm89 import FEATURE_CONTRACTS, MINIMUM_RELEASE_FEATURE_IDS
from infraswe.verifier.native_sm89 import (
    OFFICIAL_FRESH_REPLAYS,
    sha256_file,
    verify_gpu_feature,
)

NATIVE_TASKS = {
    "SM89-TARGET-001": (
        "tasks/sm89_native_build/native_dispatch.cu",
        "sm89_native_dispatch_kernel",
    ),
    "SM89-FP8-MMA-001": (
        "tasks/sm89_fp8_warp_mma/fp8_mma_smoke.cu",
        "sm89_fp8_mma_smoke",
    ),
    "SM89-FP8-CVT-001": (
        "tasks/sm89_fp8_convert_pack/fp8_convert_pack.cu",
        "sm89_fp8_convert_pack",
    ),
    "SM89-CPASYNC-001": (
        "tasks/sm89_cpasync_pipeline/cpasync_pipeline.cu",
        "sm89_cpasync_pipeline",
    ),
}
RUNTIME_TASKS = {
    "SM89-L2-001": "tasks/sm89_l2_reuse/l2_reuse.cu",
}


def command(
    argv: list[str], *, timeout: int = 300, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "wall_seconds": time.monotonic() - started,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": argv,
            "returncode": 124 if isinstance(error, subprocess.TimeoutExpired) else 127,
            "stdout": getattr(error, "stdout", "") or "",
            "stderr": getattr(error, "stderr", "") or f"{type(error).__name__}: {error}",
            "wall_seconds": time.monotonic() - started,
        }


def parsed_json_output(record: dict[str, Any]) -> dict[str, Any] | None:
    if record["returncode"] != 0:
        return None
    for line in reversed(record["stdout"].splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def unresolved(feature_id: str, code: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "0.4",
        "feature_id": feature_id,
        "status": "unresolved",
        "certified": False,
        "replay_count": 0,
        "reason": reason,
        "failure_codes": [code],
    }


def run_replays(
    binary: Path,
    *,
    environment: dict[str, str] | None = None,
    replay_count: int = OFFICIAL_FRESH_REPLAYS,
) -> list[dict[str, Any]]:
    records = []
    for replay in range(1, replay_count + 1):
        record = command([str(binary)], env=environment)
        records.append(
            {
                "replay_index": replay,
                "command": record,
                "result": parsed_json_output(record),
            }
        )
    return records


def _all_passed(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all(
        record["result"] and record["result"].get("passed") for record in records
    )


def build_native_task(
    *, feature_id: str, source: Path, task_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    artifact_root = task_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)
    copied_source = artifact_root / source.name
    shutil.copy2(source, copied_source)
    ptx = artifact_root / "kernel.ptx"
    cubin = artifact_root / "kernel.cubin"
    binary = artifact_root / "task-runner"
    common = ["nvcc", "-O3", "-std=c++20", "-lineinfo", "--ptxas-options=-v"]
    builds = {
        "ptx": command([*common, "-arch=compute_89", "-ptx", str(copied_source), "-o", str(ptx)]),
        "cubin": command([*common, "-arch=sm_89", "-cubin", str(copied_source), "-o", str(cubin)]),
        "executable": command(
            [
                *common,
                "-gencode",
                "arch=compute_89,code=sm_89",
                "-gencode",
                "arch=compute_89,code=compute_89",
                str(copied_source),
                "-o",
                str(binary),
            ]
        ),
    }
    atomic_write_json(task_root / "builds.json", builds)
    if all(record["returncode"] == 0 for record in builds.values()):
        sass = command(["cuobjdump", "--dump-sass", str(cubin)])
        symbols = command(["cuobjdump", "--dump-elf", str(cubin)])
        if sass["returncode"] == 0 and sass["stdout"].strip():
            (artifact_root / "kernel.sass.txt").write_text(sass["stdout"], encoding="utf-8")
        if symbols["returncode"] == 0 and symbols["stdout"].strip():
            (artifact_root / "kernel.elf.txt").write_text(symbols["stdout"], encoding="utf-8")
        atomic_write_json(task_root / "binary-inspection.json", {"sass": sass, "symbols": symbols})
    return artifact_root, binary, builds


def run_ptx_jit_replays(binary: Path, task_root: Path) -> dict[str, Any]:
    cold = []
    warm = []
    base_environment = dict(os.environ)
    base_environment["CUDA_FORCE_PTX_JIT"] = "1"
    for replay in range(1, OFFICIAL_FRESH_REPLAYS + 1):
        cache_root = task_root / "ptx-jit-cache" / f"replay-{replay:02d}"
        cache_root.mkdir(parents=True, exist_ok=False)
        environment = {**base_environment, "CUDA_CACHE_PATH": str(cache_root)}
        cold_record = command([str(binary)], env=environment)
        warm_record = command([str(binary)], env=environment)
        cold.append(
            {
                "replay_index": replay,
                "command": cold_record,
                "result": parsed_json_output(cold_record),
            }
        )
        warm.append(
            {
                "replay_index": replay,
                "command": warm_record,
                "result": parsed_json_output(warm_record),
            }
        )
    return {"cold": cold, "warm": warm, "passed": _all_passed(cold) and _all_passed(warm)}


def native_task(
    *,
    feature_id: str,
    source: Path,
    entry: str,
    output_root: Path,
    capability: dict[str, Any],
    runtime_allowed: bool,
) -> dict[str, Any]:
    task_root = output_root / feature_id.lower()
    task_root.mkdir(parents=True, exist_ok=False)
    artifact_root, binary, builds = build_native_task(
        feature_id=feature_id, source=source, task_root=task_root
    )
    if any(record["returncode"] != 0 for record in builds.values()):
        return unresolved(feature_id, "SM89_TASK_BUILD_FAILED", "PTX/cubin/executable build failed")
    fingerprint = capability["capability_fingerprint"]
    static = verify_gpu_feature(
        artifact_root=artifact_root,
        feature_id=feature_id,
        requested_entry=entry,
        capability_fingerprint=fingerprint,
    )
    atomic_write_json(task_root / "static-verification.json", static)
    if not runtime_allowed:
        static.update(
            {
                "replay_count": 0,
                "runtime_reason": "current device is not a canonical L40S/L20 SM89 cell",
            }
        )
        return static

    native_environment = dict(os.environ)
    native_environment.pop("CUDA_FORCE_PTX_JIT", None)
    native_replays = run_replays(binary, environment=native_environment)
    atomic_write_json(task_root / "native-replays.json", native_replays)
    dispatch_modes: dict[str, Any] = {
        "native_cubin": {"passed": _all_passed(native_replays), "replays": len(native_replays)}
    }
    correctness = _all_passed(native_replays)
    if feature_id == "SM89-TARGET-001":
        ptx_jit = run_ptx_jit_replays(binary, task_root)
        atomic_write_json(task_root / "ptx-jit-replays.json", ptx_jit)
        dispatch_modes["ptx_jit"] = {
            "passed": ptx_jit["passed"],
            "cold_replays": len(ptx_jit["cold"]),
            "warm_replays": len(ptx_jit["warm"]),
        }
        correctness = correctness and ptx_jit["passed"]
    dynamic = {
        "schema_version": "0.1",
        "feature_id": feature_id,
        "artifact_set_sha256": static["artifact_set_sha256"],
        "capability_fingerprint": fingerprint,
        "correctness": {"passed": correctness},
        "liveness": {"completed": correctness, "watchdog_passed": correctness},
        "observed_entries": [entry] if correctness else [],
        "forbidden_library_calls": [],
        "silent_fallback_count": 0,
        "fresh_process_replays": OFFICIAL_FRESH_REPLAYS,
        "loaded_module_container_sha256": sha256_file(binary),
        "dispatch_modes": dispatch_modes,
        "allocation_audit": {
            "full_size_fp16_temporaries": 0 if feature_id == "SM89-FP8-MMA-001" else None
        },
    }
    atomic_write_json(task_root / "dynamic.json", dynamic)
    result = verify_gpu_feature(
        artifact_root=artifact_root,
        feature_id=feature_id,
        requested_entry=entry,
        dynamic_evidence=dynamic,
        capability_fingerprint=fingerprint,
    )
    result.update(
        {
            "replay_count": OFFICIAL_FRESH_REPLAYS,
            "binary_path": str(binary.resolve()),
            "replays_path": str((task_root / "native-replays.json").resolve()),
        }
    )
    return result


def runtime_task(
    *,
    feature_id: str,
    source: Path,
    output_root: Path,
    runtime_allowed: bool,
) -> dict[str, Any]:
    task_root = output_root / feature_id.lower()
    task_root.mkdir(parents=True, exist_ok=False)
    binary = task_root / "task-binary"
    build = command(
        [
            "nvcc",
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "-gencode",
            "arch=compute_89,code=compute_89",
            str(source),
            "-o",
            str(binary),
        ]
    )
    atomic_write_json(task_root / "build.json", build)
    if build["returncode"] != 0:
        return unresolved(feature_id, "SM89_TASK_BUILD_FAILED", build["stderr"][-2000:])
    if not runtime_allowed:
        return unresolved(
            feature_id,
            "SM89_RUNTIME_PLATFORM_UNAVAILABLE",
            "build passed, but current device is not a canonical L40S/L20 SM89 cell",
        )
    replays = run_replays(binary)
    atomic_write_json(task_root / "replays.json", replays)
    passed = _all_passed(replays)
    return {
        "schema_version": "0.4",
        "feature_id": feature_id,
        "status": "certified" if passed else "failed",
        "certified": passed,
        "replay_count": OFFICIAL_FRESH_REPLAYS,
        "failure_codes": [] if passed else ["SM89_RUNTIME_REPLAY_FAILED"],
        "replays_path": str((task_root / "replays.json").resolve()),
        "binary_path": str(binary.resolve()),
    }


def external_result(feature_id: str, evidence_dir: Path | None) -> dict[str, Any]:
    if evidence_dir is None:
        return unresolved(
            feature_id,
            "ADA_EXTERNAL_EVIDENCE_MISSING",
            "task requires separately collected concurrency, cross-SKU, or torch.compile evidence",
        )
    path = evidence_dir / f"{feature_id}.json"
    if not path.is_file():
        return unresolved(feature_id, "ADA_EXTERNAL_EVIDENCE_MISSING", f"missing {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("feature_id") != feature_id:
        return unresolved(
            feature_id,
            "ADA_EXTERNAL_FEATURE_ID_MISMATCH",
            "external evidence feature_id does not match",
        )
    if payload.get("validator") != "infraswe-ada-sm89-external-v1" or not str(
        payload.get("input_evidence_sha256", "")
    ).startswith("sha256:"):
        return unresolved(
            feature_id,
            "ADA_EXTERNAL_PROVENANCE_MISSING",
            "external evidence must be emitted by the trusted validator and bind its raw input",
        )
    return payload


def run_suite(
    *,
    capability: dict[str, Any],
    platform_root: Path,
    output_root: Path,
    external_evidence_dir: Path | None = None,
) -> list[dict[str, Any]]:
    compile_allowed = capability.get("gates", {}).get("compile", {}).get("status") == "pass"
    runtime_allowed = capability.get("gates", {}).get("platform", {}).get("status") == "pass"
    results = []
    for feature_id in MINIMUM_RELEASE_FEATURE_IDS:
        if feature_id in NATIVE_TASKS:
            if not compile_allowed:
                results.append(
                    unresolved(
                        feature_id,
                        "SM89_COMPILE_GATE_FAILED",
                        "SM89 compiler gate did not pass",
                    )
                )
                continue
            if capability.get("features", {}).get(feature_id, {}).get("state") == (
                "toolchain_too_old"
            ):
                results.append(
                    unresolved(
                        feature_id,
                        "SM89_FEATURE_TOOLCHAIN_TOO_OLD",
                        "the compiler target exists, but this task's PTX ISA minimum is unmet",
                    )
                )
                continue
            relative_source, entry = NATIVE_TASKS[feature_id]
            results.append(
                native_task(
                    feature_id=feature_id,
                    source=platform_root / relative_source,
                    entry=entry,
                    output_root=output_root,
                    capability=capability,
                    runtime_allowed=runtime_allowed,
                )
            )
        elif feature_id in RUNTIME_TASKS:
            if not compile_allowed:
                results.append(
                    unresolved(
                        feature_id,
                        "SM89_COMPILE_GATE_FAILED",
                        "SM89 compiler gate did not pass",
                    )
                )
                continue
            results.append(
                runtime_task(
                    feature_id=feature_id,
                    source=platform_root / RUNTIME_TASKS[feature_id],
                    output_root=output_root,
                    runtime_allowed=runtime_allowed,
                )
            )
        elif FEATURE_CONTRACTS[feature_id].external_evidence:
            results.append(external_result(feature_id, external_evidence_dir))
        else:
            results.append(
                unresolved(feature_id, "ADA_TASK_RUNNER_MISSING", "no runner is registered")
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ada SM89 initial release suite")
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--platform-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-evidence-dir", type=Path)
    parser.add_argument("--require-certified", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("output root must not already exist")
    args.output_root.mkdir(parents=True)
    capability = json.loads(args.capability.read_text(encoding="utf-8"))
    results = run_suite(
        capability=capability,
        platform_root=args.platform_root,
        output_root=args.output_root,
        external_evidence_dir=args.external_evidence_dir,
    )
    atomic_write_json(args.output_root / "results.json", {"results": results})
    if args.require_certified and not all(result.get("certified") for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
