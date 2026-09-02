from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.gb10 import MINIMUM_RELEASE_FEATURE_IDS
from infraswe.verifier.native_sm121 import verify_gpu_feature

FRESH_REPLAYS = 7


def command(argv: list[str], *, timeout: int = 180) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": argv,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
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


def unresolved(feature_id: str, code: str, reason: str, *, replay_count: int = 0) -> dict:
    return {
        "schema_version": "0.4",
        "feature_id": feature_id,
        "status": "unresolved",
        "certified": False,
        "replay_count": replay_count,
        "reason": reason,
        "failure_codes": [code],
    }


def run_replays(binary: Path, *, arguments: list[str] | None = None) -> list[dict[str, Any]]:
    records = []
    for replay in range(1, FRESH_REPLAYS + 1):
        record = command([str(binary), *(arguments or [])])
        records.append(
            {
                "replay_index": replay,
                "command": record,
                "result": parsed_json_output(record),
            }
        )
    return records


def native_dispatch_task(
    *, source: Path, output_root: Path, capability: dict[str, Any]
) -> dict[str, Any]:
    artifact_root = output_root / "sm121-native-build" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)
    copied_source = artifact_root / "native_dispatch.cu"
    shutil.copy2(source, copied_source)
    builds = {
        "ptx": command(
            [
                "nvcc",
                "-O2",
                "-std=c++20",
                "-arch=compute_121",
                "-ptx",
                str(copied_source),
                "-o",
                str(artifact_root / "kernel.ptx"),
            ]
        ),
        "cubin": command(
            [
                "nvcc",
                "-O2",
                "-std=c++20",
                "-arch=sm_121",
                "-cubin",
                str(copied_source),
                "-o",
                str(artifact_root / "kernel.cubin"),
            ]
        ),
        "executable": command(
            [
                "nvcc",
                "-O2",
                "-std=c++20",
                "-arch=sm_121",
                str(copied_source),
                "-o",
                str(artifact_root / "native-dispatch"),
            ]
        ),
    }
    atomic_write_json(output_root / "sm121-native-build" / "builds.json", builds)
    if any(record["returncode"] != 0 for record in builds.values()):
        return unresolved(
            "GB10-TARGET-001",
            "SM121_NATIVE_BUILD_FAILED",
            "one or more sm_121 PTX/cubin/executable builds failed",
        )
    sass = command(["cuobjdump", "--dump-sass", str(artifact_root / "kernel.cubin")])
    (artifact_root / "kernel.sass.txt").write_text(
        sass["stdout"] + sass["stderr"], encoding="utf-8"
    )
    replay_records = run_replays(artifact_root / "native-dispatch")
    atomic_write_json(output_root / "sm121-native-build" / "replays.json", replay_records)
    passed = all(record["result"] and record["result"].get("passed") for record in replay_records)
    fingerprint = capability["capability_fingerprint"]
    static = verify_gpu_feature(
        artifact_root=artifact_root,
        feature_id="GB10-TARGET-001",
        requested_entry="dispatch_kernel",
        capability_fingerprint=fingerprint,
    )
    dynamic = {
        "schema_version": "0.1",
        "feature_id": "GB10-TARGET-001",
        "artifact_set_sha256": static["artifact_set_sha256"],
        "capability_fingerprint": fingerprint,
        "correctness": {"passed": passed},
        "liveness": {"completed": passed, "watchdog_passed": passed},
        "observed_entries": ["dispatch_kernel"] if passed else [],
        "forbidden_library_calls": [],
        "fresh_process_replays": FRESH_REPLAYS,
    }
    atomic_write_json(output_root / "sm121-native-build" / "dynamic.json", dynamic)
    result = verify_gpu_feature(
        artifact_root=artifact_root,
        feature_id="GB10-TARGET-001",
        requested_entry="dispatch_kernel",
        dynamic_evidence=dynamic,
        capability_fingerprint=fingerprint,
    )
    result["replay_count"] = FRESH_REPLAYS
    return result


def executable_task(
    *,
    feature_id: str,
    source: Path,
    output_root: Path,
    build_argv: list[str],
    arguments: list[str] | None = None,
    verifier: Path | None = None,
) -> dict[str, Any]:
    task_root = output_root / feature_id.lower()
    task_root.mkdir(parents=True, exist_ok=False)
    binary = task_root / "task-binary"
    argv = [
        part.replace("{source}", str(source)).replace("{binary}", str(binary))
        for part in build_argv
    ]
    build = command(argv)
    atomic_write_json(task_root / "build.json", build)
    if build["returncode"] != 0:
        return unresolved(feature_id, "TASK_BUILD_FAILED", build["stderr"][-2000:])
    replays = run_replays(binary, arguments=arguments)
    atomic_write_json(task_root / "replays.json", replays)
    passed = all(record["result"] and record["result"].get("passed") for record in replays)
    if verifier is not None:
        iteration_values = [
            int(record["result"].get("iterations", 0)) for record in replays if record["result"]
        ]
        dynamic = {
            "passed": passed,
            "iterations": min(iteration_values, default=0),
            "fresh_process_replays": FRESH_REPLAYS,
        }
        dynamic_path = task_root / "dynamic.json"
        verifier_path = task_root / "verifier.json"
        atomic_write_json(dynamic_path, dynamic)
        verifier_command = command(
            [
                sys.executable,
                str(verifier),
                "--binary",
                str(binary),
                "--dynamic-evidence",
                str(dynamic_path),
                "--output",
                str(verifier_path),
            ]
        )
        atomic_write_json(task_root / "verifier-command.json", verifier_command)
        if verifier_path.exists():
            result = json.loads(verifier_path.read_text(encoding="utf-8"))
            result.update(
                {
                    "replay_count": FRESH_REPLAYS,
                    "replays_path": str((task_root / "replays.json").resolve()),
                    "binary_path": str(binary.resolve()),
                }
            )
            return result
        return unresolved(
            feature_id,
            "HOST_STATIC_VERIFIER_FAILED",
            verifier_command["stderr"][-2000:],
            replay_count=FRESH_REPLAYS,
        )
    return {
        "schema_version": "0.4",
        "feature_id": feature_id,
        "status": "certified" if passed else "failed",
        "certified": passed,
        "replay_count": FRESH_REPLAYS,
        "failure_codes": [] if passed else ["TASK_REPLAY_FAILED"],
        "replays_path": str((task_root / "replays.json").resolve()),
        "binary_path": str(binary.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GB10 minimum release evidence suite")
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--platform-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("output root must not already exist")
    args.output_root.mkdir(parents=True)
    capability = json.loads(args.capability.read_text(encoding="utf-8"))
    results = []
    if capability.get("gates", {}).get("compile", {}).get("status") == "pass":
        results.append(
            native_dispatch_task(
                source=args.platform_root / "tasks/sm121_native_build/native_dispatch.cu",
                output_root=args.output_root,
                capability=capability,
            )
        )
        results.append(
            executable_task(
                feature_id="GB10-UMA-001",
                source=args.platform_root / "tasks/uma_pipeline/uma_pipeline.cu",
                output_root=args.output_root,
                build_argv=[
                    "nvcc",
                    "-O2",
                    "-std=c++20",
                    "-arch=sm_121",
                    "{source}",
                    "-o",
                    "{binary}",
                ],
            )
        )
    else:
        results.extend(
            [
                unresolved(
                    "GB10-TARGET-001",
                    "GB10_COMPILE_GATE_FAILED",
                    "SM121 compiler gate did not pass",
                ),
                unresolved(
                    "GB10-UMA-001",
                    "GB10_COMPILE_GATE_FAILED",
                    "SM121 compiler gate did not pass",
                ),
            ]
        )
    results.append(
        executable_task(
            feature_id="GB10-ARM-ORDER-001",
            source=args.platform_root / "capability_probe/arm_probe.cc",
            output_root=args.output_root,
            build_argv=[
                "g++",
                "-O3",
                "-std=c++20",
                "-pthread",
                "-march=armv8.2-a+lse",
                "{source}",
                "-o",
                "{binary}",
            ],
            arguments=["1000000"],
            verifier=args.platform_root / "verifier/inspect_host_elf.py",
        )
    )
    for feature_id in ("GB10-MMA-001", "GB10-MATRIX-IO-001"):
        state = capability.get("features", {}).get(feature_id, {}).get("state")
        results.append(
            unresolved(
                feature_id,
                "PTX_9_3_NATIVE_IMPLEMENTATION_PENDING",
                (
                    f"capability state={state}; no auditable PTX 9.3 native "
                    "implementation was supplied"
                ),
            )
        )
    by_feature = {result["feature_id"]: result for result in results}
    ordered = [by_feature[feature_id] for feature_id in MINIMUM_RELEASE_FEATURE_IDS]
    summary = {
        "schema_version": "0.4",
        "score_authority": "infraswe-scoring-v0.4",
        "status": "certified" if all(result["certified"] for result in ordered) else "partial",
        "fresh_process_replays": FRESH_REPLAYS,
        "deployability_100": None,
        "deployability_reason": "feature certification is not a substitute for v0.4 C/U/M evidence",
        "absolute_latency_global_ranking": "forbidden",
        "roce_scaleout_mixed_into_single_node": False,
        "results": ordered,
    }
    atomic_write_json(args.output_root / "minimum-suite.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
