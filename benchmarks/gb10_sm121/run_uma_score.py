from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from infraswe.io import atomic_write_json
from infraswe.models.evidence import LoadCellEvidence, RequestSample
from infraswe.models.task import TaskPackage
from infraswe.scoring.deployability import (
    build_v04_score,
    score_cell_efficiency,
    score_concurrent_stability,
    score_kernel_reuse,
    score_maintainability,
)
from infraswe.telemetry.profiler_v04 import kernel_counter_evidence, system_trace_evidence

FRESH_REPLAYS = 7
FORMAL_REQUESTS = 1200
DEFAULT_ELEMENTS = 65536
STREAMS = 4
TENANTS = 4
MEMORY_GROWTH_BUDGET = 64 << 20
PROTOCOL_ID = "gb10-uma-load-normalized-v0.4-r1"
LOAD_RATIOS = {
    "light": 0.25,
    "normal": 0.50,
    "knee": 0.80,
    "saturation": 1.00,
    "overload": 1.20,
    "burst_or_soak": 1.00,
}
FORBIDDEN_PATTERNS = {
    "tcgen05": r"\btcgen05\.",
    "tensor_memory": r"\btmem\b",
    "cublas_fallback": r"\bcublas(?:lt)?\b",
    "cutlass_fallback": r"\bcutlass\b",
    "sm100_target": r"\bsm_100a?\b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def command(
    argv: list[str],
    *,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=merged_env,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_seconds": time.time() - started,
            "timed_out": False,
            "env_overrides": dict(env or {}),
        }
    except subprocess.TimeoutExpired as error:
        return {
            "argv": argv,
            "returncode": 124,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "duration_seconds": time.time() - started,
            "timed_out": True,
            "env_overrides": dict(env or {}),
        }
    except OSError as error:
        return {
            "argv": argv,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "duration_seconds": time.time() - started,
            "timed_out": False,
            "env_overrides": dict(env or {}),
        }


def parsed_json(record: dict[str, Any]) -> dict[str, Any] | None:
    for line in reversed(str(record.get("stdout", "")).splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def geometric_mean(values: list[float]) -> float:
    if not values or any(value < 0 for value in values):
        raise ValueError("geometric mean requires non-negative values")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def workload_argv(
    binary: Path,
    *,
    mode: str = "candidate",
    allocation: str = "system",
    touch: str = "cpu-first",
    requests: int = FORMAL_REQUESTS,
    elements: int = DEFAULT_ELEMENTS,
    streams: int = STREAMS,
    arrival_rate: float = 0.0,
    slo_seconds: float = 3600.0,
    replay_index: int = 1,
    regime: str = "normal",
    samples: Path | None = None,
    burst: bool = False,
    max_workspace_bytes: int = 1 << 30,
    required_runtime_version: int = 13000,
) -> list[str]:
    argv = [
        str(binary),
        "--mode",
        mode,
        "--allocation",
        allocation,
        "--touch",
        touch,
        "--requests",
        str(requests),
        "--elements",
        str(elements),
        "--streams",
        str(streams),
        "--tenants",
        str(TENANTS),
        "--arrival-rate",
        f"{arrival_rate:.12g}",
        "--slo-us",
        f"{slo_seconds * 1e6:.12g}",
        "--replay-index",
        str(replay_index),
        "--regime",
        regime,
        "--protocol-id",
        PROTOCOL_ID,
        "--max-workspace-bytes",
        str(max_workspace_bytes),
        "--required-runtime-version",
        str(required_runtime_version),
    ]
    if samples is not None:
        argv.extend(["--samples", str(samples)])
    if burst:
        argv.append("--burst")
    return argv


def write_task_contract(
    output_root: Path,
    *,
    protocol_digest: str,
    source_digest: str,
    benchmark_cell_id: str,
) -> None:
    payload = {
        "schema_version": "0.4",
        "task": {
            "id": "gb10-uma-cpu-gpu-pipeline-v04",
            "title": "GB10 UMA CPU/GPU production pipeline",
            "track": "gb10-uma",
            "repository": "infraswe",
            "base_commit": source_digest,
            "kind": "benchmark-replacement",
            "implementation_level": "integrated",
        },
        "environment": {
            "profile": "gpu-1x-sm121-gb10-cuda130",
            "agent_mode": "docker",
            "verifier_mode": "separate",
            "network": "deny",
            "gpu_count": 1,
            "exclusive_gpu_lease": True,
            "mps": "disabled",
        },
        "replay": {"count": FRESH_REPLAYS, "require_all": True},
        "semantic_contract": {
            "path": "scoring_protocol_v04.json",
            "sha256": protocol_digest,
        },
        "backend_profile": {
            "id": "gpu-1x-sm121-gb10-cuda130",
            "adapter": "cuda-gb10-uma",
            "benchmark_cell_id": benchmark_cell_id,
        },
        "certification": {
            "hidden_correctness_required": 1.0,
            "silent_fallback_rate_max": 0.0,
            "fresh_replays": FRESH_REPLAYS,
            "require_all": True,
        },
        "concurrency": {
            "protocol_id": PROTOCOL_ID,
            "reference_saturation_anchor": "local-reference-copy-pipeline",
            "load_ratios": [0.25, 0.50, 0.80, 1.00, 1.20],
            "minimum_completed_requests_per_cell": 1000,
            "burst_or_soak": "required",
            "request_mix_sha256": protocol_digest,
        },
        "reuse_contract": {
            "sha256": protocol_digest,
            "expected_variant_budget": 4,
            "max_variant_budget": 12,
            "specialization_dimensions": ["allocation-kind", "touch-order", "target-lane"],
            "require_case_to_implementation_map": True,
            "compile_cache_observability": "required-if-applicable",
        },
        "maintainability": {
            "probe_set_sha256": protocol_digest,
            "require_capability_contract": True,
            "require_structured_failure_codes": True,
            "build_profiles": ["sm_121-runtime", "sm_121f-build", "sm_121a-build"],
        },
        "efficiency": {
            "work_model_id": "gb10-uma-f32-transform-v1",
            "regime": "memory-bound",
            "work_model_confidence_min": "high",
            "calibration_manifest_sha256": protocol_digest,
            "traffic_amplification_budget": 1.10,
        },
        "evidence": {
            "minimum_grade_for_deployability": "E2-system-trace",
            "minimum_grade_for_cell_efficiency": "E3-kernel-counter",
            "collectors": ["runtime", "nsys-system-trace", "ncu-final-case"],
        },
        "scoring": {
            "deployability_template": "deployability-v0.4",
            "cell_artifact_template": "cell-artifact-memory-v0.4",
            "absolute_latency_global_ranking": "forbidden",
            "raw_peak_performance_in_cross_cell_score": False,
        },
    }
    TaskPackage.model_validate(payload)
    atomic_write_json(output_root / "pre-registration" / "task-contract.json", payload)


def build_artifacts(
    source: Path,
    calibration_source: Path,
    output_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    artifacts = output_root / "artifacts"
    artifacts.mkdir(parents=True)
    copied_source = artifacts / source.name
    copied_calibration = artifacts / calibration_source.name
    shutil.copy2(source, copied_source)
    shutil.copy2(calibration_source, copied_calibration)
    binary = artifacts / "uma-score-workload"
    calibration_binary = artifacts / "memory-calibration"
    common = ["nvcc", "-O3", "-std=c++20", "-Xcompiler=-Wall,-Wextra,-pthread"]
    specs = {
        "sm_121_executable": [*common, "-arch=sm_121", str(copied_source), "-o", str(binary)],
        "sm_121_ptx_a": [
            *common,
            "-arch=compute_121",
            "-ptx",
            str(copied_source),
            "-o",
            str(artifacts / "uma-score-a.ptx"),
        ],
        "sm_121_ptx_b": [
            *common,
            "-arch=compute_121",
            "-ptx",
            str(copied_source),
            "-o",
            str(artifacts / "uma-score-b.ptx"),
        ],
        "sm_121f_cubin": [
            *common,
            "-arch=sm_121f",
            "-cubin",
            str(copied_source),
            "-o",
            str(artifacts / "uma-score-sm121f.cubin"),
        ],
        "sm_121a_cubin": [
            *common,
            "-arch=sm_121a",
            "-cubin",
            str(copied_source),
            "-o",
            str(artifacts / "uma-score-sm121a.cubin"),
        ],
        "calibration": [
            *common,
            "-arch=sm_121",
            str(copied_calibration),
            "-o",
            str(calibration_binary),
        ],
    }
    builds = {name: command(argv, timeout=300) for name, argv in specs.items()}
    if any(record["returncode"] != 0 for record in builds.values()):
        atomic_write_json(output_root / "build-evidence.json", builds)
        failures = [name for name, record in builds.items() if record["returncode"] != 0]
        raise RuntimeError(f"build failures: {failures}")
    sass = command(["cuobjdump", "--dump-sass", str(binary)])
    (artifacts / "uma-score.sass.txt").write_text(sass["stdout"] + sass["stderr"], encoding="utf-8")
    readelf = command(["readelf", "-h", str(binary)])
    ldd = command(["ldd", str(binary)])
    builds["sass"] = sass
    builds["readelf"] = readelf
    builds["ldd"] = ldd
    atomic_write_json(output_root / "build-evidence.json", builds)
    return binary, calibration_binary, builds


def run_calibration(calibration_binary: Path, output_root: Path) -> dict[str, Any]:
    records = []
    for replay in range(1, 4):
        record = command(
            [str(calibration_binary), "--bytes", str(256 << 20), "--iterations", "40"],
            timeout=180,
        )
        record["replay_index"] = replay
        record["parsed"] = parsed_json(record)
        records.append(record)
    if not all(record["parsed"] and record["parsed"].get("passed") for record in records):
        raise RuntimeError("same-cell memory calibration failed")
    values = [record["parsed"] for record in records]
    summary = {
        "schema_version": "0.4",
        "calibration_id": "gb10-pageable-system-memory-v1",
        "replays": records,
        "memory_bandwidth_gbps": statistics.median(
            value["memory_bandwidth_gbps"] for value in values
        ),
        "launch_floor_us": statistics.median(value["launch_floor_us"] for value in values),
        "compute_tflops": 0.0,
    }
    atomic_write_json(output_root / "calibration.json", summary)
    return summary


def run_reference_calibration(binary: Path, output_root: Path) -> dict[str, Any]:
    reference_root = output_root / "reference-anchor"
    reference_root.mkdir()
    closed_runs = []
    for replay in range(1, 4):
        record = command(workload_argv(binary, mode="reference", requests=1200), timeout=180)
        value = parsed_json(record)
        closed_runs.append({"replay_index": replay, "command": record, "parsed": value})
    if not all(run["parsed"] and run["parsed"].get("passed") for run in closed_runs):
        raise RuntimeError("reference closed-loop calibration failed")
    closed_rate = statistics.median(float(run["parsed"]["throughput_rps"]) for run in closed_runs)
    low_runs = []
    for replay in range(1, 4):
        record = command(
            workload_argv(
                binary,
                mode="reference",
                requests=600,
                arrival_rate=0.20 * closed_rate,
            ),
            timeout=180,
        )
        value = parsed_json(record)
        low_runs.append({"replay_index": replay, "command": record, "parsed": value})
    if not all(run["parsed"] and run["parsed"].get("passed") for run in low_runs):
        raise RuntimeError("reference low-load calibration failed")
    slo_seconds = max(
        0.002,
        4.0 * statistics.median(float(run["parsed"]["p95_seconds"]) for run in low_runs),
    )
    sweep = []
    accepted_rates = []
    for ratio in (0.50, 0.70, 0.85, 1.00, 1.10):
        arrival_rate = ratio * closed_rate
        runs = []
        for replay in range(1, 4):
            record = command(
                workload_argv(
                    binary,
                    mode="reference",
                    requests=1200,
                    arrival_rate=arrival_rate,
                    slo_seconds=slo_seconds,
                ),
                timeout=180,
            )
            runs.append({"replay_index": replay, "command": record, "parsed": parsed_json(record)})
        accepted = all(
            run["parsed"]
            and run["parsed"].get("passed")
            and float(run["parsed"].get("slo_goodput_ratio", 0)) >= 0.99
            and int(run["parsed"].get("queue_depth_at_offer_end", 1200)) <= 24
            for run in runs
        )
        if accepted:
            accepted_rates.append(arrival_rate)
        sweep.append(
            {
                "closed_loop_ratio": ratio,
                "arrival_rate_rps": arrival_rate,
                "accepted": accepted,
                "replays": runs,
            }
        )
    if not accepted_rates:
        raise RuntimeError("reference saturation anchor sweep found no SLO-valid point")
    summary = {
        "schema_version": "0.4",
        "reference_implementation": "host-copy-private-device-pipeline",
        "anchor_replays": 3,
        "closed_loop": {
            "median_throughput_rps": closed_rate,
            "replays": closed_runs,
        },
        "low_load": {"replays": low_runs},
        "slo_seconds": slo_seconds,
        "saturation_anchor_rps": max(accepted_rates),
        "sweep": sweep,
        "anchor_frozen_before_candidate": True,
    }
    atomic_write_json(reference_root / "reference-saturation-anchor.json", summary)
    return summary


def validate_samples(path: Path, expected: int) -> str:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            RequestSample.model_validate_json(line)
            count += 1
    if count != expected:
        raise RuntimeError(f"request sample count {count} != expected {expected}")
    return sha256_file(path)


def formal_concurrency(
    binary: Path,
    output_root: Path,
    *,
    anchor_rps: float,
    slo_seconds: float,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    concurrency_root = output_root / "concurrency"
    concurrency_root.mkdir()
    by_regime: dict[str, list[dict[str, Any]]] = {regime: [] for regime in LOAD_RATIOS}
    reference_drift = []

    def run_reference_check(replay: int, position: str) -> None:
        cell_root = concurrency_root / f"replay-{replay:02d}" / "reference-saturation"
        cell_root.mkdir(parents=True)
        samples = cell_root / "request-samples.jsonl"
        record = command(
            workload_argv(
                binary,
                mode="reference",
                requests=FORMAL_REQUESTS,
                arrival_rate=anchor_rps,
                slo_seconds=slo_seconds,
                replay_index=replay,
                regime="saturation",
                samples=samples,
            ),
            timeout=240,
        )
        value = parsed_json(record)
        digest = validate_samples(samples, FORMAL_REQUESTS) if samples.exists() else None
        accepted = bool(
            value
            and value.get("passed")
            and float(value.get("slo_goodput_ratio", 0)) >= 0.99
            and int(value.get("queue_depth_at_offer_end", FORMAL_REQUESTS)) <= 24
        )
        evidence = {
            "replay_index": replay,
            "interleave_position": position,
            "arrival_rate_rps": anchor_rps,
            "accepted": accepted,
            "command": record,
            "parsed": value,
            "request_samples": str(samples.relative_to(output_root)),
            "request_samples_sha256": digest,
        }
        atomic_write_json(cell_root / "run.json", evidence)
        reference_drift.append(evidence)

    for replay in range(1, FRESH_REPLAYS + 1):
        if replay % 2:
            run_reference_check(replay, "before-candidate")
        regime_items = list(LOAD_RATIOS.items())
        if replay % 2 == 0:
            regime_items.reverse()
        for regime, ratio in regime_items:
            cell_root = concurrency_root / f"replay-{replay:02d}" / regime
            cell_root.mkdir(parents=True)
            samples = cell_root / "request-samples.jsonl"
            argv = workload_argv(
                binary,
                requests=FORMAL_REQUESTS,
                arrival_rate=anchor_rps * ratio,
                slo_seconds=slo_seconds,
                replay_index=replay,
                regime=regime,
                samples=samples,
                burst=regime == "burst_or_soak",
            )
            record = command(argv, timeout=240)
            value = parsed_json(record)
            digest = validate_samples(samples, FORMAL_REQUESTS) if samples.exists() else None
            evidence = {
                "replay_index": replay,
                "regime": regime,
                "load_ratio": ratio,
                "arrival_rate_rps": anchor_rps * ratio,
                "command": record,
                "parsed": value,
                "request_samples": str(samples.relative_to(output_root)),
                "request_samples_sha256": digest,
            }
            atomic_write_json(cell_root / "run.json", evidence)
            by_regime[regime].append(evidence)
        if replay % 2 == 0:
            run_reference_check(replay, "after-candidate")

    load_cells = []
    for regime, records in by_regime.items():
        values = [record["parsed"] for record in records if record["parsed"]]
        if len(values) != FRESH_REPLAYS:
            raise RuntimeError(f"{regime} is missing replay results")
        p95_values = [float(value["p95_seconds"]) for value in values]
        p95_mean = statistics.mean(p95_values)
        jitter_cv = statistics.stdev(p95_values) / p95_mean if p95_mean else 0.0
        tail_scores = [min(1.0, slo_seconds / float(value["p99_seconds"])) for value in values]
        resource_scores = [
            max(
                0.0,
                1.0 - max(0, int(value["rss_growth_bytes"])) / MEMORY_GROWTH_BUDGET,
            )
            for value in values
        ]
        ratio = LOAD_RATIOS[regime]
        queue_unbounded = any(
            ratio <= 1.0
            and int(value["queue_depth_at_offer_end"]) > max(32, FORMAL_REQUESTS // 10)
            and float(value["drain_seconds"]) > 10 * slo_seconds
            for value in values
        )
        cell = LoadCellEvidence(
            protocol_id=PROTOCOL_ID,
            regime=regime,  # type: ignore[arg-type]
            load_ratio=ratio,
            offered_requests=FORMAL_REQUESTS,
            completed_requests=min(int(value["completed_requests"]) for value in values),
            slo_goodput_ratio=statistics.mean(
                float(value["slo_goodput_ratio"]) for value in values
            ),
            error_drop_rate=max(float(value["error_drop_rate"]) for value in values),
            tail_score=geometric_mean(tail_scores),
            replay_jitter_score=1.0 / (1.0 + 4.0 * jitter_cv),
            resource_stability_score=min(resource_scores),
            fairness_score=geometric_mean([float(value["fairness_jain"]) for value in values]),
            p99_status="official",
            request_samples_sha256=sha256_json(
                [record["request_samples_sha256"] for record in records]
            ),
            deadlock=any(record["command"]["timed_out"] for record in records),
            livelock=any(int(value["completed_requests"]) < FORMAL_REQUESTS for value in values),
            queue_unbounded=queue_unbounded,
            memory_growth_limit_exceeded=any(
                int(value["rss_growth_bytes"]) > MEMORY_GROWTH_BUDGET for value in values
            ),
            silent_fallback=False,
            error_drop_rate_above_max=any(float(value["error_drop_rate"]) > 0 for value in values),
        )
        load_cells.append(cell.model_dump(mode="json"))
    atomic_write_json(concurrency_root / "load-cells.json", load_cells)
    atomic_write_json(concurrency_root / "reference-drift.json", reference_drift)
    return load_cells, by_regime, reference_drift


def run_case(
    binary: Path,
    *,
    allocation: str = "system",
    touch: str = "cpu-first",
    elements: int = DEFAULT_ELEMENTS,
    streams: int = STREAMS,
    env: dict[str, str] | None = None,
    max_workspace_bytes: int = 1 << 30,
    required_runtime_version: int = 13000,
) -> dict[str, Any]:
    record = command(
        workload_argv(
            binary,
            allocation=allocation,
            touch=touch,
            requests=64,
            elements=elements,
            streams=streams,
            max_workspace_bytes=max_workspace_bytes,
            required_runtime_version=required_runtime_version,
        ),
        timeout=120,
        env=env,
    )
    return {"command": record, "parsed": parsed_json(record)}


def score_reuse_and_maintenance(
    binary: Path,
    source: Path,
    output_root: Path,
    builds: dict[str, Any],
    by_regime: dict[str, list[dict[str, Any]]],
) -> tuple[Any, Any, dict[str, Any], bool, bool]:
    case_specs = {
        "system-cpu-first": {},
        "system-gpu-first": {"touch": "gpu-first"},
        "managed-cpu-first": {"allocation": "managed"},
        "managed-gpu-first": {"allocation": "managed", "touch": "gpu-first"},
        "pinned-cpu-first": {"allocation": "pinned"},
        "odd-tail-shape": {"elements": DEFAULT_ELEMENTS + 3},
        "single-stream": {"streams": 1},
        "eight-stream": {"streams": 8},
    }
    weights = {
        "system-cpu-first": 0.25,
        "system-gpu-first": 0.15,
        "managed-cpu-first": 0.15,
        "managed-gpu-first": 0.10,
        "pinned-cpu-first": 0.10,
        "odd-tail-shape": 0.10,
        "single-stream": 0.075,
        "eight-stream": 0.075,
    }
    cases = {name: run_case(binary, **kwargs) for name, kwargs in case_specs.items()}
    case_pass = {
        name: bool(value["parsed"] and value["parsed"].get("passed"))
        for name, value in cases.items()
    }
    coverage = sum(weights[name] for name, passed in case_pass.items() if passed)

    artifacts = output_root / "artifacts"
    ptx_text = (artifacts / "uma-score-a.ptx").read_text(encoding="utf-8")
    sass_text = (artifacts / "uma-score.sass.txt").read_text(encoding="utf-8")
    source_text = source.read_text(encoding="utf-8")
    scan_text = "\n".join((ptx_text, sass_text, source_text))
    forbidden_matches = {
        name: bool(re.search(pattern, scan_text, flags=re.IGNORECASE))
        for name, pattern in FORBIDDEN_PATTERNS.items()
    }
    fallback_clean = not any(forbidden_matches.values())
    entry_count = len(re.findall(r"\.entry\s+uma_transform_kernel\b", ptx_text))
    observed_variants = entry_count
    port_profiles = {
        "sm_121-runtime": bool(case_pass["system-cpu-first"]),
        "sm_121f-build": builds["sm_121f_cubin"]["returncode"] == 0,
        "sm_121a-build": builds["sm_121a_cubin"]["returncode"] == 0,
    }
    port_reuse = (
        0.50 * port_profiles["sm_121-runtime"]
        + 0.25 * port_profiles["sm_121f-build"]
        + 0.25 * port_profiles["sm_121a-build"]
    )
    compile_events = 0
    compile_reuse = 1.0 / (1.0 + compile_events)
    reuse_evidence = {
        "schema_version": "0.4",
        "semantic_implementation_family_id": "gb10-uma-transform-family-v1",
        "source_subtree_sha256": sha256_file(source),
        "generated_ir_sha256": [sha256_file(artifacts / "uma-score-a.ptx")],
        "binary_artifact_sha256": sha256_file(binary),
        "dispatcher_rule_digest": sha256_json(case_specs),
        "compile_cache_keys": [],
        "jit_compile_count": 0,
        "recompile_count": 0,
        "runtime_specialization_cache_size": 0,
        "observed_variants": observed_variants,
        "case_to_implementation_family": {
            name: "uma_transform_kernel" if passed else None for name, passed in case_pass.items()
        },
        "case_weights": weights,
        "coverage": coverage,
        "port_profiles": port_profiles,
        "port_reuse": port_reuse,
        "forbidden_matches": forbidden_matches,
        "silent_fallback_rate": 0.0 if fallback_clean else 1.0,
        "cases": cases,
    }
    atomic_write_json(output_root / "reuse-evidence.json", reuse_evidence)
    reuse_digest = sha256_file(output_root / "reuse-evidence.json")
    reuse = score_kernel_reuse(
        coverage=coverage,
        observed_variants=observed_variants,
        expected_variant_budget=4,
        max_variant_budget=12,
        compile_reuse=compile_reuse,
        port_reuse=port_reuse,
        silent_fallback_rate=0.0 if fallback_clean else 1.0,
        evidence_digests=[reuse_digest],
    )

    negative_specs = {
        "workspace-budget-rejection": {
            "kwargs": {"max_workspace_bytes": 4096},
            "code": "GB10_WORKSPACE_BUDGET_EXCEEDED",
        },
        "capability-off-rejection": {
            "kwargs": {"env": {"INFRASWE_FORCE_NO_PAGEABLE": "1"}},
            "code": "GB10_PAGEABLE_ACCESS_UNAVAILABLE",
        },
        "allocation-failure-rejection": {
            "kwargs": {"env": {"INFRASWE_FORCE_ALLOC_FAIL": "1"}},
            "code": "GB10_ALLOCATION_INJECTED_FAILURE",
        },
        "runtime-minor-rejection": {
            "kwargs": {"required_runtime_version": 13010},
            "code": "GB10_RUNTIME_VERSION_UNSUPPORTED",
        },
    }
    negative_probes = {}
    for name, spec in negative_specs.items():
        value = run_case(binary, **spec["kwargs"])
        value["expected_failure_code"] = spec["code"]
        value["passed"] = bool(
            value["parsed"]
            and not value["parsed"].get("passed", True)
            and value["parsed"].get("failure_code") == spec["code"]
        )
        negative_probes[name] = value

    maintenance_probe_pass = {
        "odd-tail-shape": case_pass["odd-tail-shape"],
        "managed-allocation": case_pass["managed-cpu-first"],
        "mapped-pinned-allocation": case_pass["pinned-cpu-first"],
        "gpu-first-system-memory": case_pass["system-gpu-first"],
        **{name: value["passed"] for name, value in negative_probes.items()},
    }
    lifecycle_pass = all(
        int(value["parsed"]["rss_growth_bytes"]) <= MEMORY_GROWTH_BUDGET
        for records in by_regime.values()
        for value in records
        if value["parsed"]
    )
    contract_checks = {
        "capability": negative_probes["capability-off-rejection"]["passed"],
        "runtime_version": negative_probes["runtime-minor-rejection"]["passed"],
        "workspace": negative_probes["workspace-budget-rejection"]["passed"],
        "fallback_policy": fallback_clean,
        "lifecycle": lifecycle_pass,
    }
    needed_libraries = [
        line
        for line in str(builds["ldd"]["stdout"]).splitlines()
        if "=>" in line or "ld-linux" in line
    ]
    locality_checks = {
        "single_semantic_entrypoint": entry_count == 1,
        "backend_subtree_confined": "platforms/nvidia-gb10/tasks/uma_pipeline" in str(source),
        "config_only_case_extensions": all(
            value["command"]["argv"][0] == str(binary) for value in cases.values()
        ),
        "dependency_fanout_within_budget": len(needed_libraries) <= 8,
    }
    warning_free = all(
        "warning" not in str(record["stderr"]).lower()
        for name, record in builds.items()
        if name in {"sm_121_executable", "sm_121_ptx_a", "sm_121f_cubin", "sm_121a_cubin"}
    )
    build_checks = {
        "sm_121_executable": builds["sm_121_executable"]["returncode"] == 0,
        "sm_121_ptx": builds["sm_121_ptx_a"]["returncode"] == 0,
        "sm_121f_cubin": builds["sm_121f_cubin"]["returncode"] == 0,
        "sm_121a_cubin": builds["sm_121a_cubin"]["returncode"] == 0,
        "warning_free": warning_free,
        "deterministic_ptx": sha256_file(artifacts / "uma-score-a.ptx")
        == sha256_file(artifacts / "uma-score-b.ptx"),
    }
    maintainability_evidence = {
        "schema_version": "0.4",
        "contract_checks": contract_checks,
        "locality_checks": locality_checks,
        "maintenance_probes": maintenance_probe_pass,
        "negative_probe_records": negative_probes,
        "build_checks": build_checks,
        "runtime_dependencies": needed_libraries,
    }
    atomic_write_json(output_root / "maintainability-evidence.json", maintainability_evidence)
    maintainability_digest = sha256_file(output_root / "maintainability-evidence.json")

    def fraction(values: dict[str, bool]) -> float:
        return sum(values.values()) / len(values)

    maintainability = score_maintainability(
        contract=fraction(contract_checks),
        locality=fraction(locality_checks),
        tests=fraction(maintenance_probe_pass),
        build=fraction(build_checks),
        evidence_digests=[maintainability_digest],
    )
    all_maintenance_passed = all(
        all(group.values())
        for group in (contract_checks, locality_checks, maintenance_probe_pass, build_checks)
    )
    return (
        reuse,
        maintainability,
        maintainability_evidence,
        all_maintenance_passed,
        fallback_clean,
    )


def capture_system_trace(
    binary: Path,
    output_root: Path,
    *,
    anchor_rps: float,
    slo_seconds: float,
) -> tuple[Any, bool]:
    trace_root = output_root / "profilers" / "system-trace"
    trace_root.mkdir(parents=True)
    prefix = trace_root / "uma-normal"
    profile = command(
        [
            "nsys",
            "profile",
            "--trace=cuda,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            f"--output={prefix}",
            *workload_argv(
                binary,
                requests=256,
                arrival_rate=0.50 * anchor_rps,
                slo_seconds=slo_seconds,
            ),
        ],
        timeout=300,
    )
    report = prefix.with_suffix(".nsys-rep")
    stats = (
        command(
            ["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum", str(report)],
            timeout=180,
        )
        if report.exists()
        else {
            "argv": [],
            "returncode": 127,
            "stdout": "",
            "stderr": "profile report missing",
            "duration_seconds": 0.0,
            "timed_out": False,
            "env_overrides": {},
        }
    )
    (trace_root / "stats.txt").write_text(stats["stdout"] + stats["stderr"], encoding="utf-8")
    stats_text = stats["stdout"] + stats["stderr"]
    atomic_write_json(trace_root / "commands.json", {"profile": profile, "stats": stats})
    value = parsed_json(profile)
    captured = bool(
        profile["returncode"] == 0
        and report.exists()
        and stats["returncode"] == 0
        and value
        and value.get("passed")
    )
    version_record = command(["nsys", "--version"])
    raw_paths = [report, trace_root / "stats.txt"] if report.exists() else []
    evidence = system_trace_evidence(
        {"generated_kernel_count": 1} if captured and "uma_transform_kernel" in stats_text else {},
        version=(version_record["stdout"] or version_record["stderr"]).strip(),
        raw_evidence=[str(path.relative_to(output_root)) for path in raw_paths],
        raw_evidence_digests=[sha256_file(path) for path in raw_paths],
    )
    atomic_write_json(trace_root / "normalized.json", evidence.model_dump(mode="json"))
    return evidence, captured


def parse_metric_number(value: str, unit: str) -> float | None:
    normalized = value.replace(",", "").replace("%", "").strip()
    try:
        number = float(normalized)
    except ValueError:
        return None
    scale = {
        "byte": 1.0,
        "kbyte": 1e3,
        "mbyte": 1e6,
        "gbyte": 1e9,
    }.get(unit.strip().lower(), 1.0)
    return number * scale


def parse_ncu_metrics(text: str) -> dict[str, float]:
    wanted = {
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    }
    metrics: dict[str, float] = {}
    for row in csv.reader(text.splitlines()):
        for index, item in enumerate(row):
            metric = item.strip()
            if metric not in wanted:
                continue
            unit = row[index + 1] if index + 1 < len(row) else ""
            value = row[index + 2] if index + 2 < len(row) else ""
            parsed = parse_metric_number(value, unit)
            if parsed is not None:
                metrics[metric] = parsed
    return metrics


def capture_kernel_counter(
    binary: Path,
    output_root: Path,
    *,
    efficiency_elements: int,
) -> tuple[Any, float | None, bool]:
    counter_root = output_root / "profilers" / "kernel-counter"
    counter_root.mkdir(parents=True)
    metrics = [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    profile = command(
        [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "function",
            "--kernel-name",
            "regex:uma_transform_kernel",
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "--metrics",
            ",".join(metrics),
            "--csv",
            *workload_argv(
                binary,
                requests=1,
                elements=efficiency_elements,
                streams=1,
            ),
        ],
        timeout=600,
    )
    raw_log = counter_root / "ncu.csv"
    raw_log.write_text(profile["stdout"] + profile["stderr"], encoding="utf-8")
    atomic_write_json(counter_root / "command.json", profile)
    parsed = parse_ncu_metrics(profile["stdout"] + "\n" + profile["stderr"])
    actual_bytes = None
    if "dram__bytes_read.sum" in parsed and "dram__bytes_write.sum" in parsed:
        actual_bytes = parsed["dram__bytes_read.sum"] + parsed["dram__bytes_write.sum"]
    captured = profile["returncode"] == 0 and actual_bytes is not None and actual_bytes > 0
    normalized = {
        "compute_throughput_pct": parsed.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "memory_throughput_pct": parsed.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "dram_bytes_read": parsed.get("dram__bytes_read.sum"),
        "dram_bytes_write": parsed.get("dram__bytes_write.sum"),
    }
    version_record = command(["ncu", "--version"])
    evidence = kernel_counter_evidence(
        normalized if captured else None,
        applicable=True,
        version=(version_record["stdout"] or version_record["stderr"]).strip(),
        raw_evidence=[str(raw_log.relative_to(output_root))] if raw_log.exists() else [],
        raw_evidence_digests=[sha256_file(raw_log)] if raw_log.exists() else [],
    )
    atomic_write_json(counter_root / "normalized.json", evidence.model_dump(mode="json"))
    return evidence, actual_bytes, captured


def efficiency_timing(
    binary: Path,
    output_root: Path,
    *,
    elements: int,
) -> tuple[float, list[dict[str, Any]]]:
    records = []
    for replay in range(1, FRESH_REPLAYS + 1):
        record = command(
            workload_argv(binary, requests=1, elements=elements, streams=1), timeout=180
        )
        record["replay_index"] = replay
        record["parsed"] = parsed_json(record)
        records.append(record)
    if not all(record["parsed"] and record["parsed"].get("passed") for record in records):
        raise RuntimeError("unprofiled efficiency timing failed")
    timing = statistics.median(float(record["parsed"]["p50_seconds"]) for record in records)
    atomic_write_json(
        output_root / "profilers" / "unprofiled-efficiency-timing.json",
        {"schema_version": "0.4", "median_seconds": timing, "replays": records},
    )
    return timing, records


def write_manifest(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path).removeprefix('sha256:')}  ./{path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the GB10 UMA pipeline under RFC v0.4")
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("output root must not already exist")
    args.output_root.mkdir(parents=True)
    (args.output_root / "pre-registration").mkdir()
    shutil.copy2(args.protocol, args.output_root / "pre-registration" / args.protocol.name)
    capability = json.loads(args.capability.read_text(encoding="utf-8"))
    atomic_write_json(args.output_root / "capability.json", capability)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["score_authority"] != "infraswe-scoring-v0.4":
        raise RuntimeError("scoring protocol authority mismatch")
    protocol_digest = sha256_file(args.protocol)
    source_digest = sha256_file(args.source)
    benchmark_cell_id = sha256_json(
        {
            "profile_id": capability["profile_id"],
            "capability_fingerprint": capability["capability_fingerprint"],
            "toolchain": capability["toolchain"]["detected_cuda_release"],
            "protocol": protocol_digest,
        }
    )
    write_task_contract(
        args.output_root,
        protocol_digest=protocol_digest,
        source_digest=source_digest,
        benchmark_cell_id=benchmark_cell_id,
    )
    binary, calibration_binary, builds = build_artifacts(
        args.source, args.calibration_source, args.output_root
    )
    calibration = run_calibration(calibration_binary, args.output_root)
    reference = run_reference_calibration(binary, args.output_root)
    load_cells, by_regime, reference_drift = formal_concurrency(
        binary,
        args.output_root,
        anchor_rps=float(reference["saturation_anchor_rps"]),
        slo_seconds=float(reference["slo_seconds"]),
    )
    concurrency_digest = sha256_file(args.output_root / "concurrency" / "load-cells.json")
    concurrency = score_concurrent_stability(
        load_cells,
        fresh_process_replays=FRESH_REPLAYS,
        evidence_digests=[concurrency_digest],
    )
    reuse, maintainability, _maintenance_evidence, maintenance_passed, fallback_clean = (
        score_reuse_and_maintenance(binary, args.source, args.output_root, builds, by_regime)
    )
    system_evidence, system_captured = capture_system_trace(
        binary,
        args.output_root,
        anchor_rps=float(reference["saturation_anchor_rps"]),
        slo_seconds=float(reference["slo_seconds"]),
    )
    efficiency_elements = 16 * 1024 * 1024
    candidate_time, _ = efficiency_timing(binary, args.output_root, elements=efficiency_elements)
    counter_evidence, actual_bytes, counter_captured = capture_kernel_counter(
        binary,
        args.output_root,
        efficiency_elements=efficiency_elements,
    )
    minimum_bytes = float(2 * efficiency_elements * 4)
    counter_usable = bool(counter_captured and actual_bytes and actual_bytes >= minimum_bytes)
    efficiency = score_cell_efficiency(
        work_model_id="gb10-uma-f32-transform-v1",
        regime="memory-bound",
        work_model={
            "minimum_external_bytes": minimum_bytes,
            "semantic_flops": float(2 * efficiency_elements),
        },
        calibration={
            "launch_floor_us": calibration["launch_floor_us"],
            "compute_tflops": 0.0,
            "memory_bandwidth_gbps": calibration["memory_bandwidth_gbps"],
        },
        candidate_time_seconds=candidate_time,
        actual_memory_bytes=actual_bytes if counter_usable else None,
        traffic_amplification_budget=1.10,
        counter_evidence_available=counter_usable,
        counter_confidence="high" if counter_usable else "low",
        evidence_digests=[
            sha256_file(args.output_root / "calibration.json"),
            *counter_evidence.raw_evidence_digests,
        ],
    )
    formal_runs_pass = all(
        record["parsed"] and record["parsed"].get("passed") and record["command"]["returncode"] == 0
        for records in by_regime.values()
        for record in records
    )
    coverage_complete = bool(
        reuse.raw_metrics and math.isclose(float(reuse.raw_metrics["components"]["coverage"]), 1.0)
    )
    reference_drift_passed = all(record["accepted"] for record in reference_drift)
    hard_gate_status = (
        "pass"
        if formal_runs_pass
        and coverage_complete
        and fallback_clean
        and system_captured
        and reference_drift_passed
        else "fail"
        if not formal_runs_pass or not coverage_complete or not fallback_clean
        else "unresolved"
    )
    evidence_grade = (
        "E3-kernel-counter"
        if counter_usable and system_captured
        else "E2-system-trace"
        if system_captured
        else "E0-runtime"
    )
    raw_metrics = {
        "task_scope": "gb10-uma-cpu-gpu-pipeline-v04-only",
        "gb10_minimum_release_scope": "partial; not represented by this task score",
        "reference_saturation_anchor_rps": reference["saturation_anchor_rps"],
        "slo_seconds": reference["slo_seconds"],
        "load_cells": {
            regime: {
                "arrival_rate_rps": float(reference["saturation_anchor_rps"]) * ratio,
                "p50_seconds": [record["parsed"]["p50_seconds"] for record in by_regime[regime]],
                "p95_seconds": [record["parsed"]["p95_seconds"] for record in by_regime[regime]],
                "p99_seconds": [record["parsed"]["p99_seconds"] for record in by_regime[regime]],
                "throughput_rps": [
                    record["parsed"]["throughput_rps"] for record in by_regime[regime]
                ],
            }
            for regime, ratio in LOAD_RATIOS.items()
        },
        "official_timing_source": "separate-unprofiled-run",
        "profiled_timing_authoritative": False,
        "absolute_latency_global_ranking": "forbidden",
        "request_samples_root": "concurrency/",
        "baseline_candidate_interleaved": True,
        "reference_drift": [
            {
                "replay_index": record["replay_index"],
                "interleave_position": record["interleave_position"],
                "accepted": record["accepted"],
                "slo_goodput_ratio": record["parsed"]["slo_goodput_ratio"]
                if record["parsed"]
                else None,
                "throughput_rps": record["parsed"]["throughput_rps"] if record["parsed"] else None,
            }
            for record in reference_drift
        ],
        "system_trace": system_evidence.model_dump(mode="json"),
        "kernel_counter": counter_evidence.model_dump(mode="json"),
    }
    failure_codes = []
    if not maintenance_passed:
        failure_codes.append("MAINTENANCE_PROBE_SET_INCOMPLETE")
    if not coverage_complete:
        failure_codes.append("CASE_PORTFOLIO_CORRECTNESS_INCOMPLETE")
    if not system_captured:
        failure_codes.append("E2_SYSTEM_TRACE_UNAVAILABLE")
    if not reference_drift_passed:
        failure_codes.append("REFERENCE_ANCHOR_DRIFT_GATE_UNRESOLVED")
    if not counter_usable:
        failure_codes.append("E3_COUNTER_UNAVAILABLE_OR_INCOHERENT")
    dimension_results = {
        "schema_version": "0.4",
        "concurrent_stability": {
            "component": concurrency.component.model_dump(mode="json"),
            "failure_codes": list(concurrency.failure_codes),
            "raw_metrics": dict(concurrency.raw_metrics or {}),
        },
        "kernel_reuse": {
            "component": reuse.component.model_dump(mode="json"),
            "failure_codes": list(reuse.failure_codes),
            "raw_metrics": dict(reuse.raw_metrics or {}),
        },
        "maintainability": {
            "component": maintainability.component.model_dump(mode="json"),
            "failure_codes": list(maintainability.failure_codes),
            "raw_metrics": dict(maintainability.raw_metrics or {}),
        },
    }
    atomic_write_json(args.output_root / "dimension-results.json", dimension_results)
    hard_gates = {
        "schema_version": "0.4",
        "status": hard_gate_status,
        "gates": {
            "correctness": {
                "passed": formal_runs_pass and coverage_complete,
                "formal_load_runs_passed": formal_runs_pass,
                "case_portfolio_coverage_complete": coverage_complete,
            },
            "fallback": {"passed": fallback_clean, "silent_fallback_rate": 0.0},
            "liveness": {
                "passed": formal_runs_pass,
                "fresh_process_replays": FRESH_REPLAYS,
            },
            "reference_drift": {
                "passed": reference_drift_passed,
                "interleaved_replays": len(reference_drift),
            },
            "evidence": {
                "passed": system_captured,
                "system_trace_status": system_evidence.status,
                "kernel_counter_status": counter_evidence.status,
            },
        },
        "failure_codes": failure_codes,
    }
    atomic_write_json(args.output_root / "hard-gates.json", hard_gates)
    score = build_v04_score(
        hard_gate_status=hard_gate_status,  # type: ignore[arg-type]
        benchmark_cell_id=benchmark_cell_id,
        evidence_grade=evidence_grade,  # type: ignore[arg-type]
        concurrent_stability=concurrency,
        kernel_reuse=reuse,
        maintainability=maintainability,
        cell_efficiency=efficiency,
        cell_artifact_template="cell-artifact-memory-v0.4",
        raw_metrics=raw_metrics,
        additional_failure_codes=failure_codes,
    )
    score_path = args.output_root / "score.json"
    atomic_write_json(score_path, score.model_dump(mode="json"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(json.loads(score_path.read_text(encoding="utf-8")))
    summary = {
        "schema_version": "0.4",
        "task_id": "gb10-uma-cpu-gpu-pipeline-v04",
        "scope_warning": "this is not the whole GB10 minimum-release track score",
        "hard_gate_status": hard_gate_status,
        "evidence_grade": evidence_grade,
        "benchmark_cell_id": benchmark_cell_id,
        "deployability_status": score.deployability.status if score.deployability else None,
        "deployability_100": score.deployability.score_100 if score.deployability else None,
        "components": {
            name: component.model_dump(mode="json")
            for name, component in (
                score.deployability.components if score.deployability else {}
            ).items()
        },
        "cell_efficiency_status": score.cell_efficiency.status if score.cell_efficiency else None,
        "cell_artifact_status": score.cell_artifact.status if score.cell_artifact else None,
        "cell_artifact_100": score.cell_artifact.score_100 if score.cell_artifact else None,
        "reference_saturation_anchor_rps": reference["saturation_anchor_rps"],
        "slo_seconds": reference["slo_seconds"],
        "fresh_process_replays": FRESH_REPLAYS,
        "reference_drift_replays": len(reference_drift),
        "reference_drift_passed": reference_drift_passed,
        "request_samples_per_load_cell_per_replay": FORMAL_REQUESTS,
        "failure_codes": score.failure_codes,
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    write_manifest(args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
