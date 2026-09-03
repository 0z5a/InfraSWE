from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.kernel.scoring import evaluate_anchor_case
from infraswe.kernel.statistics import percentile, summarize_samples

REQUIRED_REPLAYS = (1, 2, 3)
FORMULA_VERSION = "kernel-artifact-v0.3"
FORMULA_ORIGIN = "sol-execbench-equivalent"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _legacy_hardware_class(hardware: Mapping[str, Any]) -> dict[str, Any]:
    smi = hardware.get("nvidia_smi") or {}
    return {
        "gpu_name": hardware.get("gpu_name"),
        "compute_capability": hardware.get("compute_capability"),
        "sm_count": hardware.get("sm_count"),
        "total_memory_bytes": hardware.get("total_memory_bytes"),
        "driver_version": smi.get("driver_version"),
        "torch_version": hardware.get("torch_version"),
        "torch_cuda": hardware.get("torch_cuda"),
        "cudnn_version": hardware.get("cudnn_version"),
    }


def _has_generic_hardware_identity(hardware: Mapping[str, Any]) -> bool:
    return any(
        key in hardware
        for key in (
            "accelerator_vendor",
            "architecture",
            "runtime",
            "runtime_version",
            "torch_hip",
            "rocm_smi",
        )
    )


def hardware_class(hardware: Mapping[str, Any]) -> dict[str, Any]:
    legacy = _legacy_hardware_class(hardware)
    if not _has_generic_hardware_identity(hardware):
        return legacy
    torch_hip = hardware.get("torch_hip")
    vendor = hardware.get("accelerator_vendor") or (
        "amd" if torch_hip or hardware.get("rocm_smi") else "nvidia"
    )
    architecture = hardware.get("architecture")
    if not architecture and vendor == "nvidia" and legacy["compute_capability"]:
        architecture = "sm" + str(legacy["compute_capability"]).replace(".", "")
    if not architecture and vendor == "amd":
        architecture = hardware.get("gcn_architecture")
    rocm_smi = hardware.get("rocm_smi") or {}
    return {
        "accelerator_vendor": vendor,
        "architecture": architecture,
        "runtime": hardware.get("runtime") or ("rocm" if vendor == "amd" else "cuda"),
        "runtime_version": hardware.get("runtime_version")
        or (torch_hip if vendor == "amd" else hardware.get("torch_cuda")),
        **legacy,
        "compute_unit_count": hardware.get("compute_unit_count", hardware.get("sm_count")),
        "driver_version": hardware.get("driver_version")
        or rocm_smi.get("driver_version")
        or legacy["driver_version"],
        "torch_hip": torch_hip,
    }


def hardware_cell_id(hardware: Mapping[str, Any]) -> str:
    if not _has_generic_hardware_identity(hardware):
        stable = _legacy_hardware_class(hardware)
        digest = canonical_sha256(stable).removeprefix("sha256:")[:12]
        gpu = _slug(str(stable["gpu_name"]))
        architecture = str(stable["compute_capability"]).replace(".", "")
        return f"{gpu}-sm{architecture}-{digest}"
    stable = hardware_class(hardware)
    digest = canonical_sha256(stable).removeprefix("sha256:")[:12]
    gpu = _slug(str(stable["gpu_name"]))
    vendor = _slug(str(stable["accelerator_vendor"] or "unknown"))
    architecture = _slug(str(stable["architecture"] or "unknown-architecture"))
    return f"{vendor}-{gpu}-{architecture}-{digest}"


def calibrated_anchor_us(
    work_model: Mapping[str, Any], calibration: Mapping[str, float]
) -> dict[str, Any]:
    flops = float(work_model["semantic_flops"])
    byte_count = float(work_model["minimum_external_bytes"])
    compute_us = flops / (calibration["bf16_matmul_tflops"] * 1e6)
    bandwidth_us = byte_count / (calibration["hbm_bandwidth_gbps"] * 1e3)
    launch_us = calibration["launch_floor_us"]
    components = {
        "compute_us": compute_us,
        "bandwidth_us": bandwidth_us,
        "launch_us": launch_us,
    }
    limiting_component = max(components, key=components.__getitem__)
    return {
        "latency_us": components[limiting_component],
        "limiting_component": limiting_component,
        "components_us": components,
        "kind": "calibrated-target",
        "confidence": "medium",
    }


def summarize_calibrations(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indices = sorted(int(payload["replay_index"]) for payload in payloads)
    if indices != list(REQUIRED_REPLAYS):
        raise ValueError(f"calibration replay set must be {REQUIRED_REPLAYS}, got {indices}")
    cells = {hardware_cell_id(payload["hardware"]) for payload in payloads}
    if len(cells) != 1:
        raise ValueError("calibration hardware identity drift")

    def medians(field: str) -> list[float]:
        return [float(payload[field]["median"]) for payload in payloads]

    summary = {
        "launch_floor_us": statistics.median(medians("launch_floor_us")),
        "hbm_bandwidth_gbps": statistics.median(medians("hbm_bandwidth_gbps")),
        "bf16_matmul_tflops": statistics.median(medians("bf16_matmul_tflops")),
    }
    return {
        **summary,
        "replay_count": len(payloads),
        "replay_indices": indices,
        "per_replay": {
            "launch_floor_us": medians("launch_floor_us"),
            "hbm_bandwidth_gbps": medians("hbm_bandwidth_gbps"),
            "bf16_matmul_tflops": medians("bf16_matmul_tflops"),
        },
        "anchor_kind": "calibrated-target",
        "confidence": "medium",
        "known_omissions": sorted(
            {item for payload in payloads for item in payload.get("known_omissions", [])}
        ),
    }


def _hierarchical_speedup_ci(
    replay_pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    logs_by_replay = [
        [math.log(reference / candidate) for reference, candidate in zip(refs, cands, strict=True)]
        for refs, cands in replay_pairs
    ]
    if not logs_by_replay or any(not values for values in logs_by_replay):
        raise ValueError("hierarchical bootstrap requires non-empty replay blocks")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        replay_estimates = []
        for _ in logs_by_replay:
            cluster = logs_by_replay[generator.randrange(len(logs_by_replay))]
            sample = [cluster[generator.randrange(len(cluster))] for _ in cluster]
            replay_estimates.append(statistics.median(sample))
        estimates.append(math.exp(statistics.median(replay_estimates)))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def _native_trace_passed(
    backend: str,
    cases: Sequence[Mapping[str, Any]],
    supplemental_profiles: Sequence[Mapping[str, Any]],
) -> bool:
    def event_names(profiler: Mapping[str, Any]) -> list[str]:
        events = profiler.get("device_events")
        if events is None:
            events = profiler.get("cuda_events", [])
        return [str(event.get("name", "")).lower() for event in events]

    names_by_case: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        names_by_case[str(case["case_id"])].extend(event_names(case.get("profiler", {})))
    for profile in supplemental_profiles:
        if profile.get("status") != "passed":
            continue
        names_by_case[str(profile["case_id"])].extend(event_names(profile.get("profiler", {})))
    if any(not names for names in names_by_case.values()):
        return False
    if backend == "torch-sdpa-aotriton":
        native_tokens = ("aotriton", "attn_fwd", "fmha_fwd", "flash")
        return all(
            any(
                not name.startswith("aten::")
                and any(token in name for token in native_tokens)
                for name in names
            )
            for names in names_by_case.values()
        )
    if backend.startswith("fa") or backend.startswith("torch-sdpa"):
        return all(
            any("flash" in name or "cudnn" in name for name in names)
            for names in names_by_case.values()
        )
    return all(
        any(
            name not in {"activity buffer request"} and not name.startswith("aten::")
            for name in names
        )
        for names in names_by_case.values()
    )


def _score_case(
    case_payloads: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    replay_pairs: list[tuple[list[float], list[float]]] = []
    replay_summaries = []
    for case in case_payloads:
        blocks = case["measurement"]["blocks"]
        references = [float(block["reference_latency_us"]) for block in blocks]
        candidates = [float(block["candidate_latency_us"]) for block in blocks]
        replay_pairs.append((references, candidates))
        logs = [math.log(ref / cand) for ref, cand in zip(references, candidates, strict=True)]
        replay_summaries.append(
            {
                "replay_index": case["_replay_index"],
                "reference_latency_us": summarize_samples(references),
                "candidate_latency_us": summarize_samples(candidates),
                "paired_speedup": math.exp(statistics.median(logs)),
            }
        )

    references = [value for pair in replay_pairs for value in pair[0]]
    candidates = [value for pair in replay_pairs for value in pair[1]]
    logs = [math.log(ref / cand) for ref, cand in zip(references, candidates, strict=True)]
    speedup = math.exp(statistics.median(logs))
    ci_low, ci_high = _hierarchical_speedup_ci(replay_pairs, seed=seed)
    reference_latency = statistics.median(
        item["reference_latency_us"]["median"] for item in replay_summaries
    )
    candidate_latency = statistics.median(
        item["candidate_latency_us"]["median"] for item in replay_summaries
    )
    anchor = calibrated_anchor_us(case_payloads[0]["work_model"], calibration)
    anchor_result = evaluate_anchor_case(
        baseline_latency=reference_latency,
        candidate_latency=candidate_latency,
        anchor_latency=float(anchor["latency_us"]),
        min_headroom=1.10,
        beyond_anchor_tolerance=0.03,
    ).model_dump(mode="json")
    if (
        anchor_result["anchor_score_raw"] is not None
        and not 0 <= anchor_result["anchor_score_raw"] <= 1
    ):
        anchor_result["status"] = "quarantined"
        anchor_result["failure_codes"].append("MEASUREMENT_ANCHOR_SCORE_RANGE")

    return {
        "case_id": case_payloads[0]["case_id"],
        "case_group": case_payloads[0].get("case_group", "micro"),
        "weight": float(case_payloads[0].get("weight", 1.0)),
        "shape": case_payloads[0].get("shape", {}),
        "dtype": case_payloads[0].get("dtype"),
        "work_model": case_payloads[0]["work_model"],
        "correctness_passed": all(
            bool(case["correctness"].get("passed"))
            and bool(case["correctness"].get("dynamic_input_changes_output", True))
            for case in case_payloads
        ),
        "correctness": [case["correctness"] for case in case_payloads],
        "reference_latency_us": summarize_samples(references),
        "candidate_latency_us": summarize_samples(candidates),
        "paired_speedup": speedup,
        "paired_speedup_ci95": [ci_low, ci_high],
        "anchor": anchor,
        "anchor_result": anchor_result,
        "replays": replay_summaries,
    }


def _validate_group(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    indices = sorted(int(run["replay_index"]) for run in runs)
    if indices != list(REQUIRED_REPLAYS):
        failures.append(f"EVIDENCE_REPLAY_SET:{indices}")
    if any(run.get("status") != "passed" for run in runs):
        failures.append("EXECUTION_REPLAY_FAILED")
    if any(not run.get("all_correct", False) for run in runs):
        failures.append("CORRECTNESS_MANDATORY_FAILED")
    if len({hardware_cell_id(run["hardware"]) for run in runs}) != 1:
        failures.append("EVIDENCE_HARDWARE_IDENTITY_DRIFT")
    case_sets = [{case["case_id"] for case in run.get("cases", [])} for run in runs]
    if len({tuple(sorted(case_ids)) for case_ids in case_sets}) != 1:
        failures.append("EVIDENCE_CASE_SET_DRIFT")
    provenance = {
        tuple(item.get("sha256", "") for item in run.get("implementation_provenance", []))
        for run in runs
    }
    if len(provenance) != 1:
        failures.append("EVIDENCE_IMPLEMENTATION_IDENTITY_DRIFT")
    return failures


def score_run_group(
    runs: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, float],
    supplemental_profiles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    runs = sorted(runs, key=lambda run: int(run["replay_index"]))
    backend = str(runs[0]["backend"])
    failures = _validate_group(runs)
    cases_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for case in run.get("cases", []):
            enriched = dict(case)
            enriched["_replay_index"] = int(run["replay_index"])
            cases_by_id[case["case_id"]].append(enriched)
    scored_cases = [
        _score_case(cases, calibration, seed=index + 1000)
        for index, (_, cases) in enumerate(sorted(cases_by_id.items()))
        if len(cases) == len(REQUIRED_REPLAYS)
    ]
    if len(scored_cases) != len(cases_by_id):
        failures.append("EVIDENCE_CASE_REPLAY_MISSING")
    if scored_cases and not _native_trace_passed(
        backend,
        [case for run in runs for case in run["cases"]],
        supplemental_profiles,
    ):
        failures.append("FALLBACK_NATIVE_TRACE_MISSING")

    result: dict[str, Any] = {
        "backend": backend,
        "backend_version": runs[0].get("backend_version"),
        "implementation_commit": runs[0].get("implementation_commit"),
        "implementation_mechanism": runs[0].get("implementation_mechanism"),
        "benchmark": runs[0]["benchmark"],
        "benchmark_kind": runs[0]["benchmark_kind"],
        "replay_count": len(runs),
        "replay_indices": [int(run["replay_index"]) for run in runs],
        "implementation_provenance": runs[0].get("implementation_provenance", []),
        "case_results": scored_cases,
        "supplemental_profile_count": len(supplemental_profiles),
        "failure_codes": sorted(set(failures)),
    }
    if failures:
        result.update(
            certified=False,
            verdict="fail"
            if any(code.startswith("CORRECTNESS") for code in failures)
            else "unresolved",
            disposition="valid"
            if any(code.startswith("CORRECTNESS") for code in failures)
            else "invalid",
            artifact_status="not_applicable",
            artifact_100=None,
            leaderboard_effective_artifact_100=0.0
            if any(code.startswith("CORRECTNESS") for code in failures)
            else None,
        )
        return result

    if runs[0]["benchmark_kind"] == "kernel-library":
        anchor_statuses = {case["anchor_result"]["status"] for case in scored_cases}
        if "quarantined" in anchor_statuses:
            result.update(
                certified=True,
                verdict="unresolved",
                disposition="quarantined",
                artifact_status="quarantined",
                artifact_100=None,
                leaderboard_effective_artifact_100=None,
            )
            return result
        if "not_frontier_eligible" in anchor_statuses:
            result.update(
                certified=True,
                verdict="pass",
                disposition="valid",
                artifact_status="not_frontier_eligible",
                artifact_100=None,
                leaderboard_effective_artifact_100=None,
            )
            return result
        scores = {
            case["case_id"]: float(case["anchor_result"]["anchor_score_raw"])
            for case in scored_cases
        }
        weights = {case["case_id"]: float(case["weight"]) for case in scored_cases}
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("attention case weights must sum to one")
        portfolio = sum(scores[case_id] * weights[case_id] for case_id in scores)

        public = [case for case in scored_cases if case["case_group"] == "common"]
        hidden = [case for case in scored_cases if case["case_group"] != "common"]

        def normalized_portfolio(items: Sequence[Mapping[str, Any]]) -> float:
            total = sum(float(item["weight"]) for item in items)
            return (
                sum(
                    float(item["weight"]) * float(item["anchor_result"]["anchor_score_raw"])
                    for item in items
                )
                / total
            )

        public_score = normalized_portfolio(public)
        hidden_score = normalized_portfolio(hidden)
        hidden_values = [float(case["anchor_result"]["anchor_score_raw"]) for case in hidden]
        retention = min(1.0, hidden_score / max(public_score, 1e-12))
        tail = min(
            1.0, percentile(hidden_values, 0.10) / max(statistics.median(hidden_values), 1e-12)
        )
        fallback_footprint = 1.0
        generalization = 0.50 * retention + 0.30 * tail + 0.20 * fallback_footprint
        artifact = 100 * (0.80 * portfolio + 0.20 * generalization)
        raw_speedup = math.exp(
            sum(
                weights[case_id]
                * math.log(
                    next(
                        case["paired_speedup"]
                        for case in scored_cases
                        if case["case_id"] == case_id
                    )
                )
                for case_id in scores
            )
        )
        result.update(
            components={"performance_anchor_score": portfolio, "generalization": generalization},
            component_details={
                "public_portfolio": public_score,
                "hidden_portfolio": hidden_score,
                "hidden_public_retention": retention,
                "hidden_tail": tail,
                "fallback_footprint": fallback_footprint,
                "fallback_weight": 0.0,
            },
            weighted_geometric_speedup_raw=raw_speedup,
            certified=True,
            verdict="pass",
            disposition="valid",
            artifact_status="scored",
            artifact_100=artifact,
            leaderboard_effective_artifact_100=artifact,
        )
    else:
        micro_scores = {
            case["case_id"]: (
                100 * float(case["anchor_result"]["anchor_score_raw"])
                if case["anchor_result"]["status"] == "scored"
                else None
            )
            for case in scored_cases
        }
        micro_statuses = {case["case_id"]: case["anchor_result"]["status"] for case in scored_cases}
        micro_failures = sorted(
            {
                f"{case['case_id']}:{code}"
                for case in scored_cases
                for code in case["anchor_result"]["failure_codes"]
            }
        )
        all_scored = all(status == "scored" for status in micro_statuses.values())
        result.update(
            certified=True,
            verdict="pass",
            disposition="valid",
            artifact_status="scored-per-case" if all_scored else "mixed-per-case",
            artifact_100=None,
            leaderboard_effective_artifact_100=None,
            micro_artifact_100=micro_scores,
            micro_artifact_status=micro_statuses,
            failure_codes=micro_failures,
        )
    return result


def eligibility_for_cell(
    compute_capability: str | None,
    *,
    accelerator_vendor: str | None = None,
    architecture: str | None = None,
    runtime_version: str | None = None,
    torch_version: str | None = None,
) -> dict[str, dict[str, str]]:
    if accelerator_vendor == "amd" or (architecture or "").startswith("gfx"):
        matrix = {
            "fa1": {
                "status": "not_applicable",
                "reason": "frozen FA1 artifact is a CUDA extension and has no gfx942 target",
            },
            "fa2": {
                "status": "not_applicable",
                "reason": (
                    "frozen upstream FA2 artifact in this suite is the CUDA build, not a ROCm fork"
                ),
            },
            "fa3": {
                "status": "not_applicable",
                "reason": "frozen FA3 artifact is the NVIDIA Hopper CUDA path",
            },
            "fa4": {
                "status": "not_applicable",
                "reason": "frozen flash-attn-4 CuTeDSL artifact has no ROCm/gfx942 target",
            },
        }
        exact_stack = (
            architecture == "gfx942"
            and str(runtime_version or "").startswith("6.1")
            and str(torch_version or "").split("+")[0] == "2.4.0"
        )
        if exact_stack:
            matrix.update(
                {
                    "torch-sdpa-aotriton": {
                        "status": "eligible",
                        "reason": (
                            "PyTorch 2.4.0 ROCm 6.1 Flash SDPA adapter on gfx942; "
                            "certification still requires native trace and three replays"
                        ),
                    },
                    "triton-gfx942-initial": {
                        "status": "eligible",
                        "reason": (
                            "portable Triton micro-kernel adapter registered for the exact "
                            "gfx942/ROCm 6.1 cell"
                        ),
                    },
                }
            )
        else:
            reason = "exact PyTorch 2.4.0 + ROCm 6.1 + gfx942 contract was not observed"
            matrix.update(
                {
                    "torch-sdpa-aotriton": {"status": "unresolved", "reason": reason},
                    "triton-gfx942-initial": {"status": "unresolved", "reason": reason},
                }
            )
        return matrix
    try:
        major = int(str(compute_capability).split(".")[0])
    except ValueError:
        major = -1
    if major == 8:
        return {
            "fa1": {"status": "eligible", "reason": "official SM80 path"},
            "fa2": {"status": "eligible", "reason": "official SM80 path"},
            "fa3": {"status": "eligible", "reason": "current FA3 tree includes SM80 path"},
            "fa4": {
                "status": "eligible",
                "reason": "flash-attn-4 4.0.0b28 includes an explicit SM80 CuTeDSL path",
            },
        }
    if major == 9:
        return {
            "fa1": {
                "status": "eligible",
                "reason": "frozen FA1 source rebuild supports an explicit native SM90 target",
            },
            "fa2": {
                "status": "eligible",
                "reason": "frozen FA2 source rebuild supports the Hopper SM90 target",
            },
            "fa3": {
                "status": "eligible",
                "reason": "official FA3 Hopper path targets SM90",
            },
            "fa4": {
                "status": "eligible",
                "reason": "flash-attn-4 4.0.0b28 includes an explicit SM90 CuTeDSL path",
            },
        }
    if major == 12:
        return {
            "fa1": {"status": "not_applicable", "reason": "FA1 has no SM120 target"},
            "fa2": {"status": "eligible", "reason": "current FA2 has SM120 target"},
            "fa3": {"status": "not_applicable", "reason": "FA3 CUDA path has no SM120 target"},
            "fa4": {"status": "eligible", "reason": "official FA4 Blackwell path"},
        }
    return {
        name: {"status": "unresolved", "reason": "cell architecture not registered"}
        for name in ("fa1", "fa2", "fa3", "fa4")
    }


def assemble_suite(payloads: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    calibrations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    runs: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    raw_paths: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    profiles: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    profile_paths: dict[tuple[str, str], list[str]] = defaultdict(list)
    hardware_by_cell: dict[str, Mapping[str, Any]] = {}
    for path, payload in payloads:
        hardware = payload.get("hardware")
        if not hardware:
            continue
        cell_id = hardware_cell_id(hardware)
        hardware_by_cell[cell_id] = hardware
        if payload.get("evidence_kind") == "per-case-profiler":
            profile_key = (cell_id, str(payload["backend"]))
            profiles[profile_key].append(payload)
            profile_paths[profile_key].append(path)
        elif "calibration_id" in payload:
            calibrations[cell_id].append(payload)
        elif "benchmark" in payload and "backend" in payload:
            key = (
                cell_id,
                str(payload["backend"]),
                str(payload.get("backend_version", "unknown")),
                str(payload.get("implementation_commit", "unknown")),
            )
            runs[key].append(payload)
            raw_paths[key].append(path)

    cells = []
    for cell_id in sorted(hardware_by_cell):
        calibration_error = None
        try:
            calibration = summarize_calibrations(calibrations[cell_id])
        except ValueError as error:
            calibration = None
            calibration_error = str(error)
        candidates = []
        for key in sorted(key for key in runs if key[0] == cell_id):
            if calibration is None:
                scored = {
                    "backend": key[1],
                    "backend_version": key[2],
                    "implementation_commit": key[3],
                    "certified": False,
                    "verdict": "unresolved",
                    "disposition": "invalid",
                    "artifact_status": "not_applicable",
                    "artifact_100": None,
                    "failure_codes": ["EVIDENCE_CALIBRATION_INVALID"],
                }
            else:
                profile_key = (cell_id, str(key[1]))
                scored = score_run_group(runs[key], calibration, profiles[profile_key])
            scored["raw_evidence_paths"] = sorted(raw_paths[key])
            scored["profile_evidence_paths"] = sorted(profile_paths[(cell_id, str(key[1]))])
            candidates.append(scored)
        hardware = hardware_class(hardware_by_cell[cell_id])
        cells.append(
            {
                "cell_id": cell_id,
                "hardware": hardware,
                "hardware_class_sha256": canonical_sha256(hardware),
                "calibration": calibration,
                "calibration_error": calibration_error,
                "eligibility": eligibility_for_cell(
                    hardware.get("compute_capability"),
                    accelerator_vendor=hardware.get("accelerator_vendor"),
                    architecture=hardware.get("architecture"),
                    runtime_version=hardware.get("runtime_version"),
                    torch_version=hardware.get("torch_version"),
                ),
                "candidates": candidates,
            }
        )
    formula_parameters = {
        "attention": "100*(0.80*P+0.20*G)",
        "generalization": "0.50*retention+0.30*tail+0.20*fallback_footprint",
        "micro": "100*AnchorScore",
        "replays": list(REQUIRED_REPLAYS),
        "anchor_min_headroom": 1.10,
        "anchor_beyond_tolerance": 0.03,
        "bootstrap_resamples": 10_000,
    }
    has_rocm_cell = any(cell["hardware"].get("accelerator_vendor") == "amd" for cell in cells)
    return {
        "schema_version": "0.3",
        "generated_at": utc_now(),
        "suite_id": (
            "kernel-frontier-fa1-fa4-aotriton-classic-v03"
            if has_rocm_cell
            else "kernel-frontier-fa1-fa4-classic-v03"
        ),
        "leaderboard_season": "2026q3-kernel-v1",
        "formula_version": FORMULA_VERSION,
        "formula_origin": FORMULA_ORIGIN,
        "formula_parameters": formula_parameters,
        "formula_parameters_sha256": canonical_sha256(formula_parameters),
        "cell_count": len(cells),
        "cells": cells,
    }


def load_json_evidence(root: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"score.json", "evidence-manifest.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            payloads.append((path.relative_to(root).as_posix(), payload))
    return payloads
