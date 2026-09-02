from __future__ import annotations

import copy

import pytest

from infraswe.kernel.results import assemble_suite, eligibility_for_cell


def hardware() -> dict:
    return {
        "gpu_name": "Synthetic A100",
        "compute_capability": "8.0",
        "sm_count": 108,
        "total_memory_bytes": 80 * 2**30,
        "torch_version": "2.8.0+cu128",
        "torch_cuda": "12.8",
        "cudnn_version": 91002,
        "nvidia_smi": {"driver_version": "580.65.06"},
    }


def calibration(replay_index: int) -> dict:
    summary = {"median": 100.0}
    return {
        "calibration_id": "test",
        "replay_index": replay_index,
        "hardware": hardware(),
        "launch_floor_us": {"median": 1.0},
        "hbm_bandwidth_gbps": {"median": 1000.0},
        "bf16_matmul_tflops": summary,
        "known_omissions": [],
    }


def case(case_id: str, group: str = "micro", weight: float = 1.0) -> dict:
    blocks = [
        {
            "block_index": index,
            "order": "ABBA",
            "reference_latency_us": 200.0,
            "candidate_latency_us": 150.0,
        }
        for index in range(1, 5)
    ]
    return {
        "case_id": case_id,
        "case_group": group,
        "weight": weight,
        "shape": {"elements": 1024},
        "dtype": "bfloat16",
        "work_model": {
            "semantic_flops": 10_000_000_000,
            "minimum_external_bytes": 1_000_000,
        },
        "correctness": {
            "passed": True,
            "dynamic_input_changes_output": True,
        },
        "measurement": {"blocks": blocks},
        "profiler": {"captured": True, "cuda_events": [{"name": "flash_kernel"}]},
    }


def attention_run(replay_index: int) -> dict:
    cases = [
        case("common-1", "common", 0.2),
        case("common-2", "common", 0.2),
        case("common-3", "common", 0.2),
        case("boundary", "boundary_tail", 0.2),
        case("stress", "stress_large", 0.2),
    ]
    return {
        "benchmark": "attention",
        "benchmark_kind": "kernel-library",
        "backend": "fa2",
        "backend_version": "2.test",
        "implementation_commit": "abc",
        "implementation_provenance": [{"sha256": "sha256:abc"}],
        "replay_index": replay_index,
        "hardware": hardware(),
        "status": "passed",
        "all_correct": True,
        "cases": cases,
    }


def classic_run(replay_index: int) -> dict:
    payload = attention_run(replay_index)
    payload.update(
        benchmark="classic",
        benchmark_kind="kernel-micro",
        backend="triton-fixed-config",
        backend_version="3.test",
        implementation_provenance=[{"sha256": "sha256:def"}],
        cases=[case("vector-add")],
    )
    payload["cases"][0]["profiler"]["cuda_events"][0]["name"] = "_vector_add_kernel"
    return payload


def suite_payloads(include_replay_three: bool = True):
    indices = (1, 2, 3) if include_replay_three else (1, 2)
    payloads = []
    for index in indices:
        payloads.extend(
            [
                (f"raw/calibration-{index}.json", calibration(index)),
                (f"raw/fa2-{index}.json", attention_run(index)),
                (f"raw/classic-{index}.json", classic_run(index)),
            ]
        )
    return payloads


def test_assemble_suite_scores_library_and_micro_cases() -> None:
    suite = assemble_suite(suite_payloads())
    assert suite["cell_count"] == 1
    candidates = {item["backend"]: item for item in suite["cells"][0]["candidates"]}
    fa2 = candidates["fa2"]
    assert fa2["certified"] is True
    assert fa2["components"]["performance_anchor_score"] == pytest.approx(2 / 3)
    assert fa2["components"]["generalization"] == pytest.approx(1.0)
    assert fa2["artifact_100"] == pytest.approx(100 * (0.8 * 2 / 3 + 0.2))
    assert fa2["case_results"][0]["paired_speedup_ci95"] == pytest.approx([4 / 3, 4 / 3])

    classic = candidates["triton-fixed-config"]
    assert classic["certified"] is True
    assert classic["micro_artifact_100"]["vector-add"] == pytest.approx(100 * 2 / 3)


def test_missing_calibration_replay_invalidates_cell_scores() -> None:
    suite = assemble_suite(suite_payloads(include_replay_three=False))
    cell = suite["cells"][0]
    assert cell["calibration"] is None
    assert all(candidate["disposition"] == "invalid" for candidate in cell["candidates"])


def test_micro_cases_remain_independent_when_one_has_no_headroom() -> None:
    payloads = suite_payloads()
    for _, payload in payloads:
        if payload.get("benchmark") != "classic":
            continue
        blocks = payload["cases"][0]["measurement"]["blocks"]
        for block in blocks:
            block["reference_latency_us"] = 105.0
            block["candidate_latency_us"] = 110.0
    suite = assemble_suite(payloads)
    classic = next(
        candidate
        for candidate in suite["cells"][0]["candidates"]
        if candidate["backend"] == "triton-fixed-config"
    )
    assert classic["certified"] is True
    assert classic["artifact_status"] == "mixed-per-case"
    assert classic["micro_artifact_100"]["vector-add"] is None
    assert classic["micro_artifact_status"]["vector-add"] == "not_frontier_eligible"


def test_architecture_eligibility_is_explicit() -> None:
    sm80_fa4 = eligibility_for_cell("8.0")["fa4"]
    assert sm80_fa4["status"] == "eligible"
    assert "4.0.0b28" in sm80_fa4["reason"]
    sm90 = eligibility_for_cell("9.0")
    assert all(item["status"] == "eligible" for item in sm90.values())
    assert "SM90" in sm90["fa3"]["reason"]
    assert eligibility_for_cell("12.0")["fa4"]["status"] == "eligible"
    assert eligibility_for_cell("12.0")["fa1"]["status"] == "not_applicable"


def negative_control_payloads(backend: str) -> list[tuple[str, dict]]:
    payloads: list[tuple[str, dict]] = []
    for index in (1, 2, 3):
        run = copy.deepcopy(attention_run(index))
        run["backend"] = backend
        run["backend_version"] = "negative-control-v1"
        payloads.extend(
            [
                (f"raw/calibration-{index}.json", calibration(index)),
                (f"raw/{backend}-{index}.json", run),
            ]
        )
    return payloads


def test_wrong_native_kernel_is_effective_zero() -> None:
    payloads = negative_control_payloads("garbage-zero-triton")
    for _, payload in payloads:
        if payload.get("backend") != "garbage-zero-triton":
            continue
        payload["all_correct"] = False
        for item in payload["cases"]:
            item["correctness"]["passed"] = False
            item["profiler"]["cuda_events"][0]["name"] = "_zero_kernel"
    candidate = assemble_suite(payloads)["cells"][0]["candidates"][0]
    assert candidate["certified"] is False
    assert candidate["verdict"] == "fail"
    assert candidate["leaderboard_effective_artifact_100"] == 0.0
    assert "CORRECTNESS_MANDATORY_FAILED" in candidate["failure_codes"]


def test_declared_fa_fallback_without_flash_trace_is_invalid() -> None:
    payloads = negative_control_payloads("fa-garbage-math-fallback")
    for _, payload in payloads:
        if payload.get("backend") != "fa-garbage-math-fallback":
            continue
        for item in payload["cases"]:
            item["profiler"]["cuda_events"][0]["name"] = "bmm_math_kernel"
    candidate = assemble_suite(payloads)["cells"][0]["candidates"][0]
    assert candidate["certified"] is False
    assert candidate["disposition"] == "invalid"
    assert candidate["artifact_100"] is None
    assert "FALLBACK_NATIVE_TRACE_MISSING" in candidate["failure_codes"]


def test_correct_but_slow_kernel_receives_low_score() -> None:
    payloads = negative_control_payloads("garbage-slow-fa4-waste64")
    for _, payload in payloads:
        if payload.get("backend") != "garbage-slow-fa4-waste64":
            continue
        for item in payload["cases"]:
            item["profiler"]["cuda_events"][0]["name"] = "_stream_waste_kernel"
            for block in item["measurement"]["blocks"]:
                block["candidate_latency_us"] = 400.0
    candidate = assemble_suite(payloads)["cells"][0]["candidates"][0]
    assert candidate["certified"] is True
    assert candidate["artifact_status"] == "scored"
    assert candidate["artifact_100"] == pytest.approx(40.0)
