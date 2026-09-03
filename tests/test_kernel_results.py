from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

from infraswe.kernel.results import (
    assemble_suite,
    eligibility_for_cell,
    hardware_cell_id,
    hardware_class,
)


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


def amd_hardware() -> dict:
    return {
        "accelerator_vendor": "amd",
        "architecture": "gfx942",
        "runtime": "rocm",
        "runtime_version": "6.1.2",
        "gpu_name": "AMD Instinct MI300X",
        "compute_capability": None,
        "compute_unit_count": 304,
        "sm_count": 304,
        "total_memory_bytes": 192 * 2**30,
        "torch_version": "2.4.0+rocm6.1",
        "torch_cuda": None,
        "torch_hip": "6.1.40093",
        "cudnn_version": None,
        "driver_version": "6.8.5",
        "nvidia_smi": {},
        "rocm_smi": {"driver_version": "6.8.5"},
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


def test_gfx942_rocm61_eligibility_is_exact_stack_and_fail_closed() -> None:
    matrix = eligibility_for_cell(
        None,
        accelerator_vendor="amd",
        architecture="gfx942",
        runtime_version="6.1.2",
        torch_version="2.4.0+rocm6.1",
    )
    assert matrix["torch-sdpa-aotriton"]["status"] == "eligible"
    assert matrix["triton-gfx942-initial"]["status"] == "eligible"
    assert all(matrix[name]["status"] == "not_applicable" for name in ("fa1", "fa2", "fa3", "fa4"))

    wrong_runtime = eligibility_for_cell(
        None,
        accelerator_vendor="amd",
        architecture="gfx942",
        runtime_version="6.2.0",
        torch_version="2.4.0+rocm6.1",
    )
    assert wrong_runtime["torch-sdpa-aotriton"]["status"] == "unresolved"


def test_mi300x_hardware_cell_uses_vendor_architecture_and_rocm_driver() -> None:
    manifest = amd_hardware()

    normalized = hardware_class(manifest)
    cell_id = hardware_cell_id(manifest)

    assert normalized["accelerator_vendor"] == "amd"
    assert normalized["architecture"] == "gfx942"
    assert normalized["runtime"] == "rocm"
    assert normalized["driver_version"] == "6.8.5"
    assert cell_id.startswith("amd-amd-instinct-mi300x-gfx942-")


def test_legacy_nvidia_hardware_identity_remains_byte_shape_compatible() -> None:
    normalized = hardware_class(hardware())

    assert normalized == {
        "gpu_name": "Synthetic A100",
        "compute_capability": "8.0",
        "sm_count": 108,
        "total_memory_bytes": 80 * 2**30,
        "driver_version": "580.65.06",
        "torch_version": "2.8.0+cu128",
        "torch_cuda": "12.8",
        "cudnn_version": 91002,
    }
    assert hardware_cell_id(hardware()).startswith("synthetic-a100-sm80-")


def test_mi300x_aotriton_native_trace_can_certify() -> None:
    payloads = suite_payloads()
    for _, payload in payloads:
        payload["hardware"] = amd_hardware()
        if payload.get("benchmark") == "attention":
            payload["backend"] = "torch-sdpa-aotriton"
            payload["backend_version"] = "2.4.0+rocm6.1"
            payload["implementation_commit"] = "pytorch-2.4.0-rocm6.1-aotriton"
            payload["implementation_mechanism"] = "aotriton"
            for item in payload["cases"]:
                item["profiler"] = {
                    "captured": True,
                    "device_events": [{"name": "aotriton::v2::flash::attn_fwd"}],
                }

    suite = assemble_suite(payloads)
    cell = suite["cells"][0]
    candidate = next(
        item for item in cell["candidates"] if item["backend"] == "torch-sdpa-aotriton"
    )

    assert cell["eligibility"]["torch-sdpa-aotriton"]["status"] == "eligible"
    assert candidate["certified"] is True
    assert candidate["implementation_mechanism"] == "aotriton"


def test_mi300x_aotriton_rejects_framework_flash_wrapper_without_native_kernel() -> None:
    payloads = suite_payloads()
    for _, payload in payloads:
        payload["hardware"] = amd_hardware()
        if payload.get("benchmark") == "attention":
            payload["backend"] = "torch-sdpa-aotriton"
            payload["backend_version"] = "2.4.0+rocm6.1"
            payload["implementation_commit"] = "pytorch-2.4.0-rocm6.1-aotriton"
            payload["implementation_mechanism"] = "aotriton"
            for item in payload["cases"]:
                item["profiler"] = {
                    "captured": True,
                    "device_events": [{"name": "aten::_flash_attention_forward"}],
                }

    suite = assemble_suite(payloads)
    candidate = next(
        item for item in suite["cells"][0]["candidates"] if item["backend"] == "torch-sdpa-aotriton"
    )

    assert candidate["certified"] is False
    assert candidate["disposition"] == "invalid"
    assert "FALLBACK_NATIVE_TRACE_MISSING" in candidate["failure_codes"]


def test_mi300x_report_renders_rocm_architecture_without_cuda_labels() -> None:
    payloads = suite_payloads()
    for _, payload in payloads:
        payload["hardware"] = amd_hardware()

    suite = assemble_suite(payloads)
    report_module = runpy.run_path(
        Path(__file__).parents[1] / "benchmarks/kernel_frontier/score_results.py"
    )
    markdown = report_module["markdown_report"](suite)
    rendered_html = report_module["html_report"](suite)

    assert "Vendor AMD" in markdown
    assert "Arch gfx942" in markdown
    assert "ROCM 6.1.2" in markdown
    assert "Vendor AMD" in rendered_html
    assert "Arch gfx942" in rendered_html


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
