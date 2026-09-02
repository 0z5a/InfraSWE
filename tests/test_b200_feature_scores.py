from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_summary_module():
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "kernel_frontier"
        / "summarize_b200_feature_scores.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_b200_feature_scores", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import summary module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summary_module = load_summary_module()


def native() -> dict:
    return {
        "passed": True,
        "ptx_files": ["kernel.ptx"],
        "cubin_files": ["kernel.cubin"],
        "ptx_size_bytes": 1024,
        "binary_size_bytes": 2048,
        "disassembly": [{"returncode": 0}],
    }


def measurement(value: float) -> dict:
    return {"passed": True, "latency_us_median": value}


def replay(index: int) -> dict:
    common_correctness = {"passed": True, "passed_cases": 3, "total_cases": 3}
    return {
        "suite_id": "b200-sm100-feature-score-v0.2",
        "replay_index": index,
        "hardware": {
            "gpu_name": "NVIDIA B200",
            "compute_capability": "10.0",
            "sm_count": 148,
            "cuda_runtime": "13.0",
        },
        "features": {
            "BW-TMEM-001": {
                "status": "passed",
                "correctness": common_correctness,
                "performance": {
                    "candidate": measurement(90.0 + index),
                    "portable": measurement(100.0),
                },
                "profiler": {"captured": True},
                "native": native(),
            },
            "BW-TMEM-003": {
                "status": "passed",
                "correctness": {
                    "passed": True,
                    "passed_cases": 4,
                    "total_cases": 4,
                    "invalid_alignment_rejection": {"passed": True},
                },
                "liveness": {
                    "passed": True,
                    "memory_free_before_bytes": 10_000,
                    "memory_free_after_bytes": 10_000,
                },
                "performance": {
                    "candidate_stress": measurement(10.1),
                    "control": measurement(10.0),
                },
                "native": native(),
            },
            "BW-CLC-001": {
                "status": "passed",
                "correctness": common_correctness,
                "performance": {
                    "makespan_pairs": [
                        {
                            "shape": [4096, 4096, 4096, 1],
                            "candidate": measurement(90.0 + index),
                            "baseline": measurement(100.0),
                        },
                        {
                            "shape": [4224, 4096, 4096, 1],
                            "candidate": measurement(94.0 + index),
                            "baseline": measurement(101.0),
                        },
                    ]
                },
                "profiler": {"captured": True},
                "native": native(),
            },
            "BW-TMA-001": {
                "status": "passed",
                "correctness": {"case_count": 5, "gather4": True, "scatter4": True},
                "performance": {
                    "gather4_us": 2.0,
                    "scalar_gather4_us": 4.0,
                    "scatter4_us": 2.1,
                    "scalar_scatter4_us": 4.1,
                },
                "native": native(),
            },
            "BW-TMA-002": {
                "status": "passed",
                "correctness": {"passed": True, "passed_cases": 2, "total_cases": 2},
                "performance": {
                    "candidate": measurement(80.0 + index),
                    "baseline": measurement(95.0),
                },
                "profiler": {"captured": True},
                "native": native(),
            },
        },
    }


def test_latency_anchor_score_uses_log_interpolation() -> None:
    result = summary_module.latency_anchor_score(
        [(8.0, 10.0, "a"), (9.0, 10.0, "b"), (8.5, 10.0, "c")]
    )

    assert 0.0 < result["score"] <= 1.0
    assert result["native_reference_latency_us"] == 8.0
    assert result["speedup_over_portable"] > 1.0


def test_phase1_summary_scores_core_scheduler_and_keeps_fabric_na() -> None:
    result = summary_module.build_summary([replay(1), replay(2), replay(3)])

    assert result["status"] == "scored"
    assert result["score_namespaces"]["SM100-Core"]["score_100"] > 0.0
    assert result["score_namespaces"]["SM100-Scheduler"]["score_100"] > 0.0
    assert result["score_namespaces"]["SM100-Fabric"]["score_100"] is None
    assert result["tasks"]["BW-TMEM-003"]["weights"] == summary_module.BUGFIX_WEIGHTS
