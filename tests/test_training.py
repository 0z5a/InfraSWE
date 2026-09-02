from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from infraswe.models.task import TaskPackage
from infraswe.models.training import (
    TrainingComparability,
    TrainingEvidencePackManifest,
    TrainingScoreInput,
)
from infraswe.scoring.deployability import weighted_geometric
from infraswe.scoring.training import (
    build_training_result,
    training_profiler_to_v04,
)
from infraswe.training.adapter import validate_adapter_conformance
from infraswe.training.fixtures import (
    dapo_reference_bundle,
    grpo_reference_bundle,
    sft_reference_bundle,
)
from infraswe.training.native_pytorch import NativePyTorchAdapter
from infraswe.training.probe import probe_capabilities
from infraswe.training.semantics import COMMON_GATE_NAMES, verify_training_evidence


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_cells() -> list[dict]:
    return [
        {
            "regime": regime,
            "completed_requests": 1200,
            "slo_goodput_ratio": 0.9,
            "tail_score": 0.9,
            "replay_jitter_score": 0.9,
            "resource_stability_score": 0.9,
            "fairness_score": 0.9,
        }
        for regime in ("light", "normal", "knee", "saturation", "overload", "burst_or_soak")
    ]


def _score_input(*, profiler_grade: str = "G3", reuse: bool = True) -> TrainingScoreInput:
    payload = {
        "benchmark_cell_id": "sha256:" + "c" * 64,
        "profiler_grade": profiler_grade,
        "fresh_process_replays": 7,
        "load_cells": _load_cells(),
        "reuse": {
            "shape_coverage": 0.9,
            "dtype_coverage": 0.9,
            "layout_coverage": 0.9,
            "observed_variants": 8,
            "expected_variant_budget": 8,
            "max_variant_budget": 24,
            "dispatcher_reuse": 0.9,
            "compile_cache_reuse": 0.9,
            "portability_reuse": 0.9,
            "silent_fallback_rate": 0,
        }
        if reuse
        else None,
        "maintainability": {
            "contract": 0.9,
            "locality": 0.9,
            "tests": 0.9,
            "build_reproducibility": 0.9,
        },
        "evidence_digests": ["sha256:" + "e" * 64],
        "raw_metrics": {
            "step_p50_ms": {"value": 1.0, "unit": "ms"},
            "sol_efficiency": {"value": None, "reason": "G4 counters absent"},
        },
    }
    return TrainingScoreInput.model_validate(payload)


def _comparability() -> TrainingComparability:
    return TrainingComparability(
        semantic_contract_id="sft-v1",
        hardware_cell_id="sha256:" + "c" * 64,
        normalized_execution_contract_id="single-v1",
        work_model_id="tiny-v1",
        concurrency_protocol_id="training-step-load-ladder-v1",
        leaderboard_season="test",
    )


def test_training_task_is_v04_layered_and_custom_ids_are_namespaced(project_root: Path) -> None:
    task = TaskPackage.load(project_root / "tasks/training-sft-cross-framework-v1")
    assert task.schema_version == "0.4"
    assert task.task.kind == "training-workflow"
    assert task.replay.count == 7
    assert task.profiling and task.profiling.minimum_grade_for_deployability == "G3"
    assert task.workload and task.workload.algorithm == "sft"
    assert task.validate_layout() == []

    digest_mismatch = TaskPackage.load(project_root / "tasks/training-sft-cross-framework-v1")
    assert digest_mismatch.workload
    digest_mismatch.workload.work_model.sha256 = "sha256:" + "0" * 64
    assert any("digest mismatch" in error for error in digest_mismatch.validate_layout())

    payload = task.model_dump()
    payload["trainer"]["optional"].append("org.example/trainer-x")
    namespaced = TaskPackage.model_validate(payload)
    assert namespaced.trainer and "org.example/trainer-x" in namespaced.trainer.optional

    payload = task.model_dump()
    payload["trainer"]["optional"].append("unregistered-framework")
    with pytest.raises(ValueError, match="namespaced custom id"):
        TaskPackage.model_validate(payload)

    payload = task.model_dump()
    payload["profiling"]["minimum_grade_for_deployability"] = "G2"
    with pytest.raises(ValueError, match="G3 system trace"):
        TaskPackage.model_validate(payload)


def test_positive_training_contracts_pass_all_frozen_hard_gates() -> None:
    bundles = (
        sft_reference_bundle(),
        sft_reference_bundle(muon=True),
        grpo_reference_bundle(),
        dapo_reference_bundle(),
    )
    for bundle in bundles:
        certification = verify_training_evidence(bundle)
        assert certification.status == "pass"
        assert set(certification.gates) == set(COMMON_GATE_NAMES)
        assert not certification.failure_codes


def test_sft_negative_controls_reject_packing_loss_and_rng() -> None:
    packing = sft_reference_bundle().model_copy(deep=True)
    assert packing.sft is not None
    packing.sft.observed_attention_edges.append((0, 2))
    assert "PACK_BOUNDARY_LEAK" in verify_training_evidence(packing).failure_codes

    denominator = sft_reference_bundle().model_copy(deep=True)
    assert denominator.sft is not None
    denominator.sft.observed_denominator = 4
    assert "LOSS_MASK_MISMATCH" in verify_training_evidence(denominator).failure_codes

    rng = sft_reference_bundle().model_copy(deep=True)
    assert rng.checkpoint is not None
    rng.checkpoint.rng_streams_restored.remove("dropout_rng")
    assert "RESUME_DIVERGENCE" in verify_training_evidence(rng).failure_codes


def test_online_and_muon_negative_controls_are_distinguishable() -> None:
    grpo = grpo_reference_bundle().model_copy(deep=True)
    assert grpo.grpo is not None
    grpo.grpo.samples[0].observed_advantage = 0
    assert "GRPO_GROUP_MISMATCH" in verify_training_evidence(grpo).failure_codes

    stale = grpo_reference_bundle().model_copy(deep=True)
    assert stale.grpo is not None
    stale.grpo.samples[0].policy_version = 2
    assert "POLICY_VERSION_STALE" in verify_training_evidence(stale).failure_codes

    dapo = dapo_reference_bundle().model_copy(deep=True)
    assert dapo.dapo is not None
    dapo.dapo.dynamic_sampling = False
    dapo_cert = verify_training_evidence(dapo)
    assert "DAPO_COMPONENT_MISSING" in dapo_cert.failure_codes
    assert "DYNAMIC_SAMPLING_MISMATCH" in dapo_cert.failure_codes

    muon = sft_reference_bundle(muon=True).model_copy(deep=True)
    assert muon.muon is not None
    muon.muon.parameter_groups[2].optimizer = "muon"
    assert "OPTIMIZER_GROUP_MISMATCH" in verify_training_evidence(muon).failure_codes


def test_missing_evidence_is_unresolved_and_runtime_failures_are_hard_gates() -> None:
    missing = sft_reference_bundle().model_copy(deep=True)
    missing.forward = None
    certification = verify_training_evidence(missing)
    assert certification.status == "unresolved"
    assert certification.gates["FORWARD_SEMANTICS"].status == "unresolved"

    fallback = sft_reference_bundle().model_copy(deep=True)
    assert fallback.runtime is not None
    fallback.runtime.silent_fallback_count = 1
    failed = verify_training_evidence(fallback)
    assert failed.status == "fail"
    assert "SILENT_FRAMEWORK_FALLBACK" in failed.failure_codes


def test_training_scoring_defers_to_v04_and_g3_boundary() -> None:
    bundle = sft_reference_bundle()
    certification = verify_training_evidence(bundle)
    result = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=_score_input(profiler_grade="G3"),
        comparability=_comparability(),
    )
    assert result.scoring_authority == "infraswe-scoring-v0.4"
    assert result.v04_score.deployability
    assert result.v04_score.deployability.formula_template_id == "deployability-v0.4"
    values = {
        name: float(component.value)
        for name, component in result.v04_score.deployability.components.items()
    }
    expected = 100 * weighted_geometric(
        values,
        {"concurrent_stability": 0.45, "kernel_reuse": 0.30, "maintainability": 0.25},
    )
    assert math.isclose(result.v04_score.deployability.score_100 or 0, expected)
    assert result.comparability.cross_hardware_absolute_performance is False

    g2 = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=_score_input(profiler_grade="G2"),
        comparability=_comparability(),
    )
    assert g2.v04_score.deployability
    assert g2.v04_score.deployability.status == "unresolved"
    assert g2.v04_score.deployability.score_100 is None
    assert training_profiler_to_v04("G2") == "E1-framework"
    assert training_profiler_to_v04("G3") == "E2-system-trace"
    assert training_profiler_to_v04("G4") == "E3-kernel-counter"
    with pytest.raises(ValueError, match="requires a G4"):
        training_profiler_to_v04("G3", sealed=True)


def test_training_hard_gate_failure_and_missing_dimensions_cannot_issue_score() -> None:
    bundle = sft_reference_bundle()
    assert bundle.runtime is not None
    bundle.runtime.silent_fallback_count = 1
    failed = build_training_result(
        bundle=bundle,
        certification=verify_training_evidence(bundle),
        score_input=_score_input(),
        comparability=_comparability(),
    )
    assert failed.v04_score.infra_cert == "fail"
    assert failed.v04_score.deployability
    assert failed.v04_score.deployability.score_100 is None
    assert failed.v04_score.leaderboard_effective_deployability_100 == 0

    clean = sft_reference_bundle()
    unresolved = build_training_result(
        bundle=clean,
        certification=verify_training_evidence(clean),
        score_input=_score_input(reuse=False),
        comparability=_comparability(),
    )
    assert unresolved.v04_score.deployability
    assert unresolved.v04_score.deployability.status == "unresolved"
    assert unresolved.v04_score.deployability.components["kernel_reuse"].value is None


def test_training_cell_efficiency_requires_g4_and_stays_cell_local() -> None:
    bundle = sft_reference_bundle()
    certification = verify_training_evidence(bundle)
    payload = _score_input(profiler_grade="G4").model_dump()
    payload["cell_artifact_template"] = "cell-artifact-mixed-v0.4"
    payload["cell_efficiency"] = {
        "work_model_id": "tiny-v1",
        "regime": "mixed",
        "work_model": {"minimum_external_bytes": 1e9, "semantic_flops": 1e9},
        "calibration": {
            "launch_floor_us": 1.0,
            "compute_tflops": 10.0,
            "memory_bandwidth_gbps": 100.0,
        },
        "candidate_time_seconds": 0.02,
        "actual_memory_bytes": 1e9,
        "traffic_amplification_budget": 1.0,
        "counter_evidence_available": True,
        "counter_confidence": "high",
        "evidence_digests": ["sha256:" + "f" * 64],
    }
    g4 = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=TrainingScoreInput.model_validate(payload),
        comparability=_comparability(),
    )
    assert g4.v04_score.cell_efficiency
    assert g4.v04_score.cell_efficiency.status == "scored"
    assert g4.v04_score.cell_artifact
    assert g4.v04_score.cell_artifact.status == "scored"
    assert g4.v04_score.cell_artifact.cross_cell_ranking_allowed is False

    payload["profiler_grade"] = "G3"
    g3 = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=TrainingScoreInput.model_validate(payload),
        comparability=_comparability(),
    )
    assert g3.v04_score.cell_efficiency
    assert g3.v04_score.cell_efficiency.status == "unresolved"
    assert g3.v04_score.cell_artifact
    assert g3.v04_score.cell_artifact.score_100 is None


def test_adapter_probe_never_promotes_frameworks_to_cell_certified() -> None:
    adapter = NativePyTorchAdapter()
    assert validate_adapter_conformance(adapter) == []
    manifest = probe_capabilities()
    for record in manifest.adapters.values():
        assert record.capability_level != "cell-certified"
        assert all(level != "cell-certified" for level in record.algorithms.values())


def test_evidence_pack_requires_declared_categories() -> None:
    payload = {
        "task_id": "training-sft-contract-v1",
        "sealed": False,
        "artifacts": [
            {
                "path": "semantic-contract.json",
                "sha256": "sha256:" + "a" * 64,
                "size_bytes": 10,
                "category": "contract",
            }
        ],
        "required_categories": ["contract", "checkpoint"],
    }
    with pytest.raises(ValueError, match="missing categories"):
        TrainingEvidencePackManifest.model_validate(payload)


def test_hermetic_minimum_suite_covers_positive_negative_and_score_boundaries(
    project_root: Path,
) -> None:
    module = _load_module(
        project_root / "benchmarks/training_cross_framework/run_minimum_suite.py",
        "training_minimum_suite",
    )
    result = module.run_suite()
    assert result["status"] == "pass"
    assert result["cell_certified"] is False
    assert result["official_score_published"] is False
    assert result["score_boundary"]["g3_v04_status"] == "scored"
    assert result["score_boundary"]["g2_v04_status"] == "unresolved"
    assert all(case["accepted"] for case in result["negative_controls"].values())
