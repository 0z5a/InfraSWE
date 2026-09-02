from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.models.training import (
    TrainingComparability,
    TrainingScoreInput,
)
from infraswe.scoring.training import build_training_result
from infraswe.training.fixtures import (
    dapo_reference_bundle,
    grpo_reference_bundle,
    sft_reference_bundle,
)
from infraswe.training.semantics import verify_training_evidence


def _load_cells() -> list[dict[str, Any]]:
    return [
        {
            "regime": regime,
            "completed_requests": 1200,
            "slo_goodput_ratio": 0.92,
            "tail_score": 0.88,
            "replay_jitter_score": 0.95,
            "resource_stability_score": 0.96,
            "fairness_score": 0.94,
        }
        for regime in ("light", "normal", "knee", "saturation", "overload", "burst_or_soak")
    ]


def _score_input(*, profiler_grade: str = "G3") -> TrainingScoreInput:
    return TrainingScoreInput.model_validate(
        {
            "benchmark_cell_id": "sha256:" + "c" * 64,
            "profiler_grade": profiler_grade,
            "fresh_process_replays": 7,
            "load_cells": _load_cells(),
            "reuse": {
                "shape_coverage": 0.95,
                "dtype_coverage": 0.9,
                "layout_coverage": 0.9,
                "observed_variants": 8,
                "expected_variant_budget": 8,
                "max_variant_budget": 24,
                "dispatcher_reuse": 0.9,
                "compile_cache_reuse": 0.9,
                "portability_reuse": 0.85,
                "silent_fallback_rate": 0,
            },
            "maintainability": {
                "contract": 0.95,
                "locality": 0.9,
                "tests": 0.95,
                "build_reproducibility": 0.9,
            },
            "evidence_digests": ["sha256:" + "e" * 64],
            "raw_metrics": {
                "step_p50_ms": {"value": 1.0, "unit": "ms"},
                "sol_efficiency": {
                    "value": None,
                    "reason": "fixture has no G4 hardware counters",
                },
            },
        }
    )


def _comparability() -> TrainingComparability:
    return TrainingComparability(
        semantic_contract_id="sft-packed-v1",
        hardware_cell_id="sha256:" + "c" * 64,
        normalized_execution_contract_id="training-single-v1",
        work_model_id="t0-tiny-transformer-v1",
        concurrency_protocol_id="training-step-load-ladder-v1",
        leaderboard_season="fixture-only",
    )


def _negative_cases() -> dict[str, tuple[Any, str, str | None]]:
    packing = sft_reference_bundle().model_copy(deep=True)
    assert packing.sft is not None
    packing.sft.observed_attention_edges.append((0, 2))

    denominator = sft_reference_bundle().model_copy(deep=True)
    assert denominator.sft is not None
    denominator.sft.observed_denominator = 4

    rng = sft_reference_bundle().model_copy(deep=True)
    assert rng.checkpoint is not None
    rng.checkpoint.rng_streams_restored.remove("dropout_rng")

    advantage = grpo_reference_bundle().model_copy(deep=True)
    assert advantage.grpo is not None
    advantage.grpo.samples[0].observed_advantage = 0.0

    stale = grpo_reference_bundle().model_copy(deep=True)
    assert stale.grpo is not None
    stale.grpo.samples[0].policy_version = 2

    dapo = dapo_reference_bundle().model_copy(deep=True)
    assert dapo.dapo is not None
    dapo.dapo.dynamic_sampling = False

    muon = sft_reference_bundle(muon=True).model_copy(deep=True)
    assert muon.muon is not None
    muon.muon.parameter_groups[2].optimizer = "muon"

    fallback = sft_reference_bundle().model_copy(deep=True)
    assert fallback.runtime is not None
    fallback.runtime.silent_fallback_count = 1

    deadlock = sft_reference_bundle().model_copy(deep=True)
    assert deadlock.runtime is not None
    deadlock.runtime.deadlock = True

    leak = sft_reference_bundle().model_copy(deep=True)
    assert leak.runtime is not None
    leak.runtime.resource_leaks = ["worker:17"]

    missing = sft_reference_bundle().model_copy(deep=True)
    missing.forward = None

    return {
        "packing-boundary-leak": (packing, "fail", "PACK_BOUNDARY_LEAK"),
        "loss-denominator": (denominator, "fail", "LOSS_MASK_MISMATCH"),
        "rng-resume": (rng, "fail", "RESUME_DIVERGENCE"),
        "grpo-group": (advantage, "fail", "GRPO_GROUP_MISMATCH"),
        "stale-policy": (stale, "fail", "POLICY_VERSION_STALE"),
        "dapo-component": (dapo, "fail", "DAPO_COMPONENT_MISSING"),
        "muon-group": (muon, "fail", "OPTIMIZER_GROUP_MISMATCH"),
        "silent-fallback": (fallback, "fail", "SILENT_FRAMEWORK_FALLBACK"),
        "deadlock": (deadlock, "fail", "COLLECTIVE_DEADLOCK"),
        "resource-leak": (leak, "fail", "ROLLOUT_WORKER_LEAK"),
        "missing-forward": (missing, "unresolved", None),
    }


def run_suite() -> dict[str, Any]:
    positive = {
        "sft": sft_reference_bundle(),
        "sft-muon": sft_reference_bundle(muon=True),
        "grpo-contract": grpo_reference_bundle(),
        "dapo-recipe-contract": dapo_reference_bundle(),
    }
    positive_results = {
        name: verify_training_evidence(bundle).model_dump(mode="json")
        for name, bundle in positive.items()
    }
    negative_results = {}
    suite_failures: list[str] = []
    for name, (bundle, expected_status, expected_code) in _negative_cases().items():
        certification = verify_training_evidence(bundle)
        accepted = certification.status == expected_status and (
            expected_code is None or expected_code in certification.failure_codes
        )
        if not accepted:
            suite_failures.append(f"negative-control:{name}")
        negative_results[name] = {
            "accepted": accepted,
            "expected_status": expected_status,
            "expected_code": expected_code,
            "certification": certification.model_dump(mode="json"),
        }
    for name, result in positive_results.items():
        if result["status"] != "pass":
            suite_failures.append(f"positive-control:{name}")

    bundle = positive["sft"]
    certification = verify_training_evidence(bundle)
    official_fixture = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=_score_input(profiler_grade="G3"),
        comparability=_comparability(),
    )
    g2_fixture = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=_score_input(profiler_grade="G2"),
        comparability=_comparability(),
    )
    if official_fixture.v04_score.deployability is None:
        suite_failures.append("v04-score-envelope-missing")
    elif official_fixture.v04_score.deployability.status != "scored":
        suite_failures.append("v04-g3-score-not-issued")
    if g2_fixture.v04_score.deployability is None:
        suite_failures.append("v04-g2-envelope-missing")
    elif g2_fixture.v04_score.deployability.status != "unresolved":
        suite_failures.append("v04-g2-evidence-boundary")

    return {
        "schema_version": "0.1",
        "status": "pass" if not suite_failures else "fail",
        "fixture_only": True,
        "cell_certified": False,
        "official_score_published": False,
        "scoring_authority": "infraswe-scoring-v0.4",
        "positive_controls": positive_results,
        "negative_controls": negative_results,
        "score_boundary": {
            "g3_v04_status": official_fixture.v04_score.deployability.status,
            "g3_fixture_score_100": official_fixture.v04_score.deployability.score_100,
            "g2_v04_status": g2_fixture.v04_score.deployability.status,
            "g2_fixture_score_100": g2_fixture.v04_score.deployability.score_100,
            "note": "synthetic contract test; not a hardware or leaderboard result",
        },
        "failures": suite_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hermetic training contract suite")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_suite()
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
