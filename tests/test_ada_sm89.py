from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from infraswe.kernel.ada_sm89 import (
    CANONICAL_PLATFORM_CELLS,
    FEATURE_CONTRACTS,
    MINIMUM_RELEASE_FEATURE_IDS,
    NATIVE_TARGET,
    PTX_FALLBACK_TARGET,
    capability_contract_manifest,
    target_satisfies,
)
from infraswe.models.ada_sm89 import (
    AdaSM89CapabilityManifest,
    AdaSM89CrossSKUResult,
    AdaSM89NativeResult,
)
from infraswe.models.hardware import HardwareProfile
from infraswe.scoring.ada_sm89 import architecture_overlay_score, score_cross_sku_reuse
from infraswe.verifier.native_sm89 import verify_gpu_feature as verify_gpu_feature_impl


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_gpu_feature(**kwargs):
    return verify_gpu_feature_impl(allow_provided_disassembly=True, **kwargs)


def write_gpu_artifacts(
    root: Path,
    *,
    body: str = "ret;",
    target: str = "sm_89",
    sass: str = "/*0000*/ MOV R0, R0;",
    entry: str = "sm89_native_dispatch_kernel",
) -> None:
    root.mkdir(parents=True)
    (root / "kernel.ptx").write_text(
        f""".version 9.3
.target {target}
.address_size 64
.visible .entry {entry}()
{{
  {body}
}}
""",
        encoding="utf-8",
    )
    (root / "kernel.cubin").write_bytes(b"synthetic-cubin")
    (root / "kernel.sass.txt").write_text(sass + "\n", encoding="utf-8")


def dynamic_evidence(
    feature_id: str,
    artifact_sha256: str,
    entry: str,
    *,
    module_sha256: str,
    replays: int = 7,
) -> dict:
    return {
        "schema_version": "0.1",
        "feature_id": feature_id,
        "artifact_set_sha256": artifact_sha256,
        "capability_fingerprint": "sha256:capability",
        "correctness": {"passed": True},
        "liveness": {"completed": True, "watchdog_passed": True},
        "observed_entries": [entry],
        "forbidden_library_calls": [],
        "silent_fallback_count": 0,
        "fresh_process_replays": replays,
        "loaded_module_container_sha256": module_sha256,
        "dispatch_modes": {
            "native_cubin": {"passed": True},
            "ptx_jit": {"passed": True},
        },
        "allocation_audit": {"full_size_fp16_temporaries": 0},
    }


def test_ada_contract_uses_one_exact_target_and_two_board_cells() -> None:
    manifest = capability_contract_manifest()
    assert NATIVE_TARGET == "sm_89"
    assert PTX_FALLBACK_TARGET == "compute_89"
    assert set(CANONICAL_PLATFORM_CELLS) == {"l40s-48gb-pcie", "l20-48gb-pcie"}
    assert target_satisfies("sm_89", "sm_89")
    assert not target_satisfies("sm_90", "sm_89")
    assert tuple(manifest["minimum_release_feature_ids"]) == MINIMUM_RELEASE_FEATURE_IDS
    assert manifest["scoring_authority"]["global"] == "infraswe-scoring-v0.4"
    assert manifest["scoring_authority"]["missing_evidence_policy"] == "unresolved-not-zero"
    forbidden = "\n".join(
        pattern
        for contract in FEATURE_CONTRACTS.values()
        for pattern in contract.forbidden_patterns
    )
    for token in ("wgmma", "cp\\.async\\.bulk", "tcgen05", "tmem", "multimem", "e2m1"):
        assert token in forbidden


def test_ada_probe_identifies_cells_and_is_fail_closed(tmp_path: Path, project_root: Path) -> None:
    module = load_module(
        project_root / "benchmarks/ada_sm89/capability_probe.py", "ada_sm89_capability_probe"
    )
    rows = module.parse_gpu_rows(
        "0, GPU-1, NVIDIA L40S, 46068, 580.65, 00000000:01:00.0, 0x26B510DE, 8.9\n"
        "1, GPU-2, NVIDIA L20, 46068, 580.65, 00000000:02:00.0, 0x26B610DE, 8.9\n"
    )
    assert module.identify_platform_cell(rows[0], allow_generic_sm89=False) == "l40s-48gb-pcie"
    assert module.identify_platform_cell(rows[1], allow_generic_sm89=False) == "l20-48gb-pcie"
    generic = {**rows[0], "name": "Unknown Ada", "memory_bytes": 48 * 1024**3}
    assert module.identify_platform_cell(generic, allow_generic_sm89=False) is None
    assert module.identify_platform_cell(generic, allow_generic_sm89=True) == "generic-sm89"

    result = module.probe(
        device_index=999,
        artifact_root=tmp_path / "artifacts",
        profile_id="gpu-1x-sm89-test",
        runtime_probe_source=(
            project_root / "platforms/nvidia-ada-sm89/capability_probe/cuda_probe.cc"
        ),
    )
    AdaSM89CapabilityManifest.model_validate(result)
    schema = json.loads(
        (project_root / "schemas/ada-sm89-capability.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)
    assert result["gates"]["platform"]["status"] == "fail"
    assert result["status"] in {"compile_only", "not_ready"}


def test_sm89_native_verifier_binds_dispatch_and_seven_replays(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    write_gpu_artifacts(artifacts)
    production_mode = verify_gpu_feature_impl(
        artifact_root=artifacts,
        feature_id="SM89-TARGET-001",
        requested_entry="sm89_native_dispatch_kernel",
        capability_fingerprint="sha256:capability",
    )
    assert "SM89_NATIVE_BINARY_OR_DISASSEMBLY_MISSING" in production_mode["failure_codes"]
    static = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="SM89-TARGET-001",
        requested_entry="sm89_native_dispatch_kernel",
        capability_fingerprint="sha256:capability",
    )
    assert static["status"] == "static_only"
    AdaSM89NativeResult.model_validate(static)
    dynamic = dynamic_evidence(
        "SM89-TARGET-001",
        static["artifact_set_sha256"],
        "sm89_native_dispatch_kernel",
        module_sha256=next(
            item["sha256"] for item in static["artifacts"] if item["path"] == "kernel.cubin"
        ),
    )
    certified = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="SM89-TARGET-001",
        requested_entry="sm89_native_dispatch_kernel",
        dynamic_evidence=dynamic,
        capability_fingerprint="sha256:capability",
    )
    assert certified["status"] == "certified"
    assert certified["certified"]

    dynamic["fresh_process_replays"] = 4
    insufficient = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="SM89-TARGET-001",
        requested_entry="sm89_native_dispatch_kernel",
        dynamic_evidence=dynamic,
        capability_fingerprint="sha256:capability",
    )
    assert not insufficient["certified"]
    assert "DYNAMIC_FRESH_REPLAYS_BELOW_7" in insufficient["failure_codes"]


def test_sm89_verifier_requires_fp8_ptx_and_hmma_sass(tmp_path: Path) -> None:
    artifacts = tmp_path / "fp8"
    entry = "sm89_fp8_mma_smoke"
    body = "\n".join(
        (
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32;",
            "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32;",
        )
    )
    write_gpu_artifacts(artifacts, body=body, sass="HMMA.1688.F32.F8E4M3", entry=entry)
    static = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="SM89-FP8-MMA-001",
        requested_entry=entry,
        capability_fingerprint="sha256:capability",
    )
    assert static["status"] == "static_only"
    certified = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="SM89-FP8-MMA-001",
        requested_entry=entry,
        dynamic_evidence=dynamic_evidence(
            "SM89-FP8-MMA-001",
            static["artifact_set_sha256"],
            entry,
            module_sha256=next(
                item["sha256"] for item in static["artifacts"] if item["path"] == "kernel.cubin"
            ),
        ),
        capability_fingerprint="sha256:capability",
    )
    assert certified["certified"]

    no_hmma = tmp_path / "no-hmma"
    write_gpu_artifacts(no_hmma, body=body, sass="MOV R0, R0", entry=entry)
    rejected = verify_gpu_feature(
        artifact_root=no_hmma,
        feature_id="SM89-FP8-MMA-001",
        requested_entry=entry,
        capability_fingerprint="sha256:capability",
    )
    assert "SM89_REQUIRED_SASS_PATH_MISSING" in rejected["failure_codes"]


def test_sm89_verifier_rejects_hopper_blackwell_and_fp4_paths(tmp_path: Path) -> None:
    forbidden_samples = (
        "wgmma.mma_async.sync.aligned;",
        "cp.async.bulk.tensor.2d.shared::cluster.global;",
        "tcgen05.mma;",
        "mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32;",
        "multimem.ld_reduce;",
    )
    for index, forbidden in enumerate(forbidden_samples):
        artifacts = tmp_path / f"forbidden-{index}"
        write_gpu_artifacts(artifacts, body=forbidden)
        result = verify_gpu_feature(
            artifact_root=artifacts,
            feature_id="SM89-TARGET-001",
            requested_entry="sm89_native_dispatch_kernel",
            capability_fingerprint="sha256:capability",
        )
        assert "SM89_FORBIDDEN_FEATURE_OR_FALLBACK" in result["failure_codes"]

    wrong_target = tmp_path / "wrong-target"
    write_gpu_artifacts(wrong_target, target="sm_90a")
    result = verify_gpu_feature(
        artifact_root=wrong_target,
        feature_id="SM89-TARGET-001",
        requested_entry="sm89_native_dispatch_kernel",
        capability_fingerprint="sha256:capability",
    )
    assert "SM89_REACHABLE_PTX_GATE_FAILED" in result["failure_codes"]


def test_cross_sku_reuse_is_diagnostic_and_missing_is_unresolved() -> None:
    shared = {
        "source_subtree_sha256": "sha256:source",
        "semantic_artifact_sha256": "sha256:semantic",
        "codegen_cache_key": "sha256:codegen",
    }
    cells = {
        "l40s-48gb-pcie": {
            **shared,
            "board_tuning_cache_key": "sha256:l40s-tuning",
            "realized_ratios": {"decode": 0.99, "prefill": 1.0},
        },
        "l20-48gb-pcie": {
            **shared,
            "board_tuning_cache_key": "sha256:l20-tuning",
            "realized_ratios": {"decode": 0.98, "prefill": 0.99},
        },
    }
    result = score_cross_sku_reuse(cells, shape_weights={"decode": 0.6, "prefill": 0.4})
    AdaSM89CrossSKUResult.model_validate(result)
    assert result["status"] == "diagnostic"
    assert result["deployability_100"] is None
    assert result["cross_cell_ranking_allowed"] is False

    missing = score_cross_sku_reuse(
        {"l40s-48gb-pcie": cells["l40s-48gb-pcie"]},
        shape_weights={"decode": 0.6, "prefill": 0.4},
    )
    assert missing["status"] == "unresolved"
    assert missing["score_100"] is None

    cells["l20-48gb-pcie"]["realized_ratios"]["decode"] = 0.97
    regressed = score_cross_sku_reuse(cells, shape_weights={"decode": 0.6, "prefill": 0.4})
    assert regressed["status"] == "not_deployable"
    assert "ADA_CROSS_SKU_PRODUCTION_REGRESSION" in regressed["failure_codes"]


def test_ada_overlay_cannot_replace_v04_deployability() -> None:
    component_names = {
        "concurrency_stability",
        "cross_sku_reuse",
        "maintainability",
        "production_realization",
        "local_efficiency",
        "compile_runtime",
        "evidence",
    }
    unresolved = architecture_overlay_score({name: None for name in component_names})
    assert unresolved["status"] == "unresolved"
    assert unresolved["score_100"] is None
    diagnostic = architecture_overlay_score({name: 0.8 for name in component_names})
    assert diagnostic["status"] == "diagnostic"
    assert diagnostic["deployability_100"] is None
    assert diagnostic["cross_cell_ranking_allowed"] is False


def test_ada_summary_defers_conflicting_weights_to_v04(project_root: Path) -> None:
    module = load_module(
        project_root / "platforms/nvidia-ada-sm89/verifier/score.py", "ada_sm89_score"
    )
    results = [
        {
            "feature_id": feature_id,
            "status": "certified",
            "certified": True,
            "failure_codes": [],
        }
        for feature_id in MINIMUM_RELEASE_FEATURE_IDS
    ]
    summary = module.summarize(results)
    assert summary["infra_cert"] == "pass"
    assert summary["deployability_100"] is None
    assert "C/U/M" in summary["deployability_reason"]
    assert summary["missing_evidence_is_zero"] is False
    assert summary["absolute_l40s_vs_l20_ranking_published"] is False


def test_external_evidence_validators_use_v04_and_separate_caches(project_root: Path) -> None:
    module = load_module(
        project_root / "benchmarks/ada_sm89/validate_external_evidence.py",
        "ada_sm89_external_evidence",
    )
    regimes = ("light", "normal", "knee", "saturation", "overload", "burst_or_soak")
    load_cells = [
        {
            "regime": regime,
            "completed_requests": 1000,
            "slo_goodput_ratio": 0.9,
            "tail_score": 0.9,
            "replay_jitter_score": 0.9,
            "resource_stability_score": 0.9,
            "fairness_score": 0.9,
        }
        for regime in regimes
    ]
    concurrency = module.validate_concurrency(
        {"fresh_process_replays": 7, "load_cells": load_cells}
    )
    assert concurrency["certified"]
    assert concurrency["component"]["formula_version"] == "concurrent-stability-v0.4"

    shared = {
        "codegen_cache_key": "sha256:codegen",
        "unique_graphs": 4,
        "unique_kernels": 16,
        "compile_seconds_cold": 20,
        "generated_source_mib": 2,
        "steady_compile_events": 0,
        "online_unbounded_autotune": False,
    }
    torchcompile = module.validate_torchcompile(
        {
            "fresh_process_replays": 7,
            "cells": {
                "l40s-48gb-pcie": {
                    **shared,
                    "board_tuning_cache_key": "sha256:l40s",
                },
                "l20-48gb-pcie": {
                    **shared,
                    "board_tuning_cache_key": "sha256:l20",
                },
            },
        }
    )
    assert torchcompile["certified"]
    polluted = module.validate_torchcompile(
        {
            "fresh_process_replays": 7,
            "cells": {
                "l40s-48gb-pcie": {
                    **shared,
                    "board_tuning_cache_key": "sha256:same",
                },
                "l20-48gb-pcie": {
                    **shared,
                    "board_tuning_cache_key": "sha256:same",
                },
            },
        }
    )
    assert not polluted["certified"]
    assert "ADA_TORCHCOMPILE_TUNING_CACHE_NOT_ISOLATED" in polluted["failure_codes"]


def test_suite_rejects_unvalidated_external_evidence(tmp_path: Path, project_root: Path) -> None:
    module = load_module(
        project_root / "benchmarks/ada_sm89/run_minimum_suite.py",
        "ada_sm89_minimum_suite",
    )
    feature_id = "ADA-CONCURRENCY-001"
    (tmp_path / f"{feature_id}.json").write_text(
        json.dumps({"feature_id": feature_id, "certified": True}), encoding="utf-8"
    )
    rejected = module.external_result(feature_id, tmp_path)
    assert not rejected["certified"]
    assert rejected["failure_codes"] == ["ADA_EXTERNAL_PROVENANCE_MISSING"]


def test_shared_cuda_sources_do_not_fork_on_product_name(project_root: Path) -> None:
    task_root = project_root / "platforms/nvidia-ada-sm89/tasks"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(task_root.rglob("*.cu"))
    )
    assert "L40S" not in sources
    assert "L20" not in sources


def test_l40s_and_l20_profiles_share_architecture_but_not_identity(project_root: Path) -> None:
    l40s = HardwareProfile.load(project_root / "profiles/gpu-1x-sm89-l40s-48gb-cuda133.toml")
    l20 = HardwareProfile.load(project_root / "profiles/gpu-1x-sm89-l20-48gb-cuda133.toml")
    assert l40s.architecture == l20.architecture == "sm89"
    assert l40s.compute_capability == l20.compute_capability == "8.9"
    assert l40s.runtime_version == l20.runtime_version == "13.3"
    assert l40s.gpu_model == "L40S"
    assert l20.gpu_model == "L20"
    assert l40s.id != l20.id
