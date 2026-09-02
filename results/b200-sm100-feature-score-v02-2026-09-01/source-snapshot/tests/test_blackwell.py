from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from infraswe.kernel.blackwell import (
    FEATURE_CONTRACTS,
    MVP_FEATURE_IDS,
    feature_contract_manifest,
    target_satisfies,
)
from infraswe.verifier.native_sm100 import extract_ptx_entries, verify_feature

CAPABILITY_FINGERPRINT = "sha256:" + "a" * 64


def load_benchmark_module(name: str):
    path = Path(__file__).parents[1] / "benchmarks" / "kernel_frontier" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capability_probe = load_benchmark_module("b200_capability_probe")
b200_summary = load_benchmark_module("summarize_b200_compiler_features")
parse_cuda_release = capability_probe.parse_cuda_release
parse_gpu_rows = capability_probe.parse_gpu_rows
probe = capability_probe.probe
build_summary = b200_summary.build_summary


def schema(project_root: Path, name: str) -> dict:
    return json.loads((project_root / "schemas" / name).read_text(encoding="utf-8"))


def tmem_ptx(*, target: str = "sm_100a", body: str | None = None) -> str:
    instructions = (
        body
        or """
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem], 32;
    tcgen05.mma.cta_group::1.kind::f16 [taddr], adesc, bdesc, idesc, 0, 0;
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr, 32;
    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ret;
"""
    )
    return f""".version 9.3
.target {target}
.address_size 64
.visible .entry tmem_kernel()
{{
{instructions}
}}
"""


def write_tmem_artifacts(root: Path, *, target: str = "sm_100a", body: str | None = None) -> None:
    root.mkdir(parents=True)
    (root / "kernel.ptx").write_text(tmem_ptx(target=target, body=body), encoding="utf-8")
    (root / "kernel.cubin").write_bytes(b"synthetic-cubin")
    (root / "kernel.sass.txt").write_text("/*0000*/ UTCHMMA;\n", encoding="utf-8")


def dynamic_evidence(artifact_sha256: str, replay_index: int = 1) -> dict:
    return {
        "schema_version": "0.1",
        "feature_id": "BW-TMEM-001",
        "replay_index": replay_index,
        "status": "passed",
        "artifact_set_sha256": artifact_sha256,
        "capability_fingerprint": CAPABILITY_FINGERPRINT,
        "evaluator": {"owner": "infraswe", "evaluator_sha256": "sha256:" + "c" * 64},
        "correctness": {"passed": True, "dynamic_input_changes_output": True},
        "liveness": {"completed": True, "watchdog_passed": True},
        "mutation": {"performed": True, "passed": True},
        "profiler": {"captured": True, "kernel_names": ["tmem_kernel"]},
    }


def test_blackwell_contract_freezes_targets_namespaces_and_preview() -> None:
    manifest = feature_contract_manifest()

    assert tuple(manifest["mvp_feature_ids"]) == MVP_FEATURE_IDS
    assert set(manifest["target_lanes"]) == {"sm_100", "sm_100f", "sm_100a"}
    assert FEATURE_CONTRACTS["BW-TMEM-001"].required_target == "sm_100a"
    assert FEATURE_CONTRACTS["BW-CLC-001"].namespace == "SM100-Scheduler"
    assert FEATURE_CONTRACTS["BW-PTX-PREVIEW-001"].phase == "preview-disabled"
    assert target_satisfies("sm_100a", "sm_100f")
    assert not target_satisfies("sm_100", "sm_100a")


def test_blackwell_feature_pack_matches_the_code_contract(project_root: Path) -> None:
    manifest = json.loads(
        (project_root / "benchmarks/kernel_frontier/blackwell_feature_pack_v01.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(manifest["mvp_feature_ids"]) == MVP_FEATURE_IDS
    assert manifest["scoring_contract"]["leaderboard_score_100"] is None
    Draft202012Validator(schema(project_root, "blackwell-feature-pack.schema.json")).validate(
        manifest
    )


def test_ptx_parser_ignores_comments_and_extracts_entry_body() -> None:
    entries = extract_ptx_entries(
        """.version 9.3
.target sm_100a
// .visible .entry fake() { tcgen05.mma; }
.visible .entry real()
{
  /* tcgen05.alloc; */
  ret;
}
"""
    )

    assert [entry["name"] for entry in entries] == ["real"]
    assert "tcgen05" not in entries[0]["body"]
    assert entries[0]["target"] == "sm_100a"


def test_native_sm100_is_static_only_without_dynamic_evidence(
    tmp_path: Path, project_root: Path
) -> None:
    artifact_root = tmp_path / "artifacts"
    write_tmem_artifacts(artifact_root)

    result = verify_feature(
        artifact_root=artifact_root,
        feature_id="BW-TMEM-001",
        evidence_dir=tmp_path / "evidence",
        expected_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )

    assert result["status"] == "static_only"
    assert not result["certified"]
    assert result["gates"]["reachable_ptx"]["passed"]
    assert result["gates"]["native_sass"]["passed"]
    assert result["failure_codes"] == ["DYNAMIC_EVIDENCE_MISSING"]
    Draft202012Validator(schema(project_root, "blackwell-native-result.schema.json")).validate(
        result
    )


def test_native_sm100_certifies_bound_dynamic_evidence(tmp_path: Path, project_root: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    write_tmem_artifacts(artifact_root)
    static = verify_feature(
        artifact_root=artifact_root,
        feature_id="BW-TMEM-001",
        evidence_dir=tmp_path / "static-evidence",
        expected_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )
    dynamic = dynamic_evidence(static["artifact_set_sha256"])

    result = verify_feature(
        artifact_root=artifact_root,
        feature_id="BW-TMEM-001",
        evidence_dir=tmp_path / "dynamic-evidence",
        dynamic_evidence=dynamic,
        requested_entry="tmem_kernel",
        expected_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )

    assert result["status"] == "certified"
    assert result["certified"]
    assert result["failure_codes"] == []
    Draft202012Validator(schema(project_root, "blackwell-dynamic-evidence.schema.json")).validate(
        dynamic
    )
    Draft202012Validator(schema(project_root, "blackwell-native-result.schema.json")).validate(
        result
    )


def test_native_sm100_rejects_dead_code_wrong_target_and_fallback(tmp_path: Path) -> None:
    dead_root = tmp_path / "dead"
    write_tmem_artifacts(dead_root, body="ret;")
    (dead_root / "dead-function.ptx").write_text(
        """.version 9.3
.target sm_100a
.func dead() {
  tcgen05.alloc;
  tcgen05.mma;
  tcgen05.dealloc;
  tcgen05.relinquish_alloc_permit;
}
""",
        encoding="utf-8",
    )
    dead = verify_feature(
        artifact_root=dead_root,
        feature_id="BW-TMEM-001",
        evidence_dir=tmp_path / "dead-evidence",
        expected_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )
    assert not dead["gates"]["reachable_ptx"]["passed"]

    wrong_root = tmp_path / "wrong-target"
    write_tmem_artifacts(wrong_root, target="sm_100")
    wrong = verify_feature(
        artifact_root=wrong_root,
        feature_id="BW-TMEM-001",
        evidence_dir=tmp_path / "wrong-evidence",
        expected_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )
    assert not wrong["gates"]["reachable_ptx"]["passed"]

    fallback_root = tmp_path / "fallback"
    write_tmem_artifacts(fallback_root)
    with (fallback_root / "kernel.ptx").open("a", encoding="utf-8") as handle:
        handle.write("\n.visible .func fallback() { wgmma.mma_async; }\n")
    fallback = verify_feature(
        artifact_root=fallback_root,
        feature_id="BW-TMEM-001",
        evidence_dir=tmp_path / "fallback-evidence",
        expected_capability_fingerprint=CAPABILITY_FINGERPRINT,
    )
    assert not fallback["gates"]["fallback"]["passed"]
    assert "FALLBACK_SYMBOL_DETECTED" in fallback["failure_codes"]


def test_b200_capability_probe_is_fail_closed_and_schema_valid(
    tmp_path: Path, project_root: Path
) -> None:
    result = probe(
        device_index=0,
        leased_gpu_count=1,
        artifact_root=tmp_path / "targets",
        profile_id="gpu-1x-sm100-b200-cuda133",
    )

    assert result["status"] in {"ready", "not_ready"}
    assert set(result["toolchain"]["targets"]) == {"sm_100", "sm_100f", "sm_100a"}
    assert result["score_namespaces"]["PTX-Preview"]["status"] == "disabled"
    Draft202012Validator(schema(project_root, "blackwell-capability.schema.json")).validate(result)


def test_cuda_release_and_b200_query_parsing() -> None:
    assert parse_cuda_release("Cuda compilation tools, release 13.3, V13.3.42") == "13.3"
    rows = parse_gpu_rows("0, GPU-1, NVIDIA B200, 183359, 590.1, 00000000:01:00.0, 10.0\n")
    assert rows[0]["architecture"] == "sm100"
    assert rows[0]["compute_capability"] == "10.0"


def test_summary_never_turns_evidence_coverage_into_a_performance_score(
    project_root: Path,
) -> None:
    replays = []
    for index in (1, 2, 3):
        replays.append(
            {
                "replay_index": index,
                "capability_manifest_sha256": "sha256:" + "b" * 64,
                "capability_fingerprint": CAPABILITY_FINGERPRINT,
                "hardware": {"name": "NVIDIA B200", "compute_capability": "10.0"},
                "toolchain": {"status": "supported", "detected_versions": {}},
                "features": [
                    {
                        "feature_id": feature_id,
                        "namespace": FEATURE_CONTRACTS[feature_id].namespace,
                        "status": "pending",
                        "certified": False,
                    }
                    for feature_id in MVP_FEATURE_IDS
                ],
            }
        )

    summary = build_summary(replays)

    assert summary["status"] == "evidence_pending"
    assert summary["leaderboard_score_100"] is None
    assert not summary["leaderboard_ready"]
    assert summary["score_namespaces"]["SM100-Core"]["leaderboard_score_100"] is None
    summary_schema = schema(project_root, "blackwell-feature-summary.schema.json")
    Draft202012Validator(summary_schema).validate(summary)
