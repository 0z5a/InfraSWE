from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from infraswe.kernel.gb10 import (
    FEATURE_CONTRACTS,
    MINIMUM_RELEASE_FEATURE_IDS,
    TARGET_LANES,
    capability_contract_manifest,
    target_satisfies,
)
from infraswe.models.gb10 import GB10CapabilityManifest
from infraswe.verifier.native_sm121 import verify_gpu_feature


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_gb10_contract_separates_sm121_from_sm100_and_roce() -> None:
    manifest = capability_contract_manifest()
    assert set(manifest["target_lanes"]) == {"sm_121", "sm_121f", "sm_121a"}
    assert set(TARGET_LANES) == {"sm_121", "sm_121f", "sm_121a"}
    assert target_satisfies("sm_121a", "sm_121f")
    assert not target_satisfies("sm_121", "sm_121a")
    for contract in FEATURE_CONTRACTS.values():
        assert any("tcgen05" in pattern for pattern in contract.forbidden_patterns) or (
            contract.required_target is None
        )
    assert FEATURE_CONTRACTS["GB10-ROCE-001"].leaderboard_eligible is False


def test_gb10_probe_is_fail_closed_and_schema_valid(tmp_path: Path, project_root: Path) -> None:
    module = load_module(
        project_root / "benchmarks/gb10_sm121/capability_probe.py", "gb10_capability_probe"
    )
    result = module.probe(
        device_index=0,
        artifact_root=tmp_path / "artifacts",
        profile_id="gpu-1x-sm121-gb10-cuda130",
        runtime_probe_source=project_root / "platforms/nvidia-gb10/capability_probe/cuda_probe.cc",
    )
    GB10CapabilityManifest.model_validate(result)
    schema = json.loads(
        (project_root / "schemas/gb10-capability.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(result)
    assert set(result["toolchain"]["targets"]) == set(TARGET_LANES)
    assert result["status"] in {"ready", "partial", "not_ready"}


def test_gb10_probe_reads_normalized_runtime_attributes(project_root: Path) -> None:
    module = load_module(
        project_root / "benchmarks/gb10_sm121/capability_probe.py",
        "gb10_capability_probe_attributes",
    )
    parsed = {"attributes": {"pageable_memory_access": {"returncode": 0, "value": 1}}}
    wrapped = {"parsed": parsed}
    assert module._attribute_value(parsed, "pageable_memory_access") == 1
    assert module._attribute_value(wrapped, "pageable_memory_access") == 1


def write_gpu_artifacts(root: Path, *, body: str = "ret;", target: str = "sm_121") -> None:
    root.mkdir(parents=True)
    (root / "kernel.ptx").write_text(
        f""".version 9.3
.target {target}
.address_size 64
.visible .entry dispatch_kernel()
{{
  {body}
}}
""",
        encoding="utf-8",
    )
    (root / "kernel.cubin").write_bytes(b"synthetic-cubin")
    (root / "kernel.sass.txt").write_text("/*0000*/ MOV R0, R0;\n", encoding="utf-8")


def test_sm121_native_verifier_binds_static_dynamic_and_capability(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    write_gpu_artifacts(artifacts)
    static = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="GB10-TARGET-001",
        requested_entry="dispatch_kernel",
        capability_fingerprint="sha256:capability",
    )
    assert static["status"] == "static_only"
    dynamic = {
        "schema_version": "0.1",
        "feature_id": "GB10-TARGET-001",
        "artifact_set_sha256": static["artifact_set_sha256"],
        "capability_fingerprint": "sha256:capability",
        "correctness": {"passed": True},
        "liveness": {"completed": True, "watchdog_passed": True},
        "observed_entries": ["dispatch_kernel"],
        "forbidden_library_calls": [],
    }
    certified = verify_gpu_feature(
        artifact_root=artifacts,
        feature_id="GB10-TARGET-001",
        requested_entry="dispatch_kernel",
        dynamic_evidence=dynamic,
        capability_fingerprint="sha256:capability",
    )
    assert certified["status"] == "certified"
    assert certified["certified"]


def test_sm121_verifier_rejects_sm100_tcgen05_and_accepts_blockscale_shape(
    tmp_path: Path,
) -> None:
    forbidden = tmp_path / "forbidden"
    write_gpu_artifacts(forbidden, body="tcgen05.mma;", target="sm_100a")
    rejected = verify_gpu_feature(
        artifact_root=forbidden,
        feature_id="GB10-TARGET-001",
        capability_fingerprint="sha256:capability",
    )
    assert not rejected["certified"]
    assert "GB10_FORBIDDEN_FEATURE_OR_FALLBACK" in rejected["failure_codes"]

    mma = tmp_path / "mma"
    write_gpu_artifacts(
        mma,
        target="sm_121a",
        body="mma.sync.aligned.kind::mxf4nvf4.block_scale;",
    )
    static = verify_gpu_feature(
        artifact_root=mma,
        feature_id="GB10-MMA-001",
        capability_fingerprint="sha256:capability",
    )
    assert static["status"] == "static_only"


def test_gb10_summary_defers_all_conflicting_scoring_to_v04(project_root: Path) -> None:
    module = load_module(project_root / "platforms/nvidia-gb10/verifier/score.py", "gb10_score")
    results = [
        {
            "feature_id": feature_id,
            "certified": True,
            "failure_codes": [],
        }
        for feature_id in MINIMUM_RELEASE_FEATURE_IDS
    ]
    summary = module.summarize(results)
    assert summary["status"] == "certified"
    assert summary["deployability_100"] is None
    assert "C/U/M" in summary["deployability_reason"]
    assert summary["roce_scaleout_mixed_into_single_node"] is False
