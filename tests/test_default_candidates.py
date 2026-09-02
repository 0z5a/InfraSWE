from __future__ import annotations

import json
from pathlib import Path

import pytest

import infraswe.draft.candidate_registry as candidate_registry_module
from infraswe.draft.candidate_registry import (
    build_default_candidate_registry,
    evaluate_candidate_timing_gate,
    plan_candidate_activation,
    resolve_default_candidates,
)
from infraswe.draft.defaults import build_default_draft
from infraswe.models.candidates import CandidateSelectionRequest
from infraswe.models.draft import DraftCandidate, DraftPrecompilePolicy


def _candidate(entrypoint: str, *, family: str = "generic") -> DraftCandidate:
    return DraftCandidate(
        kind="generated",
        revision="sha256:" + "a" * 64,
        intent="integrate",
        implementation_kind="cuda-native",
        entrypoints=[entrypoint],
        operator_family=family,
    )


def test_registry_separates_roles_and_pins_many_candidates() -> None:
    registry = build_default_candidate_registry()
    assert len(registry.candidates) == 39
    assert len(registry.rules) == 13
    assert registry.learned_model_used is False
    assert registry.weighted_score_used is False
    assert "peer-impl" in registry.candidates["flash_attention"].roles
    assert "host-project" not in registry.candidates["flash_attention"].roles
    assert "host-project" in registry.candidates["vllm"].roles
    for candidate in registry.candidates.values():
        assert candidate.build.registry_load_compiles_candidate is False
        assert candidate.build.registry_load_imports_candidate is False
        if candidate.source.kind == "pinned-git":
            assert candidate.source.revision and len(candidate.source.revision) == 40


def test_first_match_attention_resolution_is_metadata_only() -> None:
    resolution = resolve_default_candidates(
        CandidateSelectionRequest(
            operator_family="attention-inference-decode",
            phase="inference",
            backend="cuda",
            requested_primary_host="vllm",
        )
    )
    assert resolution.primary_peer_impl == "flashinfer"
    assert resolution.secondary_peer_impls == ["vllm", "sglang"]
    assert resolution.primary_host == "vllm"
    assert resolution.selection_side_effects == "metadata-only-no-import-no-build"
    assert resolution.compilation_started is False
    assert resolution.weighted_score_used is False
    assert all("without a score" in item.explanation for item in resolution.trace[:-2])


def test_only_explicitly_activated_selected_candidate_can_precompile() -> None:
    registry = build_default_candidate_registry()
    resolution = resolve_default_candidates(
        CandidateSelectionRequest(
            operator_family="grouped-moe-gemm",
            phase="inference",
            backend="cuda",
            requested_primary_host="sglang",
        )
    )
    plan = plan_candidate_activation(resolution)
    assert plan.activated_candidate_ids == ["deepgemm"]
    assert plan.actions[0].action == "precompile-before-timed-cases"
    assert plan.inactive_candidate_count == len(registry.candidates) - 1
    assert plan.selection_compile_seconds == 0

    pending = evaluate_candidate_timing_gate(plan)
    assert pending.timed_benchmark_allowed is False
    assert pending.timing_eligibility == "blocked"
    assert pending.blockers == ["CANDIDATE_NOT_PREPARED:deepgemm"]
    ready = evaluate_candidate_timing_gate(plan, prepared_candidate_ids=["deepgemm"])
    assert ready.timed_benchmark_allowed is True
    assert ready.timing_eligibility == "official"
    assert ready.steady_state_compile_allowed is False

    diagnostic = plan_candidate_activation(
        resolution,
        precompile_policy=DraftPrecompilePolicy(mode="off"),
    )
    diagnostic_gate = evaluate_candidate_timing_gate(diagnostic)
    assert diagnostic_gate.timed_benchmark_allowed is True
    assert diagnostic_gate.timing_eligibility == "diagnostic-only"
    assert diagnostic_gate.warnings == ["INLINE_COMPILE_DIAGNOSTIC_ONLY:deepgemm"]

    cached = plan_candidate_activation(
        resolution,
        activated_candidate_ids=["deepgemm"],
        cache_hits={"deepgemm": True},
    )
    assert cached.actions[0].action == "reuse-precompiled-artifact"
    with pytest.raises(ValueError, match="unselected"):
        plan_candidate_activation(
            resolution,
            activated_candidate_ids=["flash_attention"],
        )
    with pytest.raises(ValueError, match="only allowed"):
        plan_candidate_activation(
            resolution,
            compilation_required={"cutlass_cute": True},
        )
    with pytest.raises(ValueError, match="exactly one peer"):
        plan_candidate_activation(
            resolution,
            activated_candidate_ids=["deepgemm", "cutlass_cute"],
        )
    with pytest.raises(ValueError, match="limited to peer implementations"):
        plan_candidate_activation(
            resolution,
            activated_candidate_ids=["torch_eager"],
        )


def test_registry_digest_is_not_reserialized_on_hot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    base = build_default_candidate_registry()
    registry = type(base).model_validate(
        {**base.model_dump(mode="json"), "registry_version": "digest-cache-test"}
    )
    original = candidate_registry_module.canonical_sha256
    calls = 0

    def counted(value: object) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(candidate_registry_module, "canonical_sha256", counted)
    request = CandidateSelectionRequest(
        operator_family="attention-inference-decode",
        phase="inference",
        backend="cuda",
    )
    resolve_default_candidates(request, registry=registry)
    resolve_default_candidates(request, registry=registry)
    assert calls == 1


def test_default_draft_embeds_role_resolution_without_flattening_projects() -> None:
    draft, _ = build_default_draft(
        project="flashinfer",
        candidate=_candidate("flashinfer.decode"),
        created_by="test",
    )
    assert draft.default_candidates
    assert draft.default_candidates.primary_peer_impl == "flashinfer"
    assert draft.default_candidates.primary_host == "vllm"
    assert draft.benchmark_loop and draft.benchmark_loop.precompile.mode == "auto"


@pytest.mark.parametrize(
    ("project", "primary_peer", "primary_host", "family", "phase"),
    [
        ("cutlass-cute", "cutlass_cute", "vllm", "dense-gemm", "inference"),
        ("liger-kernel", "liger_kernel", "torchtitan", "training-fused-ops", "training"),
        ("deepgemm", "deepgemm", "vllm", "grouped-moe-gemm", "inference"),
        (
            "megatron-core",
            "deepgemm",
            "megatron_core",
            "grouped-moe-gemm",
            "training",
        ),
        ("torchtitan", "liger_kernel", "torchtitan", "training-fused-ops", "training"),
        ("verl", "liger_kernel", "verl", "training-fused-ops", "training"),
    ],
)
def test_extended_default_targets_bind_role_appropriate_candidates(
    project: str,
    primary_peer: str,
    primary_host: str,
    family: str,
    phase: str,
) -> None:
    draft, profile = build_default_draft(
        project=project,
        candidate=_candidate(f"{project}.operator"),
        created_by="test",
    )
    assert profile.repository.startswith("https://github.com/")
    assert draft.default_candidates
    assert draft.default_candidates.primary_peer_impl == primary_peer
    assert draft.default_candidates.primary_host == primary_host
    assert draft.default_candidates.request.operator_family == family
    assert draft.default_candidates.request.phase == phase


def test_materialized_registry_matches_builtin(project_root: Path) -> None:
    expected = build_default_candidate_registry().model_dump(mode="json")
    path = project_root / "catalog/default-candidates-v0.5/registry.json"
    assert json.loads(path.read_text(encoding="utf-8")) == expected
