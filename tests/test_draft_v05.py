from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from infraswe.draft.defaults import (
    DEFAULT_PROJECT_ORDER,
    build_default_catalog,
    build_default_draft,
)
from infraswe.draft.lifecycle import (
    advance_draft_state,
    audit_seal,
    canonical_sha256,
    seal_draft,
)
from infraswe.draft.precompile import decide_precompile
from infraswe.draft.resolver import parse_draft_document, resolve_draft
from infraswe.draft.selection import evidence_cache_key, select_affected_cases
from infraswe.models.draft import (
    AffectedCase,
    DraftCandidate,
    DraftRevisionEvent,
    EvidenceCacheIdentity,
    HumanReviewRecord,
    ProjectComparisonCell,
    RemoteGitDraftLocation,
)
from infraswe.models.project_score import (
    BenchmarkCostCard,
    CellEfficiencyReference,
    MergeabilityDecision,
    ProjectObjectiveResult,
    PullRequestReviewContext,
    PureTritonEligibilityEvidence,
)
from infraswe.scoring.deployability import weighted_geometric
from infraswe.scoring.project_fit import (
    PROJECT_FIT_WEIGHTS,
    audit_pure_triton,
    build_project_fit,
    build_v05_result,
    compile_mergeability_decision,
    score_benchmark_trust,
    score_evolutionary_maintainability,
    score_operational_fit,
    score_performance_reuse_utilization,
    score_project_contract_fit,
    score_pure_triton_portability,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _candidate(
    *,
    entrypoint: str = "operator.forward",
    implementation_kind: str = "framework",
) -> DraftCandidate:
    return DraftCandidate(
        kind="generated",
        revision=_digest("a"),
        intent="integrate",
        implementation_kind=implementation_kind,
        entrypoints=[entrypoint],
    )


def _comparison_cell(
    formula: str = "project-fit-kernel-v0.5",
) -> ProjectComparisonCell:
    return ProjectComparisonCell(
        target_project_profile_sha256=_digest("1"),
        target_repository_or_baseline_sha256=_digest("2"),
        change_intent="integrate",
        semantic_contract_sha256=_digest("3"),
        acceptance_contract_sha256=_digest("4"),
        probe_set_sha256=_digest("5"),
        workload_portfolio_sha256=_digest("6"),
        performance_target_sha256=_digest("7"),
        required_deployment_cell_set_sha256=_digest("8"),
        formula_template_id=formula,
        evidence_policy_id="evidence-v0.5",
        project_season="test-2026q3",
    )


def _dimensions(value: float = 0.9):
    maintainability = score_evolutionary_maintainability(
        {
            "evolution": value,
            "locality": value,
            "tests": value,
            "failure": value,
            "contract": value,
        }
    )
    contract = score_project_contract_fit(
        {
            "integration": value,
            "interface": value,
            "lifecycle": value,
            "buildtest": value,
            "policy": value,
        }
    )
    reuse = score_performance_reuse_utilization(
        {
            "attainment": value,
            "coverage": value,
            "retention": value,
            "family": value,
            "compile": value,
        }
    )
    operational = score_operational_fit(
        {
            "replay": value,
            "load": value,
            "resource": value,
            "coldsteady": value,
        }
    )
    return maintainability, contract, reuse, operational


def _ordinary_fit(*, mode: str = "official", value: float = 0.9, **overrides):
    maintainability, contract, reuse, operational = _dimensions(value)
    arguments = {
        "mode": mode,
        "infra_cert": "pass",
        "formula_template_id": "project-fit-kernel-v0.5",
        "comparison_cell": _comparison_cell(),
        "evolutionary_maintainability": maintainability,
        "project_contract_fit": contract,
        "performance_reuse_utilization": reuse,
        "operational_fit": operational,
        "fresh_process_replays": 7,
        "evidence_grade": "E2-system-trace",
        "hidden_probes_complete": True,
        "manifest_verified": True,
        "sealed_draft_sha256": _digest("9"),
    }
    arguments.update(overrides)
    return build_project_fit(**arguments)


def _review_for(draft) -> HumanReviewRecord:
    assert draft.target and draft.acceptance_contract and draft.deployment and draft.scoring
    return HumanReviewRecord(
        reviewer="upstream-maintainer@example.org",
        decision="approve",
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
        target_profile_sha256=draft.target.project_profile_sha256,
        acceptance_contract_sha256=draft.acceptance_contract.sha256,
        probe_set_sha256=draft.acceptance_contract.probe_set_sha256,
        workload_portfolio_sha256=draft.deployment.workload_portfolio.sha256,
        formula_template_id=draft.scoring.formula_template_id,
        notes_sha256=_digest("f"),
    )


def test_default_catalog_is_pinned_complete_and_machine_proposed() -> None:
    catalog = build_default_catalog()
    assert tuple(catalog.default_order) == DEFAULT_PROJECT_ORDER
    assert catalog.status == "proposed"
    assert set(catalog.entries) == set(DEFAULT_PROJECT_ORDER)
    for project, entry in catalog.entries.items():
        assert entry.profile.status == "proposed"
        assert entry.profile.repository.startswith("https://github.com/")
        assert len(entry.source_revision) == 40
        assert set(entry.artifacts) == {
            "api-abi",
            "lifecycle",
            "build-test-matrix",
            "dependency-policy",
            "fallback-policy",
            "deployment-workload-portfolio",
            "performance-acceptance-targets",
            "maintainability-probes",
        }
        assert project in entry.aliases[0]


def test_materialized_default_catalog_matches_builtin(project_root: Path) -> None:
    root = project_root / "catalog" / "default-drafts-v0.5"
    catalog = build_default_catalog()
    assert json.loads((root / "catalog.json").read_text(encoding="utf-8")) == (
        catalog.model_dump(mode="json")
    )
    for project, entry in catalog.entries.items():
        assert json.loads(
            (root / project / "project-profile.json").read_text(encoding="utf-8")
        ) == entry.profile.model_dump(mode="json")
        for kind, artifact in entry.artifacts.items():
            assert json.loads(
                (root / project / "contracts" / f"{kind}.json").read_text(encoding="utf-8")
            ) == artifact.model_dump(mode="json")


def test_default_resolution_uses_alias_then_frozen_priority() -> None:
    flashinfer = resolve_draft(candidate=_candidate(entrypoint="flashinfer.decode"))
    assert flashinfer.source_kind == "default-catalog"
    assert flashinfer.selected_default_project == "flashinfer"
    assert flashinfer.draft.draft.state == "D3-contract-proposed"
    assert flashinfer.bundled_profile and flashinfer.bundled_profile.status == "proposed"

    priority = resolve_draft(candidate=_candidate(entrypoint="generic.operator"))
    assert priority.selected_default_project == "vllm"
    assert "DEFAULT_TARGET_SELECTED_BY_PRIORITY" in priority.audit_flags
    with pytest.raises(ValueError, match="requires a candidate"):
        resolve_draft()


@pytest.mark.parametrize(
    ("entrypoint", "expected_project"),
    [
        ("nvidia/cutlass::gemm", "cutlass-cute"),
        ("linkedin/liger-kernel::cross_entropy", "liger-kernel"),
        ("deepseek-ai/deepgemm::grouped_gemm", "deepgemm"),
        ("nvidia/megatron-lm::tensor_parallel", "megatron-core"),
        ("pytorch/torchtitan::train_step", "torchtitan"),
        ("volcengine/verl::rollout", "verl"),
    ],
)
def test_extended_default_catalog_aliases_resolve_exact_target(
    entrypoint: str, expected_project: str
) -> None:
    resolution = resolve_draft(candidate=_candidate(entrypoint=entrypoint))
    assert resolution.selected_default_project == expected_project
    assert "DEFAULT_TARGET_SELECTED_BY_PRIORITY" not in resolution.audit_flags


def test_default_catalog_is_built_once_for_hot_path_resolution() -> None:
    assert build_default_catalog() is build_default_catalog()


def test_default_draft_precompiles_unavoidable_builds_before_timing() -> None:
    for project in DEFAULT_PROJECT_ORDER:
        draft, _ = build_default_draft(project=project, candidate=_candidate(), created_by="test")
        assert draft.benchmark_loop
        policy = draft.benchmark_loop.precompile
        assert policy.mode == "auto"
        miss = decide_precompile(policy, compilation_required=True, cache_hit=False)
        assert miss.action == "precompile-before-timed-cases"
        assert miss.steady_state_compile_allowed is False
        hit = decide_precompile(policy, compilation_required=True, cache_hit=True)
        assert hit.action == "reuse-precompiled-artifact"


def test_disabled_precompile_is_visible_when_compilation_is_required() -> None:
    draft, _ = build_default_draft(project="vllm", candidate=_candidate(), created_by="test")
    assert draft.benchmark_loop
    disabled = draft.benchmark_loop.precompile.model_copy(update={"mode": "off"})
    decision = decide_precompile(disabled, compilation_required=True, cache_hit=False)
    assert decision.action == "compile-inline-with-warning"
    assert decision.rationale_codes == ["PRECOMPILE_DISABLED_FOR_REQUIRED_BUILD"]


def test_local_draft_precedes_remote_and_remote_precedes_default(tmp_path: Path) -> None:
    draft, _ = build_default_draft(project="sglang", candidate=_candidate(), created_by="test")
    local = tmp_path / "draft.yaml"
    local.write_text(
        json.dumps(draft.model_dump(mode="json")),
        encoding="utf-8",
    )
    remote = RemoteGitDraftLocation(
        repository="https://example.invalid/repository.git",
        revision="main",
        path="drafts/default.yaml",
    )
    local_result = resolve_draft(
        local_draft=local,
        remote_git_draft=remote,
        remote_reader=lambda _: pytest.fail("shadowed remote reader was called"),
    )
    assert local_result.source_kind == "local"
    assert "REMOTE_GIT_DRAFT_SHADOWED_BY_LOCAL" in local_result.audit_flags

    remote_result = resolve_draft(
        remote_git_draft=remote,
        candidate=_candidate(),
        remote_reader=lambda _: json.dumps(draft.model_dump(mode="json")),
    )
    assert remote_result.source_kind == "remote-git"
    assert remote_result.draft.target and remote_result.draft.target.catalog_profile


def test_draft_parser_accepts_yaml_and_remote_paths_reject_traversal() -> None:
    draft, _ = build_default_draft(
        project="flash-attention", candidate=_candidate(), created_by="test"
    )
    yaml_text = json.dumps(draft.model_dump(mode="json"))
    assert parse_draft_document(yaml_text, source="memory").draft.id == draft.draft.id
    with pytest.raises(ValidationError, match="repository-relative"):
        RemoteGitDraftLocation(
            repository="https://example.invalid/repo.git",
            path="../draft.yaml",
        )


def test_draft_state_requires_human_review_and_seal_status() -> None:
    draft, _ = build_default_draft(project="vllm", candidate=_candidate(), created_by="test")
    payload = draft.model_dump(mode="json")
    payload["draft"]["state"] = "D4-human-reviewed"
    with pytest.raises(ValidationError, match="human-reviewed acceptance"):
        type(draft).model_validate(payload)

    payload["acceptance_contract"]["status"] = "human-reviewed"
    payload["acceptance_contract"]["human_review_sha256"] = _digest("e")
    reviewed = type(draft).model_validate(payload)
    fast = advance_draft_state(reviewed, "D5-fast-loop")
    assert fast.draft.state == "D5-fast-loop"
    with pytest.raises(ValueError, match="exactly one step"):
        advance_draft_state(fast, "D8-decided")


def test_seal_requires_matching_approved_human_maintainer_review() -> None:
    draft, _ = build_default_draft(project="vllm", candidate=_candidate(), created_by="test")
    review = _review_for(draft)
    payload = draft.model_dump(mode="json")
    payload["draft"]["state"] = "D5-fast-loop"
    payload["acceptance_contract"]["status"] = "human-reviewed"
    payload["acceptance_contract"]["human_review_sha256"] = canonical_sha256(review)
    fast = type(draft).model_validate(payload)
    sealed = seal_draft(
        fast,
        review,
        performance_target_sha256=_digest("e"),
        sealed_by="release-maintainer@example.org",
        sealed_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
    )
    assert audit_seal(sealed) == []
    assert sealed.material.candidate_sha256 == fast.candidate.revision

    tampered = sealed.model_copy(update={"sealed_by": "attacker"})
    assert audit_seal(tampered) == ["DRAFT_SEAL_DIGEST_MISMATCH"]

    mismatched = review.model_copy(update={"probe_set_sha256": _digest("0")})
    with pytest.raises(ValueError, match="does not match the Draft"):
        seal_draft(
            fast,
            mismatched,
            performance_target_sha256=_digest("e"),
            sealed_by="maintainer",
        )


def test_candidate_and_contract_loops_have_distinct_revision_semantics() -> None:
    DraftRevisionEvent(
        draft_id="draft-v05",
        from_revision=1,
        to_revision=1,
        loop_kind="candidate",
        old_candidate_sha256=_digest("a"),
        new_candidate_sha256=_digest("b"),
        old_contract_sha256=_digest("c"),
        new_contract_sha256=_digest("c"),
        actor_role="draft-owner",
        reason="candidate optimization",
    )
    with pytest.raises(ValidationError, match="project-maintainer"):
        DraftRevisionEvent(
            draft_id="draft-v05",
            from_revision=1,
            to_revision=2,
            loop_kind="contract",
            old_candidate_sha256=_digest("a"),
            new_candidate_sha256=_digest("a"),
            old_contract_sha256=_digest("c"),
            new_contract_sha256=_digest("d"),
            actor_role="draft-owner",
            reason="changed scoring after result",
        )


def test_affected_case_selection_keeps_all_mandatory_negative_categories() -> None:
    categories = [
        "positive",
        "negative-control",
        "fallback-or-unsupported",
        "hidden-adjacent",
        "build-import-load",
    ]
    cases = [
        AffectedCase(
            case_id=f"case-{index}",
            symbols=["changed.symbol"] if index == 0 else [],
            categories=[category],
        )
        for index, category in enumerate(categories)
    ]
    plan = select_affected_cases(changed_symbols=["changed.symbol"], cases=cases)
    assert plan.required_categories_present
    assert plan.coverage_confidence == "high"
    assert all(decision.selected for decision in plan.decisions)


def test_evidence_cache_identity_includes_environment_and_collector() -> None:
    identity = EvidenceCacheIdentity(
        target_repository_sha256=_digest("1"),
        project_profile_sha256=_digest("2"),
        candidate_sha256=_digest("3"),
        compiler="nvcc-12.8",
        runtime="torch-2.11",
        driver="570.211.01",
        hardware_cell="l40s-sm89",
        workload_case="sft-step",
        probe_version="probe-v1",
        collector_version="collector-v1",
        environment_digest=_digest("4"),
    )
    changed = identity.model_copy(update={"collector_version": "collector-v2"})
    assert evidence_cache_key(identity) != evidence_cache_key(changed)


def test_project_dimensions_and_ordinary_formula_are_frozen() -> None:
    fit = _ordinary_fit()
    assert fit.status == "official"
    assert math.isclose(fit.score_100 or 0, 90.0)
    assert fit.components["pure_triton_portability"].status == "not_applicable"
    numeric = {
        name: float(fit.components[name].value)
        for name in PROJECT_FIT_WEIGHTS["project-fit-kernel-v0.5"]
    }
    expected = 100 * weighted_geometric(numeric, PROJECT_FIT_WEIGHTS["project-fit-kernel-v0.5"])
    assert math.isclose(fit.score_100 or 0, expected)
    assert fit.cross_project_ranking_allowed is False


def test_missing_component_or_official_evidence_is_unresolved_without_renormalizing() -> None:
    missing = score_operational_fit(
        {"replay": 0.9, "load": 0.9, "resource": None, "coldsteady": 0.9}
    )
    maintainability, contract, reuse, _ = _dimensions()
    provisional = build_project_fit(
        mode="provisional",
        infra_cert="pass",
        formula_template_id="project-fit-kernel-v0.5",
        comparison_cell=_comparison_cell(),
        evolutionary_maintainability=maintainability,
        project_contract_fit=contract,
        performance_reuse_utilization=reuse,
        operational_fit=missing,
    )
    assert provisional.status == "provisional"
    assert provisional.score_100 is None

    three_replays = _ordinary_fit(fresh_process_replays=3)
    assert three_replays.status == "unresolved"
    assert three_replays.score_100 is None
    no_seal = _ordinary_fit(sealed_draft_sha256=None)
    assert no_seal.status == "unresolved"


def test_hard_gate_failure_does_not_issue_project_fit() -> None:
    fit = _ordinary_fit(infra_cert="fail")
    assert fit.status == "not_issued"
    assert fit.score_100 is None
    assert all(
        component.status == "not_run_due_to_gate"
        for name, component in fit.components.items()
        if name != "pure_triton_portability"
    )


def test_component_floor_cannot_be_compensated_by_other_high_scores() -> None:
    maintainability, contract, reuse, _ = _dimensions(0.95)
    low_operational = score_operational_fit(
        {"replay": 0.5, "load": 0.5, "resource": 0.5, "coldsteady": 0.5}
    )
    fit = build_project_fit(
        mode="official",
        infra_cert="pass",
        formula_template_id="project-fit-kernel-v0.5",
        comparison_cell=_comparison_cell(),
        evolutionary_maintainability=maintainability,
        project_contract_fit=contract,
        performance_reuse_utilization=reuse,
        operational_fit=low_operational,
        fresh_process_replays=7,
        evidence_grade="E3-kernel-counter",
        hidden_probes_complete=True,
        manifest_verified=True,
        sealed_draft_sha256=_digest("9"),
    )
    assert fit.status == "not_acceptable"
    decision = compile_mergeability_decision(
        infra_cert="pass", project_fit=fit, project_objectives={}
    )
    assert decision.verdict == "reject"
    assert "PROJECT_COMPONENT_FLOOR_FAILED" in decision.rationale_codes


def test_pure_triton_audit_rejects_hidden_native_paths_and_missing_profiles() -> None:
    base = {
        "implementation_kind": "triton-pure",
        "core_kernel_language": "triton",
        "backend_native_kernel_calls": [],
        "backend_specific_code_locations": ["adapters/cuda.py"],
        "backend_specific_code_is_capability_or_launch_only": True,
        "shared_semantic_implementation_family": True,
        "required_profile_evidence": {"cuda": "verified", "rocm": "verified"},
        "local_baseline_normalization": True,
        "absolute_cross_hardware_latency_ranking": False,
        "explicit_unsupported_and_fallback": True,
    }
    passed = audit_pure_triton(PureTritonEligibilityEvidence(**base))
    assert passed.status == "pass"

    hidden = dict(base)
    hidden["backend_native_kernel_calls"] = ["cuda_extension.fused_kernel"]
    failed = audit_pure_triton(PureTritonEligibilityEvidence(**hidden))
    assert failed.status == "fail"
    assert "TRITON_PURE_HIDDEN_NATIVE_PATH" in failed.failure_codes

    missing = dict(base)
    missing["required_profile_evidence"] = {"cuda": "verified", "rocm": "missing"}
    unresolved = audit_pure_triton(PureTritonEligibilityEvidence(**missing))
    assert unresolved.status == "unresolved"


def test_valid_pure_triton_uses_x_without_cross_hardware_absolute_ranking() -> None:
    maintainability, contract, reuse, operational = _dimensions(0.9)
    portability = score_pure_triton_portability(
        {
            "coverage": 0.8,
            "localretention": 0.8,
            "sharedcore": 0.8,
            "degradation": 0.8,
        }
    )
    audit = audit_pure_triton(
        PureTritonEligibilityEvidence(
            implementation_kind="triton-pure",
            core_kernel_language="triton",
            backend_specific_code_is_capability_or_launch_only=True,
            shared_semantic_implementation_family=True,
            required_profile_evidence={"cuda": "verified", "rocm": "verified"},
            local_baseline_normalization=True,
            absolute_cross_hardware_latency_ranking=False,
            explicit_unsupported_and_fallback=True,
        )
    )
    fit = build_project_fit(
        mode="official",
        infra_cert="pass",
        formula_template_id="project-fit-triton-pure-v0.5",
        comparison_cell=_comparison_cell("project-fit-triton-pure-v0.5"),
        evolutionary_maintainability=maintainability,
        project_contract_fit=contract,
        performance_reuse_utilization=reuse,
        operational_fit=operational,
        pure_triton_portability=portability,
        triton_purity_audit=audit,
        fresh_process_replays=7,
        evidence_grade="E3-kernel-counter",
        hidden_probes_complete=True,
        manifest_verified=True,
        sealed_draft_sha256=_digest("9"),
    )
    assert fit.status == "official"
    assert fit.components["pure_triton_portability"].status == "scored"


def test_benchmark_trust_is_independent_and_missing_is_unresolved() -> None:
    scored = score_benchmark_trust(
        reproducibility=0.9,
        evidence=0.8,
        statistics=0.7,
        environment=0.6,
    )
    assert scored.status == "scored"
    assert scored.score_100 is not None
    missing = score_benchmark_trust(
        reproducibility=0.9,
        evidence=None,
        statistics=0.7,
        environment=0.6,
    )
    assert missing.status == "unresolved"
    assert missing.score_100 is None


def test_v05_result_never_places_provisional_score_on_leaderboard() -> None:
    fit = _ordinary_fit(mode="provisional")
    trust = score_benchmark_trust(
        reproducibility=0.9,
        evidence=0.9,
        statistics=0.9,
        environment=0.9,
    )
    objective = ProjectObjectiveResult(
        policy="roadmap",
        status="not-tested",
    )
    decision = compile_mergeability_decision(
        infra_cert="pass",
        project_fit=fit,
        project_objectives={"edge": objective},
    )
    result = build_v05_result(
        draft_id="draft-v05",
        draft_revision=1,
        draft_state="D5-fast-loop",
        sealed_draft_sha256=None,
        target_project_profile_sha256=_digest("1"),
        target_repository_sha256=_digest("2"),
        candidate_sha256=_digest("3"),
        acceptance_contract_sha256=_digest("4"),
        infra_cert="pass",
        project_fit=fit,
        benchmark_trust=trust,
        benchmark_cost=BenchmarkCostCard(
            wall_time_seconds=1,
            accelerator_seconds=1,
            compile_seconds=0.2,
            precompile_seconds=0.2,
            cold_start_seconds=0.2,
            steady_state_seconds=0.5,
            steady_state_compile_seconds=0,
            compilation_path="precompile",
            profiler_seconds=0,
            executed_cases=4,
            skipped_cases=1,
            cache_hit_ratio=0.5,
            fast_stage_resolution_rate=0.8,
            serialization_config_compatibility="pass",
        ),
        evidence_grade="E1-framework",
        project_objectives={"edge": objective},
        cell_efficiency=CellEfficiencyReference(status="unresolved"),
        decision=decision,
    )
    assert result.project_fit.score_100 == pytest.approx(90)
    assert result.leaderboard_effective_project_fit_100 is None
    assert result.decision.verdict == "unresolved"
    assert result.project_objectives["edge"].weighted_score is None

    payload = result.model_dump(mode="json")
    payload["draft_state"] = "D3-contract-proposed"
    with pytest.raises(ValidationError, match="only valid in the D5 fast loop"):
        type(result).model_validate(payload)


def test_polarized_mergeability_requires_85_and_limits_check_to_active_review() -> None:
    below = _ordinary_fit(value=0.8499)
    at_floor = _ordinary_fit(value=0.85)
    assert (
        compile_mergeability_decision(
            infra_cert="pass", project_fit=below, project_objectives={}
        ).verdict
        == "reject"
    )
    assert (
        compile_mergeability_decision(
            infra_cert="pass", project_fit=at_floor, project_objectives={}
        ).verdict
        == "accept"
    )

    observed = datetime(2026, 9, 2, tzinfo=UTC)
    active = PullRequestReviewContext(
        created_at=observed - timedelta(days=10),
        observed_at=observed,
        last_activity_at=observed - timedelta(days=1),
        last_human_review_at=observed - timedelta(days=1),
        current_head_human_non_author_review_count=1,
        total_human_non_author_review_count=1,
    )
    active_decision = compile_mergeability_decision(
        infra_cert="pass",
        project_fit=below,
        project_objectives={},
        review_context=active,
    )
    assert active_decision.verdict == "check"
    assert "ACTIVE_NEW_PR_REVIEW_CHECK_ELIGIBLE" in active_decision.rationale_codes

    legacy = MergeabilityDecision.model_validate(
        {
            "verdict": "revise",
            "rationale_codes": ["ACTIVE_NEW_PR_REVIEW_REVISE_ELIGIBLE"],
        }
    )
    assert legacy.verdict == "check"
    assert legacy.model_dump(mode="json")["verdict"] == "check"

    stale = PullRequestReviewContext(
        created_at=observed - timedelta(days=180),
        observed_at=observed,
        last_activity_at=observed - timedelta(days=60),
        last_human_review_at=observed - timedelta(days=60),
        total_human_non_author_review_count=2,
    )
    stale_decision = compile_mergeability_decision(
        infra_cert="pass",
        project_fit=below,
        project_objectives={},
        review_context=stale,
    )
    assert stale_decision.verdict == "reject"
    assert "STALE_REVIEWED_OPEN_REJECT" in stale_decision.rationale_codes
