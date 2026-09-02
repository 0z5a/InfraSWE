from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from infraswe.cli import app
from infraswe.draft.system_defaults import (
    COMMUNICATION_PROFILE_ORDER,
    MEMORY_PROFILE_ORDER,
    build_system_profile_catalog,
    select_memory_profile,
)
from infraswe.models.draft import (
    DraftAcceptanceContract,
    DraftMetadata,
    DraftTarget,
    ProjectComparisonCell,
)
from infraswe.models.system_paths import (
    CommunicationCellIdentity,
    CommunicationContract,
    CommunicationDraftSpec,
    CommunicationEfficiencyCard,
    CommunicationLifecycle,
    CommunicationSemantics,
    CompositeSystemPathPolicy,
    MemoryTier,
    MemoryTierCapacity,
    MemoryTierCellIdentity,
    MemoryTierDeployment,
    MemoryTierDraftSpec,
    MemoryTieringContract,
    MemoryTieringEfficiencyCard,
    OffloadBaseline,
    OffloadBaselineSet,
    ResidencyTransition,
    SystemPathCandidate,
    SystemPathInfraCertEvidence,
    SystemPathLoadCell,
    SystemRetrievalBinding,
)
from infraswe.scoring.deployability import score_maintainability, weighted_geometric
from infraswe.scoring.project_fit import (
    build_project_fit,
    score_evolutionary_maintainability,
    score_performance_reuse_utilization,
    score_project_contract_fit,
)
from infraswe.scoring.system_paths import (
    build_memory_tiering_cell_artifact,
    evaluate_system_path_infra_cert,
    project_operational_projection,
    score_system_path_concurrent_stability,
    score_system_path_implementation_reuse,
    score_transfer_efficiency,
    target_attainment,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _target() -> DraftTarget:
    return DraftTarget(
        mode="repository",
        repository="target/project",
        revision=_digest("1"),
        project_profile_sha256=_digest("2"),
    )


def _comparison_cell() -> ProjectComparisonCell:
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
        formula_template_id="project-fit-system-path-v0.5.1",
        evidence_policy_id="system-path-evidence-v1",
        project_season="test-2026q3",
    )


def _retrieval(plugin: str) -> SystemRetrievalBinding:
    return SystemRetrievalBinding(
        status="complete",
        anchor_plugin=plugin,
        corpus_cutoff=datetime(2026, 9, 1, tzinfo=UTC),
        snapshot_sha256=_digest("3"),
        query_plan_sha256=_digest("4"),
        leakage_audit_sha256=_digest("5"),
        precedent_set_sha256=_digest("6"),
        trust_card_sha256=_digest("7"),
    )


def _acceptance(*, sealed: bool = False) -> DraftAcceptanceContract:
    return DraftAcceptanceContract(
        status="sealed" if sealed else "proposed",
        path="contract.yaml",
        sha256=_digest("8"),
        probe_set_sha256=_digest("9"),
        hidden_probe_policy_sha256=_digest("a"),
        human_review_sha256=_digest("b") if sealed else None,
    )


def _communication_cell() -> CommunicationCellIdentity:
    return CommunicationCellIdentity(
        cell_id="single-node-primary",
        node_count=1,
        accelerators_per_node=8,
        accelerator_model="H100-SXM",
        topology_sha256=_digest("c"),
        nic_profile="none",
        numa_binding="gpu-local",
        transport="nvlink",
        provider="nccl",
        provider_version="2.28",
        runtime_driver="cuda-13/driver-610",
        message_size_portfolio_sha256=_digest("d"),
        collective_mix_sha256=_digest("e"),
        concurrency_protocol_id="communication-concurrency-core-v1",
    )


def _communication_contract() -> CommunicationContract:
    return CommunicationContract(
        layer="collective-library",
        execution_scope="integrated",
        providers=["nccl"],
        operations=["all-reduce"],
        semantics=CommunicationSemantics(
            ordering_scope="request",
            async_completion="event-backed",
        ),
        lifecycle=CommunicationLifecycle(
            communicator_owner="runtime",
            cache_policy="project-registry",
            teardown_phase="worker-shutdown",
        ),
        required_cells=[_communication_cell()],
        message_size_portfolio_sha256=_digest("d"),
        concurrency_protocol_id="communication-concurrency-core-v1",
    )


def _memory_contract(*, profile_id: str = "kv-cache-cpu-offload-v1"):
    return MemoryTieringContract(
        profile_id=profile_id,
        object_kind="kv-cache",
        mutability="request-scoped-mutable",
        source_tier_id="device0",
        destination_tier_id="host0",
        tiers=[
            MemoryTier(
                id="device0",
                kind="device",
                capacity_bytes=80 * 2**30,
                policy="required",
            ),
            MemoryTier(
                id="host0",
                kind="host-pinned",
                numa_node=0,
                capacity_bytes=256 * 2**30,
                policy="required",
            ),
        ],
        transitions=[
            ResidencyTransition(
                source_state="DEVICE_RESIDENT",
                target_state="EVICTING",
                operation="evict",
            ),
            ResidencyTransition(
                source_state="HOST_RESIDENT",
                target_state="PREFETCHING",
                operation="prefetch",
            ),
        ],
        version_token_required=True,
        allocator_owner="memory-tier-manager",
        residency_owner="kv-manager",
        copy_stream_owner="runtime",
        capacity=MemoryTierCapacity(
            device_budget_bytes=64 * 2**30,
            host_pinned_budget_bytes=128 * 2**30,
            host_pageable_budget_bytes=0,
            queue_limit=4096,
        ),
    )


def _baseline_set(*, scoring_outcome: str = "runnable") -> OffloadBaselineSet:
    return OffloadBaselineSet(
        semantic_baseline=OffloadBaseline(mode="target-head", revision=_digest("1")),
        scoring_baseline=OffloadBaseline(
            mode="target-head-incumbent",
            revision=_digest("1"),
            outcome=scoring_outcome,
        ),
        load_anchor=OffloadBaseline(mode="local-incumbent", revision=_digest("1")),
    )


def _memory_deployment() -> MemoryTierDeployment:
    cell = MemoryTierCellIdentity(
        cell_id="gpu-local-numa0",
        gpu_model="H100-PCIe",
        gpu_count=1,
        gpu_topology_sha256=_digest("2"),
        gpu_memory_bytes=80 * 2**30,
        cpu_model="EPYC",
        cpu_socket_count=2,
        numa_topology_sha256=_digest("3"),
        host_memory_bytes=512 * 2**30,
        cpu_affinity="pinned-workers",
        numa_policy="gpu-local",
        interconnect="PCIe-Gen5-x16",
        pinned_policy="required",
        host_page_policy="4k",
        os_kernel="6.8",
        driver_runtime="cuda-13/driver-610",
        framework="runtime-v1",
        allocator="project-pinned-pool",
        background_load_policy="isolated",
    )
    return MemoryTierDeployment(
        performance_mode="fixed-device-budget",
        workload_portfolio_sha256=_digest("4"),
        required_cells=[cell],
        service_target_sha256=_digest("5"),
        residency_target_sha256=_digest("6"),
        transfer_target_sha256=_digest("7"),
    )


def _load_cells(*, domain: str = "memory-tiering", value: float = 0.9):
    protocol = (
        "memory-tiering-load-normalized-v1"
        if domain == "memory-tiering"
        else "communication-concurrency-core-v1"
    )
    ratios = {
        "light": 0.25,
        "normal": 0.50,
        "knee": 0.80,
        "saturation": 1.00,
        "overload": 1.20,
        "soak": 1.00,
    }
    return [
        SystemPathLoadCell(
            domain=domain,
            protocol_id=protocol,
            regime=regime,
            load_ratio=ratio,
            offered_work=1000,
            completed_work=1000,
            goodput_score=value,
            tail_score=value,
            jitter_score=value,
            overlap_progress_score=value,
            resource_stability_score=value,
            fairness_score=value,
            p99_status="official",
            evidence_digests=[_digest("8")],
        )
        for regime, ratio in ratios.items()
    ]


def test_communication_draft_has_no_domain_or_cross_platform_score() -> None:
    draft = CommunicationDraftSpec(
        draft=DraftMetadata(
            id="communication-path-draft",
            revision=1,
            state="D3-contract-proposed",
            created_by="test",
        ),
        target=_target(),
        candidate=SystemPathCandidate(
            domain="distributed-communication",
            kind="git-diff",
            revision=_digest("f"),
            intent="change-algorithm",
            implementation_kind="communication-native",
            entrypoints=["enqueue_collective"],
        ),
        retrieval=_retrieval("communication-v1"),
        acceptance_contract=_acceptance(),
        communication=_communication_contract(),
    )
    assert draft.scoring.communication_domain_score == "forbidden"
    assert draft.scoring.generic_cross_platform_score == "forbidden"
    assert draft.scoring.operational_fit_source == "concurrent-stability"
    payload = draft.model_dump(mode="json")
    payload["scoring"]["communication_domain_score"] = "enabled"
    with pytest.raises(ValidationError):
        CommunicationDraftSpec.model_validate(payload)


def test_memory_tier_parent_is_not_sealable_and_concrete_profile_matches_object() -> None:
    base = {
        "draft": DraftMetadata(
            id="kv-cache-offload",
            revision=1,
            state="D6-sealed",
            created_by="test",
        ),
        "target": _target(),
        "candidate": SystemPathCandidate(
            domain="memory-tiering",
            kind="git-diff",
            revision=_digest("f"),
            intent="integrate",
            implementation_kind="memory-tiering-runtime",
            entrypoints=["prefetch_kv", "evict_kv"],
        ),
        "retrieval": _retrieval("memory-tier-v1"),
        "acceptance_contract": _acceptance(sealed=True),
        "baseline_set": _baseline_set(),
        "deployment": _memory_deployment(),
    }
    with pytest.raises(ValidationError, match="parent profile is not sealable"):
        MemoryTierDraftSpec(
            **base,
            memory_tiering=_memory_contract(profile_id="memory-tiering-offload-runtime-v1"),
        )
    concrete = MemoryTierDraftSpec(
        **base,
        memory_tiering=_memory_contract(),
    )
    assert concrete.memory_tiering.profile_id == "kv-cache-cpu-offload-v1"
    payload = concrete.memory_tiering.model_dump(mode="json")
    payload["profile_id"] = "weight-cpu-offload-v1"
    with pytest.raises(ValidationError, match="does not match"):
        MemoryTieringContract.model_validate(payload)


def test_system_path_c_is_domain_specific_but_still_the_single_c_component() -> None:
    cells = _load_cells(value=0.9)
    diagnostic = score_system_path_concurrent_stability(cells, fresh_process_replays=3)
    assert diagnostic.component.status == "diagnostic"
    scored = score_system_path_concurrent_stability(cells, fresh_process_replays=5)
    assert scored.component.status == "scored"
    assert scored.component.value == pytest.approx(0.9)
    operational = project_operational_projection(scored)
    assert operational.component.value == scored.component.value
    assert operational.component.input_evidence_digests == scored.component.input_evidence_digests


def test_system_path_hard_gate_cannot_be_hidden_by_other_load_cells() -> None:
    cells = _load_cells()
    cells[3] = cells[3].model_copy(update={"hard_gate_failure_codes": ["UNBOUNDED_PREFETCH_QUEUE"]})
    result = score_system_path_concurrent_stability(cells, fresh_process_replays=7)
    assert "UNBOUNDED_PREFETCH_QUEUE:saturation" in result.failure_codes
    assert result.component.value == 0


def test_system_path_infracert_fails_memory_safety_and_never_turns_it_into_score() -> None:
    evidence = SystemPathInfraCertEvidence(
        domain="memory-tiering",
        evidence_digests=[_digest("1")],
        correctness_passed=True,
        progress_passed=True,
        lifecycle_quiescent=True,
        bounded_resources=True,
        fallback_policy_respected=True,
        residency_state_valid=True,
        version_token_valid=True,
        consumer_visibility_valid=True,
        isolation_valid=True,
        use_after_free_absent=False,
        partial_copy_absent=True,
        stale_or_lost_update_absent=True,
        prefetch_queue_bounded=False,
        host_memory_leak_absent=True,
        pageable_fallback_explicit=True,
    )
    result = evaluate_system_path_infra_cert(evidence)
    assert result.status == "fail"
    assert set(result.failure_codes) == {"UNBOUNDED_PREFETCH_QUEUE", "USE_AFTER_FREE"}
    assert not hasattr(result, "score_100")


def test_system_path_infracert_is_unresolved_when_required_oracle_is_missing() -> None:
    evidence = SystemPathInfraCertEvidence(
        domain="distributed-communication",
        evidence_digests=[_digest("1")],
        correctness_passed=True,
        progress_passed=True,
        lifecycle_quiescent=True,
        bounded_resources=True,
        fallback_policy_respected=True,
        collective_order_consistent=True,
        rank_divergence_absent=None,
        deadlock_absent=True,
    )
    result = evaluate_system_path_infra_cert(evidence)
    assert result.status == "unresolved"
    assert result.missing_checks == ["rank_divergence_absent"]


def test_system_path_reuse_has_no_portability_component() -> None:
    result = score_system_path_implementation_reuse(
        coverage=0.9,
        family_reuse=0.8,
        cache_reuse=0.7,
        observed_variants=4,
        expected_variant_budget=8,
        maximum_variant_budget=32,
    )
    assert result.component.status == "scored"
    assert result.raw_metrics["portability_component_present"] is False


def test_memory_tiering_cell_artifact_is_cell_local_and_rewards_useful_transfer() -> None:
    concurrent = score_system_path_concurrent_stability(
        _load_cells(value=0.8), fresh_process_replays=7
    )
    reuse = score_system_path_implementation_reuse(
        coverage=0.8,
        family_reuse=0.8,
        cache_reuse=0.8,
        observed_variants=4,
        expected_variant_budget=8,
        maximum_variant_budget=32,
    )
    maintainability = score_maintainability(
        contract=0.8,
        locality=0.8,
        tests=0.8,
        build=0.8,
    )
    artifact = build_memory_tiering_cell_artifact(
        concurrent_stability=concurrent,
        implementation_reuse=reuse,
        maintainability=maintainability,
        service_performance_attainment=0.8,
        device_residency_attainment=0.8,
        transfer_efficiency=0.8,
    )
    expected = 100 * weighted_geometric(
        {name: float(component.value) for name, component in artifact.components.items()},
        {
            "concurrent_stability": 0.25,
            "implementation_reuse": 0.15,
            "maintainability": 0.15,
            "service_performance_attainment": 0.20,
            "device_residency_attainment": 0.15,
            "transfer_efficiency": 0.10,
        },
    )
    assert artifact.score_100 == pytest.approx(expected)
    assert artifact.cross_cell_ranking_allowed is False
    efficient = score_transfer_efficiency(
        useful_bandwidth_attainment=0.9,
        actual_transfer_bytes=100,
        semantic_useful_bytes=100,
        traffic_amplification_budget=1.2,
        stall_attainment=0.9,
    )
    repeated = score_transfer_efficiency(
        useful_bandwidth_attainment=0.9,
        actual_transfer_bytes=300,
        semantic_useful_bytes=100,
        traffic_amplification_budget=1.2,
        stall_attainment=0.9,
    )
    assert repeated < efficient


def test_system_path_project_fit_uses_identity_o_and_ordinary_no_x_formula() -> None:
    maintainability = score_evolutionary_maintainability(
        {name: 0.9 for name in ("evolution", "locality", "tests", "failure", "contract")}
    )
    contract = score_project_contract_fit(
        {name: 0.9 for name in ("integration", "interface", "lifecycle", "buildtest", "policy")}
    )
    reuse = score_performance_reuse_utilization(
        {name: 0.9 for name in ("attainment", "coverage", "retention", "family", "compile")}
    )
    concurrent = score_system_path_concurrent_stability(
        _load_cells(value=0.9), fresh_process_replays=7
    )
    operational = project_operational_projection(concurrent)
    fit = build_project_fit(
        mode="official",
        infra_cert="pass",
        formula_template_id="project-fit-system-path-v0.5.1",
        comparison_cell=_comparison_cell(),
        evolutionary_maintainability=maintainability,
        project_contract_fit=contract,
        performance_reuse_utilization=reuse,
        operational_fit=operational,
        fresh_process_replays=7,
        evidence_grade="E2-system-trace",
        hidden_probes_complete=True,
        manifest_verified=True,
        sealed_draft_sha256=_digest("f"),
    )
    assert fit.score_100 == pytest.approx(90)


def test_attainment_cards_and_composite_policy_never_create_extra_scores() -> None:
    assert target_attainment(85, direction="higher", target=100, zero_limit=50) == 0.7
    communication = CommunicationEfficiencyCard(
        cell_identity_sha256=_digest("1"),
        latency={},
        algorithmic_bandwidth={},
        bus_bandwidth={},
        overlap={},
        rank_skew={},
        lifecycle={},
        raw_evidence_digests=[_digest("2")],
    )
    memory = MemoryTieringEfficiencyCard(
        cell_identity_sha256=_digest("1"),
        service={},
        residency={},
        transfer={},
        host_system={},
        raw_evidence_digests=[_digest("2")],
    )
    assert communication.score_status == memory.score_status == "not-a-score"
    assert not communication.cross_cell_ranking_allowed
    assert CompositeSystemPathPolicy().concurrent_stability_aggregation_count == 1


def test_system_profile_catalog_has_concrete_communication_and_memory_profiles() -> None:
    catalog = build_system_profile_catalog()
    assert catalog.profile_order == [*COMMUNICATION_PROFILE_ORDER, *MEMORY_PROFILE_ORDER]
    assert len(catalog.profiles) == 16
    abstract = catalog.profiles["memory-tiering-offload-runtime-v1"]
    assert abstract.sealable is False
    assert abstract.object_kind is None
    assert all(catalog.profiles[profile_id].sealable for profile_id in COMMUNICATION_PROFILE_ORDER)
    assert all(
        catalog.profiles[profile_id].triton_portability_score == "not-applicable"
        for profile_id in catalog.profile_order
    )
    assert select_memory_profile("kv-cache").profile_id == "kv-cache-cpu-offload-v1"
    assert select_memory_profile("training-state").profile_id == ("training-state-cpu-offload-v1")


def test_system_profile_catalog_cli_materializes_replayable_files(tmp_path) -> None:
    output = tmp_path / "system-profiles"
    result = CliRunner().invoke(
        app,
        ["draft", "system-profiles", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    catalog = build_system_profile_catalog()
    assert json.loads((output / "catalog.json").read_text(encoding="utf-8")) == (
        catalog.model_dump(mode="json")
    )
    for profile_id, profile in catalog.profiles.items():
        profile_path = output / "profiles" / f"{profile_id}.json"
        assert json.loads(profile_path.read_text(encoding="utf-8")) == (
            profile.model_dump(mode="json")
        )


def test_checked_in_system_profile_catalog_matches_builtin(project_root) -> None:
    root = project_root / "catalog" / "system-drafts-v0.5.2"
    catalog = build_system_profile_catalog()
    assert json.loads((root / "catalog.json").read_text(encoding="utf-8")) == (
        catalog.model_dump(mode="json")
    )
    for profile_id, profile in catalog.profiles.items():
        path = root / "profiles" / f"{profile_id}.json"
        assert json.loads(path.read_text(encoding="utf-8")) == profile.model_dump(mode="json")
