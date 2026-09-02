from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from infraswe.capability import (
    assert_raw_performance_comparable,
    audit_benchmark_cell,
    audit_capability_resolution,
    audit_resource_lease,
    build_attestation,
    build_benchmark_cell,
    build_registry,
    build_resource_lease,
    evaluate_candidate_capability_use,
    evaluate_resource_feasibility,
    evaluate_resource_usage,
    match_topology,
    merge_attestations,
    resolve_capabilities,
)
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    BenchmarkCellPolicy,
    CandidateCapabilityDeclaration,
    CandidateCapabilityUseObservation,
    CapabilityContract,
    CapabilityDefinition,
    CapabilityExpression,
    CapabilityProbeIdentity,
    CapabilityProofPolicy,
    CapabilityRequirement,
    CapabilityVariant,
    PhaseCapabilityContract,
    ResourceAvailability,
    ResourceEnvelope,
    ResourceLimit,
    ResourcePhaseEnvelope,
    ResourceUsageObservation,
    RunnerAcceleratorIdentity,
    RunnerHostIdentity,
    RunnerManifest,
    RunnerSelectionPolicy,
    RunnerSnapshot,
    TopologyContract,
    TopologyEdge,
    TopologyGraph,
    TopologyRelationRequirement,
    TopologyVertex,
    TopologyVertexRequirement,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _seal(model, field: str):
    material = model.model_dump(mode="json", exclude={field})
    return model.model_copy(update={field: canonical_sha256(material)})


def _requirement(capability_id: str, mode: str, proof: str) -> CapabilityExpression:
    return CapabilityExpression(
        operation="capability",
        requirement=CapabilityRequirement(
            capability_id=capability_id,
            mode=mode,
            min_proof=proof,
        ),
    )


def _fixture():
    registry = build_registry(
        registry_id="official-capabilities",
        revision=1,
        definitions=[
            CapabilityDefinition(
                capability_id="backend.cuda",
                version=1,
                domain="accelerator-backend",
                kind="semantic",
                description="A CUDA execution backend is available and usable.",
                proof_policy=CapabilityProofPolicy(minimum_default="CP3-runtime"),
                probes={"runtime": "cuda-runtime-v1"},
            ),
            CapabilityDefinition(
                capability_id="accelerator.dtype.bf16",
                version=1,
                domain="accelerator-numeric",
                kind="semantic",
                description="Native BF16 tensor operations execute correctly.",
                proof_policy=CapabilityProofPolicy(minimum_default="CP3-runtime"),
                probes={"runtime": "bf16-runtime-v1"},
            ),
            CapabilityDefinition(
                capability_id="nvidia.cuda.tma",
                version=1,
                domain="accelerator-memory",
                kind="mechanism",
                description="Native multidimensional tensor memory transfer is available.",
                proof_policy=CapabilityProofPolicy(minimum_default="CP3-runtime"),
                probes={"behavior": "cuda-tma-mechanism-v1"},
            ),
        ],
    )
    resource = ResourceEnvelope(
        policy_id="single-gpu-tma-v1",
        phases={
            "verify": ResourcePhaseEnvelope(
                resources={
                    "accelerator.count": ResourceLimit(
                        unit="count",
                        minimum_required=1,
                        reserved=1,
                        candidate_limit=1,
                        exclusive=True,
                    ),
                    "accelerator.memory.bytes": ResourceLimit(
                        unit="bytes",
                        minimum_required=24,
                        reserved=24,
                        candidate_limit=22,
                        measurement_reserve=2,
                    ),
                },
                wall_time_s=3600,
            )
        },
        envelope_sha256=_digest("0"),
    )
    resource = _seal(resource, "envelope_sha256")
    topology_contract = TopologyContract(
        contract_id="single-gpu-local-numa-v1",
        vertices=[TopologyVertexRequirement(role="worker", kind="accelerator", count=1)],
        relations=[
            TopologyRelationRequirement(
                relation_id="worker-same-node",
                pattern="same-node",
                source_role="worker",
            )
        ],
        minimum_proof="CP4-behavior",
        contract_sha256=_digest("0"),
    )
    topology_contract = _seal(topology_contract, "contract_sha256")
    expression = CapabilityExpression(
        operation="all_of",
        children=[
            _requirement("backend.cuda", "required-usable", "CP3-runtime"),
            _requirement("accelerator.dtype.bf16", "required-usable", "CP3-runtime"),
            _requirement("nvidia.cuda.tma", "required-native", "CP4-behavior"),
        ],
    )
    contract = CapabilityContract(
        policy_id="kernel-capability-v1",
        registry_sha256=registry.registry_sha256,
        variants=[
            CapabilityVariant(
                variant_id="cuda-tma-native-v1",
                priority=10,
                phases={
                    "verify": PhaseCapabilityContract(
                        requirements=expression,
                        forbidden_use=["runtime.host_round_trip_fallback"],
                        network_policy_id="no-egress-v1",
                    )
                },
                resource_envelope_sha256=resource.envelope_sha256,
                topology_contract_sha256=topology_contract.contract_sha256,
            )
        ],
        allowed_candidate_capabilities=[
            "backend.cuda",
            "accelerator.dtype.bf16",
            "nvidia.cuda.tma",
        ],
        contract_sha256=_digest("0"),
    )
    contract = _seal(contract, "contract_sha256")
    declaration = CandidateCapabilityDeclaration(
        requires=["backend.cuda"],
        uses=["nvidia.cuda.tma"],
        does_not_use=["runtime.host_round_trip_fallback"],
        declaration_sha256=_digest("0"),
    )
    declaration = _seal(declaration, "declaration_sha256")
    topology = TopologyGraph(
        vertices=[
            TopologyVertex(
                vertex_id="gpu-0",
                kind="accelerator",
                role="worker",
                attributes={"host_id": "node-a", "numa_node": 0},
            )
        ],
        graph_sha256=_digest("0"),
    )
    topology = _seal(topology, "graph_sha256")
    manifest = RunnerManifest(
        runner_id="runner-h100-01",
        revision=1,
        owner="infra-team",
        host=RunnerHostIdentity(
            architecture="x86_64",
            sockets=2,
            numa_nodes=2,
            cpu_model="test-cpu",
        ),
        accelerators=[
            RunnerAcceleratorIdentity(
                vendor="nvidia",
                model="H100-SXM",
                count=1,
                memory_bytes_each=80 * 1024**3,
            )
        ],
        software_profile={"driver": "590.1", "runtime": "cuda-13.0"},
        declared_capabilities=[
            "backend.cuda",
            "accelerator.dtype.bf16",
            "nvidia.cuda.tma",
        ],
        attestation_policy_id="official-gpu-v1",
        manifest_sha256=_digest("0"),
    )
    manifest = _seal(manifest, "manifest_sha256")
    captured = datetime(2026, 9, 2, tzinfo=UTC)
    snapshot = RunnerSnapshot(
        runner_manifest_sha256=manifest.manifest_sha256,
        captured_at=captured,
        expires_at=captured + timedelta(hours=1),
        availability="available",
        resources={
            "accelerator.count": ResourceAvailability(
                total=1, available=1, allocatable=1, exclusive_available=True
            ),
            "accelerator.memory.bytes": ResourceAvailability(
                total=80, available=75, allocatable=75
            ),
        },
        topology_sha256=topology.graph_sha256,
        snapshot_sha256=_digest("0"),
    )
    snapshot = _seal(snapshot, "snapshot_sha256")
    probe = CapabilityProbeIdentity(
        probe_id="trusted-capability-probe-v1",
        implementation_sha256=_digest("1"),
        image_sha256=_digest("2"),
        toolchain_sha256=_digest("3"),
    )
    attestations = [
        build_attestation(
            capability_id=capability_id,
            capability_definition_version=1,
            runner_snapshot_sha256=snapshot.snapshot_sha256,
            status="supported",
            proof_level=proof,
            parameters={},
            probe=probe,
            evidence_refs=["evidence://capability/" + capability_id],
            observed_at=captured,
            expires_at=captured + timedelta(hours=1),
        )
        for capability_id, proof in (
            ("backend.cuda", "CP3-runtime"),
            ("accelerator.dtype.bf16", "CP3-runtime"),
            ("nvidia.cuda.tma", "CP4-behavior"),
        )
    ]
    policy = RunnerSelectionPolicy(
        policy_id="exact-cell-registry-order-v1",
        variant_order=["cuda-tma-native-v1"],
        runner_order=[manifest.runner_id],
        probe_budget=1,
        policy_sha256=_digest("0"),
    )
    policy = _seal(policy, "policy_sha256")
    return {
        "task_seal_sha256": _digest("4"),
        "candidate_sha256": _digest("5"),
        "registry": registry,
        "contract": contract,
        "declaration": declaration,
        "runner_manifests": [manifest],
        "runner_snapshots": {manifest.manifest_sha256: snapshot},
        "attestations": attestations,
        "resource_envelopes": {resource.envelope_sha256: resource},
        "topology_contracts": {topology_contract.contract_sha256: topology_contract},
        "topology_graphs": {topology.graph_sha256: topology},
        "policy": policy,
        "now": captured + timedelta(minutes=5),
    }


def test_resolver_is_deterministic_and_selects_only_pre_run_eligible_cell() -> None:
    fixture = _fixture()
    first = resolve_capabilities(**fixture)
    second = resolve_capabilities(**fixture)
    assert first.status == "eligible"
    assert first.selected_variant_id == "cuda-tma-native-v1"
    assert first.resolution_sha256 == second.resolution_sha256
    assert audit_capability_resolution(first) == []


def test_unknown_capability_is_unresolved_not_supported_or_candidate_fail() -> None:
    fixture = _fixture()
    fixture["attestations"] = [
        item for item in fixture["attestations"] if item.capability_id != "nvidia.cuda.tma"
    ]
    result = resolve_capabilities(**fixture)
    assert result.status == "unresolved"
    assert "nvidia.cuda.tma" in result.required_probes
    assert any("CAPABILITY_PROBE_REQUIRED" in item for item in result.unresolved)


def test_contradictory_proofs_quarantine_runner() -> None:
    fixture = _fixture()
    supported = next(
        item for item in fixture["attestations"] if item.capability_id == "nvidia.cuda.tma"
    )
    unsupported = build_attestation(
        capability_id=supported.capability_id,
        capability_definition_version=1,
        runner_snapshot_sha256=supported.runner_snapshot_sha256,
        status="unsupported",
        proof_level="CP4-behavior",
        parameters={},
        probe=supported.probe,
        evidence_refs=["evidence://capability/tma-negative"],
        observed_at=supported.observed_at,
        expires_at=supported.expires_at,
    )
    fixture["attestations"].append(unsupported)
    merged, failures = merge_attestations(
        fixture["attestations"],
        registry=fixture["registry"],
        snapshot_sha256=supported.runner_snapshot_sha256,
        now=fixture["now"],
    )
    assert merged["nvidia.cuda.tma"].status == "contradictory"
    assert "CAPABILITY_PROOF_CONTRADICTION:nvidia.cuda.tma" in failures
    result = resolve_capabilities(**fixture)
    assert result.status == "runner-contradiction"


def test_capacity_and_structural_unschedulable_are_not_candidate_failures() -> None:
    fixture = _fixture()
    manifest = fixture["runner_manifests"][0]
    snapshot = fixture["runner_snapshots"][manifest.manifest_sha256]
    busy_resources = dict(snapshot.resources)
    busy_resources["accelerator.memory.bytes"] = ResourceAvailability(
        total=80, available=20, allocatable=20
    )
    busy = _seal(
        snapshot.model_copy(update={"resources": busy_resources, "snapshot_sha256": _digest("0")}),
        "snapshot_sha256",
    )
    feasibility = evaluate_resource_feasibility(
        next(iter(fixture["resource_envelopes"].values())), busy
    )
    assert feasibility.status == "capacity-unavailable"

    too_small_resources = dict(snapshot.resources)
    too_small_resources["accelerator.memory.bytes"] = ResourceAvailability(
        total=20, available=20, allocatable=20
    )
    too_small = _seal(
        snapshot.model_copy(
            update={"resources": too_small_resources, "snapshot_sha256": _digest("0")}
        ),
        "snapshot_sha256",
    )
    feasibility = evaluate_resource_feasibility(
        next(iter(fixture["resource_envelopes"].values())), too_small
    )
    assert feasibility.status == "unschedulable"


def test_resource_and_capability_runtime_failures_have_explicit_owners() -> None:
    fixture = _fixture()
    envelope = next(iter(fixture["resource_envelopes"].values()))
    candidate = evaluate_resource_usage(
        envelope,
        [
            ResourceUsageObservation(
                phase="verify",
                resource_id="accelerator.memory.bytes",
                candidate_peak=23,
            )
        ],
    )
    assert candidate.status == "VALID_FAIL"
    assert candidate.owner == "candidate"
    external = evaluate_resource_usage(
        envelope,
        [
            ResourceUsageObservation(
                phase="verify",
                resource_id="accelerator.memory.bytes",
                candidate_peak=20,
                external_interference=True,
            )
        ],
    )
    assert external.status == "INFRA_INVALID"
    capability = evaluate_candidate_capability_use(
        [
            CandidateCapabilityUseObservation(
                capability_id="runtime.host_round_trip_fallback",
                declared=False,
                forbidden=True,
                native_required=True,
                native_proved=False,
                silent_fallback=True,
                evidence_refs=["evidence://verifier/mechanism"],
            )
        ]
    )
    assert capability.status == "VALID_FAIL"
    assert any("FORBIDDEN_CAPABILITY_USED" in item for item in capability.failure_codes)


def test_topology_is_a_graph_and_changed_relation_is_a_different_cell() -> None:
    contract = TopologyContract(
        contract_id="two-gpu-same-numa",
        vertices=[TopologyVertexRequirement(role="worker", kind="accelerator", count=2)],
        relations=[
            TopologyRelationRequirement(
                relation_id="workers-local",
                pattern="same-numa",
                source_role="worker",
            )
        ],
        minimum_proof="CP4-behavior",
        contract_sha256=_digest("0"),
    )
    contract = _seal(contract, "contract_sha256")
    graph = TopologyGraph(
        vertices=[
            TopologyVertex(
                vertex_id="gpu-0",
                kind="accelerator",
                role="worker",
                attributes={"numa_node": 0},
            ),
            TopologyVertex(
                vertex_id="gpu-1",
                kind="accelerator",
                role="worker",
                attributes={"numa_node": 1},
            ),
        ],
        edges=[TopologyEdge(source="gpu-0", target="gpu-1", kind="peer-accessible")],
        graph_sha256=_digest("0"),
    )
    graph = _seal(graph, "graph_sha256")
    result = match_topology(contract, graph)
    assert result.status == "unsatisfied"
    assert "TOPOLOGY_RELATION_UNSATISFIED:workers-local" in result.failure_codes


def test_raw_performance_requires_same_pre_registered_comparison_cell() -> None:
    policy = BenchmarkCellPolicy(
        policy_id="kernel-cell-v1",
        comparison_included_fields=[
            "task.task_seal_sha256",
            "hardware.accelerator_model",
            "software.driver",
            "benchmark.workload_sha256",
        ],
        comparison_excluded_fields=["runner.snapshot_sha256"],
        policy_sha256=_digest("0"),
    )
    policy = _seal(policy, "policy_sha256")

    def cell(driver: str, snapshot: str):
        return build_benchmark_cell(
            policy=policy,
            task={"task_seal_sha256": _digest("1")},
            runner={"snapshot_sha256": snapshot},
            hardware={"accelerator_model": "H100-SXM"},
            software={"driver": driver},
            execution={"cache_policy": "warm-runtime"},
            benchmark={"workload_sha256": _digest("2")},
        )

    first = cell("590.1", _digest("3"))
    same_comparison = cell("590.1", _digest("4"))
    changed_driver = cell("591.0", _digest("4"))
    assert first.full_environment_digest != same_comparison.full_environment_digest
    assert first.comparison_cell_digest == same_comparison.comparison_cell_digest
    assert audit_benchmark_cell(first, policy) == []
    assert_raw_performance_comparable(first, same_comparison)
    with pytest.raises(ValueError, match="cross-cell"):
        assert_raw_performance_comparable(first, changed_driver)


def test_resource_lease_revalidates_snapshot_and_fails_on_cell_drift() -> None:
    fixture = _fixture()
    resolution = resolve_capabilities(**fixture)
    manifest = fixture["runner_manifests"][0]
    pre = fixture["runner_snapshots"][manifest.manifest_sha256]
    post = _seal(
        pre.model_copy(
            update={
                "captured_at": pre.captured_at + timedelta(minutes=1),
                "snapshot_sha256": _digest("0"),
            }
        ),
        "snapshot_sha256",
    )
    lease = build_resource_lease(
        resolution=resolution,
        pre_lease_snapshot=pre,
        post_lease_snapshot=post,
        allocations={"accelerators": ["gpu-0"], "cpu_set": "0-7"},
        isolation={
            "device_visibility": "exact",
            "process_namespace": "isolated",
        },
        lease_id="lease-1",
        acquired_at=post.captured_at,
        expires_at=post.captured_at + timedelta(minutes=30),
    )
    assert lease.status == "active"
    assert audit_resource_lease(lease) == []

    drifted_post = _seal(
        post.model_copy(update={"topology_sha256": _digest("f"), "snapshot_sha256": _digest("0")}),
        "snapshot_sha256",
    )
    broken = build_resource_lease(
        resolution=resolution,
        pre_lease_snapshot=pre,
        post_lease_snapshot=drifted_post,
        allocations={"accelerators": ["gpu-0"]},
        isolation={
            "device_visibility": "exact",
            "process_namespace": "isolated",
        },
        lease_id="lease-2",
        acquired_at=post.captured_at,
        expires_at=post.captured_at + timedelta(minutes=30),
    )
    assert broken.status == "broken"
    assert broken.failure_codes == ["CELL_DRIFT_BEFORE_RUN"]
