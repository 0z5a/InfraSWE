from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from infraswe.capability.logic import evaluate_capability_expression
from infraswe.capability.registry import audit_registry, merge_attestations
from infraswe.capability.resource import evaluate_resource_feasibility
from infraswe.capability.topology import match_topology
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    CandidateCapabilityDeclaration,
    CapabilityAttestation,
    CapabilityContract,
    CapabilityExpression,
    CapabilityRegistry,
    CapabilityRequirementResolution,
    CapabilityResolution,
    ExcludedRunner,
    ResourceEnvelope,
    RunnerManifest,
    RunnerSelectionPolicy,
    RunnerSnapshot,
    TopologyContract,
    TopologyGraph,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def _audit_digest(model: object, field: str) -> bool:
    material = model.model_dump(mode="json", exclude={field})
    return getattr(model, field) == canonical_sha256(material)


def _forbidden_requirements(expression: CapabilityExpression) -> set[str]:
    found: set[str] = set()
    if (
        expression.operation == "capability"
        and expression.requirement is not None
        and expression.requirement.mode == "forbidden-use"
    ):
        found.add(expression.requirement.capability_id)
    for child in expression.children:
        found.update(_forbidden_requirements(child))
    return found


def _build_resolution(
    *,
    task_seal_sha256: str,
    candidate_sha256: str,
    registry_sha256: str,
    status: str,
    policy: RunnerSelectionPolicy,
    selected_variant_id: str | None = None,
    selected_manifest_sha256: str | None = None,
    selected_snapshot_sha256: str | None = None,
    requirements: Sequence[CapabilityRequirementResolution] = (),
    topology: object | None = None,
    resources: object | None = None,
    required_probes: Sequence[str] = (),
    excluded_runners: Sequence[ExcludedRunner] = (),
    unresolved: Sequence[str] = (),
    contradictions: Sequence[str] = (),
) -> CapabilityResolution:
    identity_material = {
        "task_seal_sha256": task_seal_sha256,
        "candidate_sha256": candidate_sha256,
        "registry_sha256": registry_sha256,
        "status": status,
        "selected_variant_id": selected_variant_id,
        "selected_manifest_sha256": selected_manifest_sha256,
        "selected_snapshot_sha256": selected_snapshot_sha256,
        "policy_sha256": policy.policy_sha256,
        "excluded": [item.model_dump(mode="json") for item in excluded_runners],
    }
    resolution_id = "resolution-" + canonical_sha256(identity_material)[-20:]
    preliminary = CapabilityResolution(
        resolution_id=resolution_id,
        task_seal_sha256=task_seal_sha256,
        candidate_sha256=candidate_sha256,
        registry_sha256=registry_sha256,
        status=status,
        selected_variant_id=selected_variant_id,
        selected_runner_manifest_sha256=selected_manifest_sha256,
        selected_runner_snapshot_sha256=selected_snapshot_sha256,
        requirements=list(requirements),
        topology=topology,
        resources=resources,
        required_probes=sorted(set(required_probes)),
        excluded_runners=list(excluded_runners),
        unresolved=sorted(set(unresolved)),
        contradictions=sorted(set(contradictions)),
        policy_id=policy.policy_id,
        resolution_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"resolution_sha256"})
    return preliminary.model_copy(update={"resolution_sha256": canonical_sha256(material)})


def resolve_capabilities(
    *,
    task_seal_sha256: str,
    candidate_sha256: str,
    registry: CapabilityRegistry,
    contract: CapabilityContract,
    declaration: CandidateCapabilityDeclaration,
    runner_manifests: Sequence[RunnerManifest],
    runner_snapshots: Mapping[str, RunnerSnapshot],
    attestations: Sequence[CapabilityAttestation],
    resource_envelopes: Mapping[str, ResourceEnvelope],
    topology_contracts: Mapping[str, TopologyContract],
    topology_graphs: Mapping[str, TopologyGraph],
    policy: RunnerSelectionPolicy,
    now: datetime | None = None,
) -> CapabilityResolution:
    """Deterministically resolve variant/runner before Candidate execution."""

    if audit_registry(registry):
        raise ValueError("CapabilityRegistry is not sealable")
    if not _audit_digest(contract, "contract_sha256"):
        raise ValueError("CapabilityContract digest mismatch")
    if contract.registry_sha256 != registry.registry_sha256:
        raise ValueError("CapabilityContract registry binding mismatch")
    if not _audit_digest(policy, "policy_sha256"):
        raise ValueError("RunnerSelectionPolicy digest mismatch")
    if not _audit_digest(declaration, "declaration_sha256"):
        return _build_resolution(
            task_seal_sha256=task_seal_sha256,
            candidate_sha256=candidate_sha256,
            registry_sha256=registry.registry_sha256,
            status="candidate-declaration-ineligible",
            policy=policy,
            unresolved=["CANDIDATE_DECLARATION_DIGEST_MISMATCH"],
        )

    variant_by_id = {item.variant_id: item for item in contract.variants}
    if set(policy.variant_order) != set(variant_by_id):
        raise ValueError("RunnerSelectionPolicy must pre-register every capability variant")
    allowed = set(contract.allowed_candidate_capabilities)
    undeclared_scope = (set(declaration.requires) | set(declaration.uses)) - allowed
    forbidden = {
        capability_id
        for variant in contract.variants
        for phase in variant.phases.values()
        for capability_id in (
            set(phase.forbidden_use) | _forbidden_requirements(phase.requirements)
        )
    }
    declared_forbidden = set(declaration.uses) & forbidden
    if undeclared_scope or declared_forbidden:
        reasons = [
            *("CANDIDATE_CAPABILITY_OUTSIDE_CONTRACT:" + item for item in sorted(undeclared_scope)),
            *("CANDIDATE_DECLARES_FORBIDDEN_USE:" + item for item in sorted(declared_forbidden)),
        ]
        return _build_resolution(
            task_seal_sha256=task_seal_sha256,
            candidate_sha256=candidate_sha256,
            registry_sha256=registry.registry_sha256,
            status="candidate-declaration-ineligible",
            policy=policy,
            unresolved=reasons,
        )

    manifests_by_id = {item.runner_id: item for item in runner_manifests}
    ordered_runner_ids = [
        runner_id for runner_id in policy.runner_order if runner_id in manifests_by_id
    ]
    current = now or datetime.now(UTC)
    attestation_by_snapshot: dict[str, list[CapabilityAttestation]] = {}
    for attestation in attestations:
        attestation_by_snapshot.setdefault(attestation.runner_snapshot_sha256, []).append(
            attestation
        )

    excluded: list[ExcludedRunner] = []
    unresolved_reasons: list[str] = []
    contradiction_reasons: list[str] = []
    saw_capacity = False
    saw_unschedulable = False
    last_requirements: list[CapabilityRequirementResolution] = []
    required_probes: list[str] = []

    for variant_id in policy.variant_order:
        variant = variant_by_id[variant_id]
        resource_envelope = resource_envelopes.get(variant.resource_envelope_sha256)
        topology_contract = topology_contracts.get(variant.topology_contract_sha256)
        if resource_envelope is None or topology_contract is None:
            raise ValueError("capability variant references an unknown resource/topology contract")
        if not _audit_digest(resource_envelope, "envelope_sha256"):
            raise ValueError("ResourceEnvelope digest mismatch")
        if not _audit_digest(topology_contract, "contract_sha256"):
            raise ValueError("TopologyContract digest mismatch")

        for runner_id in ordered_runner_ids:
            manifest = manifests_by_id[runner_id]
            if not _audit_digest(manifest, "manifest_sha256"):
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_MANIFEST_DIGEST_MISMATCH",
                    )
                )
                contradiction_reasons.append("RUNNER_MANIFEST_DIGEST_MISMATCH:" + runner_id)
                continue
            snapshot = runner_snapshots.get(manifest.manifest_sha256)
            if snapshot is None:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_SNAPSHOT_MISSING",
                    )
                )
                unresolved_reasons.append("RUNNER_SNAPSHOT_MISSING:" + runner_id)
                continue
            if not _audit_digest(snapshot, "snapshot_sha256"):
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_SNAPSHOT_DIGEST_MISMATCH",
                    )
                )
                contradiction_reasons.append("RUNNER_SNAPSHOT_DIGEST_MISMATCH:" + runner_id)
                continue
            if snapshot.runner_manifest_sha256 != manifest.manifest_sha256:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_SNAPSHOT_MANIFEST_MISMATCH",
                    )
                )
                contradiction_reasons.append("RUNNER_SNAPSHOT_MANIFEST_MISMATCH:" + runner_id)
                continue
            if snapshot.expires_at <= current:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_SNAPSHOT_EXPIRED",
                    )
                )
                unresolved_reasons.append("RUNNER_SNAPSHOT_EXPIRED:" + runner_id)
                continue
            if snapshot.availability == "quarantined":
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_QUARANTINED",
                    )
                )
                contradiction_reasons.append("RUNNER_QUARANTINED:" + runner_id)
                continue
            if snapshot.availability in {"busy", "draining"}:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="CAPACITY_UNAVAILABLE",
                    )
                )
                saw_capacity = True
                continue

            facts, merge_failures = merge_attestations(
                attestation_by_snapshot.get(snapshot.snapshot_sha256, []),
                registry=registry,
                snapshot_sha256=snapshot.snapshot_sha256,
                now=current,
            )
            if any("CONTRADICTION" in item for item in merge_failures):
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_CAPABILITY_CONTRADICTION",
                    )
                )
                contradiction_reasons.extend(item + ":" + runner_id for item in merge_failures)
                continue
            if merge_failures:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="CAPABILITY_ATTESTATION_INVALID",
                    )
                )
                unresolved_reasons.extend(item + ":" + runner_id for item in merge_failures)
                continue

            phase_states: list[str] = []
            resolutions: list[CapabilityRequirementResolution] = []
            probes: list[str] = []
            for phase, phase_contract in variant.phases.items():
                state, phase_resolutions, phase_probes = evaluate_capability_expression(
                    phase_contract.requirements,
                    facts=facts,
                    registry=registry,
                    phase=phase,
                    selected_variant_id=variant_id,
                )
                phase_states.append(state)
                resolutions.extend(phase_resolutions)
                probes.extend(phase_probes)
            last_requirements = resolutions
            required_probes.extend(probes)
            if "contradictory" in phase_states:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RUNNER_CAPABILITY_CONTRADICTION",
                    )
                )
                contradiction_reasons.append("RUNNER_CAPABILITY_CONTRADICTION:" + runner_id)
                continue
            if "unsatisfied" in phase_states:
                failed = next(
                    (item for item in resolutions if item.status == "unsatisfied"),
                    None,
                )
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="REQUIRED_CAPABILITY_UNAVAILABLE",
                        capability_id=failed.capability_id if failed else None,
                        observed_status="unsatisfied",
                        proof_level=failed.proof_level if failed else None,
                    )
                )
                saw_unschedulable = True
                continue
            if "unresolved" in phase_states:
                reason = (
                    "CAPABILITY_PROOF_BUDGET_EXHAUSTED"
                    if len(set(probes)) > policy.probe_budget
                    else "CAPABILITY_PROBE_REQUIRED"
                )
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code=reason,
                    )
                )
                unresolved_reasons.append(reason + ":" + runner_id)
                continue

            graph = topology_graphs.get(snapshot.topology_sha256)
            if graph is None:
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="TOPOLOGY_GRAPH_MISSING",
                    )
                )
                unresolved_reasons.append("TOPOLOGY_GRAPH_MISSING:" + runner_id)
                continue
            topology = match_topology(topology_contract, graph)
            if topology.status == "probe-defect":
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="TOPOLOGY_PROBE_DEFECT",
                    )
                )
                unresolved_reasons.append("TOPOLOGY_PROBE_DEFECT:" + runner_id)
                continue
            if topology.status != "satisfied":
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="TOPOLOGY_REQUIREMENT_UNSATISFIED",
                    )
                )
                saw_unschedulable = True
                continue
            resources = evaluate_resource_feasibility(resource_envelope, snapshot)
            if resources.status == "capacity-unavailable":
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="CAPACITY_UNAVAILABLE",
                    )
                )
                saw_capacity = True
                continue
            if resources.status == "unschedulable":
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RESOURCE_ENVELOPE_UNSATISFIED",
                    )
                )
                saw_unschedulable = True
                continue
            if resources.status == "unresolved":
                excluded.append(
                    ExcludedRunner(
                        runner_id=runner_id,
                        variant_id=variant_id,
                        reason_code="RESOURCE_INVENTORY_UNRESOLVED",
                    )
                )
                unresolved_reasons.append("RESOURCE_INVENTORY_UNRESOLVED:" + runner_id)
                continue
            return _build_resolution(
                task_seal_sha256=task_seal_sha256,
                candidate_sha256=candidate_sha256,
                registry_sha256=registry.registry_sha256,
                status="eligible",
                policy=policy,
                selected_variant_id=variant_id,
                selected_manifest_sha256=manifest.manifest_sha256,
                selected_snapshot_sha256=snapshot.snapshot_sha256,
                requirements=resolutions,
                topology=topology,
                resources=resources,
                excluded_runners=excluded,
            )

    if contradiction_reasons:
        status = "runner-contradiction"
    elif unresolved_reasons:
        status = "unresolved"
    elif saw_capacity:
        status = "capacity-unavailable"
    else:
        status = "unschedulable"
        if not saw_unschedulable:
            unresolved_reasons.append("RUNNER_POOL_EMPTY_OR_NOT_PRE_REGISTERED")
    return _build_resolution(
        task_seal_sha256=task_seal_sha256,
        candidate_sha256=candidate_sha256,
        registry_sha256=registry.registry_sha256,
        status=status,
        policy=policy,
        requirements=last_requirements,
        required_probes=required_probes,
        excluded_runners=excluded,
        unresolved=unresolved_reasons,
        contradictions=contradiction_reasons,
    )


def audit_capability_resolution(resolution: CapabilityResolution) -> list[str]:
    material = resolution.model_dump(mode="json", exclude={"resolution_sha256"})
    return (
        []
        if resolution.resolution_sha256 == canonical_sha256(material)
        else ["CAPABILITY_RESOLUTION_DIGEST_MISMATCH"]
    )
