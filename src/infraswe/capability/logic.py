from __future__ import annotations

from typing import Any

from infraswe.models.capability import (
    PROOF_LEVEL_ORDER,
    CapabilityAttestation,
    CapabilityExpression,
    CapabilityParameterConstraint,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilityRequirementResolution,
    TrialPhase,
)


def _resolve_alias(registry: CapabilityRegistry, capability_id: str) -> str:
    aliases = {
        alias: item.capability_id
        for item in registry.definitions
        for alias in item.relationships.aliases
    }
    return aliases.get(capability_id, capability_id)


def _implied_fact(
    registry: CapabilityRegistry,
    facts: dict[str, CapabilityAttestation],
    capability_id: str,
) -> CapabilityAttestation | None:
    canonical = _resolve_alias(registry, capability_id)
    direct = facts.get(canonical) or facts.get(capability_id)
    if direct is not None:
        return direct
    definitions = {item.capability_id: item for item in registry.definitions}
    frontier = [
        source
        for source, fact in facts.items()
        if fact.status == "supported" and source in definitions
    ]
    visited: set[str] = set()
    while frontier:
        source = frontier.pop()
        if source in visited:
            continue
        visited.add(source)
        definition = definitions[source]
        implications = {_resolve_alias(registry, item) for item in definition.relationships.implies}
        if canonical in implications:
            return facts[source]
        frontier.extend(item for item in implications if item in definitions)
    return None


def _constraint_matches(value: Any, constraint: CapabilityParameterConstraint) -> bool:
    if constraint.eq is not None and value != constraint.eq:
        return False
    if constraint.gte is not None and (
        not isinstance(value, (int, float)) or value < constraint.gte
    ):
        return False
    if constraint.lte is not None and (
        not isinstance(value, (int, float)) or value > constraint.lte
    ):
        return False
    if constraint.set_contains and (
        not isinstance(value, (list, set, tuple)) or not set(constraint.set_contains) <= set(value)
    ):
        return False
    if constraint.one_of and value not in constraint.one_of:
        return False
    return not (
        constraint.count_gte is not None
        and (not hasattr(value, "__len__") or len(value) < constraint.count_gte)
    )


def _effective_minimum(requirement: CapabilityRequirement) -> int:
    floor = {
        "required-present": 1,
        "required-usable": 3,
        "required-native": 4,
        "required-absent": 1,
        "forbidden-use": 0,
        "preferred": 1,
        "optional-observe": 1,
        "explicitly-not-assumed": 0,
    }[requirement.mode]
    return max(floor, PROOF_LEVEL_ORDER[requirement.min_proof])


def _evaluate_requirement(
    requirement: CapabilityRequirement,
    *,
    facts: dict[str, CapabilityAttestation],
    registry: CapabilityRegistry,
    phase: TrialPhase,
) -> tuple[str, list[CapabilityRequirementResolution], list[str]]:
    if requirement.mode in {"forbidden-use", "explicitly-not-assumed"}:
        return (
            "satisfied",
            [
                CapabilityRequirementResolution(
                    phase=phase,
                    capability_id=requirement.capability_id,
                    mode=requirement.mode,
                    status="satisfied",
                )
            ],
            [],
        )
    fact = _implied_fact(registry, facts, requirement.capability_id)
    optional = requirement.mode in {"preferred", "optional-observe"}
    if fact is None or fact.status == "unknown":
        status = "not-applicable" if optional else "unresolved"
        resolution = CapabilityRequirementResolution(
            phase=phase,
            capability_id=requirement.capability_id,
            mode=requirement.mode,
            status=status,
            failure_code=None if optional else "CAPABILITY_PROOF_MISSING",
        )
        probes = [] if optional else [requirement.capability_id]
        return ("satisfied" if optional else "unresolved", [resolution], probes)
    if fact.status == "contradictory":
        resolution = CapabilityRequirementResolution(
            phase=phase,
            capability_id=requirement.capability_id,
            mode=requirement.mode,
            status="unsatisfied",
            proof_level=fact.proof_level,
            attestation_sha256=fact.attestation_sha256,
            failure_code="CAPABILITY_PROOF_CONTRADICTION",
        )
        return "contradictory", [resolution], []

    proof_level = PROOF_LEVEL_ORDER[fact.proof_level] if fact.proof_level else -1
    minimum = _effective_minimum(requirement)
    if requirement.mode == "required-absent":
        satisfied = fact.status == "unsupported" and proof_level >= minimum
    else:
        satisfied = fact.status == "supported" and proof_level >= minimum
    if satisfied:
        for parameter, constraint in requirement.parameters.items():
            if parameter not in fact.parameters or not _constraint_matches(
                fact.parameters[parameter], constraint
            ):
                satisfied = False
                break
    if optional and not satisfied:
        result_status = "not-applicable"
        state = "satisfied"
    else:
        result_status = "satisfied" if satisfied else "unsatisfied"
        state = result_status
    resolution = CapabilityRequirementResolution(
        phase=phase,
        capability_id=requirement.capability_id,
        mode=requirement.mode,
        status=result_status,
        proof_level=fact.proof_level,
        attestation_sha256=fact.attestation_sha256,
        failure_code=None if satisfied or optional else "CAPABILITY_REQUIREMENT_UNSATISFIED",
    )
    return state, [resolution], []


def evaluate_capability_expression(
    expression: CapabilityExpression,
    *,
    facts: dict[str, CapabilityAttestation],
    registry: CapabilityRegistry,
    phase: TrialPhase,
    selected_variant_id: str,
) -> tuple[str, list[CapabilityRequirementResolution], list[str]]:
    """Evaluate closed-world capability logic without interpreting natural language."""

    if expression.operation == "capability":
        assert expression.requirement is not None
        return _evaluate_requirement(
            expression.requirement,
            facts=facts,
            registry=registry,
            phase=phase,
        )
    if expression.operation == "conditional":
        if expression.selected_variant_is != selected_variant_id:
            return "satisfied", [], []
        return evaluate_capability_expression(
            expression.children[0],
            facts=facts,
            registry=registry,
            phase=phase,
            selected_variant_id=selected_variant_id,
        )

    evaluated = [
        evaluate_capability_expression(
            child,
            facts=facts,
            registry=registry,
            phase=phase,
            selected_variant_id=selected_variant_id,
        )
        for child in expression.children
    ]
    states = [item[0] for item in evaluated]
    resolutions = [resolution for item in evaluated for resolution in item[1]]
    probes = sorted({probe for item in evaluated for probe in item[2]})
    if "contradictory" in states:
        return "contradictory", resolutions, probes
    if expression.operation == "all_of":
        state = (
            "unsatisfied"
            if "unsatisfied" in states
            else "unresolved"
            if "unresolved" in states
            else "satisfied"
        )
    elif expression.operation == "any_of":
        state = (
            "satisfied"
            if "satisfied" in states
            else "unsatisfied"
            if all(item == "unsatisfied" for item in states)
            else "unresolved"
        )
    elif expression.operation == "one_of":
        satisfied_count = states.count("satisfied")
        state = (
            "unsatisfied"
            if satisfied_count > 1
            else "satisfied"
            if satisfied_count == 1 and "unresolved" not in states
            else "unresolved"
            if "unresolved" in states
            else "unsatisfied"
        )
    elif expression.operation == "not":
        state = {
            "satisfied": "unsatisfied",
            "unsatisfied": "satisfied",
            "unresolved": "unresolved",
        }[states[0]]
    else:
        antecedent, consequent = states
        state = (
            "satisfied"
            if antecedent == "unsatisfied"
            else consequent
            if antecedent == "satisfied"
            else "unresolved"
        )
    return state, resolutions, probes
