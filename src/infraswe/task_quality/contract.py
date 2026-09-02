from __future__ import annotations

from collections.abc import Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.task_quality import (
    INFRA_CERT_BUCKETS,
    AcceptanceObligation,
    FeasibilityWitness,
    TaskAcceptanceContract,
    TaskSpecification,
    WitnessSet,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def build_acceptance_contract(
    *,
    task_id: str,
    task_revision: int,
    obligations: Sequence[AcceptanceObligation],
) -> TaskAcceptanceContract:
    """Build a content-addressed AcceptanceContract.

    The digest excludes its own field. Callers cannot choose the authoritative
    digest and the resulting contract is suitable for a TaskSeal binding.
    """

    preliminary = TaskAcceptanceContract(
        task_id=task_id,
        task_revision=task_revision,
        obligations=list(obligations),
        contract_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"contract_sha256"})
    return preliminary.model_copy(update={"contract_sha256": canonical_sha256(material)})


def build_witness_set(witnesses: Sequence[FeasibilityWitness]) -> WitnessSet:
    preliminary = WitnessSet(witnesses=list(witnesses), witness_set_sha256=_ZERO_DIGEST)
    material = preliminary.model_dump(mode="json", exclude={"witness_set_sha256"})
    return preliminary.model_copy(update={"witness_set_sha256": canonical_sha256(material)})


def audit_acceptance_contract(
    specification: TaskSpecification,
    contract: TaskAcceptanceContract,
) -> list[str]:
    failures: list[str] = []
    if contract.task_id != specification.task_id:
        failures.append("TASK_CONTRACT_ID_MISMATCH")
    if contract.task_revision != specification.task_revision:
        failures.append("TASK_CONTRACT_REVISION_MISMATCH")

    material = contract.model_dump(mode="json", exclude={"contract_sha256"})
    if contract.contract_sha256 != canonical_sha256(material):
        failures.append("ACCEPTANCE_CONTRACT_DIGEST_MISMATCH")

    requirement_ids = {item.requirement_id for item in specification.requirements}
    scoring_requirements = {
        item.requirement_id for item in specification.requirements if item.disposition == "scoring"
    }
    obligation_ids = {item.obligation_id for item in contract.obligations}
    mapped: set[str] = set()
    for obligation in contract.obligations:
        unknown = set(obligation.source_requirements) - requirement_ids
        if unknown:
            failures.append(
                "OBLIGATION_UNKNOWN_REQUIREMENT:"
                + obligation.obligation_id
                + ":"
                + ",".join(sorted(unknown))
            )
        mapped.update(obligation.source_requirements)
        if (
            obligation.severity == "hard"
            and obligation.failure_owner == "candidate"
            and obligation.bucket not in INFRA_CERT_BUCKETS
            and not (obligation.bucket == "maintenance-probe" and obligation.release_gate)
        ):
            failures.append(
                "HARD_CANDIDATE_OBLIGATION_OUTSIDE_INFRACERT:" + obligation.obligation_id
            )

    for requirement_id in sorted(scoring_requirements - mapped):
        failures.append("SCORING_REQUIREMENT_UNMAPPED:" + requirement_id)

    if not any(
        item.bucket == "capability-obligation" and item.severity == "hard"
        for item in contract.obligations
    ):
        failures.append("HARD_CAPABILITY_OBLIGATION_MISSING")
    if not any(
        item.bucket == "regression-invariant" and item.severity == "hard"
        for item in contract.obligations
    ):
        failures.append("HARD_REGRESSION_INVARIANT_MISSING")
    if not any(item.bucket == "environment-sentinel" for item in contract.obligations):
        failures.append("ENVIRONMENT_SENTINEL_MISSING")
    if len(obligation_ids) != len(contract.obligations):
        # Normally caught by Pydantic, retained for defensive audit of constructed objects.
        failures.append("DUPLICATE_OBLIGATION_ID")
    return failures


def audit_witness_set(
    specification: TaskSpecification,
    contract: TaskAcceptanceContract,
    witness_set: WitnessSet,
) -> list[str]:
    failures: list[str] = []
    material = witness_set.model_dump(mode="json", exclude={"witness_set_sha256"})
    if witness_set.witness_set_sha256 != canonical_sha256(material):
        failures.append("WITNESS_SET_DIGEST_MISMATCH")

    obligation_ids = {item.obligation_id for item in contract.obligations}
    required = {
        item.obligation_id
        for item in contract.obligations
        if item.severity == "hard"
        and (
            item.bucket in INFRA_CERT_BUCKETS
            or (item.bucket == "maintenance-probe" and item.release_gate)
        )
    }
    covered: set[str] = set()
    for witness in witness_set.witnesses:
        if witness.target_revision != specification.target.revision:
            failures.append("WITNESS_TARGET_REVISION_MISMATCH:" + witness.witness_id)
        unknown = set(witness.covers) - obligation_ids
        if unknown:
            failures.append(
                "WITNESS_UNKNOWN_OBLIGATION:" + witness.witness_id + ":" + ",".join(sorted(unknown))
            )
        covered.update(witness.covers)
    for obligation_id in sorted(required - covered):
        failures.append("HARD_OBLIGATION_WITHOUT_WITNESS:" + obligation_id)
    return failures
