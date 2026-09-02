from __future__ import annotations

from datetime import UTC, datetime

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.task_quality import (
    HumanTaskQualificationReview,
    TaskAcceptanceContract,
    TaskQualificationReport,
    TaskSeal,
    TaskSpecification,
    WitnessSet,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def build_task_seal(
    *,
    specification: TaskSpecification,
    contract: TaskAcceptanceContract,
    witness_set: WitnessSet,
    qualification: TaskQualificationReport,
    review: HumanTaskQualificationReview,
    verifier_bundle_sha256: str,
    capability_policy_sha256: str,
    capability_registry_sha256: str,
    resource_envelope_sha256: str,
    topology_contract_sha256: str,
    benchmark_cell_policy_sha256: str,
    runner_selection_policy_sha256: str,
    benchmark_season: str,
    qualified_at: datetime | None = None,
) -> TaskSeal:
    if qualification.status not in {"QUALIFIED", "QUALIFIED_WITH_SCOPE"}:
        raise ValueError("only a qualified task can be sealed")
    qualification_material = qualification.model_dump(mode="json", exclude={"report_sha256"})
    if qualification.report_sha256 != canonical_sha256(qualification_material):
        raise ValueError("qualification report digest mismatch")
    expected = {
        "task_id": specification.task_id,
        "task_revision": specification.task_revision,
        "specification_sha256": canonical_sha256(specification),
        "contract_sha256": contract.contract_sha256,
        "witness_set_sha256": witness_set.witness_set_sha256,
    }
    observed = {
        "task_id": qualification.task_id,
        "task_revision": qualification.task_revision,
        "specification_sha256": qualification.specification_sha256,
        "contract_sha256": qualification.contract_sha256,
        "witness_set_sha256": qualification.witness_set_sha256,
    }
    mismatch = sorted(name for name in expected if expected[name] != observed[name])
    if mismatch:
        raise ValueError("qualification report binding mismatch: " + ", ".join(mismatch))
    if review.decision != "approve":
        raise ValueError("TaskSeal requires an approved human qualification review")
    review_expected = {
        "specification_sha256": qualification.specification_sha256,
        "contract_sha256": qualification.contract_sha256,
        "witness_set_sha256": qualification.witness_set_sha256,
        "mutation_suite_sha256": qualification.mutation_suite_sha256,
        "alternative_solution_set_sha256": qualification.alternative_solution_set_sha256,
        "artifact_policy_sha256": specification.artifact_policy_sha256,
        "capability_policy_sha256": specification.capability_contract_sha256,
    }
    review_mismatch = sorted(
        name for name, value in review_expected.items() if getattr(review, name) != value
    )
    if review_mismatch:
        raise ValueError("human review binding mismatch: " + ", ".join(review_mismatch))
    if capability_policy_sha256 != specification.capability_contract_sha256:
        raise ValueError("capability policy does not match TaskSpecification")

    preliminary = TaskSeal(
        task_id=specification.task_id,
        task_revision=specification.task_revision,
        qualification_status=qualification.status,
        specification_sha256=qualification.specification_sha256,
        acceptance_contract_sha256=contract.contract_sha256,
        verifier_bundle_sha256=verifier_bundle_sha256,
        witness_set_sha256=witness_set.witness_set_sha256,
        mutation_suite_sha256=qualification.mutation_suite_sha256,
        alternative_solution_set_sha256=qualification.alternative_solution_set_sha256,
        artifact_policy_sha256=specification.artifact_policy_sha256,
        capability_policy_sha256=capability_policy_sha256,
        capability_registry_sha256=capability_registry_sha256,
        capability_contract_sha256=specification.capability_contract_sha256,
        resource_envelope_sha256=resource_envelope_sha256,
        topology_contract_sha256=topology_contract_sha256,
        benchmark_cell_policy_sha256=benchmark_cell_policy_sha256,
        runner_selection_policy_sha256=runner_selection_policy_sha256,
        qualification_report_sha256=qualification.report_sha256,
        qualified_at=qualified_at or datetime.now(UTC),
        reviewers=[review.reviewer],
        benchmark_season=benchmark_season,
        task_seal_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"task_seal_sha256"})
    return preliminary.model_copy(update={"task_seal_sha256": canonical_sha256(material)})


def audit_task_seal(seal: TaskSeal) -> list[str]:
    material = seal.model_dump(mode="json", exclude={"task_seal_sha256"})
    return (
        [] if seal.task_seal_sha256 == canonical_sha256(material) else ["TASK_SEAL_DIGEST_MISMATCH"]
    )
