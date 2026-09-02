from __future__ import annotations

from collections.abc import Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.task_quality import (
    INFRA_CERT_BUCKETS,
    AlternativeValidSolutionOutcome,
    BaselineDifferential,
    HumanTaskQualificationReview,
    MutationOutcome,
    NegativeControlOutcome,
    TaskAcceptanceContract,
    TaskLeakageAudit,
    TaskQualificationReport,
    TaskSpecification,
    VerifierCoverageReport,
    VerifierFlakinessAudit,
    VerifierTrustCard,
    WitnessReplayResult,
    WitnessSet,
    weighted_mutation_adequacy,
)
from infraswe.task_quality.contract import (
    audit_acceptance_contract,
    audit_witness_set,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def _duplicate_ids(values: Sequence[str]) -> bool:
    return len(values) != len(set(values))


def _observe_exactly_once(
    *,
    label: str,
    observations: Sequence[object],
    allowed: set[str],
    required: set[str],
) -> list[str]:
    identifiers = [item.obligation_id for item in observations]
    failures: list[str] = []
    if _duplicate_ids(identifiers):
        failures.append(label + "_DUPLICATE_OBLIGATION")
    for identifier in sorted(set(identifiers) - allowed):
        failures.append(label + "_UNKNOWN_OBLIGATION:" + identifier)
    for identifier in sorted(required - set(identifiers)):
        failures.append(label + "_MISSING_OBLIGATION:" + identifier)
    return failures


def _baseline_failures(
    specification: TaskSpecification,
    contract: TaskAcceptanceContract,
    baseline: BaselineDifferential,
) -> list[str]:
    failures: list[str] = []
    if baseline.task_id != specification.task_id:
        failures.append("BASELINE_TASK_ID_MISMATCH")
    if baseline.task_revision != specification.task_revision:
        failures.append("BASELINE_TASK_REVISION_MISMATCH")
    if baseline.baseline_sha256 != specification.target.revision:
        failures.append("BASELINE_TARGET_REVISION_MISMATCH")
    if not baseline.stable:
        failures.append("BASELINE_UNSTABLE")

    obligation_by_id = {item.obligation_id: item for item in contract.obligations}
    required = {
        item.obligation_id
        for item in contract.obligations
        if item.severity == "hard" or item.bucket == "environment-sentinel"
    }
    failures.extend(
        _observe_exactly_once(
            label="BASELINE",
            observations=baseline.observations,
            allowed=set(obligation_by_id),
            required=required,
        )
    )
    for observation in baseline.observations:
        obligation = obligation_by_id.get(observation.obligation_id)
        if obligation is None:
            continue
        if observation.bucket != obligation.bucket or observation.severity != obligation.severity:
            failures.append("BASELINE_OBLIGATION_METADATA_MISMATCH:" + observation.obligation_id)
        if (
            obligation.bucket in {"regression-invariant", "environment-sentinel"}
            and obligation.severity in {"hard", "sentinel"}
            and observation.status != "PASS"
        ):
            failures.append("BASELINE_GUARD_NOT_PASSING:" + observation.obligation_id)

    if specification.change_kind in {"repair", "feature", "conformance"}:
        demonstrated_gap = any(
            observation.status == "FAIL"
            and (obligation := obligation_by_id.get(observation.obligation_id)) is not None
            and obligation.bucket in {"capability-obligation", "negative-boundary"}
            and obligation.severity == "hard"
            for observation in baseline.observations
        )
        if not demonstrated_gap:
            failures.append("BASELINE_DOES_NOT_DEMONSTRATE_TASK_GAP")
    else:
        hard_semantics = [
            observation
            for observation in baseline.observations
            if (obligation := obligation_by_id.get(observation.obligation_id)) is not None
            and obligation.severity == "hard"
            and obligation.bucket in INFRA_CERT_BUCKETS
        ]
        if any(item.status != "PASS" for item in hard_semantics):
            failures.append("OPTIMIZATION_BASELINE_SEMANTICS_NOT_PASSING")
    return failures


def _witness_replay_failures(
    contract: TaskAcceptanceContract,
    witness_set: WitnessSet,
    replays: Sequence[WitnessReplayResult],
) -> list[str]:
    failures: list[str] = []
    witness_by_id = {item.witness_id: item for item in witness_set.witnesses}
    replay_ids = [item.witness_id for item in replays]
    if _duplicate_ids(replay_ids):
        failures.append("DUPLICATE_WITNESS_REPLAY")
    for witness_id in sorted(set(witness_by_id) - set(replay_ids)):
        failures.append("WITNESS_REPLAY_MISSING:" + witness_id)
    for witness_id in sorted(set(replay_ids) - set(witness_by_id)):
        failures.append("UNKNOWN_WITNESS_REPLAY:" + witness_id)

    obligation_by_id = {item.obligation_id: item for item in contract.obligations}
    for replay in replays:
        witness = witness_by_id.get(replay.witness_id)
        if witness is None:
            continue
        if replay.witness_sha256 != witness.sha256:
            failures.append("WITNESS_REPLAY_DIGEST_MISMATCH:" + replay.witness_id)
        if replay.fresh_replays < 5:
            failures.append("WITNESS_REPLAY_COUNT_BELOW_MINIMUM:" + replay.witness_id)
        if not replay.stable:
            failures.append("WITNESS_REPLAY_UNSTABLE:" + replay.witness_id)
        required = set(witness.covers)
        failures.extend(
            _observe_exactly_once(
                label="WITNESS_" + replay.witness_id,
                observations=replay.observations,
                allowed=set(obligation_by_id),
                required=required,
            )
        )
        for observation in replay.observations:
            obligation = obligation_by_id.get(observation.obligation_id)
            if obligation is None:
                continue
            if (
                observation.bucket != obligation.bucket
                or observation.severity != obligation.severity
            ):
                failures.append(
                    "WITNESS_OBLIGATION_METADATA_MISMATCH:"
                    + replay.witness_id
                    + ":"
                    + observation.obligation_id
                )
            if observation.obligation_id in required and observation.status != "PASS":
                failures.append(
                    "WITNESS_COVERED_OBLIGATION_NOT_PASSING:"
                    + replay.witness_id
                    + ":"
                    + observation.obligation_id
                )
    return failures


def _adequacy_failures(
    contract: TaskAcceptanceContract,
    mutations: Sequence[MutationOutcome],
    negative_controls: Sequence[NegativeControlOutcome],
    alternatives: Sequence[AlternativeValidSolutionOutcome],
    *,
    minimum_mutation_adequacy: float,
) -> list[str]:
    failures: list[str] = []
    obligation_by_id = {item.obligation_id: item for item in contract.obligations}
    if not mutations:
        failures.append("MUTATION_SUITE_EMPTY")
    elif not any(item.critical for item in mutations):
        failures.append("CRITICAL_MUTANT_MISSING")
    for mutation in mutations:
        unknown = set(mutation.target_obligations) - set(obligation_by_id)
        if unknown:
            failures.append(
                "MUTANT_UNKNOWN_OBLIGATION:"
                + mutation.mutation_id
                + ":"
                + ",".join(sorted(unknown))
            )
        if mutation.critical and mutation.observed != "rejected":
            failures.append("CRITICAL_MUTANT_SURVIVED:" + mutation.mutation_id)
        if mutation.observed == "infra-invalid":
            failures.append("MUTANT_RESULT_INFRA_INVALID:" + mutation.mutation_id)
    adequacy = weighted_mutation_adequacy(list(mutations))
    if adequacy < minimum_mutation_adequacy:
        failures.append("MUTATION_ADEQUACY_BELOW_THRESHOLD")

    if not negative_controls:
        failures.append("NEGATIVE_CONTROL_SUITE_EMPTY")
    for control in negative_controls:
        if control.observed != "rejected":
            failures.append("NEGATIVE_CONTROL_NOT_REJECTED:" + control.control_id)

    if not alternatives:
        failures.append("ALTERNATIVE_VALID_SOLUTION_MISSING")
    fingerprints = [item.structure_fingerprint_sha256 for item in alternatives]
    if _duplicate_ids(fingerprints):
        failures.append("ALTERNATIVE_STRUCTURE_FINGERPRINT_DUPLICATE")
    hard_ids = {
        item.obligation_id
        for item in contract.obligations
        if item.severity == "hard"
        and (
            item.bucket in INFRA_CERT_BUCKETS
            or (item.bucket == "maintenance-probe" and item.release_gate)
        )
    }
    for alternative in alternatives:
        if alternative.observed != "accepted":
            failures.append("ALTERNATIVE_VALID_SOLUTION_REJECTED:" + alternative.solution_id)
            continue
        failures.extend(
            _observe_exactly_once(
                label="ALTERNATIVE_" + alternative.solution_id,
                observations=alternative.observations,
                allowed=set(obligation_by_id),
                required=hard_ids,
            )
        )
        by_id = {item.obligation_id: item for item in alternative.observations}
        for obligation_id in sorted(hard_ids):
            observation = by_id.get(obligation_id)
            if observation is not None and observation.status != "PASS":
                failures.append(
                    "ALTERNATIVE_HARD_OBLIGATION_NOT_PASSING:"
                    + alternative.solution_id
                    + ":"
                    + obligation_id
                )
    return failures


def qualify_task(
    *,
    specification: TaskSpecification,
    contract: TaskAcceptanceContract,
    witness_set: WitnessSet,
    baseline: BaselineDifferential,
    witness_replays: Sequence[WitnessReplayResult],
    mutations: Sequence[MutationOutcome],
    negative_controls: Sequence[NegativeControlOutcome],
    alternatives: Sequence[AlternativeValidSolutionOutcome],
    flakiness: VerifierFlakinessAudit,
    leakage: TaskLeakageAudit,
    review: HumanTaskQualificationReview | None,
    mutation_suite_sha256: str | None = None,
    alternative_solution_set_sha256: str | None = None,
    minimum_mutation_adequacy: float = 0.80,
) -> TaskQualificationReport:
    """Qualify the task/verifier before any Candidate is officially scored."""

    if not 0 <= minimum_mutation_adequacy <= 1:
        raise ValueError("minimum_mutation_adequacy must stay in [0, 1]")
    mutations_list = list(mutations)
    alternatives_list = list(alternatives)
    mutation_digest = mutation_suite_sha256 or canonical_sha256(
        [item.model_dump(mode="json") for item in mutations_list]
    )
    alternative_digest = alternative_solution_set_sha256 or canonical_sha256(
        [item.model_dump(mode="json") for item in alternatives_list]
    )
    specification_digest = canonical_sha256(specification)

    failures = audit_acceptance_contract(specification, contract)
    failures.extend(audit_witness_set(specification, contract, witness_set))
    failures.extend(_baseline_failures(specification, contract, baseline))
    failures.extend(_witness_replay_failures(contract, witness_set, list(witness_replays)))
    failures.extend(
        _adequacy_failures(
            contract,
            mutations_list,
            list(negative_controls),
            alternatives_list,
            minimum_mutation_adequacy=minimum_mutation_adequacy,
        )
    )
    if flakiness.status not in {"STABLE", "STOCHASTIC_WITH_BOUNDED_ORACLE"}:
        failures.append("VERIFIER_FLAKINESS_NOT_QUALIFIED:" + flakiness.status)
    if flakiness.pass_fail_flips and flakiness.status != "STOCHASTIC_WITH_BOUNDED_ORACLE":
        failures.append("VERIFIER_UNBOUNDED_PASS_FAIL_FLIPS")
    if leakage.status != "pass":
        failures.extend(leakage.failure_codes or ["TASK_LEAKAGE_AUDIT_NOT_PASSING"])

    expected_review_bindings = {
        "specification_sha256": specification_digest,
        "contract_sha256": contract.contract_sha256,
        "witness_set_sha256": witness_set.witness_set_sha256,
        "mutation_suite_sha256": mutation_digest,
        "alternative_solution_set_sha256": alternative_digest,
        "artifact_policy_sha256": specification.artifact_policy_sha256,
        "capability_policy_sha256": specification.capability_contract_sha256,
    }
    review_failures: list[str] = []
    if review is not None:
        for name, expected in expected_review_bindings.items():
            if getattr(review, name) != expected:
                review_failures.append("HUMAN_REVIEW_BINDING_MISMATCH:" + name)
        if review.decision == "reject":
            failures.append("HUMAN_TASK_REVIEW_REJECTED")
        elif review.decision == "request-revision":
            review_failures.append("HUMAN_TASK_REVIEW_REQUESTED_REVISION")

    all_failures = sorted(set([*failures, *review_failures]))
    if failures:
        status = "INELIGIBLE"
    elif review is None or review_failures or review.decision != "approve":
        status = "REVIEW_REQUIRED"
    else:
        status = "QUALIFIED"

    hard_ids = sorted(
        item.obligation_id
        for item in contract.obligations
        if item.severity == "hard"
        and (
            item.bucket in INFRA_CERT_BUCKETS
            or (item.bucket == "maintenance-probe" and item.release_gate)
        )
    )
    covered = sorted(
        set().union(*(set(item.covers) for item in witness_set.witnesses)) & set(hard_ids)
    )
    requirement_map = {
        requirement.requirement_id: sorted(
            item.obligation_id
            for item in contract.obligations
            if requirement.requirement_id in item.source_requirements
        )
        for requirement in specification.requirements
    }
    adequacy = weighted_mutation_adequacy(mutations_list)
    coverage_failures = [
        item
        for item in all_failures
        if any(
            marker in item
            for marker in (
                "REQUIREMENT",
                "OBLIGATION",
                "WITNESS",
                "MUTANT",
                "MUTATION",
                "NEGATIVE_CONTROL",
                "ALTERNATIVE",
            )
        )
    ]
    coverage = VerifierCoverageReport(
        requirement_to_obligations=requirement_map,
        hard_obligations=hard_ids,
        witness_covered_hard_obligations=covered,
        critical_mutants_total=sum(item.critical for item in mutations_list),
        critical_mutants_killed=sum(
            item.critical and item.observed == "rejected" for item in mutations_list
        ),
        weighted_mutation_adequacy=adequacy,
        negative_controls_total=len(negative_controls),
        negative_controls_rejected=sum(item.observed == "rejected" for item in negative_controls),
        alternative_valid_total=len(alternatives_list),
        alternative_valid_accepted=sum(item.observed == "accepted" for item in alternatives_list),
        status="pass" if not coverage_failures else "fail",
        failure_codes=coverage_failures,
    )
    es_valid = all(
        observation.status == "PASS"
        for observation in baseline.observations
        if observation.bucket == "environment-sentinel"
    )
    trust_failures = [
        item
        for item in all_failures
        if item not in review_failures and not item.startswith("HUMAN_")
    ]
    trust = VerifierTrustCard(
        status="pass" if not trust_failures else "fail",
        baseline_differential_valid=not any(item.startswith("BASELINE_") for item in failures),
        witness_replay_valid=not any(item.startswith("WITNESS_") for item in failures),
        mutation_adequacy=adequacy,
        negative_controls_valid=bool(negative_controls)
        and all(item.observed == "rejected" for item in negative_controls),
        alternative_solution_breadth_valid=bool(alternatives_list)
        and all(item.observed == "accepted" for item in alternatives_list),
        fresh_replay_stable=flakiness.status in {"STABLE", "STOCHASTIC_WITH_BOUNDED_ORACLE"},
        environment_sentinel_valid=es_valid,
        leakage_valid=leakage.status == "pass",
        failure_codes=trust_failures,
    )
    preliminary = TaskQualificationReport(
        task_id=specification.task_id,
        task_revision=specification.task_revision,
        status=status,
        specification_sha256=specification_digest,
        contract_sha256=contract.contract_sha256,
        witness_set_sha256=witness_set.witness_set_sha256,
        mutation_suite_sha256=mutation_digest,
        alternative_solution_set_sha256=alternative_digest,
        coverage=coverage,
        trust=trust,
        failure_codes=all_failures,
        report_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"report_sha256"})
    return preliminary.model_copy(update={"report_sha256": canonical_sha256(material)})
