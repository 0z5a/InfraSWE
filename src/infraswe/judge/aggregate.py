from __future__ import annotations

import statistics
from collections import defaultdict

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.judge import (
    JudgeAggregation,
    JudgeCell,
    JudgeCriterionAggregation,
    JudgeInputPackManifest,
    JudgeProfile,
    JudgeRubric,
    JudgeRunRecord,
)


def _weighted_median(values: list[tuple[float, float]]) -> float:
    positive = sorted((value, weight) for value, weight in values if weight > 0)
    if not positive:
        raise ValueError("weighted median requires a positive vote weight")
    total = sum(weight for _, weight in positive)
    threshold = total / 2
    cumulative = 0.0
    for value, weight in positive:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return positive[-1][0]


def _criterion_result(run: JudgeRunRecord, criterion_id: str):
    return next(
        (item for item in run.output.criteria if item.criterion_id == criterion_id),
        None,
    )


def _aggregate_criterion(
    criterion_id: str,
    *,
    runs: list[JudgeRunRecord],
    profile: JudgeProfile,
) -> JudgeCriterionAggregation:
    eligible = [run for run in runs if not run.candidate_family_excluded]
    total = len(eligible)
    security_incident = any(
        run.output.global_status.prompt_injection_suspected
        or (
            (result := _criterion_result(run, criterion_id)) is not None
            and (result.verdict == "possible-prompt-injection" or bool(result.security_flags))
        )
        for run in eligible
    )

    raw_votes: list[tuple[JudgeRunRecord, float, float]] = []
    for run in eligible:
        result = _criterion_result(run, criterion_id)
        if (
            run.validation_status == "valid"
            and result is not None
            and result.normalized_value is not None
        ):
            weight = run.criterion_calibration_weights.get(
                criterion_id,
                run.calibration_weight,
            )
            if weight > 0:
                raw_votes.append((run, result.normalized_value, weight))

    abstention_rate = 1.0 if total == 0 else 1 - len(raw_votes) / total
    grouped: dict[str, list[tuple[JudgeRunRecord, float, float]]] = defaultdict(list)
    for vote in raw_votes:
        grouped[vote[0].member_id].append(vote)

    required_repetitions = profile.aggregation.repetitions_per_member
    qualified: dict[str, list[tuple[JudgeRunRecord, float, float]]] = {}
    repeat_passes = 0
    repeated_members = 0
    disagreement_members: list[str] = []
    for member_id, member_votes in grouped.items():
        if len(member_votes) < required_repetitions:
            continue
        repeated_members += 1
        values = [value for _, value, _ in member_votes]
        if max(values) - min(values) <= profile.aggregation.maximum_within_member_range:
            repeat_passes += 1
            qualified[member_id] = member_votes
        else:
            disagreement_members.append(member_id)

    repeat_agreement = repeat_passes / repeated_members if repeated_members else 0.0
    votes = [vote for member_votes in qualified.values() for vote in member_votes]
    families = {run.model_family for run, _, _ in votes}
    family_values: dict[str, list[float]] = defaultdict(list)
    for run, value, _ in votes:
        family_values[run.model_family].append(value)
    family_medians = {family: statistics.median(values) for family, values in family_values.items()}
    family_range = (
        max(family_medians.values()) - min(family_medians.values())
        if len(family_medians) >= 2
        else None
    )

    failures: list[str] = []
    if security_incident:
        failures.append("JUDGE_SECURITY_REVIEW_REQUIRED")
        status = "security-review-required"
    elif disagreement_members or (
        family_range is not None and family_range > profile.aggregation.maximum_cross_family_range
    ):
        failures.extend(
            f"JUDGE_REPEAT_DISAGREEMENT:{member_id}" for member_id in disagreement_members
        )
        if (
            family_range is not None
            and family_range > profile.aggregation.maximum_cross_family_range
        ):
            failures.append("JUDGE_CROSS_FAMILY_DISAGREEMENT")
        status = "judge-disagreement"
    elif len(qualified) < profile.aggregation.minimum_valid_members:
        failures.append("JUDGE_MINIMUM_VALID_MEMBERS_NOT_MET")
        status = "unresolved"
    elif len(families) < profile.aggregation.minimum_model_families:
        failures.append("JUDGE_MINIMUM_MODEL_FAMILIES_NOT_MET")
        status = "unresolved"
    elif not votes:
        failures.append("JUDGE_REQUIRED_CRITERION_ABSTAINED")
        status = "unresolved"
    else:
        status = "valid"

    if status != "valid":
        return JudgeCriterionAggregation(
            criterion_id=criterion_id,
            status=status,
            valid_vote_count=len(votes),
            valid_member_count=len(qualified),
            valid_family_count=len(families),
            abstention_rate=abstention_rate,
            repeat_agreement=repeat_agreement,
            cross_family_range=family_range,
            failure_codes=sorted(set(failures)),
        )

    weighted = [(value, weight) for _, value, weight in votes]
    median = _weighted_median(weighted)
    mad = _weighted_median([(abs(value - median), weight) for value, weight in weighted])
    return JudgeCriterionAggregation(
        criterion_id=criterion_id,
        status="valid",
        normalized_value=median,
        weighted_mad=mad,
        valid_vote_count=len(votes),
        valid_member_count=len(qualified),
        valid_family_count=len(families),
        abstention_rate=abstention_rate,
        repeat_agreement=repeat_agreement,
        cross_family_range=family_range,
    )


def aggregate_panel(
    runs: list[JudgeRunRecord],
    *,
    profile: JudgeProfile,
    cell: JudgeCell,
    rubric: JudgeRubric,
    input_pack: JudgeInputPackManifest,
) -> JudgeAggregation:
    """Aggregate validated votes without ever inventing a global Judge score."""

    if profile.authority != "bounded-score" or profile.mode != "bounded-semantic":
        raise ValueError("official panel aggregation requires bounded-score authority")
    if not runs:
        raise ValueError("Judge panel aggregation requires run records")
    identities = {
        (
            run.output.judge_cell_sha256,
            run.output.input_pack_sha256,
            run.output.rubric_sha256,
        )
        for run in runs
    }
    expected_identity = {(cell.judge_cell_sha256, input_pack.pack_sha256, cell.rubric_sha256)}
    if identities != expected_identity:
        raise ValueError("Judge panel run identity drift")

    criteria = [
        _aggregate_criterion(item.criterion_id, runs=runs, profile=profile)
        for item in rubric.criteria
        if item.owner_type == "semantic-judge"
    ]
    if not criteria:
        raise ValueError("Judge rubric has no semantic residual criteria")
    statuses = {item.status for item in criteria}
    if "security-review-required" in statuses:
        status = "security-review-required"
    elif "judge-disagreement" in statuses:
        status = "judge-disagreement"
    elif "unresolved" in statuses:
        status = "unresolved-judge"
    else:
        status = "official"
    material = {
        "schema_version": "0.5.3",
        "judge_cell_sha256": cell.judge_cell_sha256,
        "input_pack_sha256": input_pack.pack_sha256,
        "rubric_sha256": cell.rubric_sha256,
        "policy_id": profile.aggregation.policy_id,
        "status": status,
        "top_level_score_status": "not-a-score",
        "criteria": [item.model_dump(mode="json") for item in criteria],
    }
    return JudgeAggregation.model_validate(
        {**material, "aggregation_sha256": canonical_sha256(material)}
    )


def audit_aggregation(aggregation: JudgeAggregation) -> list[str]:
    material = aggregation.model_dump(mode="json", exclude={"aggregation_sha256"})
    if aggregation.aggregation_sha256 != canonical_sha256(material):
        return ["JUDGE_AGGREGATION_DIGEST_MISMATCH"]
    return []
