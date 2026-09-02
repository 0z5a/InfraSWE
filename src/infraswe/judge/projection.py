from __future__ import annotations

import math

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.judge.aggregate import audit_aggregation
from infraswe.models.judge import (
    JudgeAggregation,
    JudgeComponentProjection,
    JudgeRubric,
    JudgeScoreProjection,
)


def _weighted_geometric(values: dict[str, float], weights: dict[str, float]) -> float:
    if set(values) != set(weights):
        raise ValueError("criterion values and weights must have identical keys")
    if not values or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("criterion weights must be nonempty and sum to 1")
    if any(value < 0 or value > 1 for value in values.values()):
        raise ValueError("criterion values must stay in [0, 1]")
    if any(value == 0 and weights[name] > 0 for name, value in values.items()):
        return 0.0
    return math.exp(sum(weights[name] * math.log(value) for name, value in values.items()))


def build_score_projection(
    rubric: JudgeRubric,
    aggregation: JudgeAggregation,
    *,
    deterministic_values: dict[str, float],
    infra_cert_status: str,
) -> JudgeScoreProjection:
    """Project bounded semantic criteria into P/M/U without adding Judge-100."""

    if infra_cert_status not in {"pass", "fail", "unresolved"}:
        raise ValueError("InfraCert status must be pass, fail, or unresolved")
    failures = audit_aggregation(aggregation)
    if failures:
        raise ValueError("cannot project invalid Judge aggregation: " + ", ".join(failures))
    if aggregation.rubric_sha256 != canonical_sha256(rubric):
        raise ValueError("Judge aggregation and rubric identity mismatch")

    aggregate_by_id = {item.criterion_id: item for item in aggregation.criteria}
    deterministic_ids = {
        item.criterion_id for item in rubric.criteria if item.owner_type == "deterministic"
    }
    missing = sorted(deterministic_ids - set(deterministic_values))
    extra = sorted(set(deterministic_values) - deterministic_ids)
    if missing or extra:
        raise ValueError(
            f"deterministic criterion ownership mismatch: missing={missing} extra={extra}"
        )
    if any(value < 0 or value > 1 for value in deterministic_values.values()):
        raise ValueError("deterministic criterion values must stay in [0, 1]")

    components: dict[str, JudgeComponentProjection] = {}
    for component in {item.owner_component for item in rubric.criteria}:
        criteria = [item for item in rubric.criteria if item.owner_component == component]
        deterministic = [item for item in criteria if item.owner_type == "deterministic"]
        semantic = [item for item in criteria if item.owner_type == "semantic-judge"]
        if not deterministic:
            raise ValueError(f"{component} requires a deterministic core projection")
        deterministic_weight = sum(item.weight_within_component for item in deterministic)
        normalized_weights = {
            item.criterion_id: item.weight_within_component / deterministic_weight
            for item in deterministic
        }
        core_values = {
            item.criterion_id: deterministic_values[item.criterion_id] for item in deterministic
        }
        core = _weighted_geometric(core_values, normalized_weights)
        judge_weight = sum(item.weight_within_component for item in semantic)
        criterion_values = dict(core_values)

        can_publish = infra_cert_status == "pass" and aggregation.status == "official"
        if not semantic:
            component_status = "deterministic-only"
            assisted = None
        elif can_publish:
            semantic_values: dict[str, float] = {}
            for item in semantic:
                aggregate = aggregate_by_id.get(item.criterion_id)
                if aggregate is None or aggregate.status != "valid":
                    raise ValueError(
                        f"official aggregation is missing valid criterion {item.criterion_id}"
                    )
                assert aggregate.normalized_value is not None
                semantic_values[item.criterion_id] = aggregate.normalized_value
            criterion_values.update(semantic_values)
            all_weights = {item.criterion_id: item.weight_within_component for item in criteria}
            assisted = _weighted_geometric(criterion_values, all_weights)
            component_status = "official"
        else:
            assisted = None
            component_status = "unresolved-judge"

        components[component] = JudgeComponentProjection(
            component=component,
            status=component_status,
            deterministic_core_projection=core,
            judge_assisted_projection=assisted,
            judge_weight_within_component=judge_weight,
            criterion_values=criterion_values,
        )

    if infra_cert_status == "fail":
        status = "hard-gate-failed"
    elif infra_cert_status == "unresolved":
        status = "hard-gate-unresolved"
    elif aggregation.status != "official":
        status = "unresolved-judge"
    else:
        status = "official"

    material = {
        "schema_version": "0.5.3",
        "infra_cert_status": infra_cert_status,
        "status": status,
        "components": {name: item.model_dump(mode="json") for name, item in components.items()},
        "cross_judge_cell_ranking_allowed": False,
    }
    return JudgeScoreProjection.model_validate(
        {**material, "projection_sha256": canonical_sha256(material)}
    )
