from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from infraswe.kernel.models import (
    AnchorCaseResult,
    Authority,
    Disposition,
    KernelAggregate,
    RoleKey,
    RoleResult,
    RoleStatus,
    Verdict,
)

STABLE_IDENTITY_FIELDS = (
    "task_package_sha256",
    "candidate_source_sha256",
    "build_artifact_sha256",
    "role_graph_sha256",
    "evaluator_sha256",
    "hardware_class_sha256",
    "environment_contract_sha256",
)


def speedup(baseline_latency: float, candidate_latency: float) -> float:
    if baseline_latency <= 0 or candidate_latency <= 0:
        raise ValueError("latencies must be positive")
    return baseline_latency / candidate_latency


def anchor_efficiency(anchor_latency: float, candidate_latency: float) -> float:
    if anchor_latency <= 0 or candidate_latency <= 0:
        raise ValueError("latencies must be positive")
    return anchor_latency / candidate_latency


def anchor_score(
    baseline_latency: float,
    candidate_latency: float,
    anchor_latency: float,
) -> float:
    if min(baseline_latency, candidate_latency, anchor_latency) <= 0:
        raise ValueError("latencies must be positive")
    if baseline_latency <= anchor_latency:
        raise ValueError("scoring baseline must be slower than anchor")
    denominator = (candidate_latency - anchor_latency) + (baseline_latency - anchor_latency)
    if denominator <= 0:
        raise ValueError("candidate is too far beyond anchor for a finite AnchorScore")
    return (baseline_latency - anchor_latency) / denominator


def evaluate_anchor_case(
    *,
    baseline_latency: float,
    candidate_latency: float,
    anchor_latency: float,
    min_headroom: float = 1.10,
    beyond_anchor_tolerance: float = 0.03,
) -> AnchorCaseResult:
    raw_speedup = speedup(baseline_latency, candidate_latency)
    efficiency = anchor_efficiency(anchor_latency, candidate_latency)
    common = {
        "scoring_baseline_latency_us": baseline_latency,
        "candidate_latency_us": candidate_latency,
        "anchor_latency_us": anchor_latency,
        "speedup_vs_scoring_baseline_raw": raw_speedup,
        "anchor_efficiency_raw": efficiency,
    }
    if baseline_latency / anchor_latency < min_headroom:
        return AnchorCaseResult(
            status="not_frontier_eligible",
            anchor_score_raw=None,
            failure_codes=["MEASUREMENT_NO_HEADROOM"],
            **common,
        )
    if candidate_latency < anchor_latency * (1 - beyond_anchor_tolerance):
        return AnchorCaseResult(
            status="quarantined",
            anchor_score_raw=None,
            failure_codes=["MEASUREMENT_BEYOND_ANCHOR"],
            **common,
        )
    return AnchorCaseResult(
        status="scored",
        anchor_score_raw=anchor_score(baseline_latency, candidate_latency, anchor_latency),
        **common,
    )


def weighted_portfolio_score(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    if set(scores) != set(weights):
        raise ValueError("score and weight case sets must match exactly")
    if not scores:
        raise ValueError("portfolio cannot be empty")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("portfolio scores must be finite and inside [0, 1]")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("portfolio weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("portfolio weights must sum to 1")
    return sum(scores[case_id] * weights[case_id] for case_id in scores)


def _invalid(*codes: str) -> KernelAggregate:
    return KernelAggregate(
        certified=False,
        verdict="unresolved",
        disposition="invalid",
        artifact_status="unscored_invalid",
        failure_codes=list(codes),
    )


def aggregate_kernel_score(
    role_results: Iterable[RoleResult],
    *,
    expected_instances: set[RoleKey],
    component_weights: Mapping[str, float],
) -> KernelAggregate:
    if not math.isclose(sum(component_weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("component weights must sum to 1")
    if any(weight < 0 for weight in component_weights.values()):
        raise ValueError("component weights must be non-negative")

    results: dict[RoleKey, RoleResult] = {}
    for result in role_results:
        key = RoleKey.from_result(result)
        if key in results:
            return _invalid(f"EVIDENCE_DUPLICATE_ROLE_INSTANCE:{key}")
        results[key] = result

    missing = sorted(expected_instances - results.keys())
    if missing:
        return _invalid(*(f"EVIDENCE_REQUIRED_ROLE_MISSING:{key}" for key in missing))
    unexpected = sorted(results.keys() - expected_instances)
    if unexpected:
        return _invalid(*(f"EVIDENCE_UNEXPECTED_ROLE_INSTANCE:{key}" for key in unexpected))

    official = [result for result in results.values() if result.authority != Authority.ADVISORY]
    for field in STABLE_IDENTITY_FIELDS:
        if len({getattr(result.identity, field) for result in official}) != 1:
            return _invalid(f"EVIDENCE_IDENTITY_DRIFT:{field}")

    if any(result.status != RoleStatus.COMPLETED for result in official):
        return _invalid("EVIDENCE_ROLE_NOT_COMPLETED")
    dispositions = {result.disposition for result in official}
    if Disposition.QUARANTINED in dispositions:
        return KernelAggregate(
            certified=False,
            verdict="unresolved",
            disposition="quarantined",
            artifact_status="quarantined",
            failure_codes=["SECURITY_OR_ANCHOR_QUARANTINE"],
        )
    if Disposition.INVALID in dispositions:
        return _invalid("EVIDENCE_INVALID_ROLE_RESULT")

    digest_authority = {
        result.result_sha256: result.authority
        for result in results.values()
        if result.result_sha256
    }
    for result in official:
        if result.authority != Authority.SCORE:
            continue
        has_advisory_input = any(
            digest_authority.get(item.result_sha256) == Authority.ADVISORY for item in result.inputs
        )
        if has_advisory_input:
            return _invalid("EVIDENCE_ADVISORY_SCORE_DEPENDENCY")

    gate_failures = sorted(
        {
            failure.code
            for result in official
            if result.authority in {Authority.GATE, Authority.AUDIT}
            and result.verdict == Verdict.FAIL
            for failure in result.failure_codes
        }
    )
    if gate_failures:
        return KernelAggregate(
            certified=False,
            verdict="fail",
            disposition="valid",
            artifact_status="not_applicable",
            artifact_100=None,
            leaderboard_effective_artifact_100=0.0,
            failure_codes=gate_failures or ["KERNEL_CERT_GATE_FAILED"],
        )

    trial_scores: dict[str, RoleResult] = {}
    for result in official:
        if result.authority != Authority.SCORE or result.scope.value != "trial":
            continue
        if result.role_id in trial_scores:
            return _invalid(f"EVIDENCE_COMPONENT_SCORE_DUPLICATE:{result.role_id}")
        trial_scores[result.role_id] = result

    components: dict[str, float] = {}
    for role_id in component_weights:
        result = trial_scores.get(role_id)
        if result is None or result.score is None:
            return _invalid(f"EVIDENCE_COMPONENT_SCORE_MISSING:{role_id}")
        components[role_id] = result.score

    artifact = 100 * sum(
        component_weights[role_id] * components[role_id] for role_id in component_weights
    )
    return KernelAggregate(
        certified=True,
        verdict="pass",
        disposition="valid",
        artifact_status="scored",
        artifact_100=artifact,
        leaderboard_effective_artifact_100=artifact,
        components=components,
    )
