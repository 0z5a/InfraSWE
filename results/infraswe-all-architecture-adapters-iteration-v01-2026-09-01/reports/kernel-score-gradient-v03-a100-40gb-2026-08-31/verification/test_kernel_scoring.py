from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from infraswe.kernel.models import (
    Authority,
    Disposition,
    FailureCode,
    MetricValue,
    RoleIdentity,
    RoleInput,
    RoleKey,
    RoleResult,
    RoleStatus,
    Scope,
    Verdict,
)
from infraswe.kernel.role_graph import RoleGraph
from infraswe.kernel.scoring import (
    aggregate_kernel_score,
    anchor_score,
    evaluate_anchor_case,
    weighted_portfolio_score,
)
from infraswe.kernel.sealing import seal_role_result, verify_role_result
from infraswe.kernel.statistics import paired_log_speedup, paired_log_speedup_ci

NOW = datetime(2026, 8, 31, tzinfo=UTC)


def _identity(replay: int = 1, **changes: str) -> RoleIdentity:
    values = {
        "task_package_sha256": "sha256:task",
        "candidate_source_sha256": "sha256:candidate",
        "build_artifact_sha256": "sha256:build",
        "role_graph_sha256": "sha256:graph",
        "evaluator_sha256": "sha256:evaluator",
        "hardware_class_sha256": "sha256:hardware",
        "environment_contract_sha256": "sha256:contract",
        "execution_environment_sha256": f"sha256:execution-{replay}",
    }
    values.update(changes)
    return RoleIdentity(**values)


def _role(
    role_id: str,
    authority: Authority,
    scope: Scope,
    *,
    replay: int | None = None,
    score: float | None = None,
    verdict: Verdict = Verdict.PASS,
    disposition: Disposition = Disposition.VALID,
    identity: RoleIdentity | None = None,
    failures: list[FailureCode] | None = None,
    inputs: list[RoleInput] | None = None,
    result_sha256: str = "",
) -> RoleResult:
    return RoleResult(
        role_id=role_id,
        role_instance_id=f"{role_id}/{replay or 'trial'}",
        authority=authority,
        scope=scope,
        status=RoleStatus.COMPLETED,
        verdict=verdict,
        disposition=disposition,
        profile="kernel-micro",
        replay_index=replay,
        identity=identity or _identity(replay or 1),
        inputs=inputs or [],
        score=score,
        metrics={
            "latency": MetricValue(
                value=9.5,
                unit="us",
                statistic="median",
                population="sealed-hidden",
            )
        },
        failure_codes=failures or [],
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        result_sha256=result_sha256,
    )


def _valid_results() -> list[RoleResult]:
    return [
        _role("correctness", Authority.GATE, Scope.REPLAY, replay=1),
        _role("correctness", Authority.GATE, Scope.REPLAY, replay=2),
        _role("replay-auditor", Authority.AUDIT, Scope.TRIAL),
        _role("anchor-scorer", Authority.SCORE, Scope.TRIAL, score=0.82),
    ]


def _expected(results: list[RoleResult]) -> set[RoleKey]:
    return {RoleKey.from_result(result) for result in results}


def test_anchor_score_keeps_baseline_and_anchor_semantics() -> None:
    assert anchor_score(20, 20, 10) == pytest.approx(0.5)
    assert anchor_score(20, 10, 10) == pytest.approx(1.0)
    assert anchor_score(20, 40, 10) < 0.5
    assert anchor_score(20, 12, 10) > anchor_score(20, 16, 10)


def test_anchor_guard_does_not_clip_beyond_anchor() -> None:
    result = evaluate_anchor_case(
        baseline_latency=20,
        candidate_latency=8,
        anchor_latency=10,
        beyond_anchor_tolerance=0.03,
    )
    assert result.status == "quarantined"
    assert result.anchor_score_raw is None
    assert result.anchor_efficiency_raw == pytest.approx(1.25)
    assert result.failure_codes == ["MEASUREMENT_BEYOND_ANCHOR"]


def test_anchor_guard_rejects_cell_without_headroom() -> None:
    result = evaluate_anchor_case(
        baseline_latency=10.5,
        candidate_latency=10.2,
        anchor_latency=10,
    )
    assert result.status == "not_frontier_eligible"
    assert "MEASUREMENT_NO_HEADROOM" in result.failure_codes


def test_role_authority_is_single_and_score_is_exclusive() -> None:
    with pytest.raises(ValidationError, match="only score authority"):
        _role("meter", Authority.METRIC, Scope.TRIAL, score=0.8)
    with pytest.raises(ValidationError, match="score authority requires"):
        _role("scorer", Authority.SCORE, Scope.TRIAL)


def test_aggregator_accepts_execution_nonce_drift() -> None:
    results = _valid_results()
    aggregate = aggregate_kernel_score(
        results,
        expected_instances=_expected(results),
        component_weights={"anchor-scorer": 1.0},
    )
    assert aggregate.certified
    assert aggregate.artifact_100 == pytest.approx(82)
    assert aggregate.leaderboard_effective_artifact_100 == pytest.approx(82)


def test_aggregator_rejects_stable_identity_drift() -> None:
    results = _valid_results()
    results[1] = _role(
        "correctness",
        Authority.GATE,
        Scope.REPLAY,
        replay=2,
        identity=_identity(2, hardware_class_sha256="sha256:wrong-hardware"),
    )
    aggregate = aggregate_kernel_score(
        results,
        expected_instances=_expected(results),
        component_weights={"anchor-scorer": 1.0},
    )
    assert aggregate.disposition == "invalid"
    assert aggregate.artifact_100 is None
    assert "EVIDENCE_IDENTITY_DRIFT:hardware_class_sha256" in aggregate.failure_codes


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("duplicate", "EVIDENCE_DUPLICATE_ROLE_INSTANCE"),
        ("missing", "EVIDENCE_REQUIRED_ROLE_MISSING"),
        ("unexpected", "EVIDENCE_UNEXPECTED_ROLE_INSTANCE"),
    ],
)
def test_aggregator_fails_closed_on_role_grid_mutation(mutation: str, code: str) -> None:
    canonical = _valid_results()
    expected = _expected(canonical)
    if mutation == "duplicate":
        results = [*canonical, canonical[0]]
    elif mutation == "missing":
        results = canonical[:-1]
    else:
        results = [*canonical, _role("extra", Authority.GATE, Scope.TRIAL)]
    aggregate = aggregate_kernel_score(
        results,
        expected_instances=expected,
        component_weights={"anchor-scorer": 1.0},
    )
    assert aggregate.disposition == "invalid"
    assert any(item.startswith(code) for item in aggregate.failure_codes)


def test_valid_candidate_gate_failure_has_effective_zero() -> None:
    results = _valid_results()
    results[0] = _role(
        "correctness",
        Authority.GATE,
        Scope.REPLAY,
        replay=1,
        verdict=Verdict.FAIL,
        failures=[
            FailureCode(
                code="CORRECTNESS_OUTPUT_MISMATCH",
                severity="error",
                owner="candidate",
                retryable=False,
            )
        ],
    )
    aggregate = aggregate_kernel_score(
        results,
        expected_instances=_expected(results),
        component_weights={"anchor-scorer": 1.0},
    )
    assert aggregate.verdict == "fail"
    assert aggregate.disposition == "valid"
    assert aggregate.artifact_100 is None
    assert aggregate.leaderboard_effective_artifact_100 == 0


def test_advisory_result_cannot_enter_scorer_dependency() -> None:
    advisory = _role(
        "explanation",
        Authority.ADVISORY,
        Scope.TRIAL,
        result_sha256="sha256:advice",
    )
    results = _valid_results()
    results[-1] = _role(
        "anchor-scorer",
        Authority.SCORE,
        Scope.TRIAL,
        score=0.82,
        inputs=[
            RoleInput(
                role_instance_id=advisory.role_instance_id,
                result_sha256="sha256:advice",
            )
        ],
    )
    results.append(advisory)
    aggregate = aggregate_kernel_score(
        results,
        expected_instances=_expected(results),
        component_weights={"anchor-scorer": 1.0},
    )
    assert aggregate.disposition == "invalid"
    assert aggregate.failure_codes == ["EVIDENCE_ADVISORY_SCORE_DEPENDENCY"]


def test_role_result_seal_detects_mutation() -> None:
    sealed = seal_role_result(_valid_results()[0], b"test-only-key")
    assert verify_role_result(sealed, b"test-only-key")
    mutated = sealed.model_copy(update={"message": "changed after sealing"})
    assert not verify_role_result(mutated, b"test-only-key")


def test_role_graph_rejects_cycle_and_advisory_score_path() -> None:
    common = {
        "schema_version": "0.3",
        "graph_id": "test",
        "edges_digest": "sha256:edges",
    }
    with pytest.raises(ValidationError, match="acyclic"):
        RoleGraph.model_validate(
            {
                **common,
                "nodes": [
                    {
                        "id": "a",
                        "authority": "gate",
                        "scope": "trial",
                        "needs": ["b"],
                        "image_digest": "sha256:image",
                        "timeout_sec": 1,
                        "on_error": "invalidate",
                    },
                    {
                        "id": "b",
                        "authority": "metric",
                        "scope": "trial",
                        "needs": ["a"],
                        "image_digest": "sha256:image",
                        "timeout_sec": 1,
                        "on_error": "invalidate",
                    },
                ],
            }
        )
    with pytest.raises(ValidationError, match="advisory role"):
        RoleGraph.model_validate(
            {
                **common,
                "nodes": [
                    {
                        "id": "critic",
                        "authority": "advisory",
                        "scope": "trial",
                        "image_digest": "sha256:image",
                        "timeout_sec": 1,
                        "on_error": "ignore",
                    },
                    {
                        "id": "score",
                        "authority": "score",
                        "scope": "trial",
                        "needs": ["critic"],
                        "image_digest": "sha256:image",
                        "timeout_sec": 1,
                        "on_error": "invalidate",
                    },
                ],
            }
        )


def test_paired_estimator_is_reciprocal_and_bootstrap_is_deterministic() -> None:
    reference = [20.0, 22.0, 19.0, 21.0, 20.5]
    candidate = [10.0, 11.0, 9.5, 10.5, 10.25]
    assert paired_log_speedup(reference, candidate) == pytest.approx(2.0)
    assert paired_log_speedup(candidate, reference) == pytest.approx(0.5)
    first = paired_log_speedup_ci(reference, candidate, resamples=500, seed=7)
    second = paired_log_speedup_ci(reference, candidate, resamples=500, seed=7)
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_portfolio_requires_exact_normalized_case_weights() -> None:
    assert weighted_portfolio_score({"a": 0.8, "b": 1.0}, {"a": 0.25, "b": 0.75}) == 0.95
    with pytest.raises(ValueError, match="case sets"):
        weighted_portfolio_score({"a": 0.8}, {"a": 0.5, "b": 0.5})
