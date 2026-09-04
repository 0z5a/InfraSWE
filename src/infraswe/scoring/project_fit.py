from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from infraswe.models.draft import DraftState, ProjectComparisonCell
from infraswe.models.project_score import (
    BenchmarkCostCard,
    BenchmarkTrustCard,
    CellEfficiencyReference,
    InfraSWEDecision,
    InfraSWEMicroscores,
    InfraSWEOverallResult,
    MergeabilityDecision,
    OrderedEvaluationGate,
    ProjectFitScore,
    ProjectObjectiveResult,
    ProjectScoreComponent,
    PullRequestReviewContext,
    PureTritonEligibilityEvidence,
    TritonPurityAudit,
    V05ScoreResult,
)
from infraswe.policy import (
    CHECK_ACTIVITY_MAX_IDLE_DAYS,
    CHECK_NEW_PR_MAX_AGE_DAYS,
    MERGE_ACCEPT_SCORE_FLOOR_100,
    STALE_REVIEWED_OPEN_MIN_AGE_DAYS,
    overall_score_decision_band,
)
from infraswe.scoring.deployability import weighted_geometric

PROJECT_FIT_WEIGHTS = {
    "project-fit-kernel-v0.5": {
        "evolutionary_maintainability": 0.40,
        "project_contract_fit": 0.30,
        "performance_reuse_utilization": 0.20,
        "operational_fit": 0.10,
    },
    "project-fit-triton-pure-v0.5": {
        "evolutionary_maintainability": 0.35,
        "project_contract_fit": 0.25,
        "pure_triton_portability": 0.20,
        "performance_reuse_utilization": 0.10,
        "operational_fit": 0.10,
    },
    "project-fit-system-path-v0.5.1": {
        "evolutionary_maintainability": 0.40,
        "project_contract_fit": 0.30,
        "performance_reuse_utilization": 0.20,
        "operational_fit": 0.10,
    },
}
PROJECT_FIT_FLOORS = {
    "project-fit-kernel-v0.5": {
        "evolutionary_maintainability": 0.60,
        "project_contract_fit": 0.60,
        "performance_reuse_utilization": 0.40,
        "operational_fit": 0.60,
    },
    "project-fit-triton-pure-v0.5": {
        "evolutionary_maintainability": 0.60,
        "project_contract_fit": 0.60,
        "pure_triton_portability": 0.50,
        "performance_reuse_utilization": 0.35,
        "operational_fit": 0.60,
    },
    "project-fit-system-path-v0.5.1": {
        "evolutionary_maintainability": 0.60,
        "project_contract_fit": 0.60,
        "performance_reuse_utilization": 0.40,
        "operational_fit": 0.60,
    },
}
INFRASWE_OVERALL_WEIGHTS = {"project_fit": 0.85, "benchmark_trust": 0.15}
PROJECT_SUBCOMPONENT_WEIGHTS = {
    "project-contract-fit-v0.5": {
        "integration": 0.30,
        "interface": 0.25,
        "lifecycle": 0.20,
        "buildtest": 0.15,
        "policy": 0.10,
    },
    "evolutionary-maintainability-v0.5": {
        "evolution": 0.35,
        "locality": 0.25,
        "tests": 0.20,
        "failure": 0.10,
        "contract": 0.10,
    },
    "performance-reuse-utilization-v0.5": {
        "attainment": 0.35,
        "coverage": 0.25,
        "retention": 0.15,
        "family": 0.15,
        "compile": 0.10,
    },
    "operational-fit-v0.5": {
        "replay": 0.30,
        "load": 0.30,
        "resource": 0.20,
        "coldsteady": 0.20,
    },
    "pure-triton-portability-v0.5": {
        "coverage": 0.35,
        "localretention": 0.25,
        "sharedcore": 0.25,
        "degradation": 0.15,
    },
}
EVIDENCE_GRADE_ORDER = {
    "E0-runtime": 0,
    "E1-framework": 1,
    "E2-system-trace": 2,
    "E3-kernel-counter": 3,
    "E4-sealed": 4,
}


@dataclass(frozen=True)
class ProjectDimensionResult:
    component: ProjectScoreComponent
    raw: Mapping[str, float] | None = None


def _score_dimension(
    values: Mapping[str, float | None],
    *,
    formula_version: str,
    evidence_digests: Sequence[str] = (),
    confidence: Literal["low", "medium", "high"] = "high",
) -> ProjectDimensionResult:
    weights = PROJECT_SUBCOMPONENT_WEIGHTS[formula_version]
    if set(values) != set(weights):
        raise ValueError(f"{formula_version} requires its exact frozen subcomponent set")
    missing = sorted(name for name, value in values.items() if value is None)
    if missing:
        return ProjectDimensionResult(
            ProjectScoreComponent(
                status="unresolved",
                value=None,
                formula_version=formula_version,
                input_evidence_digests=list(evidence_digests),
                confidence="low",
                failure_codes=["PROJECT_COMPONENT_EVIDENCE_MISSING"],
                reason="missing subcomponents: " + ", ".join(missing),
            )
        )
    numeric = {name: float(value) for name, value in values.items() if value is not None}
    score = weighted_geometric(numeric, weights)
    return ProjectDimensionResult(
        ProjectScoreComponent(
            status="scored",
            value=score,
            formula_version=formula_version,
            input_evidence_digests=list(evidence_digests),
            confidence=confidence,
        ),
        numeric,
    )


def score_project_contract_fit(
    values: Mapping[str, float | None], *, evidence_digests: Sequence[str] = ()
) -> ProjectDimensionResult:
    return _score_dimension(
        values,
        formula_version="project-contract-fit-v0.5",
        evidence_digests=evidence_digests,
    )


def score_evolutionary_maintainability(
    values: Mapping[str, float | None], *, evidence_digests: Sequence[str] = ()
) -> ProjectDimensionResult:
    return _score_dimension(
        values,
        formula_version="evolutionary-maintainability-v0.5",
        evidence_digests=evidence_digests,
    )


def score_performance_reuse_utilization(
    values: Mapping[str, float | None], *, evidence_digests: Sequence[str] = ()
) -> ProjectDimensionResult:
    return _score_dimension(
        values,
        formula_version="performance-reuse-utilization-v0.5",
        evidence_digests=evidence_digests,
    )


def score_operational_fit(
    values: Mapping[str, float | None], *, evidence_digests: Sequence[str] = ()
) -> ProjectDimensionResult:
    return _score_dimension(
        values,
        formula_version="operational-fit-v0.5",
        evidence_digests=evidence_digests,
    )


def score_pure_triton_portability(
    values: Mapping[str, float | None], *, evidence_digests: Sequence[str] = ()
) -> ProjectDimensionResult:
    return _score_dimension(
        values,
        formula_version="pure-triton-portability-v0.5",
        evidence_digests=evidence_digests,
    )


def audit_pure_triton(evidence: PureTritonEligibilityEvidence) -> TritonPurityAudit:
    if evidence.implementation_kind != "triton-pure":
        return TritonPurityAudit(
            status="not_applicable",
            evidence_digests=evidence.evidence_digests,
            reason="candidate does not claim the pure Triton implementation kind",
        )
    failures: list[str] = []
    unresolved: list[str] = []
    if evidence.core_kernel_language.lower() != "triton":
        failures.append("TRITON_PURE_CORE_NOT_TRITON")
    if evidence.backend_native_kernel_calls:
        failures.append("TRITON_PURE_HIDDEN_NATIVE_PATH")
    if not evidence.backend_specific_code_is_capability_or_launch_only:
        failures.append("TRITON_PURE_BACKEND_LOGIC_LEAK")
    if not evidence.shared_semantic_implementation_family:
        failures.append("TRITON_PURE_SHARED_CORE_MISSING")
    for profile, status in evidence.required_profile_evidence.items():
        if status == "missing":
            unresolved.append(f"TRITON_PROFILE_EVIDENCE_MISSING:{profile}")
        elif status == "unsupported":
            failures.append(f"TRITON_REQUIRED_PROFILE_UNSUPPORTED:{profile}")
    if not evidence.local_baseline_normalization:
        failures.append("TRITON_BACKEND_LOCAL_BASELINE_MISSING")
    if evidence.absolute_cross_hardware_latency_ranking:
        failures.append("TRITON_CROSS_HARDWARE_ABSOLUTE_RANKING")
    if not evidence.explicit_unsupported_and_fallback:
        failures.append("TRITON_DEGRADATION_NOT_EXPLICIT")
    if failures:
        return TritonPurityAudit(
            status="fail",
            failure_codes=sorted(failures),
            evidence_digests=evidence.evidence_digests,
        )
    if unresolved:
        return TritonPurityAudit(
            status="unresolved",
            failure_codes=sorted(unresolved),
            evidence_digests=evidence.evidence_digests,
            reason="required runner evidence is unavailable",
        )
    return TritonPurityAudit(
        status="pass",
        evidence_digests=evidence.evidence_digests,
    )


def _not_applicable_x() -> ProjectScoreComponent:
    return ProjectScoreComponent(
        status="not_applicable",
        value=None,
        formula_version="pure-triton-portability-v0.5",
        confidence="not_applicable",
        reason="ordinary candidates do not receive an edge portability score",
    )


def _not_run_component(name: str, reason: str) -> ProjectScoreComponent:
    return ProjectScoreComponent(
        status="not_run_due_to_gate",
        value=None,
        formula_version=name + "-v0.5",
        confidence="low",
        failure_codes=["PROJECT_FIT_NOT_RUN_DUE_TO_GATE"],
        reason=reason,
    )


def build_project_fit(
    *,
    mode: Literal["provisional", "official"],
    infra_cert: Literal["pass", "fail", "unresolved"],
    formula_template_id: Literal[
        "project-fit-kernel-v0.5",
        "project-fit-triton-pure-v0.5",
        "project-fit-system-path-v0.5.1",
    ],
    comparison_cell: ProjectComparisonCell,
    evolutionary_maintainability: ProjectDimensionResult,
    project_contract_fit: ProjectDimensionResult,
    performance_reuse_utilization: ProjectDimensionResult,
    operational_fit: ProjectDimensionResult,
    pure_triton_portability: ProjectDimensionResult | None = None,
    triton_purity_audit: TritonPurityAudit | None = None,
    fresh_process_replays: int = 1,
    evidence_grade: str = "E0-runtime",
    hidden_probes_complete: bool = False,
    manifest_verified: bool = False,
    sealed_draft_sha256: str | None = None,
) -> ProjectFitScore:
    if comparison_cell.formula_template_id != formula_template_id:
        raise ValueError("Project comparison cell and score use different formula templates")
    dimensions = {
        "evolutionary_maintainability": evolutionary_maintainability.component,
        "project_contract_fit": project_contract_fit.component,
        "performance_reuse_utilization": performance_reuse_utilization.component,
        "operational_fit": operational_fit.component,
        "pure_triton_portability": _not_applicable_x(),
    }
    if formula_template_id == "project-fit-triton-pure-v0.5":
        if pure_triton_portability is None or triton_purity_audit is None:
            raise ValueError("pure Triton scoring requires X evidence and a purity audit")
        dimensions["pure_triton_portability"] = pure_triton_portability.component
        if triton_purity_audit.status != "pass":
            dimensions["pure_triton_portability"] = ProjectScoreComponent(
                status=(
                    "not_run_due_to_gate" if triton_purity_audit.status == "fail" else "unresolved"
                ),
                value=None,
                formula_version="pure-triton-portability-v0.5",
                input_evidence_digests=triton_purity_audit.evidence_digests,
                confidence="low",
                failure_codes=triton_purity_audit.failure_codes,
                reason="pure Triton eligibility did not pass",
            )

    floors = PROJECT_FIT_FLOORS[formula_template_id]
    if infra_cert != "pass":
        reason = "InfraCert failed" if infra_cert == "fail" else "InfraCert is unresolved"
        gated = {}
        for name in dimensions:
            if name == "pure_triton_portability" and formula_template_id in {
                "project-fit-kernel-v0.5",
                "project-fit-system-path-v0.5.1",
            }:
                gated[name] = _not_applicable_x()
            else:
                gated[name] = _not_run_component(name, reason)
        return ProjectFitScore(
            status="not_issued" if infra_cert == "fail" else "unresolved",
            formula_template_id=formula_template_id,
            score_100=None,
            components=gated,
            component_floors=floors,
            confidence="not_applicable" if infra_cert == "fail" else "low",
            comparison_cell=comparison_cell,
            failure_codes=["INFRACERT_FAILED" if infra_cert == "fail" else "INFRACERT_UNRESOLVED"],
        )

    required_names = set(PROJECT_FIT_WEIGHTS[formula_template_id])
    scored = all(dimensions[name].status == "scored" for name in required_names)
    if mode == "official":
        readiness_failures = []
        if sealed_draft_sha256 is None:
            readiness_failures.append("DRAFT_SEAL_MISSING")
        if fresh_process_replays < 5:
            readiness_failures.append("FRESH_PROCESS_REPLAYS_BELOW_MINIMUM")
        if EVIDENCE_GRADE_ORDER.get(evidence_grade, -1) < EVIDENCE_GRADE_ORDER["E2-system-trace"]:
            readiness_failures.append("SYSTEM_TRACE_EVIDENCE_MISSING")
        if not hidden_probes_complete:
            readiness_failures.append("HIDDEN_PROBES_INCOMPLETE")
        if not manifest_verified:
            readiness_failures.append("EVIDENCE_MANIFEST_UNVERIFIED")
        if not scored:
            readiness_failures.append("PROJECT_COMPONENT_UNRESOLVED")
        official_ready = not readiness_failures
        if not official_ready or not scored:
            return ProjectFitScore(
                status="unresolved",
                formula_template_id=formula_template_id,
                score_100=None,
                components=dimensions,
                component_floors=floors,
                confidence="low",
                comparison_cell=comparison_cell,
                failure_codes=readiness_failures,
            )
    elif not scored:
        return ProjectFitScore(
            status="provisional",
            formula_template_id=formula_template_id,
            score_100=None,
            components=dimensions,
            component_floors=floors,
            confidence="low",
            comparison_cell=comparison_cell,
            failure_codes=["PROJECT_COMPONENT_UNRESOLVED"],
        )

    numeric = {name: float(dimensions[name].value) for name in required_names}
    score_100 = 100 * weighted_geometric(numeric, PROJECT_FIT_WEIGHTS[formula_template_id])
    below_floor = [name for name, floor in floors.items() if numeric[name] < floor]
    status = (
        "provisional"
        if mode == "provisional"
        else ("not_acceptable" if below_floor else "official")
    )
    return ProjectFitScore(
        status=status,
        formula_template_id=formula_template_id,
        score_100=score_100,
        components=dimensions,
        component_floors=floors,
        confidence="low" if mode == "provisional" else "high",
        comparison_cell=comparison_cell,
        failure_codes=["PROJECT_COMPONENT_FLOOR_FAILED:" + name for name in sorted(below_floor)],
    )


def score_benchmark_trust(
    *,
    reproducibility: float | None,
    evidence: float | None,
    statistics: float | None,
    environment: float | None,
    evidence_digests: Sequence[str] = (),
    failure_codes: Sequence[str] = (),
) -> BenchmarkTrustCard:
    components = {
        "reproducibility": reproducibility,
        "evidence": evidence,
        "statistics": statistics,
        "environment": environment,
    }
    if any(value is None for value in components.values()):
        return BenchmarkTrustCard(
            status="unresolved",
            score_100=None,
            components=components,
            evidence_digests=list(evidence_digests),
            failure_codes=sorted(set(failure_codes) | {"BENCHMARK_TRUST_EVIDENCE_MISSING"}),
        )
    numeric = {name: float(value) for name, value in components.items() if value is not None}
    score = 100 * weighted_geometric(
        numeric,
        {
            "reproducibility": 0.35,
            "evidence": 0.25,
            "statistics": 0.25,
            "environment": 0.15,
        },
    )
    return BenchmarkTrustCard(
        status="scored",
        score_100=score,
        components=components,
        evidence_digests=list(evidence_digests),
        failure_codes=list(failure_codes),
    )


def _review_policy_state(
    context: PullRequestReviewContext | None,
) -> Literal[
    "active-new-review",
    "stale-reviewed-open",
    "inactive-or-not-new",
    "missing",
]:
    if context is None:
        return "missing"
    age_days = (context.observed_at - context.created_at).total_seconds() / 86_400
    activity_idle_days = (context.observed_at - context.last_activity_at).total_seconds() / 86_400
    review_idle_days = (
        (context.observed_at - context.last_human_review_at).total_seconds() / 86_400
        if context.last_human_review_at is not None
        else None
    )
    current_head_review_is_recent = (
        context.current_head_human_non_author_review_count > 0
        and review_idle_days is not None
        and review_idle_days <= CHECK_ACTIVITY_MAX_IDLE_DAYS
    )
    pending_review_is_recent = (
        context.pending_human_review_request and activity_idle_days <= CHECK_ACTIVITY_MAX_IDLE_DAYS
    )
    if age_days <= CHECK_NEW_PR_MAX_AGE_DAYS and (
        current_head_review_is_recent or pending_review_is_recent
    ):
        return "active-new-review"
    if (
        age_days >= STALE_REVIEWED_OPEN_MIN_AGE_DAYS
        and context.total_human_non_author_review_count > 0
    ):
        return "stale-reviewed-open"
    return "inactive-or-not-new"


def _review_limited_failure_decision(
    *,
    context: PullRequestReviewContext | None,
    base_codes: Sequence[str],
    supported_scope: Sequence[str],
    excluded_scope: Sequence[str],
    required_actions: Sequence[str],
    unresolved_without_context: bool = False,
) -> MergeabilityDecision:
    review_state = _review_policy_state(context)
    codes = [*base_codes, "POLARIZED_DECISION_POLICY_V0_5_1"]
    if review_state == "active-new-review":
        return MergeabilityDecision(
            verdict="check",
            supported_scope=list(supported_scope),
            excluded_scope=list(excluded_scope),
            required_actions=list(required_actions),
            rationale_codes=[*codes, "ACTIVE_NEW_PR_REVIEW_CHECK_ELIGIBLE"],
        )
    if review_state == "missing" and unresolved_without_context:
        return MergeabilityDecision(
            verdict="unresolved",
            supported_scope=list(supported_scope),
            excluded_scope=list(excluded_scope),
            required_actions=list(required_actions),
            rationale_codes=[*codes, "PR_REVIEW_CONTEXT_MISSING"],
        )
    reason = (
        "STALE_REVIEWED_OPEN_REJECT"
        if review_state == "stale-reviewed-open"
        else "CHECK_REQUIRES_ACTIVE_NEW_PR_REVIEW"
    )
    return MergeabilityDecision(
        verdict="reject",
        supported_scope=list(supported_scope),
        excluded_scope=list(excluded_scope),
        required_actions=list(required_actions),
        rationale_codes=[*codes, reason],
    )


def compile_legacy_mergeability_decision(
    *,
    infra_cert: Literal["pass", "fail", "unresolved"],
    project_fit: ProjectFitScore,
    project_objectives: Mapping[str, ProjectObjectiveResult],
    supported_scope: Sequence[str] = (),
    excluded_scope: Sequence[str] = (),
    required_actions: Sequence[str] = (),
    fundamental_hard_failure: bool = True,
    review_context: PullRequestReviewContext | None = None,
) -> MergeabilityDecision:
    release_gate_failures = [
        name
        for name, objective in project_objectives.items()
        if objective.policy == "release-gate" and objective.release_gate_passed is False
    ]
    if infra_cert == "unresolved" or project_fit.status == "unresolved":
        return MergeabilityDecision(
            verdict="unresolved",
            supported_scope=list(supported_scope),
            excluded_scope=list(excluded_scope),
            required_actions=list(required_actions),
            rationale_codes=[
                "EVALUATION_EVIDENCE_UNRESOLVED",
                "POLARIZED_DECISION_POLICY_V0_5_1",
            ],
        )
    if infra_cert == "fail":
        if not fundamental_hard_failure:
            return _review_limited_failure_decision(
                context=review_context,
                base_codes=["INFRACERT_FAILED"],
                supported_scope=supported_scope,
                excluded_scope=excluded_scope,
                required_actions=required_actions,
            )
        return MergeabilityDecision(
            verdict="reject",
            required_actions=list(required_actions),
            rationale_codes=[
                "INFRACERT_FAILED",
                "POLARIZED_DECISION_POLICY_V0_5_1",
            ],
        )
    if project_fit.status in {"provisional", "not_issued"}:
        return _review_limited_failure_decision(
            context=review_context,
            base_codes=["OFFICIAL_PROJECT_FIT_NOT_AVAILABLE"],
            supported_scope=list(supported_scope),
            excluded_scope=list(excluded_scope),
            required_actions=list(required_actions),
            unresolved_without_context=True,
        )
    score = float(project_fit.score_100 or 0)
    if project_fit.status == "not_acceptable":
        if score < 60:
            return MergeabilityDecision(
                verdict="reject",
                supported_scope=list(supported_scope),
                excluded_scope=list(excluded_scope),
                required_actions=list(required_actions),
                rationale_codes=[
                    "PROJECT_COMPONENT_FLOOR_FAILED",
                    "PROJECT_FIT_REJECT_BAND",
                    "POLARIZED_DECISION_POLICY_V0_5_1",
                ],
            )
        return _review_limited_failure_decision(
            context=review_context,
            base_codes=["PROJECT_COMPONENT_FLOOR_FAILED"],
            supported_scope=supported_scope,
            excluded_scope=excluded_scope,
            required_actions=required_actions,
        )
    if score < MERGE_ACCEPT_SCORE_FLOOR_100:
        return _review_limited_failure_decision(
            context=review_context,
            base_codes=["PROJECT_FIT_BELOW_MERGE_FLOOR_85"],
            supported_scope=supported_scope,
            excluded_scope=excluded_scope,
            required_actions=required_actions,
        )
    if release_gate_failures and not excluded_scope:
        return MergeabilityDecision(
            verdict="reject",
            supported_scope=list(supported_scope),
            excluded_scope=list(excluded_scope),
            required_actions=list(required_actions),
            rationale_codes=[
                *["PROJECT_RELEASE_GATE_FAILED:" + name for name in release_gate_failures],
                "POLARIZED_DECISION_POLICY_V0_5_1",
            ],
        )
    verdict = "accept_with_scope" if excluded_scope else "accept"
    codes = [
        "PROJECT_FIT_ACCEPT_BAND_85_PLUS",
        "POLARIZED_DECISION_POLICY_V0_5_1",
    ]
    if release_gate_failures:
        codes.extend("PROJECT_RELEASE_GATE_FAILED:" + name for name in release_gate_failures)
    return MergeabilityDecision(
        verdict=verdict,
        supported_scope=list(supported_scope),
        excluded_scope=list(excluded_scope),
        required_actions=list(required_actions),
        rationale_codes=codes,
    )


def _dimension_gate(
    project_fit: ProjectFitScore,
    *component_names: str,
) -> tuple[Literal["pass", "fail", "unresolved"], list[str]]:
    failures: list[str] = []
    unresolved: list[str] = []
    for name in component_names:
        component = project_fit.components[name]
        if component.status != "scored" or component.value is None:
            unresolved.append(f"MICROSCORE_COMPONENT_UNRESOLVED:{name}")
        elif component.value < project_fit.component_floors[name]:
            failures.append(f"MICROSCORE_COMPONENT_FLOOR_FAILED:{name}")
    if failures:
        return "fail", failures
    if unresolved:
        return "unresolved", unresolved
    return "pass", ["ORDERED_GATE_PASSED"]


def _ordered_gates(
    *,
    infra_cert: Literal["pass", "fail", "unresolved"],
    project_fit: ProjectFitScore,
    benchmark_trust: BenchmarkTrustCard,
    project_objectives: Mapping[str, ProjectObjectiveResult],
    excluded_scope: Sequence[str],
) -> tuple[list[OrderedEvaluationGate], float | None]:
    maintainability_status, maintainability_codes = _dimension_gate(
        project_fit, "evolutionary_maintainability"
    )
    gates = [
        OrderedEvaluationGate(
            name="maintainability",
            status=maintainability_status,
            rationale_codes=maintainability_codes,
        )
    ]
    if maintainability_status != "pass":
        gates.extend(
            [
                OrderedEvaluationGate(
                    name="deployability",
                    status="not-run",
                    rationale_codes=["BLOCKED_BY_MAINTAINABILITY_GATE"],
                ),
                OrderedEvaluationGate(
                    name="performance",
                    status="not-run",
                    rationale_codes=["BLOCKED_BY_MAINTAINABILITY_GATE"],
                ),
                OrderedEvaluationGate(
                    name="overall-score",
                    status="not-run",
                    rationale_codes=["BLOCKED_BY_MAINTAINABILITY_GATE"],
                ),
            ]
        )
        return gates, None

    if infra_cert == "fail":
        deployability_status: Literal["pass", "fail", "unresolved"] = "fail"
        deployability_codes = ["INFRACERT_FAILED"]
    elif infra_cert == "unresolved" or project_fit.status in {
        "provisional",
        "unresolved",
        "not_issued",
    }:
        deployability_status = "unresolved"
        deployability_codes = ["DEPLOYABILITY_EVIDENCE_UNRESOLVED"]
    else:
        deployability_status, deployability_codes = _dimension_gate(
            project_fit, "project_contract_fit", "operational_fit"
        )
    gates.append(
        OrderedEvaluationGate(
            name="deployability",
            status=deployability_status,
            rationale_codes=deployability_codes,
        )
    )
    if deployability_status != "pass":
        gates.extend(
            [
                OrderedEvaluationGate(
                    name="performance",
                    status="not-run",
                    rationale_codes=["BLOCKED_BY_DEPLOYABILITY_GATE"],
                ),
                OrderedEvaluationGate(
                    name="overall-score",
                    status="not-run",
                    rationale_codes=["BLOCKED_BY_DEPLOYABILITY_GATE"],
                ),
            ]
        )
        return gates, None

    performance_names = ["performance_reuse_utilization"]
    if project_fit.formula_template_id == "project-fit-triton-pure-v0.5":
        performance_names.append("pure_triton_portability")
    performance_status, performance_codes = _dimension_gate(project_fit, *performance_names)
    release_gate_failures = [
        name
        for name, objective in project_objectives.items()
        if objective.policy == "release-gate" and objective.release_gate_passed is False
    ]
    if release_gate_failures and not excluded_scope:
        performance_status = "fail"
        performance_codes = [
            *performance_codes,
            *["PROJECT_RELEASE_GATE_FAILED:" + name for name in release_gate_failures],
        ]
    elif release_gate_failures:
        performance_codes = [
            *performance_codes,
            *[
                "PROJECT_RELEASE_GATE_ISOLATED_TO_EXCLUDED_SCOPE:" + name
                for name in release_gate_failures
            ],
        ]
    gates.append(
        OrderedEvaluationGate(
            name="performance",
            status=performance_status,
            rationale_codes=performance_codes,
        )
    )
    if performance_status != "pass":
        gates.append(
            OrderedEvaluationGate(
                name="overall-score",
                status="not-run",
                rationale_codes=["BLOCKED_BY_PERFORMANCE_GATE"],
            )
        )
        return gates, None

    if project_fit.score_100 is None or benchmark_trust.score_100 is None:
        gates.append(
            OrderedEvaluationGate(
                name="overall-score",
                status="unresolved",
                rationale_codes=["OVERALL_MICROSCORE_UNRESOLVED"],
            )
        )
        return gates, None
    overall_score_100 = 100 * weighted_geometric(
        {
            "project_fit": project_fit.score_100 / 100,
            "benchmark_trust": benchmark_trust.score_100 / 100,
        },
        INFRASWE_OVERALL_WEIGHTS,
    )
    band = overall_score_decision_band(overall_score_100)
    gates.append(
        OrderedEvaluationGate(
            name="overall-score",
            status="pass",
            rationale_codes=[f"OVERALL_SCORE_BAND:{band}"],
        )
    )
    return gates, overall_score_100


def compile_infraswe_assessment(
    *,
    infra_cert: Literal["pass", "fail", "unresolved"],
    project_fit: ProjectFitScore,
    benchmark_trust: BenchmarkTrustCard,
    project_objectives: Mapping[str, ProjectObjectiveResult],
    supported_scope: Sequence[str] = (),
    excluded_scope: Sequence[str] = (),
    required_actions: Sequence[str] = (),
) -> tuple[float | None, list[OrderedEvaluationGate], InfraSWEDecision]:
    gates, overall_score_100 = _ordered_gates(
        infra_cert=infra_cert,
        project_fit=project_fit,
        benchmark_trust=benchmark_trust,
        project_objectives=project_objectives,
        excluded_scope=excluded_scope,
    )
    first_blocking = next(
        (gate for gate in gates[:3] if gate.status in {"fail", "unresolved"}),
        None,
    )
    if first_blocking is not None:
        classification = "reject" if first_blocking.status == "fail" else "check"
        rationale = [
            f"ORDERED_HARD_GATE:{first_blocking.name}:{first_blocking.status}",
            *first_blocking.rationale_codes,
        ]
    elif overall_score_100 is None:
        classification = "check"
        rationale = ["OVERALL_SCORE_UNRESOLVED_AFTER_HARD_GATES"]
    else:
        classification = overall_score_decision_band(overall_score_100)
        rationale = [
            f"OVERALL_SCORE_CLASSIFICATION:{classification}",
            "ABOVE_65_SCORE_IS_EVALUATION_ONLY"
            if classification == "accept"
            else "FIXED_OVERALL_SCORE_BANDS",
        ]
    return (
        overall_score_100,
        gates,
        InfraSWEDecision(
            classification=classification,
            acceptance_scope=(
                "limited"
                if classification == "accept" and excluded_scope
                else "full"
                if classification == "accept"
                else "not-applicable"
            ),
            supported_scope=list(supported_scope),
            excluded_scope=list(excluded_scope),
            required_actions=list(required_actions),
            rationale_codes=rationale,
        ),
    )


def compile_mergeability_decision(
    *,
    infra_cert: Literal["pass", "fail", "unresolved"],
    project_fit: ProjectFitScore,
    benchmark_trust: BenchmarkTrustCard,
    project_objectives: Mapping[str, ProjectObjectiveResult],
    supported_scope: Sequence[str] = (),
    excluded_scope: Sequence[str] = (),
    required_actions: Sequence[str] = (),
) -> InfraSWEDecision:
    """Return the three-class result after ordered gates and the sole overall score."""

    return compile_infraswe_assessment(
        infra_cert=infra_cert,
        project_fit=project_fit,
        benchmark_trust=benchmark_trust,
        project_objectives=project_objectives,
        supported_scope=supported_scope,
        excluded_scope=excluded_scope,
        required_actions=required_actions,
    )[2]


def build_v05_result(
    *,
    draft_id: str,
    draft_revision: int,
    draft_state: DraftState,
    sealed_draft_sha256: str | None,
    target_project_profile_sha256: str,
    target_repository_sha256: str,
    candidate_sha256: str,
    acceptance_contract_sha256: str,
    infra_cert: Literal["pass", "fail", "unresolved"],
    project_fit: ProjectFitScore,
    benchmark_trust: BenchmarkTrustCard,
    benchmark_cost: BenchmarkCostCard,
    evidence_grade: Literal[
        "E0-runtime", "E1-framework", "E2-system-trace", "E3-kernel-counter", "E4-sealed"
    ],
    project_objectives: Mapping[str, ProjectObjectiveResult],
    cell_efficiency: CellEfficiencyReference,
    decision: MergeabilityDecision,
    raw_metrics: Mapping[str, object] | None = None,
    failure_codes: Sequence[str] = (),
    audit_flags: Sequence[str] = (),
) -> V05ScoreResult:
    if infra_cert == "fail" or project_fit.status == "not_acceptable":
        effective = 0.0
    elif infra_cert == "unresolved" or project_fit.status != "official":
        effective = None
    else:
        effective = project_fit.score_100
    return V05ScoreResult(
        draft_id=draft_id,
        draft_revision=draft_revision,
        draft_state=draft_state,
        sealed_draft_sha256=sealed_draft_sha256,
        target_project_profile_sha256=target_project_profile_sha256,
        target_repository_sha256=target_repository_sha256,
        candidate_sha256=candidate_sha256,
        acceptance_contract_sha256=acceptance_contract_sha256,
        infra_cert=infra_cert,
        project_fit=project_fit,
        leaderboard_effective_project_fit_100=effective,
        benchmark_trust=benchmark_trust,
        benchmark_cost=benchmark_cost,
        evidence_grade=evidence_grade,
        project_objectives=dict(project_objectives),
        cell_efficiency=cell_efficiency,
        decision=decision,
        raw_metrics=dict(raw_metrics or {}),
        failure_codes=list(failure_codes),
        audit_flags=list(audit_flags),
    )


def build_infraswe_result(
    *,
    draft_id: str,
    draft_revision: int,
    draft_state: DraftState,
    sealed_draft_sha256: str | None,
    target_project_profile_sha256: str,
    target_repository_sha256: str,
    candidate_sha256: str,
    acceptance_contract_sha256: str,
    infra_cert: Literal["pass", "fail", "unresolved"],
    project_fit: ProjectFitScore,
    benchmark_trust: BenchmarkTrustCard,
    benchmark_cost: BenchmarkCostCard,
    evidence_grade: Literal[
        "E0-runtime", "E1-framework", "E2-system-trace", "E3-kernel-counter", "E4-sealed"
    ],
    project_objectives: Mapping[str, ProjectObjectiveResult],
    cell_efficiency: CellEfficiencyReference,
    supported_scope: Sequence[str] = (),
    excluded_scope: Sequence[str] = (),
    required_actions: Sequence[str] = (),
    evaluation_engine: Literal["infraswe", "external"] = "infraswe",
    evaluation_scope: Literal["full", "staged"] = "full",
    seal_enabled: bool = True,
    raw_metrics: Mapping[str, object] | None = None,
    failure_codes: Sequence[str] = (),
    audit_flags: Sequence[str] = (),
) -> InfraSWEOverallResult:
    """Build the default single-composite result with nested microscores."""

    overall_score_100, ordered_gates, decision = compile_infraswe_assessment(
        infra_cert=infra_cert,
        project_fit=project_fit,
        benchmark_trust=benchmark_trust,
        project_objectives=project_objectives,
        supported_scope=supported_scope,
        excluded_scope=excluded_scope,
        required_actions=required_actions,
    )
    return InfraSWEOverallResult(
        draft_id=draft_id,
        draft_revision=draft_revision,
        draft_state=draft_state,
        sealed_draft_sha256=sealed_draft_sha256,
        target_project_profile_sha256=target_project_profile_sha256,
        target_repository_sha256=target_repository_sha256,
        candidate_sha256=candidate_sha256,
        acceptance_contract_sha256=acceptance_contract_sha256,
        infra_cert=infra_cert,
        overall_score_100=overall_score_100,
        microscores=InfraSWEMicroscores(
            project_fit=project_fit,
            benchmark_trust=benchmark_trust,
        ),
        ordered_gates=ordered_gates,
        benchmark_cost=benchmark_cost,
        evidence_grade=evidence_grade,
        project_objectives=dict(project_objectives),
        cell_efficiency=cell_efficiency,
        decision=decision,
        evaluation_engine=evaluation_engine,
        evaluation_scope=evaluation_scope,
        seal_enabled=seal_enabled,
        raw_metrics=dict(raw_metrics or {}),
        failure_codes=list(failure_codes),
        audit_flags=list(audit_flags),
    )
