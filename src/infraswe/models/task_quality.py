from __future__ import annotations

import math
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

ObligationBucket = Literal[
    "capability-obligation",
    "regression-invariant",
    "negative-boundary",
    "mechanism-proof",
    "safety-liveness",
    "environment-sentinel",
    "maintenance-probe",
]
ObligationStatus = Literal[
    "PASS",
    "FAIL",
    "INFRA_INVALID",
    "BENCHMARK_DEFECT",
    "UNRESOLVED",
    "NOT_APPLICABLE",
]
CandidateResultStatus = Literal[
    "VALID_PASS",
    "VALID_FAIL",
    "INFRA_INVALID",
    "BENCHMARK_DEFECT",
    "NOT_APPLICABLE",
    "UNRESOLVED",
]

INFRA_CERT_BUCKETS = frozenset(
    {
        "capability-obligation",
        "regression-invariant",
        "negative-boundary",
        "mechanism-proof",
        "safety-liveness",
    }
)


class TaskQualityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TaskTarget(TaskQualityModel):
    repository: str
    revision: Digest


class TaskRequirement(TaskQualityModel):
    requirement_id: str = Field(pattern=r"^REQ-[A-Z0-9][A-Z0-9_-]*$")
    statement: str = Field(min_length=12)
    visibility: Literal[
        "public",
        "public-contract-hidden-instance",
        "private-qualification",
    ]
    disposition: Literal["scoring", "non-scoring-informational"] = "scoring"


class TaskSpecification(TaskQualityModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    task_revision: int = Field(ge=1)
    change_kind: Literal["repair", "feature", "optimization", "conformance"]
    target: TaskTarget
    change_intent: str = Field(min_length=12)
    requirements: list[TaskRequirement] = Field(min_length=1)
    allowed_changes: list[str] = Field(min_length=1)
    forbidden_changes: list[str] = Field(min_length=1)
    supported_behavior: list[str] = Field(min_length=1)
    unsupported_behavior: list[str] = Field(default_factory=list)
    correctness_tolerance_policy_id: str
    resource_expectation_id: str
    public_validation_interface: list[str] = Field(min_length=1)
    artifact_policy_sha256: Digest
    capability_contract_sha256: Digest

    @model_validator(mode="after")
    def ids_and_paths_are_unambiguous(self) -> TaskSpecification:
        identifiers = [item.requirement_id for item in self.requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Task requirement ids must be unique")
        for pattern in [*self.allowed_changes, *self.forbidden_changes]:
            path = PurePosixPath(pattern)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Task modification scopes must stay repository-relative")
        if set(self.allowed_changes) & set(self.forbidden_changes):
            raise ValueError("Task allowed and forbidden change scopes cannot overlap exactly")
        return self


class ObligationOracle(TaskQualityModel):
    kind: Literal[
        "semantic",
        "differential",
        "property",
        "mechanism",
        "liveness",
        "resource-meter",
        "environment-sentinel",
        "maintenance",
    ]
    reference_id: str
    tolerance_profile_id: str | None = None


class ObligationRepeatPolicy(TaskQualityModel):
    seeds: list[int] = Field(default_factory=list)
    fresh_processes: int = Field(default=1, ge=1)


class AcceptanceObligation(TaskQualityModel):
    obligation_id: str = Field(pattern=r"^(CO|RI|NB|MP|SL|ES|MT)-[A-Z0-9][A-Z0-9_-]*$")
    source_requirements: list[str] = Field(default_factory=list)
    profile_provenance: list[str] = Field(default_factory=list)
    bucket: ObligationBucket
    severity: Literal["hard", "quality", "sentinel", "advisory"]
    release_gate: bool = False
    scope: dict[str, Any] = Field(default_factory=dict)
    oracle: ObligationOracle
    repeat: ObligationRepeatPolicy = Field(default_factory=ObligationRepeatPolicy)
    failure_owner: Literal["candidate", "benchmark", "infrastructure", "task-author"]
    evidence_owner: Literal[
        "pristine-verifier",
        "trusted-meter",
        "environment-sentinel",
        "deterministic-maintenance-probe",
    ]
    visibility: Literal[
        "public",
        "public-contract-hidden-cases",
        "private-qualification",
    ]
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def bucket_prefix_and_authority_are_coherent(self) -> AcceptanceObligation:
        expected_prefix = {
            "capability-obligation": "CO-",
            "regression-invariant": "RI-",
            "negative-boundary": "NB-",
            "mechanism-proof": "MP-",
            "safety-liveness": "SL-",
            "environment-sentinel": "ES-",
            "maintenance-probe": "MT-",
        }[self.bucket]
        if not self.obligation_id.startswith(expected_prefix):
            raise ValueError("obligation id prefix must match its bucket")
        if self.severity == "hard" and not (self.source_requirements or self.profile_provenance):
            raise ValueError("hard obligation requires requirement or profile provenance")
        if self.bucket == "environment-sentinel":
            if self.failure_owner == "candidate" or self.evidence_owner != "environment-sentinel":
                raise ValueError("environment sentinel cannot be Candidate-owned")
            if self.severity not in {"sentinel", "hard"}:
                raise ValueError("environment sentinel requires sentinel severity")
        if self.bucket == "maintenance-probe" and self.severity == "hard" and not self.release_gate:
            raise ValueError("hard maintenance probe must be an explicit release gate")
        if self.release_gate and self.bucket != "maintenance-probe":
            raise ValueError("release_gate is reserved for maintenance probes")
        return self


class TaskAcceptanceContract(TaskQualityModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    task_revision: int = Field(ge=1)
    obligations: list[AcceptanceObligation] = Field(min_length=1)
    contract_sha256: Digest

    @model_validator(mode="after")
    def obligations_are_unique(self) -> TaskAcceptanceContract:
        identifiers = [item.obligation_id for item in self.obligations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Acceptance obligation ids must be unique")
        if not any(item.bucket == "capability-obligation" for item in self.obligations):
            raise ValueError("Acceptance contract requires a capability obligation")
        if not any(item.bucket == "environment-sentinel" for item in self.obligations):
            raise ValueError("Acceptance contract requires an environment sentinel")
        return self


class FeasibilityWitness(TaskQualityModel):
    witness_id: str
    kind: Literal[
        "reference-patch",
        "reference-implementation",
        "oracle-adapter",
        "constructive-trace",
        "approved-upstream",
        "minimal-executable",
    ]
    source_locator: str
    sha256: Digest
    target_revision: Digest
    build_recipe_sha256: Digest
    covers: list[str] = Field(min_length=1)
    known_limitations: list[str] = Field(default_factory=list)
    license_status: Literal["redistributable", "private-qualification-only", "reviewed-exception"]
    reviewer: str
    grading_usage: Literal["forbidden"] = "forbidden"


class WitnessSet(TaskQualityModel):
    schema_version: Literal["0.1"] = "0.1"
    witnesses: list[FeasibilityWitness] = Field(min_length=1)
    witness_set_sha256: Digest

    @model_validator(mode="after")
    def witnesses_are_unique(self) -> WitnessSet:
        identifiers = [item.witness_id for item in self.witnesses]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Witness ids must be unique")
        return self


class ObligationObservation(TaskQualityModel):
    obligation_id: str
    bucket: ObligationBucket
    severity: Literal["hard", "quality", "sentinel", "advisory"]
    status: ObligationStatus
    evidence_refs: list[str] = Field(default_factory=list)
    failure_code: str | None = None

    @model_validator(mode="after")
    def decided_status_has_evidence(self) -> ObligationObservation:
        if self.status in {"PASS", "FAIL"} and not self.evidence_refs:
            raise ValueError("decided obligation status requires evidence refs")
        if self.status == "FAIL" and not self.failure_code:
            raise ValueError("failed obligation requires a failure code")
        return self


class BaselineDifferential(TaskQualityModel):
    task_id: str
    task_revision: int = Field(ge=1)
    baseline_sha256: Digest
    fresh_replays: int = Field(ge=3)
    stable: bool
    observations: list[ObligationObservation] = Field(min_length=1)


class WitnessReplayResult(TaskQualityModel):
    witness_id: str
    witness_sha256: Digest
    fresh_replays: int = Field(ge=1)
    stable: bool
    observations: list[ObligationObservation] = Field(min_length=1)


class MutationOutcome(TaskQualityModel):
    mutation_id: str
    patch_sha256: Digest
    target_obligations: list[str] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    critical: bool = False
    expected: Literal["rejected"] = "rejected"
    observed: Literal["rejected", "survived", "infra-invalid"]
    evidence_refs: list[str] = Field(min_length=1)


class NegativeControlOutcome(TaskQualityModel):
    control_id: str
    kind: Literal[
        "do-nothing",
        "output-only-fake",
        "benchmark-modification",
        "silent-fallback",
        "hard-coded-public-cases",
        "unsupported-claim",
        "invalid-resource-claim",
        "stale-cache",
        "wrong-architecture-dispatch",
    ]
    observed: Literal["rejected", "accepted", "infra-invalid"]
    evidence_refs: list[str] = Field(min_length=1)


class AlternativeValidSolutionOutcome(TaskQualityModel):
    solution_id: str
    implementation_sha256: Digest
    structure_fingerprint_sha256: Digest
    source: Literal["maintainer", "independent-author", "agent-human-validated"]
    expected: Literal["accepted"] = "accepted"
    observed: Literal["accepted", "rejected", "infra-invalid"]
    observations: list[ObligationObservation] = Field(min_length=1)


class VerifierFlakinessAudit(TaskQualityModel):
    fresh_replays: int = Field(ge=5)
    status: Literal[
        "STABLE",
        "STOCHASTIC_WITH_BOUNDED_ORACLE",
        "FLAKY_VERIFIER",
        "INFRA_UNSTABLE",
        "UNKNOWN",
    ]
    pass_fail_flips: int = Field(ge=0)
    explanation: str | None = None

    @model_validator(mode="after")
    def flips_are_explained(self) -> VerifierFlakinessAudit:
        if self.pass_fail_flips and self.status == "STABLE":
            raise ValueError("stable verifier cannot have PASS/FAIL flips")
        if self.status == "STOCHASTIC_WITH_BOUNDED_ORACLE" and not self.explanation:
            raise ValueError("bounded stochastic verifier requires an explanation")
        return self


class TaskLeakageAudit(TaskQualityModel):
    status: Literal["pass", "fail", "unresolved"]
    instruction_hidden_case_leak: bool = False
    witness_artifact_leak: bool = False
    future_fix_leak: bool = False
    cache_leak: bool = False
    judge_pack_witness_leak: bool = False
    failure_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_findings(self) -> TaskLeakageAudit:
        leaked = any(
            (
                self.instruction_hidden_case_leak,
                self.witness_artifact_leak,
                self.future_fix_leak,
                self.cache_leak,
                self.judge_pack_witness_leak,
            )
        )
        if leaked and self.status != "fail":
            raise ValueError("confirmed task leakage requires status=fail")
        if self.status != "pass" and not self.failure_codes:
            raise ValueError("non-passing task leakage audit requires failure codes")
        return self


class HumanTaskQualificationReview(TaskQualityModel):
    reviewer: str
    reviewed_at: datetime
    decision: Literal["approve", "reject", "request-revision"]
    specification_sha256: Digest
    contract_sha256: Digest
    witness_set_sha256: Digest
    mutation_suite_sha256: Digest
    alternative_solution_set_sha256: Digest
    artifact_policy_sha256: Digest
    capability_policy_sha256: Digest
    rationale: str

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> HumanTaskQualificationReview:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Task qualification review timestamp must be timezone-aware")
        return self


class VerifierCoverageReport(TaskQualityModel):
    requirement_to_obligations: dict[str, list[str]]
    hard_obligations: list[str]
    witness_covered_hard_obligations: list[str]
    critical_mutants_total: int = Field(ge=0)
    critical_mutants_killed: int = Field(ge=0)
    weighted_mutation_adequacy: float = Field(ge=0, le=1)
    negative_controls_total: int = Field(ge=0)
    negative_controls_rejected: int = Field(ge=0)
    alternative_valid_total: int = Field(ge=0)
    alternative_valid_accepted: int = Field(ge=0)
    status: Literal["pass", "fail", "unresolved"]
    failure_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts_are_coherent(self) -> VerifierCoverageReport:
        pairs = (
            (self.critical_mutants_killed, self.critical_mutants_total),
            (self.negative_controls_rejected, self.negative_controls_total),
            (self.alternative_valid_accepted, self.alternative_valid_total),
        )
        if any(observed > total for observed, total in pairs):
            raise ValueError("Verifier coverage passed count cannot exceed total")
        if self.status == "pass" and self.failure_codes:
            raise ValueError("passing verifier coverage cannot carry failures")
        if self.status != "pass" and not self.failure_codes:
            raise ValueError("non-passing verifier coverage requires failures")
        return self


class VerifierTrustCard(TaskQualityModel):
    status: Literal["pass", "fail", "unresolved", "legacy-unknown"]
    baseline_differential_valid: bool
    witness_replay_valid: bool
    mutation_adequacy: float = Field(ge=0, le=1)
    negative_controls_valid: bool
    alternative_solution_breadth_valid: bool
    fresh_replay_stable: bool
    environment_sentinel_valid: bool
    leakage_valid: bool
    failure_codes: list[str] = Field(default_factory=list)
    candidate_score_effect: Literal[False] = False


class TaskQualificationReport(TaskQualityModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    task_revision: int = Field(ge=1)
    status: Literal[
        "QUALIFIED",
        "QUALIFIED_WITH_SCOPE",
        "REVIEW_REQUIRED",
        "INELIGIBLE",
        "REVOKED",
    ]
    specification_sha256: Digest
    contract_sha256: Digest
    witness_set_sha256: Digest
    mutation_suite_sha256: Digest
    alternative_solution_set_sha256: Digest
    coverage: VerifierCoverageReport
    trust: VerifierTrustCard
    failure_codes: list[str] = Field(default_factory=list)
    report_sha256: Digest
    candidate_score_effect: Literal[False] = False


class TaskSeal(TaskQualityModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    task_revision: int = Field(ge=1)
    qualification_status: Literal["QUALIFIED", "QUALIFIED_WITH_SCOPE"]
    specification_sha256: Digest
    acceptance_contract_sha256: Digest
    verifier_bundle_sha256: Digest
    witness_set_sha256: Digest
    mutation_suite_sha256: Digest
    alternative_solution_set_sha256: Digest
    artifact_policy_sha256: Digest
    capability_policy_sha256: Digest
    capability_registry_sha256: Digest
    capability_contract_sha256: Digest
    resource_envelope_sha256: Digest
    topology_contract_sha256: Digest
    benchmark_cell_policy_sha256: Digest
    runner_selection_policy_sha256: Digest
    qualification_report_sha256: Digest
    qualified_at: datetime
    reviewers: list[str] = Field(min_length=1)
    benchmark_season: str
    task_seal_sha256: Digest

    @model_validator(mode="after")
    def qualified_timestamp_is_aware(self) -> TaskSeal:
        if self.qualified_at.tzinfo is None or self.qualified_at.utcoffset() is None:
            raise ValueError("Task Seal timestamp must be timezone-aware")
        return self


class VerifierResultDiagnostics(TaskQualityModel):
    bucket_pass_fractions: dict[ObligationBucket, float]
    first_failure: str | None = None
    all_failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fractions_are_bounded(self) -> VerifierResultDiagnostics:
        if any(value < 0 or value > 1 for value in self.bucket_pass_fractions.values()):
            raise ValueError("obligation pass fractions must stay in [0, 1]")
        return self


class VerifierResult(TaskQualityModel):
    schema_version: Literal["0.1"] = "0.1"
    task_id: str
    task_seal_sha256: Digest
    candidate_sha256: Digest
    environment_status: Literal["PASS", "INFRA_INVALID", "BENCHMARK_DEFECT"]
    obligations: list[ObligationObservation]
    infra_cert: Literal[0, 1] | None
    result_status: CandidateResultStatus
    diagnostics: VerifierResultDiagnostics
    verifier_result_sha256: Digest

    @model_validator(mode="after")
    def result_status_and_infra_cert_are_coherent(self) -> VerifierResult:
        identifiers = [item.obligation_id for item in self.obligations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Verifier result obligation ids must be unique")
        if self.environment_status != "PASS":
            expected = self.environment_status
            if self.result_status != expected or self.infra_cert is not None:
                raise ValueError("invalid environment cannot become a Candidate result")
        if self.result_status == "VALID_PASS" and self.infra_cert != 1:
            raise ValueError("VALID_PASS requires InfraCert=1")
        if self.result_status == "VALID_FAIL" and self.infra_cert != 0:
            raise ValueError("VALID_FAIL requires InfraCert=0")
        if self.result_status in {"UNRESOLVED", "NOT_APPLICABLE"} and self.infra_cert is not None:
            raise ValueError("unresolved/not-applicable result cannot publish InfraCert")
        return self


def weighted_mutation_adequacy(outcomes: list[MutationOutcome]) -> float:
    if not outcomes:
        return 0.0
    total = sum(item.weight for item in outcomes)
    killed = sum(item.weight for item in outcomes if item.observed == "rejected")
    return killed / total if total else 0.0


def bucket_pass_fraction(observations: list[ObligationObservation]) -> float:
    decided = [item for item in observations if item.status in {"PASS", "FAIL"}]
    if not decided:
        return 0.0
    value = sum(item.status == "PASS" for item in decided) / len(decided)
    assert math.isfinite(value)
    return value
