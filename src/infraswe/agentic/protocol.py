from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel

from infraswe.artifact_boundary.evidence import audit_evidence_pack
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.agentic import (
    AgentHarnessProfile,
    AgenticDraftSpec,
    AlgorithmProfile,
    BranchRecord,
    CreditAssignmentMap,
    EpisodeOutcomeSeal,
    EpisodeSeal,
    ExternalPolicyState,
    FeedbackItem,
    FeedbackPack,
    GangLeaseRecord,
    GroupManifest,
    LegacyExperienceManifest,
    LegacyExperienceRecord,
    LogprobFidelityReport,
    ModelBoundaryPolicy,
    ModelBoundaryTrace,
    PolicyCell,
    PolicySnapshot,
    RewardAnchor,
    RewardEvent,
    RewardPack,
    RewardProfile,
    RewardQualification,
    RLBatchManifest,
    RolloutFabricProfile,
    RolloutRequest,
    RuntimeCapabilityReport,
    SandboxProfile,
    SandboxSnapshot,
    TokenModulation,
    TokenRange,
    TrainingRunSeal,
    TrajectoryEnvelope,
    VerifierOutcomePack,
)
from infraswe.models.artifact_boundary import EvidencePackManifest, TrialSeal

_ZERO_DIGEST = "sha256:" + "0" * 64

_DIGEST_FIELDS: dict[type[BaseModel], str] = {
    PolicySnapshot: "policy_snapshot_sha256",
    ExternalPolicyState: "external_policy_state_sha256",
    AgentHarnessProfile: "harness_profile_sha256",
    PolicyCell: "policy_cell_sha256",
    ModelBoundaryTrace: "trace_sha256",
    LogprobFidelityReport: "fidelity_report_sha256",
    SandboxProfile: "sandbox_profile_sha256",
    SandboxSnapshot: "snapshot_sha256",
    BranchRecord: "branch_sha256",
    TrajectoryEnvelope: "trajectory_sha256",
    RolloutRequest: "request_sha256",
    EpisodeSeal: "episode_seal_sha256",
    VerifierOutcomePack: "outcome_sha256",
    RewardQualification: "qualification_sha256",
    RewardProfile: "reward_profile_sha256",
    RewardEvent: "event_sha256",
    FeedbackPack: "feedback_pack_sha256",
    CreditAssignmentMap: "credit_map_sha256",
    RewardPack: "reward_pack_sha256",
    EpisodeOutcomeSeal: "episode_outcome_sha256",
    AlgorithmProfile: "algorithm_profile_sha256",
    GroupManifest: "group_manifest_sha256",
    TrainingRunSeal: "training_run_seal_sha256",
    RLBatchManifest: "batch_sha256",
    RolloutFabricProfile: "fabric_profile_sha256",
    GangLeaseRecord: "lease_sha256",
    RuntimeCapabilityReport: "report_sha256",
    AgenticDraftSpec: "draft_sha256",
    LegacyExperienceRecord: "record_sha256",
    LegacyExperienceManifest: "manifest_sha256",
}

_LEAKAGE_PATTERNS = (
    re.compile(r"(?i)(?:tests?|fixtures?)/(?:hidden|private)/"),
    re.compile(r"(?i)\bhidden[_ -]?(?:input|case|answer|oracle)\b"),
    re.compile(r"(?i)\bprivate[_ -]?witness\b"),
    re.compile(r"(?i)\bsecret[_ -]?(?:probe|argument|token)\b"),
    re.compile(r"(?i)\bheldout[_ -]?(?:task|identity|answer)\b"),
    re.compile(r"(?i)\bfuture[_ -]?fix[_ -]?(?:symbol|patch)\b"),
)


def digest_field(model_type: type[BaseModel]) -> str:
    try:
        return _DIGEST_FIELDS[model_type]
    except KeyError as error:
        raise TypeError(f"{model_type.__name__} is not a sealed v0.6 artifact") from error


def build_sealed[ModelT: BaseModel](model_type: type[ModelT], /, **payload: Any) -> ModelT:
    """Validate and content-address one v0.6 protocol artifact."""

    field = digest_field(model_type)
    provisional = model_type.model_validate({**payload, field: _ZERO_DIGEST})
    material = provisional.model_dump(mode="json", exclude={field})
    return model_type.model_validate(
        {**provisional.model_dump(mode="json"), field: canonical_sha256(material)}
    )


def reseal[ModelT: BaseModel](model: ModelT) -> ModelT:
    field = digest_field(type(model))
    payload = model.model_dump(mode="json", exclude={field})
    return build_sealed(type(model), **payload)


def audit_sealed(model: BaseModel) -> list[str]:
    field = digest_field(type(model))
    observed = cast(str, getattr(model, field))
    material = model.model_dump(mode="json", exclude={field})
    expected = canonical_sha256(material)
    return [] if observed == expected else [f"{type(model).__name__.upper()}_DIGEST_MISMATCH"]


def audit_policy_cell_bindings(
    cell: PolicyCell,
    *,
    policy: PolicySnapshot,
    harness: AgentHarnessProfile,
    sandbox: SandboxProfile,
    external_state: ExternalPolicyState | None = None,
) -> list[str]:
    failures = [
        *audit_sealed(cell),
        *audit_sealed(policy),
        *audit_sealed(harness),
        *audit_sealed(sandbox),
    ]
    bindings = {
        "POLICY_CELL_POLICY_MISMATCH": (
            cell.policy_snapshot_sha256,
            policy.policy_snapshot_sha256,
        ),
        "POLICY_CELL_UPDATE_MODE_MISMATCH": (cell.update_mode, policy.update_mode),
        "POLICY_CELL_HARNESS_MISMATCH": (
            cell.harness_profile_sha256,
            harness.harness_profile_sha256,
        ),
        "POLICY_CELL_SKILL_PACK_MISMATCH": (
            cell.skill_pack_sha256,
            harness.skill_pack_sha256,
        ),
        "POLICY_CELL_COMPACTION_POLICY_MISMATCH": (
            cell.compaction_policy_sha256,
            harness.context_compaction_policy_sha256,
        ),
        "POLICY_CELL_SANDBOX_MISMATCH": (
            cell.sandbox_profile_sha256,
            sandbox.sandbox_profile_sha256,
        ),
    }
    failures.extend(code for code, pair in bindings.items() if pair[0] != pair[1])
    if external_state is None:
        if cell.external_policy_state_sha256 is not None:
            failures.append("POLICY_CELL_EXTERNAL_STATE_MISSING")
    else:
        failures.extend(audit_sealed(external_state))
        if cell.external_policy_state_sha256 != external_state.external_policy_state_sha256:
            failures.append("POLICY_CELL_EXTERNAL_STATE_MISMATCH")
        if external_state.base_policy_snapshot_sha256 != policy.policy_snapshot_sha256:
            failures.append("EXTERNAL_STATE_BASE_POLICY_MISMATCH")
    return sorted(set(failures))


def build_logprob_fidelity_report(
    *,
    trajectory_sha256: str,
    rollout_train_logprob_correlation: float,
    mean_abs_delta_logp: float,
    importance_ratio_quantiles: dict[str, float],
    mask_alignment_rate: float,
    model_boundary_policy: ModelBoundaryPolicy,
) -> LogprobFidelityReport:
    minimum_correlation = float(model_boundary_policy.minimum_logprob_correlation)
    maximum_delta = float(model_boundary_policy.maximum_mean_abs_delta_logp)
    minimum_alignment = float(model_boundary_policy.minimum_mask_alignment_rate)
    eligible = (
        rollout_train_logprob_correlation >= minimum_correlation
        and mean_abs_delta_logp <= maximum_delta
        and mask_alignment_rate >= minimum_alignment
    )
    return build_sealed(
        LogprobFidelityReport,
        trajectory_sha256=trajectory_sha256,
        rollout_train_logprob_correlation=rollout_train_logprob_correlation,
        mean_abs_delta_logp=mean_abs_delta_logp,
        importance_ratio_quantiles=importance_ratio_quantiles,
        mask_alignment_rate=mask_alignment_rate,
        minimum_correlation=minimum_correlation,
        maximum_mean_abs_delta_logp=maximum_delta,
        minimum_mask_alignment_rate=minimum_alignment,
        policy_gradient_eligible=eligible,
    )


def audit_trajectory_bindings(
    trajectory: TrajectoryEnvelope,
    *,
    policy: PolicySnapshot,
    harness: AgentHarnessProfile,
    sandbox: SandboxProfile,
    fidelity: LogprobFidelityReport | None = None,
) -> list[str]:
    failures = [
        *audit_sealed(trajectory),
        *audit_sealed(policy),
        *audit_sealed(harness),
        *audit_sealed(sandbox),
    ]
    bindings = {
        "TRAJECTORY_POLICY_MISMATCH": (
            trajectory.policy_snapshot_sha256,
            policy.policy_snapshot_sha256,
        ),
        "TRAJECTORY_HARNESS_MISMATCH": (
            trajectory.harness_profile_sha256,
            harness.harness_profile_sha256,
        ),
        "TRAJECTORY_SANDBOX_MISMATCH": (
            trajectory.sandbox_profile_sha256,
            sandbox.sandbox_profile_sha256,
        ),
    }
    failures.extend(code for code, pair in bindings.items() if pair[0] != pair[1])
    for trace in trajectory.traces:
        failures.extend(audit_sealed(trace))
    for branch in trajectory.branches:
        failures.extend(audit_sealed(branch))
    if trajectory.policy_gradient_eligible:
        if fidelity is None:
            failures.append("TRAJECTORY_LOGPROB_FIDELITY_MISSING")
        else:
            failures.extend(audit_sealed(fidelity))
            if fidelity.trajectory_sha256 != trajectory.trajectory_sha256:
                failures.append("TRAJECTORY_LOGPROB_FIDELITY_BINDING_MISMATCH")
            if not fidelity.policy_gradient_eligible:
                failures.append("TRAJECTORY_LOGPROB_FIDELITY_REJECTED")
    return sorted(set(failures))


def sign_preserving_modulation(
    *,
    span: TokenRange,
    verifier_advantage: float,
    teacher_signal: float,
    threshold: float,
    strength: float,
    temperature: float,
    minimum_multiplier: float,
    maximum_multiplier: float,
) -> TokenModulation:
    if temperature <= 0 or strength < 0:
        raise ValueError("modulation temperature must be positive and strength non-negative")
    if not 0 < minimum_multiplier <= maximum_multiplier:
        raise ValueError("modulation multipliers must be positive and ordered")
    active = abs(teacher_signal) > threshold
    delta = strength * math.tanh(teacher_signal / temperature) if active else 0.0
    multiplier = min(max(1 + delta, minimum_multiplier), maximum_multiplier)
    return TokenModulation(
        span=span,
        verifier_advantage=verifier_advantage,
        multiplier=multiplier,
        modulated_advantage=verifier_advantage * multiplier,
    )


def build_feedback_pack(
    *,
    task_seal_sha256: str,
    episode_seal_sha256: str,
    feedback_profile_sha256: str,
    visibility: str,
    task_split: str,
    items: list[FeedbackItem],
) -> FeedbackPack:
    leaked = any(
        pattern.search(item.public_summary) for item in items for pattern in _LEAKAGE_PATTERNS
    )
    leakage_scan = "blocked" if leaked else "pass"
    teacher_eligible = (
        leakage_scan == "pass"
        and task_split == "train"
        and visibility in {"POLICY_VISIBLE", "TEACHER_VISIBLE_REDACTED"}
    )
    return build_sealed(
        FeedbackPack,
        task_seal_sha256=task_seal_sha256,
        episode_seal_sha256=episode_seal_sha256,
        feedback_profile_sha256=feedback_profile_sha256,
        visibility=visibility,
        task_split=task_split,
        items=items,
        leakage_scan=leakage_scan,
        teacher_eligible=teacher_eligible,
    )


def _event_checks(
    events: list[RewardEvent],
    *,
    episode_id: str,
    outcome: VerifierOutcomePack,
    evidence_pack: EvidencePackManifest,
    feedback: FeedbackPack | None,
) -> None:
    evidence_by_id = {item.evidence_id: item for item in evidence_pack.artifacts}
    for obligation in outcome.obligations:
        expected_authorities = (
            {"VERIFIER", "INFRASTRUCTURE"} if obligation.bucket == "ES" else {"VERIFIER"}
        )
        for evidence_ref in obligation.evidence_refs:
            artifact = evidence_by_id.get(evidence_ref)
            if artifact is None:
                raise ValueError("VERIFIER_OUTCOME_EVIDENCE_REF_UNRESOLVED")
            if artifact.authority not in expected_authorities:
                raise ValueError("VERIFIER_OUTCOME_EVIDENCE_AUTHORITY_MISMATCH")
    fact_ids = [item.fact_id for item in events]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("REWARD_DUPLICATE_FACT")
    event_ids = [item.event_id for item in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("REWARD_DUPLICATE_EVENT_ID")
    obligation_ids = {item.obligation_id for item in outcome.obligations}
    for event in events:
        if audit_sealed(event):
            raise ValueError("REWARD_EVENT_DIGEST_MISMATCH")
        if event.scope.episode_id != episode_id:
            raise ValueError("REWARD_EVENT_EPISODE_MISMATCH")
        if (
            event.scope.obligation_id is not None
            and event.scope.obligation_id not in obligation_ids
        ):
            raise ValueError("REWARD_EVENT_OBLIGATION_MISMATCH")
        resolved = []
        for evidence_ref in event.evidence_refs:
            artifact = evidence_by_id.get(evidence_ref)
            if artifact is None:
                raise ValueError("REWARD_EVENT_EVIDENCE_REF_UNRESOLVED")
            resolved.append(artifact)
        required_authorities: set[str] = set()
        if event.authority == "OFFICIAL_HARD":
            required_authorities = {"VERIFIER"}
        elif event.authority == "VALIDITY":
            required_authorities = {"VERIFIER", "INFRASTRUCTURE"}
        elif event.authority == "OFFICIAL_COMPONENT":
            if event.kind == "performance":
                required_authorities = {"METER"}
            elif event.kind == "official-semantic":
                required_authorities = {"BOUNDED_JUDGE"}
            else:
                required_authorities = {"VERIFIER", "METER", "BOUNDED_JUDGE"}
        if required_authorities and not any(
            artifact.authority in required_authorities for artifact in resolved
        ):
            raise ValueError("REWARD_EVENT_EVIDENCE_AUTHORITY_MISMATCH")
        if event.kind == "performance":
            if event.authority != "OFFICIAL_COMPONENT":
                raise ValueError("PERFORMANCE_REWARD_REQUIRES_TRUSTED_METER")
            if event.scope.comparison_cell_sha256 != outcome.benchmark_cell_sha256:
                raise ValueError("CROSS_CELL_PERFORMANCE_REWARD_FORBIDDEN")
            if not outcome.infra_cert:
                raise ValueError("PERFORMANCE_REWARD_REQUIRES_INFRACERT")
        if event.kind == "teacher-shaping" and (feedback is None or not feedback.teacher_eligible):
            raise ValueError("TEACHER_SHAPING_REQUIRES_ELIGIBLE_FEEDBACK")


def compile_reward_pack(
    *,
    episode_seal: EpisodeSeal,
    outcome: VerifierOutcomePack,
    evidence_pack: EvidencePackManifest,
    trial_seal: TrialSeal,
    qualification: RewardQualification,
    profile: RewardProfile,
    events: list[RewardEvent],
    credit_profile_sha256: str,
    credit_map: CreditAssignmentMap | None = None,
    feedback: FeedbackPack | None = None,
    official_projection_path: str = "scoring/official-score.json",
    training_projection_path: str = "reward/training-projection.json",
) -> RewardPack:
    for artifact in (episode_seal, outcome, qualification, profile):
        if audit_sealed(artifact):
            raise ValueError(f"{type(artifact).__name__} digest mismatch")
    evidence_failures = audit_evidence_pack(evidence_pack, trial_seal=trial_seal)
    if evidence_failures:
        raise ValueError(
            "RewardCompiler requires a sealed EvidencePack: " + ",".join(evidence_failures)
        )
    bindings = {
        "REWARD_EPISODE_OUTCOME_MISMATCH": (
            outcome.episode_seal_sha256,
            episode_seal.episode_seal_sha256,
        ),
        "REWARD_EPISODE_EVIDENCE_MISMATCH": (
            episode_seal.evidence_pack_sha256,
            evidence_pack.evidence_pack_sha256,
        ),
        "REWARD_OUTCOME_EVIDENCE_MISMATCH": (
            outcome.evidence_pack_sha256,
            evidence_pack.evidence_pack_sha256,
        ),
        "REWARD_OUTCOME_CELL_MISMATCH": (
            outcome.benchmark_cell_sha256,
            episode_seal.benchmark_cell_sha256,
        ),
        "REWARD_TASK_QUALIFICATION_MISMATCH": (
            qualification.task_seal_sha256,
            episode_seal.task_seal_sha256,
        ),
        "REWARD_PROFILE_QUALIFICATION_MISMATCH": (
            qualification.reward_profile_sha256,
            profile.reward_profile_sha256,
        ),
        "REWARD_OUTCOME_TASK_MISMATCH": (
            outcome.task_seal_sha256,
            episode_seal.task_seal_sha256,
        ),
        "REWARD_TRIAL_TASK_MISMATCH": (
            trial_seal.task_seal_sha256,
            episode_seal.task_seal_sha256,
        ),
        "REWARD_TRIAL_DRAFT_MISMATCH": (
            trial_seal.draft_seal_sha256,
            episode_seal.draft_seal_sha256,
        ),
        "REWARD_TRIAL_ARTIFACT_MISMATCH": (
            trial_seal.candidate_artifact_manifest_sha256,
            episode_seal.candidate_artifact_manifest_sha256,
        ),
        "REWARD_TRIAL_CAPABILITY_MISMATCH": (
            trial_seal.capability_resolution_sha256,
            episode_seal.capability_resolution_sha256,
        ),
        "REWARD_TRIAL_LEASE_MISMATCH": (
            trial_seal.resource_lease_sha256,
            episode_seal.resource_lease_sha256,
        ),
        "REWARD_TRIAL_CELL_MISMATCH": (
            trial_seal.benchmark_cell_sha256,
            episode_seal.benchmark_cell_sha256,
        ),
    }
    mismatches = [code for code, pair in bindings.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError(",".join(mismatches))
    if qualification.status != "qualified":
        raise ValueError("REWARD_INSTRUMENT_NOT_QUALIFIED")
    invalid_episode_statuses = {
        "INFRA_INVALID",
        "BENCHMARK_DEFECT",
        "ROLLOUT_DEFECT",
        "STALE_REJECTED",
        "CENSORED",
    }
    if (not outcome.validity.valid) != (episode_seal.status in invalid_episode_statuses):
        raise ValueError("REWARD_EPISODE_VALIDITY_STATUS_MISMATCH")
    if (
        outcome.validity.valid
        and outcome.infra_cert
        and episode_seal.status
        not in {
            "VALID_PASS",
            "JUDGE_UNRESOLVED",
        }
    ):
        raise ValueError("REWARD_PASS_STATUS_MISMATCH")
    if (
        outcome.validity.valid
        and not outcome.infra_cert
        and episode_seal.status
        not in {
            "VALID_FAIL",
            "POLICY_BUDGET_EXCEEDED",
            "SECURITY_REJECTED",
        }
    ):
        raise ValueError("REWARD_FAIL_STATUS_MISMATCH")
    if credit_map is not None:
        if audit_sealed(credit_map):
            raise ValueError("REWARD_CREDIT_MAP_DIGEST_MISMATCH")
        if credit_map.episode_id != episode_seal.episode_id:
            raise ValueError("REWARD_CREDIT_MAP_EPISODE_MISMATCH")
        event_ids = {item.event_id for item in events}
        if not set(credit_map.anchor_reward_event_ids).issubset(event_ids):
            raise ValueError("REWARD_CREDIT_MAP_ANCHOR_EVENT_MISSING")
    if feedback is not None:
        if audit_sealed(feedback):
            raise ValueError("REWARD_FEEDBACK_PACK_DIGEST_MISMATCH")
        if feedback.episode_seal_sha256 != episode_seal.episode_seal_sha256:
            raise ValueError("REWARD_FEEDBACK_PACK_EPISODE_MISMATCH")
        if feedback.task_seal_sha256 != episode_seal.task_seal_sha256:
            raise ValueError("REWARD_FEEDBACK_PACK_TASK_MISMATCH")
    _event_checks(
        events,
        episode_id=episode_seal.episode_id,
        outcome=outcome,
        evidence_pack=evidence_pack,
        feedback=feedback,
    )

    invalid = not outcome.validity.valid or episode_seal.training_mask == 0
    if invalid:
        anchor = RewardAnchor(infra_cert=None, scalar_band="masked", scalar_value=None)
        training_mask = 0
        validity = "invalid"
    elif not outcome.infra_cert:
        base = (profile.hard_fail_floor + profile.hard_fail_ceiling) / 2
        shaping = sum(
            item.value
            for item in events
            if item.status == "VALID" and item.authority == "TRAINING_SHAPING"
        )
        shaping = min(max(shaping, -profile.process_shaping_cap), profile.process_shaping_cap)
        value = min(max(base + shaping, profile.hard_fail_floor), profile.hard_fail_ceiling)
        anchor = RewardAnchor(infra_cert=0, scalar_band="hard-fail", scalar_value=value)
        training_mask = 1
        validity = "valid"
    else:
        attained = any(
            item.status == "VALID" and item.kind == "performance" and item.value > 0
            for item in events
        )
        base = (profile.hard_pass_floor + profile.hard_pass_ceiling) / 2 if attained else 0.0
        shaping = sum(
            item.value
            for item in events
            if item.status == "VALID" and item.authority == "TRAINING_SHAPING"
        )
        shaping = min(max(shaping, -profile.process_shaping_cap), profile.process_shaping_cap)
        value = min(max(base + shaping, profile.hard_pass_floor), profile.hard_pass_ceiling)
        anchor = RewardAnchor(
            infra_cert=1,
            scalar_band="pass-attainment" if attained else "pass-no-attainment",
            scalar_value=value,
        )
        training_mask = 1
        validity = "valid"

    return build_sealed(
        RewardPack,
        episode_seal_sha256=episode_seal.episode_seal_sha256,
        evidence_pack_sha256=evidence_pack.evidence_pack_sha256,
        verifier_outcome_sha256=outcome.outcome_sha256,
        reward_profile_sha256=profile.reward_profile_sha256,
        reward_qualification_sha256=qualification.qualification_sha256,
        credit_profile_sha256=credit_profile_sha256,
        validity=validity,
        training_mask=training_mask,
        anchor=anchor,
        event_sha256s=[item.event_sha256 for item in events],
        credit_map_sha256=credit_map.credit_map_sha256 if credit_map else None,
        feedback_pack_sha256=feedback.feedback_pack_sha256 if feedback else None,
        official_projection_path=official_projection_path,
        official_projection_independently_reproducible=True,
        training_projection_path=training_projection_path,
        training_reward_affects_official_score=False,
        revoked=False,
        revocation_reason=None,
    )


def build_episode_outcome_seal(
    *,
    episode_seal: EpisodeSeal,
    reward_pack: RewardPack,
) -> EpisodeOutcomeSeal:
    """Bind execution/evidence and reward without creating a mutual digest cycle."""

    if audit_sealed(episode_seal):
        raise ValueError("EPISODE_OUTCOME_EPISODE_DIGEST_MISMATCH")
    if audit_sealed(reward_pack):
        raise ValueError("EPISODE_OUTCOME_REWARD_DIGEST_MISMATCH")
    if reward_pack.episode_seal_sha256 != episode_seal.episode_seal_sha256:
        raise ValueError("EPISODE_OUTCOME_BINDING_MISMATCH")
    return build_sealed(
        EpisodeOutcomeSeal,
        episode_seal_sha256=episode_seal.episode_seal_sha256,
        reward_pack_sha256=reward_pack.reward_pack_sha256,
    )


def audit_episode_outcome_seal(
    outcome_seal: EpisodeOutcomeSeal,
    *,
    episode_seal: EpisodeSeal,
    reward_pack: RewardPack,
) -> list[str]:
    failures = [
        *audit_sealed(outcome_seal),
        *audit_sealed(episode_seal),
        *audit_sealed(reward_pack),
    ]
    if outcome_seal.episode_seal_sha256 != episode_seal.episode_seal_sha256:
        failures.append("EPISODE_OUTCOME_EPISODE_BINDING_MISMATCH")
    if outcome_seal.reward_pack_sha256 != reward_pack.reward_pack_sha256:
        failures.append("EPISODE_OUTCOME_REWARD_BINDING_MISMATCH")
    if reward_pack.episode_seal_sha256 != episode_seal.episode_seal_sha256:
        failures.append("EPISODE_OUTCOME_REWARD_EPISODE_MISMATCH")
    return sorted(set(failures))


def validate_rl_batch(
    batch: RLBatchManifest,
    *,
    training_run: TrainingRunSeal,
    algorithm: AlgorithmProfile,
    known_policy_snapshots: set[str],
    revoked_reward_packs: set[str] | None = None,
) -> list[str]:
    failures = [*audit_sealed(batch), *audit_sealed(training_run), *audit_sealed(algorithm)]
    revoked = revoked_reward_packs or set()
    if batch.training_run_seal_sha256 != training_run.training_run_seal_sha256:
        failures.append("BATCH_TRAINING_RUN_MISMATCH")
    if batch.algorithm_profile_sha256 != algorithm.algorithm_profile_sha256:
        failures.append("BATCH_ALGORITHM_PROFILE_MISMATCH")
    if training_run.algorithm_profile_sha256 != algorithm.algorithm_profile_sha256:
        failures.append("TRAINING_RUN_ALGORITHM_PROFILE_MISMATCH")
    if batch.target_policy_snapshot_sha256 not in known_policy_snapshots:
        failures.append("BATCH_TARGET_POLICY_UNKNOWN")
    if batch.proximal_policy_snapshot_sha256 not in known_policy_snapshots:
        failures.append("BATCH_PROXIMAL_POLICY_UNKNOWN")

    behavior = {item.behavior_policy_snapshot_sha256 for item in batch.members}
    tasks = {item.task_seal_sha256 for item in batch.members}
    cells = {item.benchmark_cell_sha256 for item in batch.members}
    reward_schemas = {item.reward_schema_version for item in batch.members}
    if len(behavior) != 1:
        failures.append("BATCH_MIXED_BEHAVIOR_POLICY")
    if len(tasks) != 1:
        failures.append("BATCH_MIXED_TASK_GROUP")
    if len(cells) != 1:
        failures.append("BATCH_MIXED_COMPARISON_CELL")
    if len(reward_schemas) != 1:
        failures.append("BATCH_MIXED_REWARD_SCHEMA")

    for member in batch.members:
        suffix = ":" + member.episode_outcome_sha256
        if member.behavior_policy_snapshot_sha256 not in known_policy_snapshots:
            failures.append("UNKNOWN_BEHAVIOR_POLICY" + suffix)
        if member.reward_revoked or member.reward_pack_sha256 in revoked:
            failures.append("REVOKED_REWARD_PACK" + suffix)
        if member.feedback_leakage_blocked and member.training_mask:
            failures.append("FEEDBACK_LEAKAGE_IN_TRAINING_MEMBER" + suffix)
        if member.episode_status == "SECURITY_REJECTED":
            expected_security_mask = int(
                algorithm.loss.security_rejected_training == "valid-negative"
            )
            if member.training_mask != expected_security_mask:
                failures.append("SECURITY_REJECTION_MASK_POLICY_MISMATCH" + suffix)
        if (
            member.training_mask
            and not member.policy_gradient_eligible
            and algorithm.llm_weights_update
        ):
            failures.append("TRAJECTORY_NOT_POLICY_GRADIENT_ELIGIBLE" + suffix)
        if (
            member.policy_lag > algorithm.policy_staleness.max_versions
            and algorithm.policy_staleness.over_limit == "reject"
        ):
            failures.append("STALENESS_LIMIT_EXCEEDED" + suffix)
        if algorithm.family == "steppo" and member.training_mask and not member.valid_step_count:
            failures.append("STEPPO_STEP_MAP_MISSING" + suffix)
    return sorted(set(failures))


def build_runtime_capability_report(
    *,
    gpu_count: int,
    gpu_topology_attested: bool,
    rootless_sandbox_enforced: bool,
    exact_token_gateway_available: bool,
    hosted_policy_exact_tokens_available: bool,
    trainer_adapter_available: bool,
    distributed_gang_enforced: bool,
    observed_at: datetime | None = None,
) -> RuntimeCapabilityReport:
    checks = {
        "NO_ACCELERATOR": gpu_count > 0,
        "GPU_TOPOLOGY_NOT_ATTESTED": gpu_topology_attested,
        "ROOTLESS_SANDBOX_NOT_ENFORCED": rootless_sandbox_enforced,
        "EXACT_TOKEN_GATEWAY_UNAVAILABLE": exact_token_gateway_available,
        "TRAINER_ADAPTER_UNAVAILABLE": trainer_adapter_available,
        "DISTRIBUTED_GANG_NOT_ENFORCED": distributed_gang_enforced,
    }
    reasons = [code for code, passed in checks.items() if not passed]
    return build_sealed(
        RuntimeCapabilityReport,
        observed_at=observed_at or datetime.now(UTC),
        gpu_count=gpu_count,
        gpu_topology_attested=gpu_topology_attested,
        rootless_sandbox_enforced=rootless_sandbox_enforced,
        exact_token_gateway_available=exact_token_gateway_available,
        hosted_policy_exact_tokens_available=hosted_policy_exact_tokens_available,
        trainer_adapter_available=trainer_adapter_available,
        distributed_gang_enforced=distributed_gang_enforced,
        production_ready=not reasons,
        unavailable_reasons=reasons,
    )
