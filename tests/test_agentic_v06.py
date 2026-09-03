from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from infraswe.agentic import (
    audit_episode_outcome_seal,
    audit_policy_cell_bindings,
    audit_sealed,
    audit_trajectory_bindings,
    build_episode_outcome_seal,
    build_feedback_pack,
    build_legacy_experience_manifest,
    build_logprob_fidelity_report,
    build_runtime_capability_report,
    build_sealed,
    compile_reward_pack,
    reseal,
    sign_preserving_modulation,
    validate_rl_batch,
)
from infraswe.artifact_boundary import build_evidence_pack, build_trial_seal
from infraswe.cli import app
from infraswe.models.agentic import (
    AdapterIdentity,
    AgentHarnessProfile,
    AgenticEpisode,
    AlgorithmProfile,
    BackpressureProfile,
    ClippingProfile,
    CreditAssignmentMap,
    CreditStepAssignment,
    DecodingPolicy,
    EpisodeEvent,
    EpisodeSeal,
    ExcludedTokenRange,
    ExternalPolicyState,
    FabricPool,
    FeedbackItem,
    GangDeviceAllocation,
    GangLeaseRecord,
    HarnessLimits,
    ModelArtifactIdentity,
    ModelBoundaryPolicy,
    ModelBoundaryTrace,
    OverlongPolicy,
    PolicyCell,
    PolicySnapshot,
    PolicyStalenessProfile,
    RewardEvent,
    RewardProfile,
    RewardQualification,
    RewardScope,
    RLBatchManifest,
    RLBatchMember,
    RolloutFabricProfile,
    SamplingMetadata,
    SandboxDevicePolicy,
    SandboxFilesystemPolicy,
    SandboxNetworkPolicy,
    SandboxProcessPolicy,
    SandboxProfile,
    SandboxSnapshotPolicy,
    ServingIdentity,
    StepTokenSpan,
    TokenRange,
    TrainingBudgets,
    TrainingInfrastructure,
    TrainingLossProfile,
    TrainingRunSeal,
    TrainingSamplingProfile,
    TrainingSplits,
    TrajectoryEnvelope,
    TrajectoryStep,
    VerifierObligationOutcome,
    VerifierOutcomePack,
    VerifierValidity,
)
from infraswe.models.artifact_boundary import EvidenceArtifact, EvidenceProducerIdentity
from infraswe.schema import schema_documents


def _d(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def identities():
    policy = build_sealed(
        PolicySnapshot,
        policy_id="policy-a",
        policy_version=3,
        update_mode="none",
        base_model=ModelArtifactIdentity(
            family="qwen-like",
            weights_sha256=_d("weights"),
            config_sha256=_d("config"),
            tokenizer_sha256=_d("tokenizer"),
            chat_template_sha256=_d("template"),
        ),
        adapter=None,
        serving=ServingIdentity(
            engine_sha256=_d("engine"),
            dtype="bf16",
            tensor_parallel=2,
        ),
        decoding_defaults=DecodingPolicy(
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=4096,
            stop_policy_sha256=_d("stop"),
        ),
        created_from_training_run_sha256=None,
    )
    harness = build_sealed(
        AgentHarnessProfile,
        harness_id="native-v1",
        harness_image_sha256=_d("harness-image"),
        control_flow_version=1,
        prompt_builder_sha256=_d("prompt"),
        context_compaction_policy_sha256=_d("compaction"),
        tool_schema_sha256=_d("tools"),
        tool_result_serialization_sha256=_d("tool-serialization"),
        skill_pack_sha256=_d("skills"),
        model_boundary=ModelBoundaryPolicy(protocol="openai-compatible-proxy"),
        limits=HarnessLimits(
            max_model_calls=32,
            max_tool_calls=128,
            max_generated_tokens=8192,
            wall_time_s=3600,
        ),
        hidden_chain_of_thought_capture=False,
    )
    sandbox = build_sealed(
        SandboxProfile,
        sandbox_id="rootless-v1",
        mode="agent-rootless",
        image_sha256=_d("sandbox-image"),
        rootless=True,
        privileged=False,
        filesystem=SandboxFilesystemPolicy(),
        network=SandboxNetworkPolicy(mode="disabled"),
        devices=SandboxDevicePolicy(),
        process=SandboxProcessPolicy(seccomp_policy_sha256=_d("seccomp")),
        snapshots=SandboxSnapshotPolicy(),
        enforcement="declarative-only",
        runtime_attestation_sha256=None,
        reward_authority="none",
    )
    cell = build_sealed(
        PolicyCell,
        policy_snapshot_sha256=policy.policy_snapshot_sha256,
        update_mode="none",
        external_policy_state_sha256=None,
        harness_profile_sha256=harness.harness_profile_sha256,
        skill_pack_sha256=harness.skill_pack_sha256,
        tool_policy_sha256=_d("tool-policy"),
        prompt_policy_sha256=harness.prompt_builder_sha256,
        compaction_policy_sha256=harness.context_compaction_policy_sha256,
        decoding_policy_sha256=_d("decoding"),
        feedback_visibility_policy_sha256=_d("feedback-visibility"),
        sandbox_profile_sha256=sandbox.sandbox_profile_sha256,
    )
    return policy, harness, sandbox, cell


def _trace(policy: PolicySnapshot) -> ModelBoundaryTrace:
    return build_sealed(
        ModelBoundaryTrace,
        model_call_id="call-1",
        logical_turn_id="turn-1",
        policy_snapshot_sha256=policy.policy_snapshot_sha256,
        behavior_policy_version=policy.policy_version,
        request_sha256=_d("request"),
        response_sha256=_d("response"),
        input_token_ids=[1, 2],
        output_token_ids=[3, 4, 5],
        output_token_roles=["assistant", "tool-serialization", "assistant"],
        trainable_mask=[True, False, True],
        rollout_logprobs=[-0.1, -0.2, -0.3],
        sampling=SamplingMetadata(
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=32,
            seed=7,
        ),
        history_before_sha256=_d("history-before"),
        history_after_sha256=_d("history-after"),
        compaction_applied=True,
        pre_compaction_history_sha256=_d("pre-compaction"),
    )


def _trajectory(identities) -> tuple[TrajectoryEnvelope, object]:
    policy, harness, sandbox, _ = identities
    trace = _trace(policy)
    trajectory = build_sealed(
        TrajectoryEnvelope,
        episode_id="episode-1",
        task_seal_sha256=_d("task"),
        policy_snapshot_sha256=policy.policy_snapshot_sha256,
        harness_profile_sha256=harness.harness_profile_sha256,
        sandbox_profile_sha256=sandbox.sandbox_profile_sha256,
        trajectory_fidelity="exact-model-boundary",
        harness_fidelity="native-exact",
        exact_token_ids_available=True,
        policy_gradient_eligible=True,
        traces=[trace],
        steps=[
            TrajectoryStep(
                step_id=0,
                branch_id="main",
                parent_step_id=None,
                policy_snapshot_sha256=policy.policy_snapshot_sha256,
                observation_sha256=_d("observation"),
                model_call_id=trace.model_call_id,
                workspace_before_sha256=_d("workspace-before"),
                workspace_after_sha256=_d("workspace-after"),
                phase="EDIT",
                wall_time_ms=12,
            )
        ],
        step_token_spans=[
            StepTokenSpan(
                step_id=0,
                model_call_id=trace.model_call_id,
                global_span=TokenRange(start=0, end=3),
                trainable_ranges=[TokenRange(start=0, end=1), TokenRange(start=2, end=3)],
                excluded_ranges=[
                    ExcludedTokenRange(
                        reason="tool-serialization",
                        span=TokenRange(start=1, end=2),
                    )
                ],
            )
        ],
        branches=[],
        allowed_uses=["policy-gradient", "external-policy", "qualitative-audit"],
    )
    fidelity = build_logprob_fidelity_report(
        trajectory_sha256=trajectory.trajectory_sha256,
        rollout_train_logprob_correlation=0.999,
        mean_abs_delta_logp=0.001,
        importance_ratio_quantiles={"p50": 1.0, "p99": 1.01},
        mask_alignment_rate=1.0,
        model_boundary_policy=harness.model_boundary,
    )
    return trajectory, fidelity


def _evidence() -> tuple[object, object]:
    trial = build_trial_seal(
        task_seal_sha256=_d("task"),
        draft_seal_sha256=_d("draft"),
        artifact_policy_sha256=_d("artifact-policy"),
        cache_policy_sha256=_d("cache"),
        capability_resolution_sha256=_d("capability"),
        runner_attestation_sha256=_d("runner"),
        candidate_artifact_manifest_sha256=_d("candidate"),
        build_environment_sha256=_d("build"),
        verifier_environment_sha256=_d("verifier-env"),
        meter_environment_sha256=_d("meter-env"),
        resource_lease_sha256=_d("lease"),
        benchmark_cell_sha256=_d("cell"),
        environment_sentinel_policy_sha256=_d("sentinel"),
        start_time=datetime(2026, 9, 3, tzinfo=UTC),
    )
    verifier_producer = EvidenceProducerIdentity(
        producer_id="verifier-v1",
        role="pristine-verifier",
        implementation_sha256=_d("verifier-impl"),
        image_sha256=_d("verifier-image"),
        configuration_sha256=_d("verifier-config"),
    )
    verifier_artifacts = [
        EvidenceArtifact(
            evidence_id=f"evidence://verifier/{name}",
            relative_path=f"verifier/{name}.json",
            media_type="application/json",
            sha256=_d("evidence-" + name),
            size_bytes=0,
            origin_trust="T2_PRISTINE_REEXECUTED",
            authority="VERIFIER",
            producer=verifier_producer,
        )
        for name in ("core", "co", "ri", "nb", "mp", "sl", "es")
    ]
    meter_artifact = EvidenceArtifact(
        evidence_id="evidence://meter/performance",
        relative_path="meter/performance.json",
        media_type="application/json",
        sha256=_d("meter-evidence"),
        size_bytes=0,
        origin_trust="T3_TRUSTED_METERED",
        authority="METER",
        producer=EvidenceProducerIdentity(
            producer_id="meter-v1",
            role="trusted-meter",
            implementation_sha256=_d("meter-impl"),
            image_sha256=_d("meter-image"),
            configuration_sha256=_d("meter-config"),
        ),
    )
    evidence = build_evidence_pack(
        trial_seal=trial,
        draft_seal_sha256=trial.draft_seal_sha256,
        candidate_artifact_manifest_sha256=trial.candidate_artifact_manifest_sha256,
        runner_attestation_sha256=trial.runner_attestation_sha256,
        verifier_result_sha256=_d("verifier-result"),
        measurement_set_sha256=_d("measurements"),
        score_input_sha256=_d("score-input"),
        artifacts=[*verifier_artifacts, meter_artifact],
    )
    return trial, evidence


def _episode(evidence, *, status: str = "VALID_PASS") -> EpisodeSeal:
    invalid = status in {"INFRA_INVALID", "BENCHMARK_DEFECT", "ROLLOUT_DEFECT", "CENSORED"}
    failed = status not in {"VALID_PASS", "JUDGE_UNRESOLVED"}
    return build_sealed(
        EpisodeSeal,
        episode_id="episode-1",
        task_seal_sha256=_d("task"),
        draft_seal_sha256=_d("draft"),
        policy_snapshot_sha256=_d("policy"),
        external_policy_state_sha256=None,
        policy_cell_sha256=_d("policy-cell"),
        harness_profile_sha256=_d("harness"),
        sandbox_profile_sha256=_d("sandbox"),
        capability_resolution_sha256=_d("capability"),
        resource_lease_sha256=_d("lease"),
        benchmark_cell_sha256=_d("cell"),
        trajectory_envelope_sha256=_d("trajectory"),
        candidate_artifact_manifest_sha256=_d("candidate"),
        evidence_pack_sha256=evidence.evidence_pack_sha256,
        status=status,
        failure_owner="infrastructure" if invalid else "candidate" if failed else "none",
        failure_code="RUNNER_LOST" if invalid else "CANDIDATE_FAILURE" if failed else None,
        training_mask=0 if invalid else 1,
        started_at=datetime(2026, 9, 3, tzinfo=UTC),
        sealed_at=datetime(2026, 9, 3, 0, 1, tzinfo=UTC),
    )


def _outcome(episode: EpisodeSeal, *, hard_pass: bool, invalid: bool = False):
    obligations = []
    for bucket in ("CO", "RI", "NB", "MP", "SL", "ES"):
        failed = (not hard_pass and not invalid and bucket == "MP") or (invalid and bucket == "ES")
        obligations.append(
            VerifierObligationOutcome(
                obligation_id=f"{bucket}-CORE",
                bucket=bucket,
                status="fail" if failed else "pass",
                evidence_refs=[f"evidence://verifier/{bucket.lower()}"],
                failure_code=f"{bucket}_FAILED" if failed else None,
            )
        )
    return build_sealed(
        VerifierOutcomePack,
        task_seal_sha256=episode.task_seal_sha256,
        episode_seal_sha256=episode.episode_seal_sha256,
        evidence_pack_sha256=episode.evidence_pack_sha256,
        benchmark_cell_sha256=episode.benchmark_cell_sha256,
        validity=VerifierValidity(
            environment_sentinel="fail" if invalid else "pass",
            artifact_boundary="pass",
            trial_seal="pass",
        ),
        obligations=obligations,
        infra_cert=1 if hard_pass and not invalid else 0,
        first_failure="ES-CORE" if invalid else None if hard_pass else "MP-CORE",
        failure_owner="infrastructure" if invalid else "none" if hard_pass else "candidate",
    )


def _reward_profile_and_qualification() -> tuple[RewardProfile, RewardQualification]:
    profile = build_sealed(
        RewardProfile,
        profile_id="verifier-anchor-v1",
        process_shaping_cap=0.1,
    )
    qualification = build_sealed(
        RewardQualification,
        task_seal_sha256=_d("task"),
        reward_profile_sha256=profile.reward_profile_sha256,
        source_authority_audit="pass",
        hard_outcome_monotonicity="pass",
        invalid_censoring_audit="pass",
        performance_noise_audit="pass",
        group_variance_feasibility="pass",
        feedback_leakage_audit="pass",
        reward_hack_mutation_audit="pass",
        cost_dominance_audit="pass",
        teacher_sign_anchor_audit="pass",
        training_replay_stability="pass",
        status="qualified",
    )
    return profile, qualification


def _event(episode_id: str, *, kind: str, authority: str, value: float = 1.0, cell=None):
    return build_sealed(
        RewardEvent,
        event_id=f"event-{kind}",
        fact_id=f"fact-{kind}",
        kind=kind,
        value=value,
        status="VALID",
        authority=authority,
        owner="trusted-producer",
        visibility="ADVANTAGE_ONLY",
        scope=RewardScope(episode_id=episode_id, comparison_cell_sha256=cell),
        evidence_refs=[]
        if authority in {"TRAINING_SHAPING", "NONE"}
        else [
            "evidence://meter/performance" if kind == "performance" else "evidence://verifier/core"
        ],
        producer_sha256=_d("producer" + kind),
    )


def _algorithm() -> AlgorithmProfile:
    return build_sealed(
        AlgorithmProfile,
        profile_id="dapo-v1",
        family="dapo",
        policy_granularity="token",
        advantage_granularity="episode",
        credit_overlay="verifier-step-token-v1",
        clipping=ClippingProfile(lower=0.2, upper=0.28),
        sampling=TrainingSamplingProfile(
            group_size=8,
            dynamic_replenishment=True,
            require_nonzero_valid_reward_variance=True,
        ),
        loss=TrainingLossProfile(normalization="valid-trainable-token"),
        overlong=OverlongPolicy(policy_budget_exceeded="soft-shape"),
        policy_staleness=PolicyStalenessProfile(max_versions=2, over_limit="reject"),
        llm_weights_update=True,
    )


def _training_run(algorithm: AlgorithmProfile) -> TrainingRunSeal:
    return build_sealed(
        TrainingRunSeal,
        run_id="train-1",
        initial_policy_snapshot_sha256=_d("behavior"),
        algorithm_profile_sha256=algorithm.algorithm_profile_sha256,
        sampler_profile_sha256=_d("sampler"),
        reward_profile_sha256=_d("reward-profile"),
        credit_profile_sha256=_d("credit-profile"),
        feedback_profile_sha256=_d("feedback-profile"),
        task_splits=TrainingSplits(
            train_set_sha256=_d("train"),
            dev_set_sha256=_d("dev"),
            heldout_set_sha256=_d("heldout"),
        ),
        infrastructure=TrainingInfrastructure(
            rollout_fabric_sha256=_d("rollout-fabric"),
            learner_fabric_sha256=_d("learner-fabric"),
            runner_pool_policy_sha256=_d("runner-pool"),
        ),
        budgets=TrainingBudgets(
            valid_episodes=8,
            total_episode_attempts=12,
            environment_gpu_hours=10,
            learner_gpu_hours=5,
            judge_gpu_hours=1,
            wall_time_s=3600,
        ),
        checkpoint_policy_sha256=_d("checkpoint"),
        stop_policy_sha256=_d("stop"),
    )


def test_policy_snapshot_binds_tokenizer_template_and_detects_tamper(identities) -> None:
    policy, _, _, _ = identities
    with pytest.raises(ValidationError, match="frozen"):
        policy.policy_version = 4
    changed = reseal(
        policy.model_copy(
            update={
                "base_model": policy.base_model.model_copy(
                    update={"tokenizer_sha256": _d("different-tokenizer")}
                )
            }
        )
    )
    assert changed.policy_snapshot_sha256 != policy.policy_snapshot_sha256
    tampered = policy.model_copy(update={"policy_version": 4})
    assert audit_sealed(tampered) == ["POLICYSNAPSHOT_DIGEST_MISMATCH"]


def test_adapter_identity_is_required_exactly_for_adapter_mode(identities) -> None:
    policy, _, _, _ = identities
    payload = policy.model_dump(mode="json")
    payload["update_mode"] = "adapter"
    payload["adapter"] = None
    with pytest.raises(ValidationError, match="adapter identity"):
        PolicySnapshot.model_validate(payload)
    payload["adapter"] = AdapterIdentity(
        kind="lora",
        weights_sha256=_d("lora"),
        config_sha256=_d("lora-config"),
    ).model_dump(mode="json")
    assert reseal(PolicySnapshot.model_validate(payload)).adapter is not None


def test_external_policy_state_is_frozen_inside_policy_cell(identities) -> None:
    policy, harness, sandbox, _ = identities
    policy = reseal(policy.model_copy(update={"update_mode": "external-state"}))
    state = build_sealed(
        ExternalPolicyState,
        state_id="bandit-1",
        kind="contextual-bandit",
        base_policy_snapshot_sha256=policy.policy_snapshot_sha256,
        feature_schema_sha256=_d("features"),
        action_taxonomy_sha256=_d("actions"),
        state_blob_sha256=_d("state"),
        update_count=12,
    )
    cell = build_sealed(
        PolicyCell,
        policy_snapshot_sha256=policy.policy_snapshot_sha256,
        update_mode="external-state",
        external_policy_state_sha256=state.external_policy_state_sha256,
        harness_profile_sha256=harness.harness_profile_sha256,
        skill_pack_sha256=harness.skill_pack_sha256,
        tool_policy_sha256=_d("tool-policy"),
        prompt_policy_sha256=harness.prompt_builder_sha256,
        compaction_policy_sha256=harness.context_compaction_policy_sha256,
        decoding_policy_sha256=_d("decoding"),
        feedback_visibility_policy_sha256=_d("feedback"),
        sandbox_profile_sha256=sandbox.sandbox_profile_sha256,
    )
    assert not audit_policy_cell_bindings(
        cell,
        policy=policy,
        harness=harness,
        sandbox=sandbox,
        external_state=state,
    )
    assert "POLICY_CELL_EXTERNAL_STATE_MISSING" in audit_policy_cell_bindings(
        cell,
        policy=policy,
        harness=harness,
        sandbox=sandbox,
    )


def test_model_boundary_mask_excludes_tool_tokens(identities) -> None:
    policy, _, _, _ = identities
    payload = _trace(policy).model_dump(mode="json")
    payload["trainable_mask"] = [True, True, True]
    with pytest.raises(ValidationError, match="assistant-owned"):
        ModelBoundaryTrace.model_validate(payload)


def test_exact_trajectory_and_logprob_gate(identities) -> None:
    policy, harness, sandbox, _ = identities
    trajectory, fidelity = _trajectory(identities)
    assert not audit_trajectory_bindings(
        trajectory,
        policy=policy,
        harness=harness,
        sandbox=sandbox,
        fidelity=fidelity,
    )
    failed = build_logprob_fidelity_report(
        trajectory_sha256=trajectory.trajectory_sha256,
        rollout_train_logprob_correlation=0.5,
        mean_abs_delta_logp=0.2,
        importance_ratio_quantiles={"p50": 1.0},
        mask_alignment_rate=0.9,
        model_boundary_policy=harness.model_boundary,
    )
    assert not failed.policy_gradient_eligible
    assert "TRAJECTORY_LOGPROB_FIDELITY_REJECTED" in audit_trajectory_bindings(
        trajectory,
        policy=policy,
        harness=harness,
        sandbox=sandbox,
        fidelity=failed,
    )


def test_trajectory_token_spans_must_reproduce_boundary_mask(identities) -> None:
    trajectory, _ = _trajectory(identities)
    payload = trajectory.model_dump(mode="json")
    payload["step_token_spans"][0]["trainable_ranges"] = [
        {"start": 0, "end": 2},
        {"start": 2, "end": 3},
    ]
    payload["step_token_spans"][0]["excluded_ranges"] = []
    with pytest.raises(ValidationError, match="model-boundary mask"):
        TrajectoryEnvelope.model_validate(payload)


def test_mixed_policy_trajectory_fails_closed(identities) -> None:
    trajectory, _ = _trajectory(identities)
    payload = trajectory.model_dump(mode="json")
    payload["steps"][0]["policy_snapshot_sha256"] = _d("other-policy")
    with pytest.raises(ValidationError, match="mixed-policy"):
        TrajectoryEnvelope.model_validate(payload)


def test_reconstructed_trajectory_cannot_claim_policy_gradient(identities) -> None:
    policy, harness, sandbox, _ = identities
    with pytest.raises(ValidationError, match="never policy-gradient"):
        build_sealed(
            TrajectoryEnvelope,
            episode_id="legacy",
            task_seal_sha256=_d("task"),
            policy_snapshot_sha256=policy.policy_snapshot_sha256,
            harness_profile_sha256=harness.harness_profile_sha256,
            sandbox_profile_sha256=sandbox.sandbox_profile_sha256,
            trajectory_fidelity="reconstructed",
            harness_fidelity="transcript-only",
            exact_token_ids_available=False,
            policy_gradient_eligible=True,
            traces=[],
            steps=[],
            step_token_spans=[],
            branches=[],
            allowed_uses=["policy-gradient"],
        )


def test_sandbox_profiles_forbid_privileged_agent_and_fake_attestation(identities) -> None:
    _, _, sandbox, _ = identities
    payload = sandbox.model_dump(mode="json")
    payload["privileged"] = True
    with pytest.raises(ValidationError, match="rootless and unprivileged"):
        SandboxProfile.model_validate(payload)
    payload = sandbox.model_dump(mode="json")
    payload["enforcement"] = "runtime-attested"
    payload["runtime_attestation_sha256"] = None
    with pytest.raises(ValidationError, match="requires exactly one attestation"):
        SandboxProfile.model_validate(payload)


def test_episode_state_machine_and_invalid_mask_are_structured() -> None:
    now = datetime(2026, 9, 3, tzinfo=UTC)
    events = [
        EpisodeEvent(stage="E0_REQUESTED", observed_at=now, detail_sha256=_d("e0")),
        EpisodeEvent(stage="E2_IDENTITIES_BOUND", observed_at=now, detail_sha256=_d("e2")),
    ]
    with pytest.raises(ValidationError, match="frozen state machine"):
        AgenticEpisode(
            episode_id="episode-1",
            rollout_request_sha256=_d("request"),
            pinned_policy_snapshot_sha256=_d("policy"),
            events=events,
            status="INFRA_INVALID",
            failure_owner="infrastructure",
            failure_code="RUNNER_LOST",
            training_mask=0,
        )
    with pytest.raises(ValidationError, match="training mask"):
        AgenticEpisode(
            episode_id="episode-1",
            rollout_request_sha256=_d("request"),
            pinned_policy_snapshot_sha256=_d("policy"),
            events=events[:1],
            status="INFRA_INVALID",
            failure_owner="infrastructure",
            failure_code="RUNNER_LOST",
            training_mask=1,
        )


def test_security_rejection_mask_is_explicitly_configurable() -> None:
    event = EpisodeEvent(
        stage="E0_REQUESTED",
        observed_at=datetime(2026, 9, 3, tzinfo=UTC),
        detail_sha256=_d("security-rejection"),
    )
    for mask in (0, 1):
        episode = AgenticEpisode(
            episode_id=f"security-{mask}",
            rollout_request_sha256=_d("request"),
            pinned_policy_snapshot_sha256=_d("policy"),
            events=[event],
            status="SECURITY_REJECTED",
            failure_owner="policy",
            failure_code="BENCHMARK_TAMPER_ATTEMPT",
            training_mask=mask,
        )
        assert episode.training_mask == mask


def test_invalid_infrastructure_reward_is_masked_not_negative() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence, status="INFRA_INVALID")
    outcome = _outcome(episode, hard_pass=False, invalid=True)
    profile, qualification = _reward_profile_and_qualification()
    reward = compile_reward_pack(
        episode_seal=episode,
        outcome=outcome,
        evidence_pack=evidence,
        trial_seal=trial,
        qualification=qualification,
        profile=profile,
        events=[],
        credit_profile_sha256=_d("credit-profile"),
    )
    assert reward.validity == "invalid"
    assert reward.training_mask == 0
    assert reward.anchor.scalar_value is None


def test_hard_fail_cannot_be_crossed_by_process_shaping() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence, status="VALID_FAIL").model_copy(
        update={"failure_owner": "candidate", "failure_code": "MP_FAILED"}
    )
    episode = reseal(episode)
    outcome = _outcome(episode, hard_pass=False)
    profile, qualification = _reward_profile_and_qualification()
    reward = compile_reward_pack(
        episode_seal=episode,
        outcome=outcome,
        evidence_pack=evidence,
        trial_seal=trial,
        qualification=qualification,
        profile=profile,
        events=[
            _event(
                episode.episode_id,
                kind="process-shaping",
                authority="TRAINING_SHAPING",
                value=100,
            )
        ],
        credit_profile_sha256=_d("credit-profile"),
    )
    assert reward.anchor.scalar_band == "hard-fail"
    assert reward.anchor.scalar_value <= profile.hard_fail_ceiling


def test_revoked_valid_reward_is_masked_without_rewriting_anchor() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence)
    outcome = _outcome(episode, hard_pass=True)
    profile, qualification = _reward_profile_and_qualification()
    reward = compile_reward_pack(
        episode_seal=episode,
        outcome=outcome,
        evidence_pack=evidence,
        trial_seal=trial,
        qualification=qualification,
        profile=profile,
        events=[],
        credit_profile_sha256=_d("credit-profile"),
    )
    revoked = reseal(
        reward.model_copy(
            update={
                "training_mask": 0,
                "revoked": True,
                "revocation_reason": "TASK_SEAL_REVOKED",
            }
        )
    )
    assert revoked.anchor == reward.anchor
    assert revoked.training_mask == 0
    assert revoked.revoked


def test_episode_outcome_seal_closes_the_digest_dag() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence)
    outcome = _outcome(episode, hard_pass=True)
    profile, qualification = _reward_profile_and_qualification()
    reward = compile_reward_pack(
        episode_seal=episode,
        outcome=outcome,
        evidence_pack=evidence,
        trial_seal=trial,
        qualification=qualification,
        profile=profile,
        events=[],
        credit_profile_sha256=_d("credit-profile"),
    )
    sealed = build_episode_outcome_seal(
        episode_seal=episode,
        reward_pack=reward,
    )
    assert sealed.episode_seal_sha256 == episode.episode_seal_sha256
    assert sealed.reward_pack_sha256 == reward.reward_pack_sha256
    assert not audit_episode_outcome_seal(
        sealed,
        episode_seal=episode,
        reward_pack=reward,
    )


def test_candidate_reward_spoof_has_no_authority() -> None:
    with pytest.raises(ValidationError, match="no reward authority"):
        _event(
            "episode-1",
            kind="candidate-self-report",
            authority="OFFICIAL_HARD",
        )
    event = _event("episode-1", kind="candidate-self-report", authority="NONE")
    assert event.authority == "NONE"


def test_cross_cell_performance_reward_is_rejected() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence)
    outcome = _outcome(episode, hard_pass=True)
    profile, qualification = _reward_profile_and_qualification()
    event = _event(
        episode.episode_id,
        kind="performance",
        authority="OFFICIAL_COMPONENT",
        cell=_d("another-cell"),
    )
    with pytest.raises(ValueError, match="CROSS_CELL"):
        compile_reward_pack(
            episode_seal=episode,
            outcome=outcome,
            evidence_pack=evidence,
            trial_seal=trial,
            qualification=qualification,
            profile=profile,
            events=[event],
            credit_profile_sha256=_d("credit-profile"),
        )


def test_reward_compiler_rejects_outcome_from_another_cell() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence)
    outcome = reseal(
        _outcome(episode, hard_pass=True).model_copy(
            update={"benchmark_cell_sha256": _d("another-cell")}
        )
    )
    profile, qualification = _reward_profile_and_qualification()
    with pytest.raises(ValueError, match="REWARD_OUTCOME_CELL_MISMATCH"):
        compile_reward_pack(
            episode_seal=episode,
            outcome=outcome,
            evidence_pack=evidence,
            trial_seal=trial,
            qualification=qualification,
            profile=profile,
            events=[],
            credit_profile_sha256=_d("credit-profile"),
        )


def test_reward_event_evidence_must_resolve_with_matching_authority() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence)
    outcome = _outcome(episode, hard_pass=True)
    profile, qualification = _reward_profile_and_qualification()
    event = _event(
        episode.episode_id,
        kind="performance",
        authority="OFFICIAL_COMPONENT",
        cell=episode.benchmark_cell_sha256,
    )
    unresolved = reseal(event.model_copy(update={"evidence_refs": ["evidence://meter/missing"]}))
    with pytest.raises(ValueError, match="EVIDENCE_REF_UNRESOLVED"):
        compile_reward_pack(
            episode_seal=episode,
            outcome=outcome,
            evidence_pack=evidence,
            trial_seal=trial,
            qualification=qualification,
            profile=profile,
            events=[unresolved],
            credit_profile_sha256=_d("credit-profile"),
        )
    wrong_authority = reseal(
        event.model_copy(update={"evidence_refs": ["evidence://verifier/core"]})
    )
    with pytest.raises(ValueError, match="EVIDENCE_AUTHORITY_MISMATCH"):
        compile_reward_pack(
            episode_seal=episode,
            outcome=outcome,
            evidence_pack=evidence,
            trial_seal=trial,
            qualification=qualification,
            profile=profile,
            events=[wrong_authority],
            credit_profile_sha256=_d("credit-profile"),
        )


def test_duplicate_reward_fact_is_rejected() -> None:
    trial, evidence = _evidence()
    episode = _episode(evidence)
    outcome = _outcome(episode, hard_pass=True)
    profile, qualification = _reward_profile_and_qualification()
    event = _event(episode.episode_id, kind="correctness", authority="OFFICIAL_HARD")
    with pytest.raises(ValueError, match="DUPLICATE_FACT"):
        compile_reward_pack(
            episode_seal=episode,
            outcome=outcome,
            evidence_pack=evidence,
            trial_seal=trial,
            qualification=qualification,
            profile=profile,
            events=[event, event],
            credit_profile_sha256=_d("credit-profile"),
        )


def test_feedback_leakage_and_heldout_are_blocked() -> None:
    leaked = build_feedback_pack(
        task_seal_sha256=_d("task"),
        episode_seal_sha256=_d("episode"),
        feedback_profile_sha256=_d("feedback"),
        visibility="TEACHER_VISIBLE_REDACTED",
        task_split="train",
        items=[FeedbackItem(kind="obligation-failure", public_summary="read tests/hidden/x")],
    )
    assert leaked.leakage_scan == "blocked"
    assert not leaked.teacher_eligible
    heldout = build_feedback_pack(
        task_seal_sha256=_d("task"),
        episode_seal_sha256=_d("episode"),
        feedback_profile_sha256=_d("feedback"),
        visibility="TEACHER_VISIBLE_REDACTED",
        task_split="heldout",
        items=[FeedbackItem(kind="obligation-failure", public_summary="bounded failure")],
    )
    assert heldout.leakage_scan == "pass"
    assert not heldout.teacher_eligible
    advantage_only = build_feedback_pack(
        task_seal_sha256=_d("task"),
        episode_seal_sha256=_d("episode"),
        feedback_profile_sha256=_d("feedback"),
        visibility="ADVANTAGE_ONLY",
        task_split="train",
        items=[],
    )
    assert not advantage_only.teacher_eligible


def test_teacher_modulation_preserves_verifier_sign() -> None:
    negative = sign_preserving_modulation(
        span=TokenRange(start=0, end=2),
        verifier_advantage=-1,
        teacher_signal=100,
        threshold=0.1,
        strength=10,
        temperature=1,
        minimum_multiplier=0.2,
        maximum_multiplier=2,
    )
    assert negative.modulated_advantage < 0
    assert 0.2 <= negative.multiplier <= 2


def test_detection_step_cannot_receive_causal_blame() -> None:
    with pytest.raises(ValidationError, match="zero causal credit"):
        CreditStepAssignment(
            step_id=2,
            role="detection-only",
            weight=0.5,
            reason_code="test-observed-failure",
        )
    credit = build_sealed(
        CreditAssignmentMap,
        profile_id="step-v1",
        episode_id="episode-1",
        anchor_reward_event_ids=["event-hard"],
        step_assignments=[
            CreditStepAssignment(
                step_id=1,
                role="pivotal-suspect",
                weight=0.6,
                reason_code="introduced-change",
            )
        ],
        token_modulations=[],
        unresolved_credit_mass=0.4,
    )
    assert credit.unresolved_credit_mass == 0.4


def test_dapo_requires_exact_token_normalization() -> None:
    algorithm = _algorithm()
    payload = algorithm.model_dump(mode="json")
    payload["loss"]["normalization"] = "valid-sequence"
    with pytest.raises(ValidationError, match="DAPO"):
        AlgorithmProfile.model_validate(payload)


def test_batch_rejects_stale_unknown_or_revoked_experience() -> None:
    algorithm = _algorithm()
    training_run = _training_run(algorithm)
    member = RLBatchMember(
        episode_outcome_sha256=_d("episode-outcome"),
        episode_seal_sha256=_d("episode-seal"),
        trajectory_sha256=_d("trajectory"),
        reward_pack_sha256=_d("reward"),
        task_seal_sha256=_d("task"),
        behavior_policy_snapshot_sha256=_d("behavior"),
        benchmark_cell_sha256=_d("cell"),
        episode_status="VALID_PASS",
        training_mask=1,
        policy_gradient_eligible=True,
        valid_token_count=3,
        valid_step_count=1,
        policy_lag=3,
    )
    batch = build_sealed(
        RLBatchManifest,
        batch_id="batch-1",
        training_run_seal_sha256=training_run.training_run_seal_sha256,
        target_policy_snapshot_sha256=_d("target"),
        proximal_policy_snapshot_sha256=_d("proximal"),
        algorithm_profile_sha256=algorithm.algorithm_profile_sha256,
        members=[member],
        group_manifest_sha256s=[_d("group")],
        valid_token_count=3,
        valid_step_count=1,
        policy_lag_distribution={"max": 3},
        sampler_selection_log_sha256=_d("sampler-log"),
    )
    failures = validate_rl_batch(
        batch,
        training_run=training_run,
        algorithm=algorithm,
        known_policy_snapshots={_d("target"), _d("proximal")},
        revoked_reward_packs={member.reward_pack_sha256},
    )
    assert any(item.startswith("UNKNOWN_BEHAVIOR_POLICY") for item in failures)
    assert any(item.startswith("STALENESS_LIMIT_EXCEEDED") for item in failures)
    assert any(item.startswith("REVOKED_REWARD_PACK") for item in failures)


def test_batch_applies_frozen_security_rejection_policy() -> None:
    algorithm = _algorithm()
    training_run = _training_run(algorithm)
    member = RLBatchMember(
        episode_outcome_sha256=_d("security-outcome"),
        episode_seal_sha256=_d("security-seal"),
        trajectory_sha256=_d("security-trajectory"),
        reward_pack_sha256=_d("security-reward"),
        task_seal_sha256=_d("task"),
        behavior_policy_snapshot_sha256=_d("behavior"),
        benchmark_cell_sha256=_d("cell"),
        episode_status="SECURITY_REJECTED",
        training_mask=1,
        policy_gradient_eligible=True,
        valid_token_count=3,
        valid_step_count=1,
        policy_lag=0,
    )
    batch = build_sealed(
        RLBatchManifest,
        batch_id="security-batch",
        training_run_seal_sha256=training_run.training_run_seal_sha256,
        target_policy_snapshot_sha256=_d("target"),
        proximal_policy_snapshot_sha256=_d("proximal"),
        algorithm_profile_sha256=algorithm.algorithm_profile_sha256,
        members=[member],
        group_manifest_sha256s=[_d("group")],
        valid_token_count=3,
        valid_step_count=1,
        policy_lag_distribution={"max": 0},
        sampler_selection_log_sha256=_d("sampler-log"),
    )
    failures = validate_rl_batch(
        batch,
        training_run=training_run,
        algorithm=algorithm,
        known_policy_snapshots={_d("target"), _d("proximal"), _d("behavior")},
    )
    assert any(item.startswith("SECURITY_REJECTION_MASK_POLICY_MISMATCH") for item in failures)


def test_partial_gang_allocation_never_starts() -> None:
    allocation = GangDeviceAllocation(rank=0, device_id="GPU-0", node_id="node-a", numa_node=0)
    with pytest.raises(ValidationError, match="complete atomic allocation"):
        build_sealed(
            GangLeaseRecord,
            lease_id="lease-1",
            requested_gpu_count=2,
            status="active",
            allocations=[allocation],
            topology_sha256=_d("topology"),
            workload_started=True,
        )
    blocked = build_sealed(
        GangLeaseRecord,
        lease_id="lease-2",
        requested_gpu_count=2,
        status="capacity-unavailable",
        allocations=[allocation],
        topology_sha256=_d("topology"),
        workload_started=False,
    )
    assert blocked.status == "capacity-unavailable"


def test_official_fabric_rejects_gpu_pool_overlap() -> None:
    pools = [
        FabricPool(
            pool_id="policy",
            kind="policy",
            worker_profile_sha256=_d("policy-worker"),
            device_ids=["GPU-0"],
            node_ids=["node-a"],
        ),
        FabricPool(
            pool_id="environment",
            kind="environment",
            worker_profile_sha256=_d("environment-worker"),
            device_ids=["GPU-0"],
            node_ids=["node-a"],
        ),
    ]
    with pytest.raises(ValidationError, match="cannot overlap"):
        build_sealed(
            RolloutFabricProfile,
            fabric_id="fabric-1",
            pools=pools,
            backpressure=BackpressureProfile(
                rollout_admission_limit=8,
                sandbox_concurrency_limit=2,
                environment_gpu_seconds_limit=3600,
                trajectory_buffer_high_water=16,
                reward_queue_limit=16,
                maximum_policy_lag=2,
            ),
            measurement_isolation="MI1",
            topology_sha256=_d("topology"),
        )


def test_runtime_capability_report_fails_closed_without_external_enforcement() -> None:
    report = build_runtime_capability_report(
        gpu_count=2,
        gpu_topology_attested=False,
        rootless_sandbox_enforced=False,
        exact_token_gateway_available=False,
        hosted_policy_exact_tokens_available=False,
        trainer_adapter_available=False,
        distributed_gang_enforced=False,
        observed_at=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert not report.production_ready
    assert "ROOTLESS_SANDBOX_NOT_ENFORCED" in report.unavailable_reasons
    assert "TRAINER_ADAPTER_UNAVAILABLE" in report.unavailable_reasons
    payload = report.model_dump(mode="json")
    payload["unavailable_reasons"] = []
    with pytest.raises(ValidationError, match="reasons disagree"):
        type(report).model_validate(payload)


def test_legacy_migration_never_fabricates_tokens_or_reward(tmp_path: Path) -> None:
    root = tmp_path / "training-bulk"
    group = root / "groups" / "group-0000"
    group.mkdir(parents=True)
    artifacts = {
        "input-lock.json": {
            "profile": "training",
            "cases": [
                {
                    "case_id": "project-pr-1",
                    "project": "project",
                    "repository": "org/project",
                    "pull_number": 1,
                    "acquisition_status": "acquired",
                },
                {
                    "case_id": "project-pr-2",
                    "project": "project",
                    "repository": "org/project",
                    "pull_number": 2,
                    "acquisition_status": "invalid",
                },
            ],
        },
        "exact-head-evidence.json": {
            "records": [
                {"case_id": "project-pr-1", "status": "completed"},
                {"case_id": "project-pr-2", "status": "prewarm_failed"},
            ]
        },
        "judgment-locks.json": {
            "locks": [
                {
                    "material": {
                        "case_id": "project-pr-1",
                        "decision": "accept",
                    }
                },
                {
                    "material": {
                        "case_id": "project-pr-2",
                        "decision": "unresolved",
                    }
                },
            ]
        },
        "outcome-reveal.json": {
            "cases": [
                {
                    "case_id": "project-pr-1",
                    "machine_decision": "accept",
                    "oracle_decision": "accept",
                },
                {
                    "case_id": "project-pr-2",
                    "machine_decision": "unresolved",
                    "oracle_decision": "unresolved",
                },
            ]
        },
        "oracle-audit.json": {"cases": []},
    }
    for name, value in artifacts.items():
        (group / name).write_text(json.dumps(value), encoding="utf-8")
    manifest = build_legacy_experience_manifest(
        [root],
        manifest_id="legacy-test",
        generated_at=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert manifest.attempted_records == 2
    assert manifest.valid_records == 1
    assert manifest.invalid_records == 1
    assert manifest.policy_gradient_eligible_records == 0
    assert manifest.fabricated_token_records == 0
    assert manifest.qualified_reward_records == 0
    assert all(item.reward_pack_sha256 is None for item in manifest.records)
    assert all(item.trajectory_fidelity == "reconstructed" for item in manifest.records)
    assert manifest.records[1].experience_status == "invalid"
    assert manifest.records[1].invalid_reason == "LEGACY_ACQUISITION_INVALID"


def test_v06_schemas_are_strict_and_registered() -> None:
    schemas = schema_documents()
    assert sum("v0.6" in name for name in schemas) == 29
    assert "policy-snapshot-v0.6.schema.json" in schemas
    assert "legacy-experience-manifest-v0.6.schema.json" in schemas
    assert schemas["policy-snapshot-v0.6.schema.json"]["additionalProperties"] is False


def test_rl_policy_cli_validates_digest(tmp_path: Path, identities) -> None:
    policy, _, _, _ = identities
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.model_dump(mode="json")), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["rl", "policy", "validate", str(path)])
    assert result.exit_code == 0, result.output
    payload = policy.model_dump(mode="json")
    payload["policy_version"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = runner.invoke(app, ["rl", "policy", "validate", str(path)])
    assert result.exit_code == 4


def test_rl_fabric_preflight_reports_unavailable_instead_of_claiming_support(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capabilities.json"
    result = CliRunner().invoke(
        app,
        ["rl", "fabric", "preflight", "--gpu-count", "2", "--output", str(output)],
    )
    assert result.exit_code == 5
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["production_ready"] is False
    assert "ROOTLESS_SANDBOX_NOT_ENFORCED" in payload["unavailable_reasons"]
