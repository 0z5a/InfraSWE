from __future__ import annotations

import math
from datetime import datetime
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.models.draft import Digest

EpisodeStatus = Literal[
    "VALID_PASS",
    "VALID_FAIL",
    "POLICY_BUDGET_EXCEEDED",
    "INFRA_INVALID",
    "BENCHMARK_DEFECT",
    "ROLLOUT_DEFECT",
    "JUDGE_UNRESOLVED",
    "STALE_REJECTED",
    "SECURITY_REJECTED",
    "CENSORED",
]
FailureOwner = Literal[
    "candidate",
    "policy",
    "task",
    "benchmark",
    "infrastructure",
    "sandbox",
    "rollout",
    "judge",
    "teacher",
    "trainer",
    "none",
]
EpisodeStage = Literal[
    "E0_REQUESTED",
    "E1_ADMISSION_VALIDATED",
    "E2_IDENTITIES_BOUND",
    "E3_CAPABILITY_RESOLVED",
    "E4_RESOURCE_LEASED",
    "E5_SANDBOX_READY",
    "E6_POLICY_ENDPOINT_BOUND",
    "E7_INTERACTING",
    "E7B_SNAPSHOT_BRANCH",
    "E8_AGENT_TERMINATED",
    "E9_ARTIFACT_FROZEN",
    "E10_PRISTINE_APPLY_BUILD",
    "E11_EXECUTABLE_VERIFY",
    "E12_TRUSTED_MEASURE",
    "E13_OPTIONAL_JUDGE_FEEDBACK",
    "E14_REWARD_COMPILED",
    "E15_EPISODE_SEALED",
    "E16_RESOURCE_RELEASED_AUDITED",
]

EPISODE_STAGE_ORDER = {
    stage: index
    for index, stage in enumerate(
        (
            "E0_REQUESTED",
            "E1_ADMISSION_VALIDATED",
            "E2_IDENTITIES_BOUND",
            "E3_CAPABILITY_RESOLVED",
            "E4_RESOURCE_LEASED",
            "E5_SANDBOX_READY",
            "E6_POLICY_ENDPOINT_BOUND",
            "E7_INTERACTING",
            "E7B_SNAPSHOT_BRANCH",
            "E8_AGENT_TERMINATED",
            "E9_ARTIFACT_FROZEN",
            "E10_PRISTINE_APPLY_BUILD",
            "E11_EXECUTABLE_VERIFY",
            "E12_TRUSTED_MEASURE",
            "E13_OPTIONAL_JUDGE_FEEDBACK",
            "E14_REWARD_COMPILED",
            "E15_EPISODE_SEALED",
            "E16_RESOURCE_RELEASED_AUDITED",
        )
    )
}
EPISODE_STAGE_TRANSITIONS: dict[str, set[str]] = {
    "E0_REQUESTED": {"E1_ADMISSION_VALIDATED"},
    "E1_ADMISSION_VALIDATED": {"E2_IDENTITIES_BOUND"},
    "E2_IDENTITIES_BOUND": {"E3_CAPABILITY_RESOLVED"},
    "E3_CAPABILITY_RESOLVED": {"E4_RESOURCE_LEASED"},
    "E4_RESOURCE_LEASED": {"E5_SANDBOX_READY"},
    "E5_SANDBOX_READY": {"E6_POLICY_ENDPOINT_BOUND"},
    "E6_POLICY_ENDPOINT_BOUND": {"E7_INTERACTING"},
    "E7_INTERACTING": {"E7B_SNAPSHOT_BRANCH", "E8_AGENT_TERMINATED"},
    "E7B_SNAPSHOT_BRANCH": {"E7B_SNAPSHOT_BRANCH", "E8_AGENT_TERMINATED"},
    "E8_AGENT_TERMINATED": {"E9_ARTIFACT_FROZEN"},
    "E9_ARTIFACT_FROZEN": {"E10_PRISTINE_APPLY_BUILD"},
    "E10_PRISTINE_APPLY_BUILD": {"E11_EXECUTABLE_VERIFY"},
    "E11_EXECUTABLE_VERIFY": {"E12_TRUSTED_MEASURE"},
    "E12_TRUSTED_MEASURE": {"E13_OPTIONAL_JUDGE_FEEDBACK", "E14_REWARD_COMPILED"},
    "E13_OPTIONAL_JUDGE_FEEDBACK": {"E14_REWARD_COMPILED"},
    "E14_REWARD_COMPILED": {"E15_EPISODE_SEALED"},
    "E15_EPISODE_SEALED": {"E16_RESOURCE_RELEASED_AUDITED"},
    "E16_RESOURCE_RELEASED_AUDITED": set(),
}


class AgenticModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class ModelArtifactIdentity(AgenticModel):
    family: str
    weights_sha256: Digest
    config_sha256: Digest
    tokenizer_sha256: Digest
    chat_template_sha256: Digest


class AdapterIdentity(AgenticModel):
    kind: Literal["lora", "adapter", "prefix", "policy-head"]
    weights_sha256: Digest
    config_sha256: Digest


class ServingIdentity(AgenticModel):
    engine_sha256: Digest
    quantization_sha256: Digest | None = None
    dtype: str
    tensor_parallel: int = Field(default=1, ge=1)
    expert_parallel: int = Field(default=1, ge=1)
    routing_replay_policy_id: str | None = None
    router_dtype: str | None = None
    capacity_policy_id: str | None = None


class DecodingPolicy(AgenticModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(ge=1)
    stop_policy_sha256: Digest


class PolicySnapshot(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    policy_id: str
    policy_version: int = Field(ge=0)
    update_mode: Literal["none", "external-state", "adapter", "full-parameter"]
    base_model: ModelArtifactIdentity
    adapter: AdapterIdentity | None = None
    serving: ServingIdentity
    decoding_defaults: DecodingPolicy
    created_from_training_run_sha256: Digest | None = None
    policy_snapshot_sha256: Digest

    @model_validator(mode="after")
    def adapter_matches_update_mode(self) -> PolicySnapshot:
        if (self.update_mode == "adapter") != (self.adapter is not None):
            raise ValueError("adapter identity is required exactly for adapter update mode")
        return self


class ExternalPolicyState(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    state_id: str
    kind: Literal[
        "contextual-bandit",
        "mcts",
        "tactic-value",
        "precedent-memory",
        "reranker",
    ]
    base_policy_snapshot_sha256: Digest
    feature_schema_sha256: Digest
    action_taxonomy_sha256: Digest
    state_blob_sha256: Digest
    update_count: int = Field(ge=0)
    external_policy_state_sha256: Digest


class ModelBoundaryPolicy(AgenticModel):
    protocol: str
    capture_raw_token_ids: Literal[True] = True
    capture_sampling_metadata: Literal[True] = True
    capture_rollout_logprobs: Literal["required", "optional"] = "optional"
    trainer_recompute_required: Literal[True] = True
    minimum_logprob_correlation: float = Field(default=0.99, ge=-1, le=1)
    maximum_mean_abs_delta_logp: float = Field(default=0.05, ge=0)
    minimum_mask_alignment_rate: float = Field(default=1.0, ge=0, le=1)


class HarnessLimits(AgenticModel):
    max_model_calls: int = Field(ge=1)
    max_tool_calls: int = Field(ge=0)
    max_generated_tokens: int = Field(ge=1)
    wall_time_s: int = Field(ge=1)


class AgentHarnessProfile(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    harness_id: str
    harness_image_sha256: Digest
    control_flow_version: int = Field(ge=1)
    prompt_builder_sha256: Digest
    context_compaction_policy_sha256: Digest
    tool_schema_sha256: Digest
    tool_result_serialization_sha256: Digest
    skill_pack_sha256: Digest
    model_boundary: ModelBoundaryPolicy
    limits: HarnessLimits
    hidden_chain_of_thought_capture: Literal[False] = False
    harness_profile_sha256: Digest


class PolicyCell(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    policy_snapshot_sha256: Digest
    update_mode: Literal["none", "external-state", "adapter", "full-parameter"]
    external_policy_state_sha256: Digest | None = None
    harness_profile_sha256: Digest
    skill_pack_sha256: Digest
    tool_policy_sha256: Digest
    prompt_policy_sha256: Digest
    compaction_policy_sha256: Digest
    decoding_policy_sha256: Digest
    feedback_visibility_policy_sha256: Digest
    sandbox_profile_sha256: Digest
    policy_cell_sha256: Digest

    @model_validator(mode="after")
    def external_state_is_part_of_policy_identity(self) -> PolicyCell:
        if (self.update_mode == "external-state") != (
            self.external_policy_state_sha256 is not None
        ):
            raise ValueError("external-state mode requires exactly one frozen external state")
        return self


TokenRole = Literal["assistant", "tool-serialization", "system", "user", "environment", "padding"]


class SamplingMetadata(AgenticModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(ge=1)
    seed: int
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelBoundaryTrace(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    model_call_id: str
    logical_turn_id: str
    policy_snapshot_sha256: Digest
    behavior_policy_version: int = Field(ge=0)
    request_sha256: Digest
    response_sha256: Digest
    input_token_ids: list[int]
    output_token_ids: list[int]
    output_token_roles: list[TokenRole]
    trainable_mask: list[bool]
    rollout_logprobs: list[float] | None = None
    sampling: SamplingMetadata
    history_before_sha256: Digest
    history_after_sha256: Digest
    compaction_applied: bool = False
    pre_compaction_history_sha256: Digest | None = None
    trace_sha256: Digest

    @model_validator(mode="after")
    def exact_tokens_and_masks_align(self) -> ModelBoundaryTrace:
        length = len(self.output_token_ids)
        if length == 0:
            raise ValueError("model boundary trace requires output token ids")
        if len(self.output_token_roles) != length or len(self.trainable_mask) != length:
            raise ValueError("output tokens, roles, and trainable mask must align exactly")
        if self.rollout_logprobs is not None and len(self.rollout_logprobs) != length:
            raise ValueError("rollout logprobs must align with output token ids")
        if any(token < 0 for token in [*self.input_token_ids, *self.output_token_ids]):
            raise ValueError("token ids must be non-negative")
        if any(
            mask and role != "assistant"
            for mask, role in zip(self.trainable_mask, self.output_token_roles, strict=True)
        ):
            raise ValueError("only assistant-owned tokens may enter the trainable mask")
        if self.compaction_applied and self.pre_compaction_history_sha256 is None:
            raise ValueError("compaction requires the pre-compaction history digest")
        if not self.compaction_applied and self.pre_compaction_history_sha256 is not None:
            raise ValueError("non-compacted calls cannot claim a pre-compaction digest")
        return self


class LogprobFidelityReport(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    trajectory_sha256: Digest
    rollout_train_logprob_correlation: float = Field(ge=-1, le=1)
    mean_abs_delta_logp: float = Field(ge=0)
    importance_ratio_quantiles: dict[str, float]
    mask_alignment_rate: float = Field(ge=0, le=1)
    minimum_correlation: float = Field(ge=-1, le=1)
    maximum_mean_abs_delta_logp: float = Field(ge=0)
    minimum_mask_alignment_rate: float = Field(ge=0, le=1)
    policy_gradient_eligible: bool
    fidelity_report_sha256: Digest

    @model_validator(mode="after")
    def eligibility_matches_thresholds(self) -> LogprobFidelityReport:
        expected = (
            self.rollout_train_logprob_correlation >= self.minimum_correlation
            and self.mean_abs_delta_logp <= self.maximum_mean_abs_delta_logp
            and self.mask_alignment_rate >= self.minimum_mask_alignment_rate
        )
        if self.policy_gradient_eligible != expected:
            raise ValueError("logprob fidelity eligibility disagrees with frozen thresholds")
        return self


class SandboxFilesystemPolicy(AgenticModel):
    base: Literal["read-only"] = "read-only"
    workspace: Literal["copy-on-write"] = "copy-on-write"
    verifier_assets: Literal["not-mounted"] = "not-mounted"
    host_mounts: Literal["forbidden"] = "forbidden"
    shared_cache_write: Literal["forbidden"] = "forbidden"


class SandboxNetworkPolicy(AgenticModel):
    mode: Literal["disabled", "allowlist"]
    destinations: list[str] = Field(default_factory=list)
    public_web: Literal["forbidden"] = "forbidden"

    @model_validator(mode="after")
    def destinations_match_mode(self) -> SandboxNetworkPolicy:
        if self.mode == "disabled" and self.destinations:
            raise ValueError("disabled sandbox networking cannot list destinations")
        if self.mode == "allowlist" and not self.destinations:
            raise ValueError("allowlist networking requires a non-empty destination list")
        return self


class SandboxDevicePolicy(AgenticModel):
    visibility: Literal["exact-lease"] = "exact-lease"
    device_control: Literal["restricted"] = "restricted"
    unallocated_peer_access: Literal["forbidden"] = "forbidden"


class SandboxProcessPolicy(AgenticModel):
    pid_namespace: Literal["isolated"] = "isolated"
    seccomp_policy_sha256: Digest
    capability_policy: Literal["minimal"] = "minimal"


class SandboxSnapshotPolicy(AgenticModel):
    mode: Literal["cow-filesystem-plus-logical-state"] = "cow-filesystem-plus-logical-state"
    live_process_memory: Literal["forbidden"] = "forbidden"
    host_sockets: Literal["forbidden"] = "forbidden"
    gpu_context_snapshot: Literal["forbidden"] = "forbidden"
    verifier_private_files: Literal["forbidden"] = "forbidden"
    secret_credentials: Literal["forbidden"] = "forbidden"


class SandboxProfile(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    sandbox_id: str
    mode: Literal["agent-rootless", "replay-rootless", "pristine", "shared-legacy"]
    image_sha256: Digest
    rootless: bool
    privileged: bool
    filesystem: SandboxFilesystemPolicy
    network: SandboxNetworkPolicy
    devices: SandboxDevicePolicy
    process: SandboxProcessPolicy
    snapshots: SandboxSnapshotPolicy
    enforcement: Literal["declarative-only", "runtime-attested"]
    runtime_attestation_sha256: Digest | None = None
    reward_authority: Literal["none", "diagnostic", "official-pristine"]
    sandbox_profile_sha256: Digest

    @model_validator(mode="after")
    def security_claims_are_fail_closed(self) -> SandboxProfile:
        if self.mode in {"agent-rootless", "replay-rootless"} and (
            not self.rootless or self.privileged
        ):
            raise ValueError("agent and replay sandboxes must be rootless and unprivileged")
        if self.mode == "shared-legacy" and self.reward_authority != "diagnostic":
            raise ValueError("shared legacy sandboxes have diagnostic reward authority only")
        if self.mode != "pristine" and self.reward_authority == "official-pristine":
            raise ValueError("only a pristine sandbox can claim official pristine authority")
        if (self.enforcement == "runtime-attested") != (
            self.runtime_attestation_sha256 is not None
        ):
            raise ValueError("runtime-attested enforcement requires exactly one attestation")
        return self


class SandboxSnapshot(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    snapshot_id: str
    episode_id: str
    branch_id: str
    after_step_id: int = Field(ge=0)
    workspace_sha256: Digest
    logical_harness_state_sha256: Digest
    public_tool_output_sha256s: list[Digest] = Field(default_factory=list)
    allowlisted_dependency_sha256s: list[Digest] = Field(default_factory=list)
    live_process_memory_included: Literal[False] = False
    gpu_context_included: Literal[False] = False
    verifier_private_files_included: Literal[False] = False
    secret_credentials_included: Literal[False] = False
    snapshot_sha256: Digest


class BranchRecord(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    episode_id: str
    parent_branch_id: str
    branch_id: str
    snapshot_sha256: Digest
    pivotal_step_id: int = Field(ge=0)
    prefix_sha256: Digest
    restored_workspace_sha256: Digest
    runtime_reinitialized: Literal[True] = True
    independent_trajectory: Literal[True] = True
    branch_sha256: Digest

    @model_validator(mode="after")
    def branch_is_distinct(self) -> BranchRecord:
        if self.parent_branch_id == self.branch_id:
            raise ValueError("a branch must have a distinct identity")
        return self


class ToolEvent(AgenticModel):
    tool_call_id: str
    tool: str
    args_sha256: Digest
    result_sha256: Digest
    exit_status: int | None = None
    authority: Literal["NONE", "DIAGNOSTIC", "CANDIDATE_CLAIM"] = "DIAGNOSTIC"


class TrajectoryStep(AgenticModel):
    step_id: int = Field(ge=0)
    branch_id: str
    parent_step_id: int | None = Field(default=None, ge=0)
    policy_snapshot_sha256: Digest
    observation_sha256: Digest
    model_call_id: str | None = None
    tool_calls: list[ToolEvent] = Field(default_factory=list)
    workspace_before_sha256: Digest
    workspace_after_sha256: Digest
    snapshot_sha256: Digest | None = None
    phase: Literal[
        "INSPECT",
        "EDIT",
        "BUILD",
        "PUBLIC_VERIFY",
        "PROFILE_DIAGNOSTIC",
        "ROLLBACK",
        "FINALIZE",
    ]
    wall_time_ms: int = Field(ge=0)


class TokenRange(AgenticModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def end_follows_start(self) -> TokenRange:
        if self.end <= self.start:
            raise ValueError("token range end must follow start")
        return self


class ExcludedTokenRange(AgenticModel):
    reason: Literal["system", "tool-serialization", "environment", "padding", "untrusted-mask"]
    span: TokenRange


class StepTokenSpan(AgenticModel):
    step_id: int = Field(ge=0)
    model_call_id: str
    global_span: TokenRange
    trainable_ranges: list[TokenRange] = Field(default_factory=list)
    excluded_ranges: list[ExcludedTokenRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def subranges_stay_inside_global_span(self) -> StepTokenSpan:
        ranges = [*self.trainable_ranges, *(item.span for item in self.excluded_ranges)]
        if any(
            item.start < self.global_span.start or item.end > self.global_span.end
            for item in ranges
        ):
            raise ValueError("step token subranges must stay inside the global span")
        return self


class TrajectoryEnvelope(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    episode_id: str
    task_seal_sha256: Digest
    policy_snapshot_sha256: Digest
    harness_profile_sha256: Digest
    sandbox_profile_sha256: Digest
    trajectory_fidelity: Literal["exact-model-boundary", "reconstructed"]
    harness_fidelity: Literal["native-exact", "transcript-only"]
    exact_token_ids_available: bool
    policy_gradient_eligible: bool
    traces: list[ModelBoundaryTrace] = Field(default_factory=list)
    steps: list[TrajectoryStep] = Field(default_factory=list)
    step_token_spans: list[StepTokenSpan] = Field(default_factory=list)
    branches: list[BranchRecord] = Field(default_factory=list)
    allowed_uses: list[
        Literal[
            "policy-gradient",
            "external-policy",
            "curriculum",
            "offline-retrieval",
            "qualitative-audit",
            "human-study",
        ]
    ] = Field(min_length=1)
    trajectory_sha256: Digest

    @model_validator(mode="after")
    def trajectory_identity_and_fidelity_are_coherent(self) -> TrajectoryEnvelope:
        trace_ids = [item.model_call_id for item in self.traces]
        step_ids = [item.step_id for item in self.steps]
        branch_ids = [item.branch_id for item in self.branches]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("model call ids must be unique")
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("trajectory step ids must be unique")
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("trajectory branch ids must be unique")
        if any(
            item.policy_snapshot_sha256 != self.policy_snapshot_sha256
            for item in [*self.traces, *self.steps]
        ):
            raise ValueError("mixed-policy episode trajectories are forbidden")
        if any(
            step.model_call_id is not None and step.model_call_id not in set(trace_ids)
            for step in self.steps
        ):
            raise ValueError("trajectory steps must reference captured model calls")
        if any(item.model_call_id not in set(trace_ids) for item in self.step_token_spans):
            raise ValueError("step token spans must reference captured model calls")
        steps_by_id = {item.step_id: item for item in self.steps}
        traces_by_id = {item.model_call_id: item for item in self.traces}
        span_calls = [item.model_call_id for item in self.step_token_spans]
        if len(span_calls) != len(set(span_calls)):
            raise ValueError("each captured model call requires exactly one token span")
        for span in self.step_token_spans:
            step = steps_by_id.get(span.step_id)
            if step is None or step.model_call_id != span.model_call_id:
                raise ValueError("step token spans must bind their trajectory step")
            trace = traces_by_id[span.model_call_id]
            if span.global_span.end - span.global_span.start != len(trace.output_token_ids):
                raise ValueError("global token span length must match captured output tokens")
            claimed: list[bool | None] = [None] * len(trace.output_token_ids)
            for token_range in span.trainable_ranges:
                for index in range(
                    token_range.start - span.global_span.start,
                    token_range.end - span.global_span.start,
                ):
                    if claimed[index] is not None:
                        raise ValueError("trainable and excluded token ranges cannot overlap")
                    claimed[index] = True
            for excluded in span.excluded_ranges:
                for index in range(
                    excluded.span.start - span.global_span.start,
                    excluded.span.end - span.global_span.start,
                ):
                    if claimed[index] is not None:
                        raise ValueError("trainable and excluded token ranges cannot overlap")
                    claimed[index] = False
            if any(value is None for value in claimed):
                raise ValueError("token ranges must cover every captured output token")
            if claimed != trace.trainable_mask:
                raise ValueError("step token ranges disagree with the model-boundary mask")
        reconstructed = self.trajectory_fidelity == "reconstructed"
        if reconstructed:
            if self.harness_fidelity != "transcript-only" or self.exact_token_ids_available:
                raise ValueError("reconstructed trajectories must be transcript-only")
            if self.policy_gradient_eligible or "policy-gradient" in self.allowed_uses:
                raise ValueError("reconstructed trajectories are never policy-gradient eligible")
        else:
            if self.harness_fidelity != "native-exact" or not self.exact_token_ids_available:
                raise ValueError("exact trajectories require native exact token capture")
            if not self.traces:
                raise ValueError("exact trajectories require model boundary traces")
            if set(span_calls) != set(trace_ids):
                raise ValueError("exact trajectories require a token span for every model call")
        return self


class RolloutBudget(AgenticModel):
    generated_tokens: int = Field(ge=1)
    model_calls: int = Field(ge=1)
    tool_calls: int = Field(ge=0)
    wall_time_s: int = Field(ge=1)
    environment_gpu_s: int = Field(ge=0)


class RolloutRequest(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    request_id: str
    task_seal_sha256: Digest
    draft_seal_sha256: Digest
    policy_snapshot_sha256: Digest
    external_policy_state_sha256: Digest | None = None
    harness_profile_sha256: Digest
    sandbox_profile_sha256: Digest
    algorithm_profile_sha256: Digest
    feedback_visibility_policy_sha256: Digest
    seed: int
    group_id: str
    group_index: int = Field(ge=0)
    budget: RolloutBudget
    requested_resource_class: str
    request_sha256: Digest


class EpisodeEvent(AgenticModel):
    stage: EpisodeStage
    observed_at: datetime
    detail_sha256: Digest

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> EpisodeEvent:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("episode event timestamp must be timezone-aware")
        return self


class AgenticEpisode(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    episode_id: str
    rollout_request_sha256: Digest
    pinned_policy_snapshot_sha256: Digest
    events: list[EpisodeEvent] = Field(min_length=1)
    status: EpisodeStatus
    failure_owner: FailureOwner = "none"
    failure_code: str | None = None
    training_mask: Literal[0, 1]

    @model_validator(mode="after")
    def state_machine_and_failure_are_coherent(self) -> AgenticEpisode:
        orders = [EPISODE_STAGE_ORDER[item.stage] for item in self.events]
        stages = [item.stage for item in self.events]
        if orders[0] != 0 or any(
            right not in EPISODE_STAGE_TRANSITIONS[left] for left, right in pairwise(stages)
        ):
            raise ValueError("episode events must follow the frozen state machine")
        masked = {
            "INFRA_INVALID",
            "BENCHMARK_DEFECT",
            "ROLLOUT_DEFECT",
            "STALE_REJECTED",
            "CENSORED",
        }
        if self.status != "SECURITY_REJECTED" and self.training_mask != (
            0 if self.status in masked else 1
        ):
            raise ValueError("episode training mask disagrees with disposition")
        if self.status in {"VALID_PASS", "JUDGE_UNRESOLVED"}:
            if self.failure_owner != "none" or self.failure_code is not None:
                raise ValueError("non-failure episode status cannot carry failure ownership")
        elif not self.failure_code or self.failure_owner == "none":
            raise ValueError("failed, invalid, or censored episodes require failure ownership")
        return self


class EpisodeSeal(AgenticModel):
    """Immutable execution/evidence seal created before a RewardPack.

    RewardPack points to this seal. EpisodeOutcomeSeal then binds both objects, avoiding
    the impossible mutual-digest cycle in the prose RFC example.
    """

    schema_version: Literal["0.6"] = "0.6"
    episode_id: str
    task_seal_sha256: Digest
    draft_seal_sha256: Digest
    policy_snapshot_sha256: Digest
    external_policy_state_sha256: Digest | None = None
    policy_cell_sha256: Digest
    harness_profile_sha256: Digest
    sandbox_profile_sha256: Digest
    capability_resolution_sha256: Digest
    resource_lease_sha256: Digest
    benchmark_cell_sha256: Digest
    trajectory_envelope_sha256: Digest
    candidate_artifact_manifest_sha256: Digest
    evidence_pack_sha256: Digest
    status: EpisodeStatus
    failure_owner: FailureOwner
    failure_code: str | None = None
    training_mask: Literal[0, 1]
    started_at: datetime
    sealed_at: datetime
    episode_seal_sha256: Digest

    @model_validator(mode="after")
    def seal_window_and_mask_are_coherent(self) -> EpisodeSeal:
        for timestamp in (self.started_at, self.sealed_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("episode seal timestamps must be timezone-aware")
        if self.sealed_at < self.started_at:
            raise ValueError("episode cannot seal before it starts")
        masked = self.status in {
            "INFRA_INVALID",
            "BENCHMARK_DEFECT",
            "ROLLOUT_DEFECT",
            "STALE_REJECTED",
            "CENSORED",
        }
        if self.status != "SECURITY_REJECTED" and self.training_mask != (0 if masked else 1):
            raise ValueError("episode seal mask disagrees with status")
        if self.status in {"VALID_PASS", "JUDGE_UNRESOLVED"}:
            if self.failure_owner != "none" or self.failure_code is not None:
                raise ValueError("non-failure episode seal cannot carry failure ownership")
        elif not self.failure_code or self.failure_owner == "none":
            raise ValueError("failed, invalid, or censored episode seals require failure ownership")
        return self


class VerifierValidity(AgenticModel):
    environment_sentinel: Literal["pass", "fail", "unresolved"]
    artifact_boundary: Literal["pass", "fail", "unresolved"]
    trial_seal: Literal["pass", "fail", "unresolved"]

    @property
    def valid(self) -> bool:
        return all(value == "pass" for value in self.model_dump().values())


class VerifierObligationOutcome(AgenticModel):
    obligation_id: str
    bucket: Literal["CO", "RI", "NB", "MP", "SL", "ES", "MT"]
    status: Literal["pass", "fail", "partial", "unresolved", "not-applicable"]
    evidence_refs: list[str] = Field(min_length=1)
    failure_code: str | None = None

    @model_validator(mode="after")
    def failure_has_code(self) -> VerifierObligationOutcome:
        if self.status in {"fail", "unresolved"} and not self.failure_code:
            raise ValueError("failed or unresolved obligations require a failure code")
        if self.status not in {"fail", "unresolved"} and self.failure_code is not None:
            raise ValueError("passing obligations cannot carry a failure code")
        return self


class VerifierOutcomePack(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    task_seal_sha256: Digest
    episode_seal_sha256: Digest
    evidence_pack_sha256: Digest
    benchmark_cell_sha256: Digest
    validity: VerifierValidity
    obligations: list[VerifierObligationOutcome] = Field(min_length=1)
    infra_cert: Literal[0, 1]
    first_failure: str | None = None
    failure_owner: FailureOwner = "none"
    outcome_sha256: Digest

    @model_validator(mode="after")
    def hard_outcome_matches_obligations(self) -> VerifierOutcomePack:
        required = {"CO", "RI", "NB", "MP", "SL", "ES"}
        buckets = [item.bucket for item in self.obligations]
        if not required.issubset(buckets):
            raise ValueError("verifier outcome must cover CO/RI/NB/MP/SL/ES")
        if len([item.obligation_id for item in self.obligations]) != len(
            {item.obligation_id for item in self.obligations}
        ):
            raise ValueError("verifier obligation ids must be unique")
        hard = [item for item in self.obligations if item.bucket in required]
        expected = self.validity.valid and all(item.status == "pass" for item in hard)
        if self.infra_cert != int(expected):
            raise ValueError("InfraCert disagrees with validity and hard obligations")
        failures = [item for item in hard if item.status != "pass"]
        failure_ids = {item.obligation_id for item in failures}
        if (
            self.validity.valid
            and failures
            and (
                self.first_failure not in failure_ids
                or self.failure_owner not in {"candidate", "policy"}
            )
        ):
            raise ValueError("valid hard failure requires first failure and policy ownership")
        if (
            self.validity.valid
            and not failures
            and (self.first_failure is not None or self.failure_owner != "none")
        ):
            raise ValueError("passing verifier outcome cannot carry failure ownership")
        if not self.validity.valid:
            if self.failure_owner in {"candidate", "policy", "none"}:
                raise ValueError("invalid verifier outcome requires non-policy failure ownership")
            if self.first_failure is None:
                raise ValueError("invalid verifier outcome requires a first failure")
        return self


class RewardQualification(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    task_seal_sha256: Digest
    reward_profile_sha256: Digest
    source_authority_audit: Literal["pass", "fail"]
    hard_outcome_monotonicity: Literal["pass", "fail"]
    invalid_censoring_audit: Literal["pass", "fail"]
    performance_noise_audit: Literal["pass", "fail"]
    group_variance_feasibility: Literal["pass", "fail"]
    feedback_leakage_audit: Literal["pass", "fail"]
    reward_hack_mutation_audit: Literal["pass", "fail"]
    cost_dominance_audit: Literal["pass", "fail"]
    teacher_sign_anchor_audit: Literal["pass", "fail"]
    training_replay_stability: Literal["pass", "fail"]
    status: Literal["qualified", "rejected"]
    qualification_sha256: Digest

    @model_validator(mode="after")
    def status_matches_audits(self) -> RewardQualification:
        ignored = {
            "schema_version",
            "task_seal_sha256",
            "reward_profile_sha256",
            "status",
            "qualification_sha256",
        }
        passed = all(
            value == "pass" for name, value in self.model_dump().items() if name not in ignored
        )
        if self.status != ("qualified" if passed else "rejected"):
            raise ValueError("reward qualification status disagrees with audits")
        return self


class RewardProfile(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    profile_id: str
    hard_anchor: Literal["infra-cert"] = "infra-cert"
    hard_fail_floor: float = -2.0
    hard_fail_ceiling: float = -1.0
    hard_pass_floor: float = 0.0
    hard_pass_ceiling: float = 1.0
    process_shaping_cap: float = Field(default=0.1, ge=0)
    performance_scope: Literal["same-comparison-cell"] = "same-comparison-cell"
    duplicate_fact_policy: Literal["reject"] = "reject"
    reward_profile_sha256: Digest

    @model_validator(mode="after")
    def hard_margin_cannot_be_crossed(self) -> RewardProfile:
        if not (
            self.hard_fail_floor
            <= self.hard_fail_ceiling
            < self.hard_pass_floor
            <= self.hard_pass_ceiling
        ):
            raise ValueError("hard fail and pass reward bands must be disjoint and ordered")
        if self.process_shaping_cap >= self.hard_pass_floor - self.hard_fail_ceiling:
            raise ValueError("training-only shaping cap must stay below the hard band margin")
        return self


class RewardScope(AgenticModel):
    episode_id: str
    obligation_id: str | None = None
    step_ids: list[int] = Field(default_factory=list)
    token_ranges: list[TokenRange] = Field(default_factory=list)
    comparison_cell_sha256: Digest | None = None


class RewardEvent(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    event_id: str
    fact_id: str
    kind: Literal[
        "correctness",
        "regression",
        "negative-boundary",
        "mechanism-proof",
        "safety-liveness",
        "environment-validity",
        "performance",
        "official-semantic",
        "teacher-shaping",
        "trajectory-critic",
        "policy-cost",
        "process-shaping",
        "candidate-self-report",
    ]
    value: float
    status: Literal["VALID", "INVALID", "UNRESOLVED"]
    authority: Literal[
        "OFFICIAL_HARD",
        "OFFICIAL_COMPONENT",
        "VALIDITY",
        "AUDIT_ONLY",
        "TRAINING_SHAPING",
        "DIAGNOSTIC",
        "NONE",
    ]
    owner: str
    visibility: Literal[
        "POLICY_VISIBLE",
        "TEACHER_VISIBLE_REDACTED",
        "ADVANTAGE_ONLY",
        "PRIVATE_AUDIT_ONLY",
    ]
    scope: RewardScope
    evidence_refs: list[str] = Field(default_factory=list)
    producer_sha256: Digest
    event_sha256: Digest

    @model_validator(mode="after")
    def authority_matches_source(self) -> RewardEvent:
        if self.kind == "candidate-self-report" and self.authority != "NONE":
            raise ValueError("Candidate self-reports have no reward authority")
        if self.authority in {"OFFICIAL_HARD", "OFFICIAL_COMPONENT", "VALIDITY"} and not (
            self.evidence_refs
        ):
            raise ValueError("authoritative reward events require evidence references")
        if self.kind == "teacher-shaping" and self.authority != "TRAINING_SHAPING":
            raise ValueError("teacher output is training shaping only")
        return self


class FeedbackItem(AgenticModel):
    kind: Literal[
        "obligation-failure",
        "compiler-diagnostic",
        "runtime-diagnostic",
        "semantic-judge-critique",
        "sibling-summary",
    ]
    public_summary: str
    evidence_ref: str | None = None
    hidden_details_included: Literal[False] = False
    normalized_code: str | None = None
    source_excerpt_policy: str | None = None


class FeedbackPack(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    task_seal_sha256: Digest
    episode_seal_sha256: Digest
    feedback_profile_sha256: Digest
    visibility: Literal[
        "POLICY_VISIBLE",
        "TEACHER_VISIBLE_REDACTED",
        "ADVANTAGE_ONLY",
        "PRIVATE_AUDIT_ONLY",
    ]
    task_split: Literal["train", "dev", "heldout"]
    items: list[FeedbackItem] = Field(default_factory=list)
    leakage_scan: Literal["pass", "blocked"]
    teacher_eligible: bool
    feedback_pack_sha256: Digest

    @model_validator(mode="after")
    def leakage_and_split_are_fail_closed(self) -> FeedbackPack:
        allowed = (
            self.leakage_scan == "pass"
            and self.task_split == "train"
            and self.visibility in {"POLICY_VISIBLE", "TEACHER_VISIBLE_REDACTED"}
        )
        if self.teacher_eligible != allowed:
            raise ValueError("teacher eligibility requires train split and a passing leakage scan")
        return self


class CreditStepAssignment(AgenticModel):
    step_id: int = Field(ge=0)
    role: Literal[
        "causal-candidate",
        "pivotal-suspect",
        "detection-only",
        "reusable-positive",
        "unresolved",
    ]
    weight: float = Field(ge=0, le=1)
    reason_code: str
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def detection_is_not_mistaken_for_cause(self) -> CreditStepAssignment:
        if self.role == "detection-only" and self.weight != 0:
            raise ValueError("failure detection steps receive zero causal credit by default")
        return self


class TokenModulation(AgenticModel):
    span: TokenRange
    verifier_advantage: float
    multiplier: float = Field(gt=0)
    modulated_advantage: float

    @model_validator(mode="after")
    def verifier_sign_is_preserved(self) -> TokenModulation:
        expected = self.verifier_advantage * self.multiplier
        if not math.isclose(self.modulated_advantage, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("token modulation must be a positive multiple of verifier advantage")
        return self


class CreditAssignmentMap(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    profile_id: str
    episode_id: str
    anchor_reward_event_ids: list[str] = Field(min_length=1)
    step_assignments: list[CreditStepAssignment] = Field(default_factory=list)
    token_modulations: list[TokenModulation] = Field(default_factory=list)
    unresolved_credit_mass: float = Field(ge=0, le=1)
    credit_map_sha256: Digest

    @model_validator(mode="after")
    def assigned_mass_is_bounded(self) -> CreditAssignmentMap:
        assigned = sum(item.weight for item in self.step_assignments)
        if assigned + self.unresolved_credit_mass > 1 + 1e-9:
            raise ValueError("assigned and unresolved credit mass cannot exceed one")
        return self


class RewardAnchor(AgenticModel):
    infra_cert: Literal[0, 1] | None
    scalar_band: Literal["masked", "hard-fail", "pass-no-attainment", "pass-attainment"]
    scalar_value: float | None


class RewardPack(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    episode_seal_sha256: Digest
    evidence_pack_sha256: Digest
    verifier_outcome_sha256: Digest
    reward_profile_sha256: Digest
    reward_qualification_sha256: Digest
    credit_profile_sha256: Digest
    validity: Literal["valid", "invalid"]
    training_mask: Literal[0, 1]
    anchor: RewardAnchor
    event_sha256s: list[Digest]
    credit_map_sha256: Digest | None = None
    feedback_pack_sha256: Digest | None = None
    official_projection_path: str
    official_projection_independently_reproducible: Literal[True] = True
    training_projection_path: str
    training_reward_affects_official_score: Literal[False] = False
    revoked: bool = False
    revocation_reason: str | None = None
    reward_pack_sha256: Digest

    @model_validator(mode="after")
    def invalidity_and_revocation_are_censored(self) -> RewardPack:
        if self.validity == "invalid":
            if self.training_mask != 0 or self.anchor.scalar_band != "masked":
                raise ValueError("invalid RewardPack must be masked")
            if self.anchor.scalar_value is not None or self.anchor.infra_cert is not None:
                raise ValueError("invalid RewardPack cannot synthesize a negative anchor")
        elif self.anchor.infra_cert is None:
            raise ValueError("valid RewardPack requires a verifier anchor")
        elif self.revoked:
            if self.training_mask != 0:
                raise ValueError("revoked RewardPack must be masked")
        elif self.training_mask != 1:
            raise ValueError("valid RewardPack requires a verifier anchor and mask=1")
        if self.revoked and (self.training_mask != 0 or not self.revocation_reason):
            raise ValueError("revoked RewardPack must be masked and explain the revocation")
        if not self.revoked and self.revocation_reason is not None:
            raise ValueError("non-revoked RewardPack cannot carry a revocation reason")
        return self


class EpisodeOutcomeSeal(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    episode_seal_sha256: Digest
    reward_pack_sha256: Digest
    episode_outcome_sha256: Digest


class ClippingProfile(AgenticModel):
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)

    @model_validator(mode="after")
    def upper_is_not_lower(self) -> ClippingProfile:
        if self.upper < self.lower:
            raise ValueError("upper clipping bound cannot be below lower clipping bound")
        return self


class TrainingSamplingProfile(AgenticModel):
    group_size: int = Field(ge=1)
    dynamic_replenishment: bool
    require_nonzero_valid_reward_variance: bool
    official_dynamic_dropping: Literal[False] = False
    preserve_attempt_ledger: Literal[True] = True


class TrainingLossProfile(AgenticModel):
    normalization: Literal[
        "valid-trainable-token",
        "valid-step",
        "valid-sequence",
        "external-policy-update",
    ]
    invalid_trajectory_masking: Literal["required"] = "required"
    security_rejected_training: Literal["mask", "valid-negative"] = "mask"


class OverlongPolicy(AgenticModel):
    policy_budget_exceeded: Literal["soft-shape", "hard-fail", "mask"]
    infrastructure_preempted: Literal["mask"] = "mask"
    controller_cancelled: Literal["mask"] = "mask"


class PolicyStalenessProfile(AgenticModel):
    measure: Literal["completed-learner-updates"] = "completed-learner-updates"
    max_versions: int = Field(ge=0)
    mixed_policy_episode: Literal["forbidden"] = "forbidden"
    over_limit: Literal["reject", "off-policy-correction"]


class AlgorithmProfile(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    profile_id: str
    family: Literal["dapo", "gspo", "steppo", "ppo", "grpo", "rloo", "external-policy"]
    policy_granularity: Literal["token", "step", "episode", "external-action"]
    advantage_granularity: Literal["token", "step", "episode", "group"]
    credit_overlay: str | None = None
    clipping: ClippingProfile | None = None
    sampling: TrainingSamplingProfile
    loss: TrainingLossProfile
    overlong: OverlongPolicy
    policy_staleness: PolicyStalenessProfile
    llm_weights_update: bool
    algorithm_profile_sha256: Digest

    @model_validator(mode="after")
    def algorithm_requirements_are_explicit(self) -> AlgorithmProfile:
        if self.family == "dapo" and (
            self.clipping is None
            or self.policy_granularity != "token"
            or self.loss.normalization != "valid-trainable-token"
        ):
            raise ValueError("DAPO requires frozen clipping and valid-token normalization")
        if self.family == "steppo" and self.policy_granularity != "step":
            raise ValueError("StepPO requires step policy granularity")
        if self.family == "external-policy":
            if self.llm_weights_update or self.policy_granularity != "external-action":
                raise ValueError("external policy profiles cannot claim LLM weight updates")
        elif not self.llm_weights_update:
            raise ValueError("neural algorithm profiles must declare LLM weight updates")
        return self


class GroupManifest(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    group_id: str
    task_seal_sha256: Digest
    policy_snapshot_sha256: Digest
    benchmark_cell_sha256: Digest
    reward_profile_sha256: Digest
    requested_size: int = Field(ge=1)
    attempts: int = Field(ge=1)
    valid_members: int = Field(ge=0)
    invalid_infra: int = Field(ge=0)
    zero_variance_replenished: int = Field(ge=0)
    final_episode_outcome_sha256s: list[Digest]
    sampling_probability_log_sha256: Digest
    group_manifest_sha256: Digest

    @model_validator(mode="after")
    def attempts_and_membership_are_coherent(self) -> GroupManifest:
        if self.valid_members != len(self.final_episode_outcome_sha256s):
            raise ValueError("valid member count must match final group membership")
        if self.valid_members > self.requested_size or self.attempts < self.valid_members:
            raise ValueError("group attempt and membership counts are inconsistent")
        if self.invalid_infra > self.attempts - self.valid_members:
            raise ValueError("invalid infrastructure count exceeds discarded attempts")
        if len(self.final_episode_outcome_sha256s) != len(set(self.final_episode_outcome_sha256s)):
            raise ValueError("group episode outcomes must be unique")
        return self


class TrainingSplits(AgenticModel):
    train_set_sha256: Digest
    dev_set_sha256: Digest
    heldout_set_sha256: Digest

    @model_validator(mode="after")
    def splits_are_distinct(self) -> TrainingSplits:
        values = self.model_dump().values()
        if len(set(values)) != 3:
            raise ValueError("train, dev, and heldout set identities must be distinct")
        return self


class TrainingInfrastructure(AgenticModel):
    rollout_fabric_sha256: Digest
    learner_fabric_sha256: Digest
    runner_pool_policy_sha256: Digest


class TrainingBudgets(AgenticModel):
    valid_episodes: int = Field(ge=1)
    total_episode_attempts: int = Field(ge=1)
    environment_gpu_hours: float = Field(ge=0)
    learner_gpu_hours: float = Field(ge=0)
    judge_gpu_hours: float = Field(ge=0)
    wall_time_s: int = Field(ge=1)

    @model_validator(mode="after")
    def attempts_cover_valid_episodes(self) -> TrainingBudgets:
        if self.total_episode_attempts < self.valid_episodes:
            raise ValueError("total episode attempts cannot be below valid episode budget")
        return self


class TrainingRunSeal(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    run_id: str
    initial_policy_snapshot_sha256: Digest
    algorithm_profile_sha256: Digest
    sampler_profile_sha256: Digest
    reward_profile_sha256: Digest
    credit_profile_sha256: Digest
    feedback_profile_sha256: Digest
    task_splits: TrainingSplits
    infrastructure: TrainingInfrastructure
    budgets: TrainingBudgets
    checkpoint_policy_sha256: Digest
    stop_policy_sha256: Digest
    heldout_checkpoint_selection: Literal["forbidden"] = "forbidden"
    training_run_seal_sha256: Digest


class RLBatchMember(AgenticModel):
    episode_outcome_sha256: Digest
    episode_seal_sha256: Digest
    trajectory_sha256: Digest
    reward_pack_sha256: Digest
    task_seal_sha256: Digest
    behavior_policy_snapshot_sha256: Digest
    benchmark_cell_sha256: Digest
    reward_schema_version: Literal["0.6"] = "0.6"
    episode_status: EpisodeStatus
    training_mask: Literal[0, 1]
    policy_gradient_eligible: bool
    feedback_leakage_blocked: bool = False
    reward_revoked: bool = False
    valid_token_count: int = Field(ge=0)
    valid_step_count: int = Field(ge=0)
    policy_lag: int = Field(ge=0)

    @model_validator(mode="after")
    def masked_members_have_no_gradient_tokens(self) -> RLBatchMember:
        masked_status = self.episode_status in {
            "INFRA_INVALID",
            "BENCHMARK_DEFECT",
            "ROLLOUT_DEFECT",
            "STALE_REJECTED",
            "CENSORED",
        }
        if self.episode_status != "SECURITY_REJECTED" and self.training_mask != (
            0 if masked_status else 1
        ):
            raise ValueError("batch member mask disagrees with episode status")
        if self.training_mask == 0 and (self.valid_token_count or self.valid_step_count):
            raise ValueError("masked batch members cannot claim valid gradient units")
        if self.policy_gradient_eligible and self.training_mask != 1:
            raise ValueError("policy-gradient eligible members require mask=1")
        return self


class RLBatchManifest(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    batch_id: str
    training_run_seal_sha256: Digest
    target_policy_snapshot_sha256: Digest
    proximal_policy_snapshot_sha256: Digest
    algorithm_profile_sha256: Digest
    members: list[RLBatchMember] = Field(min_length=1)
    group_manifest_sha256s: list[Digest] = Field(min_length=1)
    valid_token_count: int = Field(ge=0)
    valid_step_count: int = Field(ge=0)
    policy_lag_distribution: dict[str, float]
    sampler_selection_log_sha256: Digest
    batch_sha256: Digest

    @model_validator(mode="after")
    def declared_counts_match_members(self) -> RLBatchManifest:
        if self.valid_token_count != sum(item.valid_token_count for item in self.members):
            raise ValueError("batch valid token count does not match members")
        if self.valid_step_count != sum(item.valid_step_count for item in self.members):
            raise ValueError("batch valid step count does not match members")
        if len([item.episode_outcome_sha256 for item in self.members]) != len(
            {item.episode_outcome_sha256 for item in self.members}
        ):
            raise ValueError("batch members must be unique")
        return self


class FabricPool(AgenticModel):
    pool_id: str
    kind: Literal["policy", "environment", "learner", "judge", "build-cpu", "metadata"]
    worker_profile_sha256: Digest
    device_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(min_length=1)
    independently_deployable: Literal[True] = True


class BackpressureProfile(AgenticModel):
    rollout_admission_limit: int = Field(ge=1)
    sandbox_concurrency_limit: int = Field(ge=1)
    environment_gpu_seconds_limit: int = Field(ge=1)
    trajectory_buffer_high_water: int = Field(ge=1)
    reward_queue_limit: int = Field(ge=1)
    maximum_policy_lag: int = Field(ge=0)


class RolloutFabricProfile(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    fabric_id: str
    pools: list[FabricPool] = Field(min_length=2)
    backpressure: BackpressureProfile
    measurement_isolation: Literal["MI0", "MI1", "MI2", "MI3"]
    trainer_direct_sandbox_control: Literal[False] = False
    trainer_direct_verifier_control: Literal[False] = False
    topology_sha256: Digest
    fabric_profile_sha256: Digest

    @model_validator(mode="after")
    def pools_are_separate_for_official_measurement(self) -> RolloutFabricProfile:
        ids = [item.pool_id for item in self.pools]
        if len(ids) != len(set(ids)):
            raise ValueError("fabric pool ids must be unique")
        kinds = {item.kind for item in self.pools}
        if not {"policy", "environment"}.issubset(kinds):
            raise ValueError("rollout fabric requires policy and environment pools")
        if self.measurement_isolation != "MI0":
            by_kind = {kind: set() for kind in kinds}
            for pool in self.pools:
                by_kind[pool.kind].update(pool.device_ids)
            environment = by_kind.get("environment", set())
            for kind in {"policy", "learner", "judge"}:
                if environment & by_kind.get(kind, set()):
                    raise ValueError("official environment devices cannot overlap other GPU pools")
        return self


class GangDeviceAllocation(AgenticModel):
    rank: int = Field(ge=0)
    device_id: str
    node_id: str
    numa_node: int = Field(ge=0)
    nic_id: str | None = None


class GangLeaseRecord(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    lease_id: str
    requested_gpu_count: int = Field(ge=1)
    status: Literal["active", "unschedulable", "capacity-unavailable", "broken", "released"]
    allocations: list[GangDeviceAllocation] = Field(default_factory=list)
    topology_sha256: Digest
    atomic_allocation: Literal[True] = True
    workload_started: bool
    lease_sha256: Digest

    @model_validator(mode="after")
    def partial_gang_never_starts(self) -> GangLeaseRecord:
        complete = len(self.allocations) == self.requested_gpu_count
        ranks = [item.rank for item in self.allocations]
        devices = [item.device_id for item in self.allocations]
        if len(ranks) != len(set(ranks)) or len(devices) != len(set(devices)):
            raise ValueError("gang allocation ranks and devices must be unique")
        if complete and set(ranks) != set(range(self.requested_gpu_count)):
            raise ValueError("complete gang allocations require contiguous ranks")
        if self.status == "active" and (not complete or not self.workload_started):
            raise ValueError("active gang lease requires the complete atomic allocation")
        if not complete and self.workload_started:
            raise ValueError("partial gang allocation must fail before workload start")
        return self


class RuntimeCapabilityReport(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    observed_at: datetime
    gpu_count: int = Field(ge=0)
    gpu_topology_attested: bool
    rootless_sandbox_enforced: bool
    exact_token_gateway_available: bool
    hosted_policy_exact_tokens_available: bool
    trainer_adapter_available: bool
    distributed_gang_enforced: bool
    production_ready: bool
    unavailable_reasons: list[str]
    report_sha256: Digest

    @model_validator(mode="after")
    def production_claim_requires_all_capabilities(self) -> RuntimeCapabilityReport:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("runtime capability timestamp must be timezone-aware")
        checks = (
            ("NO_ACCELERATOR", self.gpu_count > 0),
            ("GPU_TOPOLOGY_NOT_ATTESTED", self.gpu_topology_attested),
            ("ROOTLESS_SANDBOX_NOT_ENFORCED", self.rootless_sandbox_enforced),
            ("EXACT_TOKEN_GATEWAY_UNAVAILABLE", self.exact_token_gateway_available),
            ("TRAINER_ADAPTER_UNAVAILABLE", self.trainer_adapter_available),
            ("DISTRIBUTED_GANG_NOT_ENFORCED", self.distributed_gang_enforced),
        )
        expected_reasons = [code for code, passed in checks if not passed]
        if self.production_ready != all(passed for _, passed in checks):
            raise ValueError("production readiness disagrees with observed capabilities")
        if self.unavailable_reasons != expected_reasons:
            raise ValueError("runtime capability reasons disagree with observed capabilities")
        return self


class AgenticDraftSpec(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    draft_id: str
    revision: int = Field(ge=1)
    task_seal_sha256: Digest
    task_split: Literal["train", "dev", "heldout", "official"]
    track: Literal[
        "frozen-agent-artifact",
        "external-policy-learning",
        "online-adaptation",
        "agentic-rl-training",
        "rl-systems",
        "task-authoring-and-verifier-audit",
    ]
    policy_cell_sha256: Digest
    rollout_budget: RolloutBudget
    algorithm_profile_sha256: Digest
    reward_profile_sha256: Digest
    credit_profile_sha256: Digest
    feedback_profile_sha256: Digest
    rollout_fabric_sha256: Digest
    measurement_isolation: Literal["MI0", "MI1", "MI2", "MI3"]
    mixed_policy_episode: Literal["forbidden"] = "forbidden"
    candidate_score_formula_unchanged: Literal[True] = True
    training_reward_affects_official_score: Literal[False] = False
    draft_sha256: Digest


class LegacyExperienceRecord(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    case_id: str
    domain: Literal["training", "inference", "communication", "other"]
    project: str
    repository: str
    pull_number: int = Field(ge=1)
    group_id: str
    source_artifact_sha256s: dict[str, Digest]
    acquisition_status: Literal["acquired", "invalid"]
    oracle_availability: Literal["available", "invalid"]
    experience_status: Literal["valid", "invalid"]
    invalid_reason: str | None = None
    execution_status: str | None = None
    machine_decision: Literal["accept", "check", "reject", "unresolved"] | None = None
    oracle_decision: Literal["accept", "check", "reject", "unresolved"] | None = None
    trajectory_fidelity: Literal["reconstructed"] = "reconstructed"
    harness_fidelity: Literal["transcript-only"] = "transcript-only"
    exact_token_ids_available: Literal[False] = False
    policy_gradient_eligible: Literal[False] = False
    reward_qualification: Literal["not-qualified"] = "not-qualified"
    reward_pack_sha256: None = None
    allowed_uses: list[
        Literal[
            "external-policy",
            "curriculum",
            "offline-retrieval",
            "qualitative-audit",
        ]
    ] = Field(min_length=1)
    record_sha256: Digest

    @model_validator(mode="after")
    def validity_has_a_reason_and_usable_oracle(self) -> LegacyExperienceRecord:
        invalid = (
            self.acquisition_status == "invalid"
            or self.oracle_availability == "invalid"
            or self.oracle_decision in {None, "unresolved"}
        )
        if self.experience_status != ("invalid" if invalid else "valid"):
            raise ValueError("legacy experience validity disagrees with source outcomes")
        if invalid != (self.invalid_reason is not None):
            raise ValueError("invalid legacy experience requires exactly one reason")
        return self


class LegacyExperienceManifest(AgenticModel):
    schema_version: Literal["0.6"] = "0.6"
    manifest_id: str
    generated_at: datetime
    source_roots: list[str] = Field(min_length=1)
    records: list[LegacyExperienceRecord] = Field(min_length=1)
    attempted_records: int = Field(ge=1)
    valid_records: int = Field(ge=0)
    invalid_records: int = Field(ge=0)
    policy_gradient_eligible_records: Literal[0] = 0
    fabricated_token_records: Literal[0] = 0
    qualified_reward_records: Literal[0] = 0
    manifest_sha256: Digest

    @model_validator(mode="after")
    def counts_and_legacy_safety_match(self) -> LegacyExperienceManifest:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("legacy manifest timestamp must be timezone-aware")
        if self.attempted_records != len(self.records):
            raise ValueError("legacy attempted record count must match records")
        if self.valid_records + self.invalid_records != self.attempted_records:
            raise ValueError("legacy valid and invalid counts must cover all attempts")
        observed_invalid = sum(item.experience_status == "invalid" for item in self.records)
        if observed_invalid != self.invalid_records:
            raise ValueError("legacy invalid count disagrees with records")
        identifiers = [item.case_id for item in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("legacy experience case ids must be unique")
        return self
