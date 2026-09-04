from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.models import KernelAggregate, RoleResult
from infraswe.kernel.role_graph import RoleGraph
from infraswe.models.ada_sm89 import (
    AdaSM89CapabilityManifest,
    AdaSM89CrossSKUResult,
    AdaSM89NativeResult,
)
from infraswe.models.agentic import (
    AgentHarnessProfile,
    AgenticDraftSpec,
    AlgorithmProfile,
    BranchRecord,
    CreditAssignmentMap,
    EpisodeOutcomeSeal,
    EpisodeSeal,
    ExternalPolicyState,
    FeedbackPack,
    GangLeaseRecord,
    GroupManifest,
    LegacyExperienceManifest,
    LogprobFidelityReport,
    ModelBoundaryTrace,
    PolicyCell,
    PolicySnapshot,
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
    TrainingRunSeal,
    TrajectoryEnvelope,
    VerifierOutcomePack,
)
from infraswe.models.artifact import ArtifactManifest
from infraswe.models.artifact_boundary import (
    ArtifactPolicy,
    CachePolicy,
    CandidateArtifactManifest,
    EvidenceArtifact,
    EvidencePackManifest,
    MeasurementIntegrityReport,
    OfficialTimingSample,
    PristineApplyResult,
    PristineBase,
    PristineBuildResult,
    ScoreEvidenceBinding,
    TransportEnvelope,
    TrialSeal,
    WorkspaceFreezeAttestation,
)
from infraswe.models.candidates import (
    CandidateActivationPlan,
    CandidateTimingGate,
    DefaultCandidateRegistry,
    DefaultCandidateResolution,
)
from infraswe.models.capability import (
    BenchmarkCellManifest,
    BenchmarkCellPolicy,
    CandidateCapabilityDeclaration,
    CapabilityAttestation,
    CapabilityContract,
    CapabilityRegistry,
    CapabilityResolution,
    EnvironmentSentinelResult,
    ResourceEnvelope,
    ResourceFeasibilityResult,
    ResourceLease,
    RunnerManifest,
    RunnerSelectionPolicy,
    RunnerSnapshot,
    TopologyContract,
    TopologyGraph,
)
from infraswe.models.communication_phase import (
    CommunicationPhaseRegressionPolicy,
    CommunicationPhaseRegressionResult,
    CommunicationPhaseTraceRecord,
    CommunicationPhaseTraceSet,
)
from infraswe.models.draft import (
    AffectedCasePlan,
    DefaultDraftCatalog,
    DraftSourceResolution,
    DraftSpec,
    ProjectComparisonCell,
    SealedDraft,
    TargetProjectProfile,
)
from infraswe.models.evidence import LoadCellEvidence, ProfilerEvidence, RequestSample
from infraswe.models.gb10 import GB10CapabilityManifest
from infraswe.models.history import (
    BlindEvaluationEvidence,
    HistoricalCalibrationReport,
    HistoricalExplainableJudgmentLock,
    HistoricalGroundTruth,
    HistoricalJudgmentReviewEvidence,
    HistoricalPolarizedDecisionOracle,
    HistoricalPRCandidate,
    HistoricalPredictionLock,
    HistoricalReviewEvidence,
    HistoricalReviewFinalityEvidence,
)
from infraswe.models.judge import (
    JudgeAggregation,
    JudgeCalibrationReport,
    JudgeCell,
    JudgeCostCard,
    JudgeDriftSentinel,
    JudgeInputPackManifest,
    JudgeInputPackSpec,
    JudgeOutput,
    JudgeProfile,
    JudgeRubric,
    JudgeRunRecord,
    JudgeScoreProjection,
    JudgeTrustCard,
    JudgeVerifierAuditResult,
)
from infraswe.models.project_score import InfraSWEOverallResult, TritonPurityAudit, V05ScoreResult
from infraswe.models.retrieval import FootprintExtractionRequest, RetrievalBundle
from infraswe.models.score import ScoreResult
from infraswe.models.system_paths import (
    CommunicationDraftSpec,
    CommunicationEfficiencyCard,
    MemoryTierDraftSpec,
    MemoryTieringEfficiencyCard,
    SystemDraftProfileCatalog,
    SystemPathInfraCertEvidence,
    SystemPathInfraCertResult,
    SystemPathLoadCell,
)
from infraswe.models.task import TaskPackage
from infraswe.models.task_quality import (
    TaskAcceptanceContract,
    TaskQualificationReport,
    TaskSeal,
    TaskSpecification,
    VerifierResult,
    WitnessSet,
)
from infraswe.models.training import (
    TrainingCapabilityManifest,
    TrainingCertification,
    TrainingEvidenceBundle,
    TrainingEvidencePackManifest,
    TrainingResult,
    TrainingScoreInput,
)


def schema_documents() -> dict[str, dict[str, Any]]:
    return {
        "task.schema.json": TaskPackage.model_json_schema(),
        "artifact.schema.json": ArtifactManifest.model_json_schema(),
        "communication-phase-trace-record-v0.1.schema.json": (
            CommunicationPhaseTraceRecord.model_json_schema()
        ),
        "communication-phase-trace-set-v0.1.schema.json": (
            CommunicationPhaseTraceSet.model_json_schema()
        ),
        "communication-phase-regression-policy-v0.1.schema.json": (
            CommunicationPhaseRegressionPolicy.model_json_schema()
        ),
        "communication-phase-regression-result-v0.1.schema.json": (
            CommunicationPhaseRegressionResult.model_json_schema()
        ),
        "agentic-draft-v0.6.schema.json": AgenticDraftSpec.model_json_schema(),
        "policy-snapshot-v0.6.schema.json": PolicySnapshot.model_json_schema(),
        "external-policy-state-v0.6.schema.json": ExternalPolicyState.model_json_schema(),
        "agent-harness-profile-v0.6.schema.json": AgentHarnessProfile.model_json_schema(),
        "policy-cell-v0.6.schema.json": PolicyCell.model_json_schema(),
        "model-boundary-trace-v0.6.schema.json": ModelBoundaryTrace.model_json_schema(),
        "logprob-fidelity-v0.6.schema.json": LogprobFidelityReport.model_json_schema(),
        "sandbox-profile-v0.6.schema.json": SandboxProfile.model_json_schema(),
        "sandbox-snapshot-v0.6.schema.json": SandboxSnapshot.model_json_schema(),
        "branch-record-v0.6.schema.json": BranchRecord.model_json_schema(),
        "trajectory-envelope-v0.6.schema.json": TrajectoryEnvelope.model_json_schema(),
        "rollout-request-v0.6.schema.json": RolloutRequest.model_json_schema(),
        "episode-seal-v0.6.schema.json": EpisodeSeal.model_json_schema(),
        "verifier-outcome-v0.6.schema.json": VerifierOutcomePack.model_json_schema(),
        "reward-qualification-v0.6.schema.json": RewardQualification.model_json_schema(),
        "reward-profile-v0.6.schema.json": RewardProfile.model_json_schema(),
        "reward-event-v0.6.schema.json": RewardEvent.model_json_schema(),
        "feedback-pack-v0.6.schema.json": FeedbackPack.model_json_schema(),
        "credit-assignment-v0.6.schema.json": CreditAssignmentMap.model_json_schema(),
        "reward-pack-v0.6.schema.json": RewardPack.model_json_schema(),
        "episode-outcome-seal-v0.6.schema.json": EpisodeOutcomeSeal.model_json_schema(),
        "algorithm-profile-v0.6.schema.json": AlgorithmProfile.model_json_schema(),
        "group-manifest-v0.6.schema.json": GroupManifest.model_json_schema(),
        "training-run-seal-v0.6.schema.json": TrainingRunSeal.model_json_schema(),
        "rl-batch-manifest-v0.6.schema.json": RLBatchManifest.model_json_schema(),
        "rollout-fabric-profile-v0.6.schema.json": RolloutFabricProfile.model_json_schema(),
        "gang-lease-v0.6.schema.json": GangLeaseRecord.model_json_schema(),
        "runtime-capability-report-v0.6.schema.json": (RuntimeCapabilityReport.model_json_schema()),
        "legacy-experience-manifest-v0.6.schema.json": (
            LegacyExperienceManifest.model_json_schema()
        ),
        "task-specification-v0.1.schema.json": TaskSpecification.model_json_schema(),
        "task-acceptance-contract-v0.1.schema.json": (TaskAcceptanceContract.model_json_schema()),
        "task-witness-set-v0.1.schema.json": WitnessSet.model_json_schema(),
        "task-qualification-v0.1.schema.json": TaskQualificationReport.model_json_schema(),
        "task-seal-v0.1.schema.json": TaskSeal.model_json_schema(),
        "verifier-result-v0.1.schema.json": VerifierResult.model_json_schema(),
        "artifact-policy-v0.1.schema.json": ArtifactPolicy.model_json_schema(),
        "workspace-freeze-v0.1.schema.json": (WorkspaceFreezeAttestation.model_json_schema()),
        "candidate-artifact-manifest-v0.1.schema.json": (
            CandidateArtifactManifest.model_json_schema()
        ),
        "transport-envelope-v0.1.schema.json": TransportEnvelope.model_json_schema(),
        "pristine-base-v0.1.schema.json": PristineBase.model_json_schema(),
        "pristine-apply-result-v0.1.schema.json": (PristineApplyResult.model_json_schema()),
        "pristine-build-result-v0.1.schema.json": (PristineBuildResult.model_json_schema()),
        "cache-policy-v0.1.schema.json": CachePolicy.model_json_schema(),
        "official-timing-sample-v0.1.schema.json": (OfficialTimingSample.model_json_schema()),
        "measurement-integrity-v0.1.schema.json": (MeasurementIntegrityReport.model_json_schema()),
        "trial-seal-v0.1.schema.json": TrialSeal.model_json_schema(),
        "evidence-artifact-v0.1.schema.json": EvidenceArtifact.model_json_schema(),
        "evidence-pack-v0.1.schema.json": EvidencePackManifest.model_json_schema(),
        "score-evidence-binding-v0.1.schema.json": (ScoreEvidenceBinding.model_json_schema()),
        "capability-registry-v0.1.schema.json": CapabilityRegistry.model_json_schema(),
        "capability-contract-v0.1.schema.json": CapabilityContract.model_json_schema(),
        "candidate-capability-declaration-v0.1.schema.json": (
            CandidateCapabilityDeclaration.model_json_schema()
        ),
        "capability-attestation-v0.1.schema.json": CapabilityAttestation.model_json_schema(),
        "runner-manifest-v0.1.schema.json": RunnerManifest.model_json_schema(),
        "runner-snapshot-v0.1.schema.json": RunnerSnapshot.model_json_schema(),
        "resource-envelope-v0.1.schema.json": ResourceEnvelope.model_json_schema(),
        "resource-feasibility-v0.1.schema.json": (ResourceFeasibilityResult.model_json_schema()),
        "topology-contract-v0.1.schema.json": TopologyContract.model_json_schema(),
        "topology-graph-v0.1.schema.json": TopologyGraph.model_json_schema(),
        "benchmark-cell-policy-v0.1.schema.json": (BenchmarkCellPolicy.model_json_schema()),
        "benchmark-cell-v0.1.schema.json": BenchmarkCellManifest.model_json_schema(),
        "runner-selection-policy-v0.1.schema.json": (RunnerSelectionPolicy.model_json_schema()),
        "capability-resolution-v0.1.schema.json": (CapabilityResolution.model_json_schema()),
        "resource-lease-v0.1.schema.json": ResourceLease.model_json_schema(),
        "environment-sentinel-v0.1.schema.json": (EnvironmentSentinelResult.model_json_schema()),
        "result.schema.json": ScoreResult.model_json_schema(),
        "target-project-profile-v0.5.schema.json": TargetProjectProfile.model_json_schema(),
        "draft-v0.5.schema.json": DraftSpec.model_json_schema(),
        "sealed-draft-v0.5.schema.json": SealedDraft.model_json_schema(),
        "project-comparison-cell-v0.5.schema.json": ProjectComparisonCell.model_json_schema(),
        "affected-case-plan-v0.5.schema.json": AffectedCasePlan.model_json_schema(),
        "default-draft-catalog-v0.5.schema.json": DefaultDraftCatalog.model_json_schema(),
        "default-candidate-registry-v0.5.schema.json": (
            DefaultCandidateRegistry.model_json_schema()
        ),
        "default-candidate-resolution-v0.5.schema.json": (
            DefaultCandidateResolution.model_json_schema()
        ),
        "candidate-activation-plan-v0.5.schema.json": CandidateActivationPlan.model_json_schema(),
        "candidate-timing-gate-v0.5.schema.json": CandidateTimingGate.model_json_schema(),
        "draft-source-resolution-v0.5.schema.json": DraftSourceResolution.model_json_schema(),
        "triton-purity-audit-v0.5.schema.json": TritonPurityAudit.model_json_schema(),
        "project-score-v0.5.schema.json": V05ScoreResult.model_json_schema(),
        "infraswe-overall-result-v0.1.schema.json": InfraSWEOverallResult.model_json_schema(),
        "historical-pr-candidate-v0.5.schema.json": HistoricalPRCandidate.model_json_schema(),
        "historical-pr-evidence-v0.5.schema.json": BlindEvaluationEvidence.model_json_schema(),
        "historical-pr-prediction-lock-v0.5.schema.json": (
            HistoricalPredictionLock.model_json_schema()
        ),
        "historical-pr-ground-truth-v0.5.schema.json": HistoricalGroundTruth.model_json_schema(),
        "historical-pr-explainable-judgment-v0.5.schema.json": (
            HistoricalExplainableJudgmentLock.model_json_schema()
        ),
        "historical-pr-review-evidence-v0.5.schema.json": (
            HistoricalReviewEvidence.model_json_schema()
        ),
        "historical-judgment-review-evidence-v0.5.schema.json": (
            HistoricalJudgmentReviewEvidence.model_json_schema()
        ),
        "historical-review-finality-evidence-v0.5.schema.json": (
            HistoricalReviewFinalityEvidence.model_json_schema()
        ),
        "historical-pr-calibration-v0.5.schema.json": (
            HistoricalCalibrationReport.model_json_schema()
        ),
        "historical-polarized-decision-oracle-v0.5.1.schema.json": (
            HistoricalPolarizedDecisionOracle.model_json_schema()
        ),
        "judge-profile-v0.5.3.schema.json": JudgeProfile.model_json_schema(),
        "judge-rubric-v0.5.3.schema.json": JudgeRubric.model_json_schema(),
        "judge-calibration-v0.5.3.schema.json": JudgeCalibrationReport.model_json_schema(),
        "judge-drift-v0.5.3.schema.json": JudgeDriftSentinel.model_json_schema(),
        "judge-cell-v0.5.3.schema.json": JudgeCell.model_json_schema(),
        "judge-input-spec-v0.5.3.schema.json": JudgeInputPackSpec.model_json_schema(),
        "judge-input-pack-v0.5.3.schema.json": JudgeInputPackManifest.model_json_schema(),
        "judge-output-v0.5.3.schema.json": JudgeOutput.model_json_schema(),
        "judge-run-v0.5.3.schema.json": JudgeRunRecord.model_json_schema(),
        "judge-aggregation-v0.5.3.schema.json": JudgeAggregation.model_json_schema(),
        "judge-score-projection-v0.5.3.schema.json": JudgeScoreProjection.model_json_schema(),
        "judge-trust-v0.5.3.schema.json": JudgeTrustCard.model_json_schema(),
        "judge-cost-v0.5.3.schema.json": JudgeCostCard.model_json_schema(),
        "judge-verifier-audit-v0.5.3.schema.json": (JudgeVerifierAuditResult.model_json_schema()),
        "precedent-retrieval-bundle-v0.5.1.schema.json": (RetrievalBundle.model_json_schema()),
        "precedent-footprint-request-v0.5.1.schema.json": (
            FootprintExtractionRequest.model_json_schema()
        ),
        "communication-draft-v0.5.1.schema.json": CommunicationDraftSpec.model_json_schema(),
        "memory-tier-draft-v0.5.2.schema.json": MemoryTierDraftSpec.model_json_schema(),
        "system-path-load-cell-v0.5.2.schema.json": SystemPathLoadCell.model_json_schema(),
        "system-path-infracert-evidence-v0.5.2.schema.json": (
            SystemPathInfraCertEvidence.model_json_schema()
        ),
        "system-path-infracert-result-v0.5.2.schema.json": (
            SystemPathInfraCertResult.model_json_schema()
        ),
        "communication-efficiency-card-v0.5.1.schema.json": (
            CommunicationEfficiencyCard.model_json_schema()
        ),
        "memory-tiering-efficiency-card-v0.5.2.schema.json": (
            MemoryTieringEfficiencyCard.model_json_schema()
        ),
        "system-draft-profile-catalog-v0.5.2.schema.json": (
            SystemDraftProfileCatalog.model_json_schema()
        ),
        "request-sample-v0.4.schema.json": RequestSample.model_json_schema(),
        "load-cell-v0.4.schema.json": LoadCellEvidence.model_json_schema(),
        "profiler-evidence-v0.4.schema.json": ProfilerEvidence.model_json_schema(),
        "ada-sm89-capability.schema.json": AdaSM89CapabilityManifest.model_json_schema(),
        "ada-sm89-cross-sku.schema.json": AdaSM89CrossSKUResult.model_json_schema(),
        "ada-sm89-native-result.schema.json": AdaSM89NativeResult.model_json_schema(),
        "gb10-capability.schema.json": GB10CapabilityManifest.model_json_schema(),
        "training-capability.schema.json": TrainingCapabilityManifest.model_json_schema(),
        "training-evidence.schema.json": TrainingEvidenceBundle.model_json_schema(),
        "training-cert.schema.json": TrainingCertification.model_json_schema(),
        "training-score-input.schema.json": TrainingScoreInput.model_json_schema(),
        "training-result.schema.json": TrainingResult.model_json_schema(),
        "training-evidence-pack.schema.json": TrainingEvidencePackManifest.model_json_schema(),
        "kernel-role-result.schema.json": RoleResult.model_json_schema(),
        "kernel-role-graph.schema.json": RoleGraph.model_json_schema(),
        "kernel-aggregate.schema.json": KernelAggregate.model_json_schema(),
    }


def write_schema_documents(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, schema in schema_documents().items():
        atomic_write_json(output / name, schema)


def rendered_schema_bytes(schema: Mapping[str, Any]) -> bytes:
    return (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()


def stale_schema_names(output: Path) -> list[str]:
    stale = []
    for name, schema in schema_documents().items():
        path = output / name
        if not path.is_file() or path.read_bytes() != rendered_schema_bytes(schema):
            stale.append(name)
    return stale
