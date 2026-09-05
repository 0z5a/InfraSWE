from __future__ import annotations

import functools
import http.server
import shlex
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from infraswe import __version__
from infraswe.agentic import (
    audit_episode_outcome_seal,
    audit_sealed,
    build_legacy_experience_manifest,
    build_runtime_capability_report,
    validate_rl_batch,
)
from infraswe.agents import CliAgent, NoopAgent, OracleAgent
from infraswe.artifact_boundary import (
    audit_artifact_policy,
    audit_candidate_manifest,
    audit_evidence_pack,
    audit_score_binding,
    collect_candidate_artifacts,
)
from infraswe.capability import (
    assert_raw_performance_comparable,
    audit_capability_resolution,
    audit_registry,
    resolve_capabilities,
)
from infraswe.draft.candidate_registry import build_default_candidate_registry
from infraswe.draft.defaults import DEFAULT_PROJECT_ORDER, build_default_catalog
from infraswe.draft.lifecycle import audit_seal, canonical_sha256, seal_draft
from infraswe.draft.resolver import parse_draft_document, resolve_draft
from infraswe.draft.system_defaults import build_system_profile_catalog
from infraswe.environments.hardware_manifest import write_hardware_manifest
from infraswe.io import atomic_write_json, atomic_write_jsonl
from infraswe.judge import (
    aggregate_panel,
    audit_input_pack,
    audit_judge_cell,
    audit_profile_eligibility,
    build_input_pack,
    build_judge_cell,
    build_score_projection,
    build_trust_card,
    validate_judge_output,
)
from infraswe.models.agentic import (
    AgentHarnessProfile,
    AlgorithmProfile,
    EpisodeOutcomeSeal,
    EpisodeSeal,
    PolicySnapshot,
    RewardPack,
    RLBatchManifest,
    RolloutFabricProfile,
    TrainingRunSeal,
)
from infraswe.models.artifact_boundary import (
    ArtifactPolicy,
    CandidateArtifactManifest,
    EvidencePackManifest,
    ScoreEvidenceBinding,
    TrialSeal,
    WorkspaceFreezeAttestation,
)
from infraswe.models.capability import (
    BenchmarkCellManifest,
    CandidateCapabilityDeclaration,
    CapabilityAttestation,
    CapabilityContract,
    CapabilityRegistry,
    CapabilityResolution,
    ResourceEnvelope,
    RunnerManifest,
    RunnerSelectionPolicy,
    RunnerSnapshot,
    TopologyContract,
    TopologyGraph,
)
from infraswe.models.communication_phase import (
    CommunicationPhaseRegressionPolicy,
    CommunicationPhaseTraceSet,
)
from infraswe.models.draft import (
    DraftCandidate,
    HumanReviewRecord,
    RemoteGitDraftLocation,
    SealedDraft,
)
from infraswe.models.hardware import HardwareProfile, validate_hardware_manifest
from infraswe.models.judge import (
    JudgeAggregation,
    JudgeCalibrationReport,
    JudgeCell,
    JudgeDriftSentinel,
    JudgeInputPackManifest,
    JudgeInputPackSpec,
    JudgeOutput,
    JudgeProfile,
    JudgeRubric,
    JudgeRunRecord,
)
from infraswe.models.retrieval import (
    CandidateFootprint,
    FootprintExtractionRequest,
    HumanRuleDecision,
    PrecedentGraphEdge,
    PrecedentRecord,
    PrecedentSet,
    QueryPlan,
    RepositorySnapshot,
    RetrievalBundle,
    RuleCandidate,
)
from infraswe.models.score import ScoreResult
from infraswe.models.task import TaskPackage
from infraswe.models.task_quality import (
    AlternativeValidSolutionOutcome,
    BaselineDifferential,
    HumanTaskQualificationReview,
    MutationOutcome,
    NegativeControlOutcome,
    TaskAcceptanceContract,
    TaskLeakageAudit,
    TaskQualificationReport,
    TaskSeal,
    TaskSpecification,
    VerifierFlakinessAudit,
    WitnessReplayResult,
    WitnessSet,
)
from infraswe.models.training import (
    TrainingComparability,
    TrainingEvidenceBundle,
    TrainingScoreInput,
)
from infraswe.models.trial import TrialRecord
from infraswe.retrieval import (
    PrecedentStore,
    apply_human_rule_decisions,
    audit_leakage,
    audit_precedent_set_digest,
    audit_retrieval_bundle_digest,
    build_default_query_plan,
    build_precedent_set,
    build_retrieval_assessment,
    build_retrieval_bundle,
    compile_rule_candidates,
    contract_executable_rules,
    detect_conflicts,
    execute_retrieval,
    extract_candidate_footprint,
)
from infraswe.runner import TrialRunner
from infraswe.schema import schema_documents, stale_schema_names, write_schema_documents
from infraswe.scoring.communication_phase import evaluate_communication_phase_regression
from infraswe.scoring.report import write_reports
from infraswe.scoring.training import build_training_result
from infraswe.task_quality import (
    audit_acceptance_contract,
    audit_task_seal,
    audit_witness_set,
    build_task_seal,
    qualify_task,
)
from infraswe.training.probe import probe_capabilities
from infraswe.training.semantics import verify_training_evidence

app = typer.Typer(no_args_is_help=True, help="Executable benchmark harness for Infra agents.")
task_app = typer.Typer(no_args_is_help=True, help="Inspect and certify task packages.")
schema_app = typer.Typer(no_args_is_help=True, help="Export protocol JSON Schemas.")
lease_app = typer.Typer(no_args_is_help=True, help="Inspect resource leases and hardware.")
training_app = typer.Typer(
    no_args_is_help=True, help="Probe and verify cross-framework training evidence."
)
communication_app = typer.Typer(
    no_args_is_help=True, help="Normalize and regress communication-phase evidence."
)
draft_app = typer.Typer(
    no_args_is_help=True,
    help="Resolve, inspect, and seal project-conditioned Draft v0.5 documents.",
)
precedent_app = typer.Typer(
    no_args_is_help=True,
    help="Build and query deterministic local precedent indexes.",
)
judge_app = typer.Typer(
    no_args_is_help=True,
    help="Build and audit sealed v0.5.3 LLM-as-a-Judge evidence.",
)
judge_profile_app = typer.Typer(no_args_is_help=True, help="Validate Judge profiles.")
judge_cell_app = typer.Typer(no_args_is_help=True, help="Seal and audit Judge Cells.")
judge_pack_app = typer.Typer(no_args_is_help=True, help="Build and audit Judge input packs.")
artifact_app = typer.Typer(no_args_is_help=True, help="Collect and audit v0.1 Candidate artifacts.")
evidence_app = typer.Typer(
    no_args_is_help=True, help="Audit sealed v0.1 EvidencePacks and score bindings."
)
capability_app = typer.Typer(
    no_args_is_help=True, help="Resolve v0.1 capability/resource/topology contracts."
)
cell_app = typer.Typer(no_args_is_help=True, help="Audit BenchmarkCell comparability.")
rl_app = typer.Typer(no_args_is_help=True, help="Audit v0.6 agentic RL protocol artifacts.")
rl_policy_app = typer.Typer(no_args_is_help=True, help="Validate immutable policy snapshots.")
rl_harness_app = typer.Typer(no_args_is_help=True, help="Validate exact-token harness profiles.")
rl_episode_app = typer.Typer(no_args_is_help=True, help="Inspect sealed agentic episodes.")
rl_reward_app = typer.Typer(no_args_is_help=True, help="Inspect verifier-anchored rewards.")
rl_batch_app = typer.Typer(no_args_is_help=True, help="Validate trainer-neutral RL batches.")
rl_legacy_app = typer.Typer(no_args_is_help=True, help="Migrate legacy offline experience.")
rl_fabric_app = typer.Typer(no_args_is_help=True, help="Audit rollout fabric capabilities.")
rl_train_app = typer.Typer(no_args_is_help=True, help="Audit immutable training run seals.")
app.add_typer(task_app, name="task")
app.add_typer(schema_app, name="schema")
app.add_typer(lease_app, name="lease")
app.add_typer(training_app, name="training")
app.add_typer(communication_app, name="communication")
app.add_typer(draft_app, name="draft")
app.add_typer(precedent_app, name="precedent")
app.add_typer(judge_app, name="judge")
app.add_typer(artifact_app, name="artifact")
app.add_typer(evidence_app, name="evidence")
app.add_typer(capability_app, name="capability")
app.add_typer(cell_app, name="cell")
app.add_typer(rl_app, name="rl")
judge_app.add_typer(judge_profile_app, name="profile")
judge_app.add_typer(judge_cell_app, name="cell")
judge_app.add_typer(judge_pack_app, name="pack")
rl_app.add_typer(rl_policy_app, name="policy")
rl_app.add_typer(rl_harness_app, name="harness")
rl_app.add_typer(rl_episode_app, name="episode")
rl_app.add_typer(rl_reward_app, name="reward")
rl_app.add_typer(rl_batch_app, name="batch")
rl_app.add_typer(rl_legacy_app, name="legacy")
rl_app.add_typer(rl_fabric_app, name="fabric")
rl_app.add_typer(rl_train_app, name="train")
console = Console()


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise typer.BadParameter(f"cannot parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{path} must contain an object")
    return payload


def _read_sequence(path: Path) -> list[Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise typer.BadParameter(f"cannot parse {path}: {error}") from error
    if not isinstance(payload, list):
        raise typer.BadParameter(f"{path} must contain an array")
    return payload


def _read_sealed[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
    *,
    exit_code: int,
) -> ModelT:
    try:
        model = model_type.model_validate(_read_mapping(path))
        failures = audit_sealed(model)
    except (TypeError, ValueError, ValidationError) as error:
        console.print(f"[red]INVALID[/red] {error}")
        raise typer.Exit(exit_code) from error
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(exit_code)
    return model


def _select_agent(name: str, command: str | None, task: TaskPackage):
    if name == "noop":
        return NoopAgent()
    if name == "oracle":
        return OracleAgent(task)
    if name == "cli":
        if not command:
            raise typer.BadParameter("--agent-command is required when --agent=cli")
        return CliAgent(shlex.split(command))
    raise typer.BadParameter("agent must be one of: noop, oracle, cli")


@app.command("run")
def run_command(
    task_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    agent: Annotated[str, typer.Option("--agent")] = "noop",
    agent_command: Annotated[str | None, typer.Option("--agent-command")] = None,
    executor: Annotated[str | None, typer.Option("--executor")] = None,
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs"),
    ttl: Annotated[int, typer.Option("--ttl", min=1)] = 180,
    max_infra_cost: Annotated[float | None, typer.Option("--max-infra-cost", min=0)] = None,
) -> None:
    """Run Agent -> artifact -> destroy -> fresh verifier replay -> score."""
    task = TaskPackage.load(task_path)
    selected_executor = executor or (
        "docker" if task.environment.agent_mode == "docker" else "local"
    )
    runner = TrialRunner(
        task=task,
        agent=_select_agent(agent, agent_command, task),
        runs_root=runs_root,
        executor_kind=selected_executor,
        ttl_minutes=ttl,
        max_infra_cost_usd=max_infra_cost,
    )
    result = runner.run()
    table = Table(title=f"InfraSWE · {result.record.task_id}")
    table.add_column("Run")
    table.add_column("Stable")
    table.add_column("Core")
    table.add_column("InfraExt")
    table.add_column("InfraTotal")
    table.add_row(
        result.record.trial_id,
        "yes" if result.score.stable_resolved_at_1 else "no",
        f"{result.score.core_100:.2f}",
        f"{result.score.infra_ext_100:.2f}",
        f"{result.score.infra_total:.2f}",
    )
    console.print(table)
    console.print(f"Artifacts: [link={result.run_dir}]{result.run_dir}[/link]")
    if not result.score.stable_resolved_at_1:
        raise typer.Exit(1)


@task_app.command("validate")
def validate_task(
    task_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate task.toml and the required hidden/public package layout."""
    task = TaskPackage.load(task_path)
    errors = task.validate_layout()
    if errors:
        for error in errors:
            console.print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)
    console.print(
        f"[green]valid[/green] {task.task.id} · {task.environment.profile} · "
        f"{task.replay.count} replay(s)"
    )


@task_app.command("certify")
def certify_task(
    task_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    executor: Annotated[str, typer.Option("--executor")] = "docker",
    runs_root: Annotated[Path, typer.Option("--runs-root")] = Path("runs/certification"),
) -> None:
    """Prove that the base fails and the trusted solution passes all fresh replays."""
    task = TaskPackage.load(task_path)
    layout_errors = task.validate_layout()
    if layout_errors:
        raise typer.BadParameter("invalid task: " + "; ".join(layout_errors))
    base = TrialRunner(
        task=task,
        agent=NoopAgent(),
        runs_root=runs_root,
        executor_kind=executor,
        ttl_minutes=30,
    ).run()
    oracle = TrialRunner(
        task=task,
        agent=OracleAgent(task),
        runs_root=runs_root,
        executor_kind=executor,
        ttl_minutes=30,
    ).run()
    regression_nonempty = all(
        any(name.startswith("regression.") for name in replay.assertions)
        for replay in oracle.record.replays
    )
    passed = (
        not base.score.resolved_at_1 and oracle.score.stable_resolved_at_1 and regression_nonempty
    )
    console.print(f"base-fails: {'yes' if not base.score.resolved_at_1 else 'no'}")
    console.print(
        f"gold-pass-x{task.replay.count}: {'yes' if oracle.score.stable_resolved_at_1 else 'no'}"
    )
    console.print(f"regression-oracle-nonempty: {'yes' if regression_nonempty else 'no'}")
    console.print(f"certification runs: {runs_root.resolve()}")
    if not passed:
        raise typer.Exit(1)


@task_app.command("audit-contract")
def audit_task_contract(
    specification_path: Annotated[
        Path, typer.Option("--specification", exists=True, readable=True)
    ],
    contract_path: Annotated[Path, typer.Option("--contract", exists=True, readable=True)],
    witness_set_path: Annotated[
        Path | None, typer.Option("--witness-set", exists=True, readable=True)
    ] = None,
) -> None:
    """Audit TaskSpecification ↔ AcceptanceContract ↔ optional WitnessSet."""

    specification = TaskSpecification.model_validate(_read_mapping(specification_path))
    contract = TaskAcceptanceContract.model_validate(_read_mapping(contract_path))
    failures = audit_acceptance_contract(specification, contract)
    if witness_set_path is not None:
        witness_set = WitnessSet.model_validate(_read_mapping(witness_set_path))
        failures.extend(audit_witness_set(specification, contract, witness_set))
    if failures:
        for failure in sorted(set(failures)):
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(
        f"[green]valid[/green] task={specification.task_id} "
        f"revision={specification.task_revision} obligations={len(contract.obligations)}"
    )


@task_app.command("qualify")
def qualify_task_command(
    specification_path: Annotated[
        Path, typer.Option("--specification", exists=True, readable=True)
    ],
    contract_path: Annotated[Path, typer.Option("--contract", exists=True, readable=True)],
    witness_set_path: Annotated[Path, typer.Option("--witness-set", exists=True, readable=True)],
    baseline_path: Annotated[Path, typer.Option("--baseline", exists=True, readable=True)],
    witness_replays_path: Annotated[
        Path, typer.Option("--witness-replays", exists=True, readable=True)
    ],
    mutations_path: Annotated[Path, typer.Option("--mutations", exists=True, readable=True)],
    controls_path: Annotated[Path, typer.Option("--negative-controls", exists=True, readable=True)],
    alternatives_path: Annotated[Path, typer.Option("--alternatives", exists=True, readable=True)],
    flakiness_path: Annotated[Path, typer.Option("--flakiness", exists=True, readable=True)],
    leakage_path: Annotated[Path, typer.Option("--leakage", exists=True, readable=True)],
    review_path: Annotated[
        Path | None, typer.Option("--review", exists=True, readable=True)
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("task-qualification-v0.1.json"),
) -> None:
    """Fail-closed Task/Verifier qualification; this never scores a Candidate."""

    report = qualify_task(
        specification=TaskSpecification.model_validate(_read_mapping(specification_path)),
        contract=TaskAcceptanceContract.model_validate(_read_mapping(contract_path)),
        witness_set=WitnessSet.model_validate(_read_mapping(witness_set_path)),
        baseline=BaselineDifferential.model_validate(_read_mapping(baseline_path)),
        witness_replays=[
            WitnessReplayResult.model_validate(item)
            for item in _read_sequence(witness_replays_path)
        ],
        mutations=[MutationOutcome.model_validate(item) for item in _read_sequence(mutations_path)],
        negative_controls=[
            NegativeControlOutcome.model_validate(item) for item in _read_sequence(controls_path)
        ],
        alternatives=[
            AlternativeValidSolutionOutcome.model_validate(item)
            for item in _read_sequence(alternatives_path)
        ],
        flakiness=VerifierFlakinessAudit.model_validate(_read_mapping(flakiness_path)),
        leakage=TaskLeakageAudit.model_validate(_read_mapping(leakage_path)),
        review=(
            HumanTaskQualificationReview.model_validate(_read_mapping(review_path))
            if review_path is not None
            else None
        ),
    )
    atomic_write_json(output, report.model_dump(mode="json"))
    console.print(
        f"status={report.status} failures={len(report.failure_codes)} output={output.resolve()}"
    )
    if report.status not in {"QUALIFIED", "QUALIFIED_WITH_SCOPE"}:
        raise typer.Exit(3 if report.status == "REVIEW_REQUIRED" else 2)


@task_app.command("seal")
def seal_task(
    specification_path: Annotated[
        Path, typer.Option("--specification", exists=True, readable=True)
    ],
    contract_path: Annotated[Path, typer.Option("--contract", exists=True, readable=True)],
    witness_set_path: Annotated[Path, typer.Option("--witness-set", exists=True, readable=True)],
    qualification_path: Annotated[
        Path, typer.Option("--qualification", exists=True, readable=True)
    ],
    review_path: Annotated[Path, typer.Option("--review", exists=True, readable=True)],
    verifier_bundle_sha256: Annotated[str, typer.Option("--verifier-bundle-sha256")],
    capability_policy_sha256: Annotated[str, typer.Option("--capability-policy-sha256")],
    capability_registry_sha256: Annotated[str, typer.Option("--capability-registry-sha256")],
    resource_envelope_sha256: Annotated[str, typer.Option("--resource-envelope-sha256")],
    topology_contract_sha256: Annotated[str, typer.Option("--topology-contract-sha256")],
    benchmark_cell_policy_sha256: Annotated[str, typer.Option("--benchmark-cell-policy-sha256")],
    runner_selection_policy_sha256: Annotated[
        str, typer.Option("--runner-selection-policy-sha256")
    ],
    benchmark_season: Annotated[str, typer.Option("--benchmark-season")],
    output: Annotated[Path, typer.Option("--output")] = Path("task-seal-v0.1.json"),
) -> None:
    """Seal only a qualified, human-approved v0.1 task."""

    seal = build_task_seal(
        specification=TaskSpecification.model_validate(_read_mapping(specification_path)),
        contract=TaskAcceptanceContract.model_validate(_read_mapping(contract_path)),
        witness_set=WitnessSet.model_validate(_read_mapping(witness_set_path)),
        qualification=TaskQualificationReport.model_validate(_read_mapping(qualification_path)),
        review=HumanTaskQualificationReview.model_validate(_read_mapping(review_path)),
        verifier_bundle_sha256=verifier_bundle_sha256,
        capability_policy_sha256=capability_policy_sha256,
        capability_registry_sha256=capability_registry_sha256,
        resource_envelope_sha256=resource_envelope_sha256,
        topology_contract_sha256=topology_contract_sha256,
        benchmark_cell_policy_sha256=benchmark_cell_policy_sha256,
        runner_selection_policy_sha256=runner_selection_policy_sha256,
        benchmark_season=benchmark_season,
    )
    atomic_write_json(output, seal.model_dump(mode="json"))
    console.print(f"[green]sealed[/green] task={seal.task_id} output={output.resolve()}")


@task_app.command("audit-seal")
def audit_task_seal_command(
    seal_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    seal = TaskSeal.model_validate(_read_mapping(seal_path))
    failures = audit_task_seal(seal)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(f"[green]valid-seal[/green] task={seal.task_id} revision={seal.task_revision}")


@schema_app.command("export")
def export_schemas(
    output: Annotated[Path, typer.Option("--output")] = Path("schemas"),
) -> None:
    write_schema_documents(output)
    console.print(f"wrote {len(schema_documents())} schemas to {output.resolve()}")


@schema_app.command("check")
def check_schemas(
    output: Annotated[Path, typer.Option("--output")] = Path("schemas"),
) -> None:
    """Fail when checked-in protocol schemas differ from the Pydantic models."""
    stale = stale_schema_names(output)
    if stale:
        for name in stale:
            console.print(f"[red]STALE[/red] {name}")
        raise typer.Exit(1)
    console.print(f"[green]fresh[/green] {len(schema_documents())} schema(s)")


@precedent_app.command("index")
def index_precedents(
    snapshot_path: Annotated[Path, typer.Option("--snapshot", exists=True, readable=True)],
    records_path: Annotated[Path, typer.Option("--records", exists=True, readable=True)],
    edges_path: Annotated[
        Path | None,
        typer.Option("--edges", exists=True, readable=True),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path(".infraswe/index.sqlite"),
) -> None:
    """Load a frozen snapshot plus normalized JSONL records into SQLite/FTS5."""

    snapshot = RepositorySnapshot.model_validate(_read_mapping(snapshot_path))
    records = [
        PrecedentRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    edges = (
        [
            PrecedentGraphEdge.model_validate_json(line)
            for line in edges_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if edges_path is not None
        else []
    )
    record_ids = {record.precedent_id for record in records}
    dangling_edges = [
        edge
        for edge in edges
        if edge.source_id not in record_ids or edge.target_id not in record_ids
    ]
    if dangling_edges:
        raise typer.BadParameter("all graph edge endpoints must exist in --records")
    with PrecedentStore(output) as store:
        store.set_metadata("schema_version", "0.5.1")
        store.set_metadata("snapshot", snapshot.model_dump(mode="json"))
        store.set_metadata("snapshot_sha256", canonical_sha256(snapshot))
        for record in records:
            store.upsert_record(record)
        for edge in edges:
            store.add_edge(edge)
    console.print(
        f"[green]indexed[/green] records={len(records)} edges={len(edges)} "
        f"snapshot={snapshot.repository}@{snapshot.revision} output={output.resolve()}"
    )


@precedent_app.command("footprint")
def extract_precedent_footprint(
    request_path: Annotated[Path, typer.Option("--request", exists=True, readable=True)],
    source_root: Annotated[
        Path,
        typer.Option("--source-root", exists=True, readable=True, file_okay=False),
    ],
    output: Annotated[Path, typer.Option("--output")] = Path("candidate-footprint.json"),
) -> None:
    """Extract deterministic symbol/build/lifecycle/domain anchors from frozen files."""

    request = FootprintExtractionRequest.model_validate(_read_mapping(request_path))
    root = source_root.resolve()
    sources: dict[str, str] = {}
    for relative in request.files:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise typer.BadParameter(f"footprint source is missing or escapes root: {relative}")
        sources[relative] = path.read_text(encoding="utf-8")
    footprint = extract_candidate_footprint(request, sources)
    atomic_write_json(output, footprint.model_dump(mode="json"))
    domain = "communication" if footprint.communication is not None else "memory-tiering"
    console.print(
        f"[green]extracted[/green] domain={domain} files={len(footprint.files)} "
        f"symbols={len(footprint.symbols)} output={output.resolve()}"
    )


@precedent_app.command("plan")
def plan_precedents(
    snapshot_path: Annotated[Path, typer.Option("--snapshot", exists=True, readable=True)],
    footprint_path: Annotated[Path, typer.Option("--footprint", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("query-plan.json"),
    forbidden_source_id: Annotated[list[str] | None, typer.Option("--forbid-source-id")] = None,
) -> None:
    """Freeze the deterministic multi-channel retrieval plan."""

    snapshot = RepositorySnapshot.model_validate(_read_mapping(snapshot_path))
    footprint = CandidateFootprint.model_validate(_read_mapping(footprint_path))
    plan = build_default_query_plan(
        target_snapshot_sha256=canonical_sha256(snapshot),
        footprint=footprint,
        corpus_cutoff=snapshot.corpus_cutoff,
        forbidden_source_ids=forbidden_source_id or (),
    )
    atomic_write_json(output, plan.model_dump(mode="json"))
    console.print(f"[green]planned[/green] passes={len(plan.passes)} output={output.resolve()}")


@precedent_app.command("retrieve")
def retrieve_precedents(
    index_path: Annotated[Path, typer.Option("--index", exists=True, readable=True)],
    footprint_path: Annotated[Path, typer.Option("--footprint", exists=True, readable=True)],
    plan_path: Annotated[Path, typer.Option("--plan", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("retrieval"),
) -> None:
    """Run deterministic retrieval, leakage audit, and PrecedentSet sealing."""

    footprint = CandidateFootprint.model_validate(_read_mapping(footprint_path))
    plan = QueryPlan.model_validate(_read_mapping(plan_path))
    with PrecedentStore(index_path) as store:
        if store.get_metadata("snapshot_sha256") != plan.target_snapshot_sha256:
            raise typer.BadParameter("query plan and precedent index bind different snapshots")
        snapshot_payload = store.get_metadata("snapshot")
        if not isinstance(snapshot_payload, dict):
            raise typer.BadParameter("precedent index is missing its repository snapshot")
        snapshot = RepositorySnapshot.model_validate(snapshot_payload)
        hits, fused, records = execute_retrieval(store, footprint, plan)
        leakage = audit_leakage(records, plan)
        edges = store.edges_between(leakage.allowed_precedent_ids)
    allowed_ids = set(leakage.allowed_precedent_ids)
    allowed_records = [record for record in records if record.precedent_id in allowed_ids]
    conflicts = detect_conflicts(allowed_records)
    coverage, trust = build_retrieval_assessment(
        snapshot=snapshot,
        footprint=footprint,
        plan=plan,
        records=allowed_records,
        conflicts=conflicts,
        leakage_audit=leakage,
    )
    atomic_write_json(
        output / "channel-results.json",
        [item.model_dump(mode="json") for item in hits],
    )
    atomic_write_json(
        output / "fused-ranking.json",
        [item.model_dump(mode="json") for item in fused],
    )
    atomic_write_json(
        output / "precedents.json",
        [item.model_dump(mode="json") for item in records],
    )
    atomic_write_json(output / "leakage-audit.json", leakage.model_dump(mode="json"))
    atomic_write_json(
        output / "conflict-sets.json",
        [item.model_dump(mode="json") for item in conflicts],
    )
    atomic_write_json(output / "coverage.json", coverage.model_dump(mode="json"))
    atomic_write_json(output / "trust-card.json", trust.model_dump(mode="json"))
    exclusions = {item.precedent_id: item.reason for item in leakage.exclusions}
    atomic_write_jsonl(
        output / "omitted.jsonl",
        [
            {
                "precedent_id": record.precedent_id,
                "reason": exclusions[record.precedent_id],
                "record": record.model_dump(mode="json"),
            }
            for record in records
            if record.precedent_id in exclusions
        ],
    )
    if leakage.status != "pass":
        console.print(f"[red]{leakage.status}[/red] leakage audit; no PrecedentSet was sealed")
        raise typer.Exit(1)
    precedent_set = build_precedent_set(
        draft_id=footprint.draft_id,
        draft_revision=footprint.draft_revision,
        target_snapshot_sha256=plan.target_snapshot_sha256,
        query_plan=plan,
        records=records,
        graph_edges=edges,
        conflicts=conflicts,
        leakage_audit=leakage,
        omitted_records_path="retrieval/omitted.jsonl",
    )
    rules = compile_rule_candidates(precedent_set.records)
    bundle = build_retrieval_bundle(
        snapshot=snapshot,
        footprint=footprint,
        query_plan=plan,
        channel_hits=hits,
        fused_ranking=fused,
        leakage_audit=leakage,
        coverage=coverage,
        rules=rules,
        trust=trust,
        precedent_set=precedent_set,
    )
    atomic_write_json(output / "precedent-set.json", precedent_set.model_dump(mode="json"))
    atomic_write_json(
        output / "rule-candidates.json",
        [item.model_dump(mode="json") for item in rules],
    )
    atomic_write_json(output / "retrieval-bundle.json", bundle.model_dump(mode="json"))
    console.print(
        f"[green]retrieved[/green] records={len(precedent_set.records)} "
        f"coverage={coverage.status} digest={precedent_set.digest} "
        f"output={output.resolve()}"
    )


@precedent_app.command("audit")
def audit_precedent_set(
    precedent_set_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify a sealed PrecedentSet canonical digest."""

    precedent_set = PrecedentSet.model_validate(_read_mapping(precedent_set_path))
    if not audit_precedent_set_digest(precedent_set):
        console.print("[red]invalid[/red] PrecedentSet digest mismatch")
        raise typer.Exit(1)
    console.print(f"[green]valid[/green] {precedent_set.digest}")


@precedent_app.command("review-rules")
def review_precedent_rules(
    rules_path: Annotated[Path, typer.Option("--rules", exists=True, readable=True)],
    decisions_path: Annotated[
        Path,
        typer.Option("--decisions", exists=True, readable=True),
    ],
    edited_rules_path: Annotated[
        Path | None,
        typer.Option("--edited-rules", exists=True, readable=True),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("reviewed-rules.json"),
    contract_output: Annotated[Path, typer.Option("--contract-output")] = Path(
        "contract-rules.json"
    ),
) -> None:
    """Apply digest-bound human decisions and emit D3-executable rules."""

    rules = [RuleCandidate.model_validate(item) for item in _read_sequence(rules_path)]
    decisions = [
        HumanRuleDecision.model_validate_json(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    edited_rules = (
        [
            RuleCandidate.model_validate_json(line)
            for line in edited_rules_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if edited_rules_path is not None
        else []
    )
    reviewed = apply_human_rule_decisions(
        rules,
        decisions,
        edited_rules=edited_rules,
    )
    executable = contract_executable_rules(reviewed)
    atomic_write_json(output, [item.model_dump(mode="json") for item in reviewed])
    atomic_write_json(
        contract_output,
        [item.model_dump(mode="json") for item in executable],
    )
    console.print(
        f"[green]reviewed[/green] decisions={len(decisions)} "
        f"contract_rules={len(executable)} output={output.resolve()}"
    )


@precedent_app.command("audit-bundle")
def audit_retrieval_bundle(
    bundle_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify a retrieval bundle and its nested PrecedentSet digests."""

    bundle = RetrievalBundle.model_validate(_read_mapping(bundle_path))
    valid = audit_retrieval_bundle_digest(bundle) and audit_precedent_set_digest(
        bundle.precedent_set
    )
    if not valid:
        console.print("[red]invalid[/red] retrieval bundle digest mismatch")
        raise typer.Exit(1)
    console.print(f"[green]valid[/green] {bundle.bundle_sha256}")


@judge_profile_app.command("validate")
def validate_judge_profile(
    profile_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    calibration_path: Annotated[
        Path | None,
        typer.Option("--calibration", exists=True, readable=True),
    ] = None,
    drift_path: Annotated[
        Path | None,
        typer.Option("--drift", exists=True, readable=True),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Validate snapshot, calibration, drift, panel, and weight-cap eligibility."""

    profile = JudgeProfile.model_validate(_read_mapping(profile_path))
    calibration = (
        JudgeCalibrationReport.model_validate(_read_mapping(calibration_path))
        if calibration_path is not None
        else None
    )
    drift = (
        JudgeDriftSentinel.model_validate(_read_mapping(drift_path))
        if drift_path is not None
        else None
    )
    failures = audit_profile_eligibility(
        profile,
        calibration=calibration,
        drift=drift,
    )
    status = (
        "eligible"
        if not failures and profile.authority == "bounded-score"
        else "audit-only"
        if not failures
        else "ineligible"
    )
    payload = {
        "schema_version": "0.5.3",
        "profile_id": profile.profile_id,
        "authority": profile.authority,
        "status": status,
        "failure_codes": failures,
    }
    if output is not None:
        atomic_write_json(output, payload)
    console.print(f"status={status} authority={profile.authority} failures={len(failures)}")
    for failure in failures:
        console.print(f"[red]FAIL[/red] {failure}")
    if failures:
        raise typer.Exit(2)


@judge_cell_app.command("seal")
def seal_judge_cell(
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, readable=True),
    ],
    rubric_path: Annotated[
        Path,
        typer.Option("--rubric", exists=True, readable=True),
    ],
    calibration_path: Annotated[
        Path,
        typer.Option("--calibration", exists=True, readable=True),
    ],
    drift_path: Annotated[
        Path,
        typer.Option("--drift", exists=True, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output")] = Path("judge-cell.json"),
    trust_output: Annotated[Path, typer.Option("--trust-output")] = Path("judge-trust.json"),
) -> None:
    """Seal an eligible multi-family Judge Cell and emit its trust card."""

    profile = JudgeProfile.model_validate(_read_mapping(profile_path))
    rubric = JudgeRubric.model_validate(_read_mapping(rubric_path))
    calibration = JudgeCalibrationReport.model_validate(_read_mapping(calibration_path))
    drift = JudgeDriftSentinel.model_validate(_read_mapping(drift_path))
    cell = build_judge_cell(profile, rubric, calibration, drift)
    trust = build_trust_card(
        profile,
        domain=rubric.domain,
        calibration=calibration,
        drift=drift,
        cell=cell,
    )
    atomic_write_json(output, cell.model_dump(mode="json"))
    atomic_write_json(trust_output, trust.model_dump(mode="json"))
    console.print(
        f"[green]sealed[/green] cell={cell.judge_cell_sha256} "
        f"trust={trust.status} output={output.resolve()}"
    )


@judge_cell_app.command("audit")
def audit_judge_cell_command(
    cell_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Recompute the Judge Cell digest."""

    cell = JudgeCell.model_validate(_read_mapping(cell_path))
    failures = audit_judge_cell(cell)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(f"[green]valid[/green] {cell.judge_cell_sha256}")


@judge_pack_app.command("build")
def build_judge_input_pack(
    spec_path: Annotated[
        Path,
        typer.Option("--spec", exists=True, readable=True),
    ],
    source_root: Annotated[
        Path,
        typer.Option("--source-root", exists=True, readable=True, file_okay=False),
    ],
    output: Annotated[Path, typer.Option("--output")] = Path("judge/input-pack"),
) -> None:
    """Build the minimal content-addressed pack and untrusted-content boundary."""

    spec = JudgeInputPackSpec.model_validate(_read_mapping(spec_path))
    manifest = build_input_pack(spec, source_root=source_root, output=output)
    console.print(
        f"[green]built[/green] artifacts={len(manifest.artifacts)} "
        f"digest={manifest.pack_sha256} output={output.resolve()}"
    )


@judge_pack_app.command("audit")
def audit_judge_input_pack(
    manifest_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Verify the pack digest, every content digest, and candidate boundaries."""

    manifest = JudgeInputPackManifest.model_validate(_read_mapping(manifest_path))
    failures = audit_input_pack(manifest, root=manifest_path.parent)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(f"[green]valid[/green] {manifest.pack_sha256}")


@judge_app.command("validate-output")
def validate_judge_output_command(
    output_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, readable=True),
    ],
    cell_path: Annotated[Path, typer.Option("--cell", exists=True, readable=True)],
    rubric_path: Annotated[
        Path,
        typer.Option("--rubric", exists=True, readable=True),
    ],
    pack_path: Annotated[Path, typer.Option("--pack", exists=True, readable=True)],
    member_id: Annotated[str, typer.Option("--member-id")],
    repetition: Annotated[int, typer.Option("--repetition", min=1)],
    decoding_seed: Annotated[int, typer.Option("--decoding-seed")],
    candidate_agent_family: Annotated[
        str | None,
        typer.Option("--candidate-agent-family"),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("judge-run.json"),
) -> None:
    """Resolve evidence refs and make a raw Judge output eligible or invalid."""

    run = validate_judge_output(
        JudgeOutput.model_validate(_read_mapping(output_path)),
        profile=JudgeProfile.model_validate(_read_mapping(profile_path)),
        cell=JudgeCell.model_validate(_read_mapping(cell_path)),
        rubric=JudgeRubric.model_validate(_read_mapping(rubric_path)),
        input_pack=JudgeInputPackManifest.model_validate(_read_mapping(pack_path)),
        member_id=member_id,
        repetition=repetition,
        decoding_seed=decoding_seed,
        candidate_agent_family=candidate_agent_family,
    )
    atomic_write_json(output, run.model_dump(mode="json"))
    console.print(
        f"status={run.validation_status} excluded={run.candidate_family_excluded} "
        f"failures={len(run.failure_codes)} output={output.resolve()}"
    )
    if run.validation_status != "valid":
        raise typer.Exit(2)


@judge_app.command("aggregate")
def aggregate_judge_runs(
    runs_dir: Annotated[
        Path,
        typer.Option("--runs", exists=True, readable=True, file_okay=False),
    ],
    profile_path: Annotated[
        Path,
        typer.Option("--profile", exists=True, readable=True),
    ],
    cell_path: Annotated[Path, typer.Option("--cell", exists=True, readable=True)],
    rubric_path: Annotated[
        Path,
        typer.Option("--rubric", exists=True, readable=True),
    ],
    pack_path: Annotated[Path, typer.Option("--pack", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("judge/aggregation.json"),
) -> None:
    """Apply the sealed weighted-median, repeat, family, and abstention policy."""

    paths = sorted(runs_dir.glob("*.json"))
    if not paths:
        raise typer.BadParameter("--runs contains no JSON JudgeRun records")
    runs = [JudgeRunRecord.model_validate(_read_mapping(path)) for path in paths]
    aggregation = aggregate_panel(
        runs,
        profile=JudgeProfile.model_validate(_read_mapping(profile_path)),
        cell=JudgeCell.model_validate(_read_mapping(cell_path)),
        rubric=JudgeRubric.model_validate(_read_mapping(rubric_path)),
        input_pack=JudgeInputPackManifest.model_validate(_read_mapping(pack_path)),
    )
    atomic_write_json(output, aggregation.model_dump(mode="json"))
    console.print(
        f"status={aggregation.status} criteria={len(aggregation.criteria)} "
        f"top-level={aggregation.top_level_score_status} output={output.resolve()}"
    )
    if aggregation.status != "official":
        raise typer.Exit(2)


@judge_app.command("project")
def project_judge_semantic_residual(
    rubric_path: Annotated[Path, typer.Option("--rubric", exists=True, readable=True)],
    aggregation_path: Annotated[
        Path,
        typer.Option("--aggregation", exists=True, readable=True),
    ],
    deterministic_values_path: Annotated[
        Path,
        typer.Option("--deterministic-values", exists=True, readable=True),
    ],
    infra_cert: Annotated[str, typer.Option("--infra-cert")],
    output: Annotated[Path, typer.Option("--output")] = Path("judge/semantic-projection.json"),
) -> None:
    """Project criterion-level Judge evidence into P/M/U; never create Judge-100."""

    deterministic_payload = _read_mapping(deterministic_values_path)
    values = {str(name): float(value) for name, value in deterministic_payload.items()}
    projection = build_score_projection(
        JudgeRubric.model_validate(_read_mapping(rubric_path)),
        JudgeAggregation.model_validate(_read_mapping(aggregation_path)),
        deterministic_values=values,
        infra_cert_status=infra_cert,
    )
    atomic_write_json(output, projection.model_dump(mode="json"))
    console.print(
        f"status={projection.status} components={','.join(sorted(projection.components))} "
        f"output={output.resolve()}"
    )
    if projection.status != "official":
        raise typer.Exit(2)


@draft_app.command("validate")
def validate_draft(
    draft_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate a JSON or YAML Draft against the v0.5 state invariants."""

    draft = parse_draft_document(
        draft_path.read_text(encoding="utf-8"), source=str(draft_path.resolve())
    )
    console.print(
        f"[green]valid[/green] {draft.draft.id} revision={draft.draft.revision} "
        f"state={draft.draft.state}"
    )


@draft_app.command("defaults")
def materialize_default_drafts(
    output: Annotated[Path, typer.Option("--output")] = Path("catalog/default-drafts-v0.5"),
) -> None:
    """Materialize the pinned ten-project default Draft catalog."""

    catalog = build_default_catalog()
    atomic_write_json(output / "catalog.json", catalog.model_dump(mode="json"))
    for project, entry in catalog.entries.items():
        project_dir = output / project
        atomic_write_json(
            project_dir / "project-profile.json", entry.profile.model_dump(mode="json")
        )
        for kind, artifact in entry.artifacts.items():
            atomic_write_json(
                project_dir / "contracts" / f"{kind}.json",
                artifact.model_dump(mode="json"),
            )
    console.print(
        f"status={catalog.status} projects={','.join(catalog.default_order)} "
        f"output={output.resolve()}"
    )


@draft_app.command("candidates")
def materialize_default_candidates(
    output: Annotated[Path, typer.Option("--output")] = Path("catalog/default-candidates-v0.5"),
) -> None:
    """Materialize the role-separated, build-lazy default candidate registry."""

    registry = build_default_candidate_registry()
    atomic_write_json(output / "registry.json", registry.model_dump(mode="json"))
    console.print(
        f"status={registry.status} candidates={len(registry.candidates)} "
        f"rules={len(registry.rules)} output={output.resolve()}"
    )


@draft_app.command("system-profiles")
def materialize_system_profiles(
    output: Annotated[Path, typer.Option("--output")] = Path("catalog/system-drafts-v0.5.2"),
) -> None:
    """Materialize communication and memory-tiering Draft profiles."""

    catalog = build_system_profile_catalog()
    atomic_write_json(output / "catalog.json", catalog.model_dump(mode="json"))
    for profile_id, profile in catalog.profiles.items():
        atomic_write_json(
            output / "profiles" / f"{profile_id}.json",
            profile.model_dump(mode="json"),
        )
    console.print(
        f"status={catalog.status} profiles={len(catalog.profiles)} output={output.resolve()}"
    )


@draft_app.command("resolve")
def resolve_draft_command(
    local_draft: Annotated[Path | None, typer.Option("--local-draft")] = None,
    remote_repository: Annotated[str | None, typer.Option("--remote-repository")] = None,
    remote_revision: Annotated[str, typer.Option("--remote-revision")] = "HEAD",
    remote_path: Annotated[str | None, typer.Option("--remote-path")] = None,
    candidate_path: Annotated[Path | None, typer.Option("--candidate")] = None,
    default_project: Annotated[str | None, typer.Option("--default-project")] = None,
    target_hint: Annotated[str | None, typer.Option("--target-hint")] = None,
    created_by: Annotated[str, typer.Option("--created-by")] = "infraswe-cli",
    output: Annotated[Path, typer.Option("--output")] = Path("draft-source-resolution.json"),
) -> None:
    """Resolve local > remote Git > pinned built-in default Draft."""

    if (remote_repository is None) != (remote_path is None):
        raise typer.BadParameter("--remote-repository and --remote-path must be provided together")
    remote = (
        RemoteGitDraftLocation(
            repository=remote_repository,
            revision=remote_revision,
            path=remote_path,
        )
        if remote_repository is not None and remote_path is not None
        else None
    )
    candidate = (
        DraftCandidate.model_validate(_read_mapping(candidate_path))
        if candidate_path is not None
        else None
    )
    if default_project is not None and default_project not in DEFAULT_PROJECT_ORDER:
        raise typer.BadParameter(
            "--default-project must be one of: " + ", ".join(DEFAULT_PROJECT_ORDER)
        )
    resolution = resolve_draft(
        local_draft=local_draft,
        remote_git_draft=remote,
        candidate=candidate,
        default_project=default_project,
        target_hint=target_hint,
        created_by=created_by,
    )
    atomic_write_json(output, resolution.model_dump(mode="json"))
    console.print(
        f"source={resolution.source_kind} target="
        f"{resolution.selected_default_project or 'explicit'} "
        f"state={resolution.draft.draft.state} output={output.resolve()}"
    )


@draft_app.command("seal")
def seal_draft_command(
    draft_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    review_path: Annotated[Path, typer.Option("--review", exists=True, readable=True)],
    performance_target_sha256: Annotated[str, typer.Option("--performance-target-sha256")],
    sealed_by: Annotated[str, typer.Option("--sealed-by")],
    output: Annotated[Path, typer.Option("--output")] = Path("sealed-draft.json"),
) -> None:
    """Seal only a D5 Draft with a matching approved project-maintainer review."""

    draft = parse_draft_document(
        draft_path.read_text(encoding="utf-8"), source=str(draft_path.resolve())
    )
    review = HumanReviewRecord.model_validate(_read_mapping(review_path))
    sealed = seal_draft(
        draft,
        review,
        performance_target_sha256=performance_target_sha256,
        sealed_by=sealed_by,
    )
    atomic_write_json(output, sealed.model_dump(mode="json"))
    console.print(f"[green]sealed[/green] {sealed.draft_id} output={output.resolve()}")


@draft_app.command("audit-seal")
def audit_draft_seal(
    seal_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Recompute the canonical seal digest and fail on mutation."""

    seal = SealedDraft.model_validate(_read_mapping(seal_path))
    failures = audit_seal(seal)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(1)
    console.print(f"[green]valid-seal[/green] {seal.draft_id} revision={seal.draft_revision}")


@artifact_app.command("lint-policy")
def lint_artifact_policy(
    policy_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    policy = ArtifactPolicy.model_validate(_read_mapping(policy_path))
    failures = audit_artifact_policy(policy)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(f"[green]valid-policy[/green] {policy.policy_id}")


@artifact_app.command("collect")
def collect_artifact_command(
    root: Annotated[Path, typer.Option("--root", exists=True, file_okay=False)],
    requested_path: Annotated[Path, typer.Option("--requested", exists=True, readable=True)],
    policy_path: Annotated[Path, typer.Option("--policy", exists=True, readable=True)],
    freeze_path: Annotated[Path, typer.Option("--freeze", exists=True, readable=True)],
    task_seal_sha256: Annotated[str, typer.Option("--task-seal-sha256")],
    candidate_id: Annotated[str, typer.Option("--candidate-id")],
    output: Annotated[Path, typer.Option("--output")] = Path("candidate-artifacts"),
) -> None:
    requested = _read_mapping(requested_path)
    if not all(isinstance(name, str) and isinstance(path, str) for name, path in requested.items()):
        raise typer.BadParameter("--requested must map logical names to relative paths")
    manifest = collect_candidate_artifacts(
        root=root,
        requested=requested,
        output=output,
        policy=ArtifactPolicy.model_validate(_read_mapping(policy_path)),
        task_seal_sha256=task_seal_sha256,
        candidate_id=candidate_id,
        freeze=WorkspaceFreezeAttestation.model_validate(_read_mapping(freeze_path)),
    )
    manifest_path = output / "artifact-manifest.json"
    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    console.print(f"artifacts={len(manifest.artifacts)} manifest={manifest_path.resolve()}")


@artifact_app.command("inspect-manifest")
def inspect_artifact_manifest(
    manifest_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    root: Annotated[Path | None, typer.Option("--root", exists=True, file_okay=False)] = None,
) -> None:
    manifest = CandidateArtifactManifest.model_validate(_read_mapping(manifest_path))
    failures = audit_candidate_manifest(manifest, root=root)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(
        f"[green]valid-manifest[/green] candidate={manifest.candidate_id} "
        f"artifacts={len(manifest.artifacts)}"
    )


@evidence_app.command("verify")
def verify_evidence_pack(
    pack_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    trial_seal_path: Annotated[Path, typer.Option("--trial-seal", exists=True, readable=True)],
    root: Annotated[Path | None, typer.Option("--root", exists=True, file_okay=False)] = None,
) -> None:
    pack = EvidencePackManifest.model_validate(_read_mapping(pack_path))
    trial_seal = TrialSeal.model_validate(_read_mapping(trial_seal_path))
    failures = audit_evidence_pack(pack, trial_seal=trial_seal, root=root)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(
        f"[green]valid-evidence-pack[/green] evidence={len(pack.artifacts)} "
        f"digest={pack.evidence_pack_sha256}"
    )


@evidence_app.command("trace-score")
def trace_score_evidence(
    binding_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pack_path: Annotated[Path, typer.Option("--pack", exists=True, readable=True)],
) -> None:
    binding = ScoreEvidenceBinding.model_validate(_read_mapping(binding_path))
    pack = EvidencePackManifest.model_validate(_read_mapping(pack_path))
    failures = audit_score_binding(binding, pack)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(
        f"[green]valid-score-binding[/green] component={binding.component_id} "
        f"refs={len(binding.evidence_refs)}"
    )


@capability_app.command("registry-validate")
def validate_capability_registry(
    registry_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    registry = CapabilityRegistry.model_validate(_read_mapping(registry_path))
    failures = audit_registry(registry)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(
        f"[green]valid-registry[/green] definitions={len(registry.definitions)} "
        f"digest={registry.registry_sha256}"
    )


@capability_app.command("resolve")
def resolve_capability_command(
    task_seal_sha256: Annotated[str, typer.Option("--task-seal-sha256")],
    candidate_sha256: Annotated[str, typer.Option("--candidate-sha256")],
    registry_path: Annotated[Path, typer.Option("--registry", exists=True, readable=True)],
    contract_path: Annotated[Path, typer.Option("--contract", exists=True, readable=True)],
    declaration_path: Annotated[Path, typer.Option("--declaration", exists=True, readable=True)],
    runners_path: Annotated[Path, typer.Option("--runners", exists=True, readable=True)],
    snapshots_path: Annotated[Path, typer.Option("--snapshots", exists=True, readable=True)],
    attestations_path: Annotated[Path, typer.Option("--attestations", exists=True, readable=True)],
    resources_path: Annotated[
        Path, typer.Option("--resource-envelopes", exists=True, readable=True)
    ],
    topology_contracts_path: Annotated[
        Path, typer.Option("--topology-contracts", exists=True, readable=True)
    ],
    topology_graphs_path: Annotated[
        Path, typer.Option("--topology-graphs", exists=True, readable=True)
    ],
    policy_path: Annotated[Path, typer.Option("--policy", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("capability-resolution-v0.1.json"),
) -> None:
    snapshots = [RunnerSnapshot.model_validate(item) for item in _read_sequence(snapshots_path)]
    resources = [ResourceEnvelope.model_validate(item) for item in _read_sequence(resources_path)]
    topology_contracts = [
        TopologyContract.model_validate(item) for item in _read_sequence(topology_contracts_path)
    ]
    topology_graphs = [
        TopologyGraph.model_validate(item) for item in _read_sequence(topology_graphs_path)
    ]
    resolution = resolve_capabilities(
        task_seal_sha256=task_seal_sha256,
        candidate_sha256=candidate_sha256,
        registry=CapabilityRegistry.model_validate(_read_mapping(registry_path)),
        contract=CapabilityContract.model_validate(_read_mapping(contract_path)),
        declaration=CandidateCapabilityDeclaration.model_validate(_read_mapping(declaration_path)),
        runner_manifests=[
            RunnerManifest.model_validate(item) for item in _read_sequence(runners_path)
        ],
        runner_snapshots={item.runner_manifest_sha256: item for item in snapshots},
        attestations=[
            CapabilityAttestation.model_validate(item) for item in _read_sequence(attestations_path)
        ],
        resource_envelopes={item.envelope_sha256: item for item in resources},
        topology_contracts={item.contract_sha256: item for item in topology_contracts},
        topology_graphs={item.graph_sha256: item for item in topology_graphs},
        policy=RunnerSelectionPolicy.model_validate(_read_mapping(policy_path)),
    )
    atomic_write_json(output, resolution.model_dump(mode="json"))
    console.print(
        f"status={resolution.status} probes={len(resolution.required_probes)} "
        f"output={output.resolve()}"
    )
    exit_codes = {
        "eligible": 0,
        "unresolved": 3,
        "runner-contradiction": 7,
        "candidate-declaration-ineligible": 9,
        "unschedulable": 5,
        "capacity-unavailable": 5,
    }
    if exit_codes[resolution.status]:
        raise typer.Exit(exit_codes[resolution.status])


@capability_app.command("audit-resolution")
def audit_capability_resolution_command(
    resolution_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    resolution = CapabilityResolution.model_validate(_read_mapping(resolution_path))
    failures = audit_capability_resolution(resolution)
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(2)
    console.print(
        f"[green]valid-resolution[/green] status={resolution.status} "
        f"digest={resolution.resolution_sha256}"
    )


@cell_app.command("compare")
def compare_benchmark_cells(
    left_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    right_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    left = BenchmarkCellManifest.model_validate(_read_mapping(left_path))
    right = BenchmarkCellManifest.model_validate(_read_mapping(right_path))
    try:
        assert_raw_performance_comparable(left, right)
    except ValueError as error:
        console.print(f"[red]NOT_COMPARABLE[/red] {error}")
        raise typer.Exit(5) from error
    console.print(f"[green]RAW_PERFORMANCE_COMPARABLE[/green] cell={left.comparison_cell_digest}")


@lease_app.command("preflight")
def lease_preflight(
    profile: Annotated[str, typer.Option("--profile")],
    output: Annotated[Path, typer.Option("--output")] = Path("hardware-manifest.json"),
    profiles_dir: Annotated[Path, typer.Option("--profiles-dir")] = Path("profiles"),
) -> None:
    manifest = write_hardware_manifest(output, profile)
    profile_path = Path(profile)
    if not profile_path.is_file():
        profile_path = profiles_dir / f"{profile}.toml"
    if not profile_path.is_file():
        raise typer.BadParameter(f"hardware profile not found: {profile}")
    validation = validate_hardware_manifest(HardwareProfile.load(profile_path), manifest)
    manifest["validation"] = validation
    atomic_write_json(output, manifest)
    console.print(
        f"profile={profile} gpu_count={manifest['gpu_count']} "
        f"passed={validation['passed']} manifest={output.resolve()}"
    )
    for warning in validation["warnings"]:
        console.print(f"[yellow]WARNING[/yellow] {warning}")
    for error in validation["errors"]:
        console.print(f"[red]ERROR[/red] {error}")
    if not validation["passed"]:
        raise typer.Exit(1)


@rl_policy_app.command("validate")
def validate_rl_policy(
    policy_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    policy = _read_sealed(policy_path, PolicySnapshot, exit_code=4)
    console.print(
        f"[green]valid-policy[/green] id={policy.policy_id} "
        f"version={policy.policy_version} digest={policy.policy_snapshot_sha256}"
    )


@rl_harness_app.command("validate")
def validate_rl_harness(
    harness_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    harness = _read_sealed(harness_path, AgentHarnessProfile, exit_code=4)
    console.print(
        f"[green]valid-harness[/green] id={harness.harness_id} "
        f"digest={harness.harness_profile_sha256}"
    )


@rl_episode_app.command("inspect")
def inspect_rl_episode(
    episode_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    episode = _read_sealed(episode_path, EpisodeSeal, exit_code=6)
    console.print(
        f"[green]valid-episode-seal[/green] id={episode.episode_id} "
        f"status={episode.status} mask={episode.training_mask}"
    )


@rl_episode_app.command("inspect-outcome")
def inspect_rl_episode_outcome(
    outcome_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    episode_path: Annotated[Path, typer.Option("--episode", exists=True, readable=True)],
    reward_path: Annotated[Path, typer.Option("--reward", exists=True, readable=True)],
) -> None:
    outcome = _read_sealed(outcome_path, EpisodeOutcomeSeal, exit_code=6)
    episode = _read_sealed(episode_path, EpisodeSeal, exit_code=6)
    reward = _read_sealed(reward_path, RewardPack, exit_code=6)
    failures = audit_episode_outcome_seal(
        outcome,
        episode_seal=episode,
        reward_pack=reward,
    )
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(6)
    console.print(
        "[green]valid-episode-outcome[/green] "
        f"episode={outcome.episode_seal_sha256} reward={outcome.reward_pack_sha256}"
    )


@rl_reward_app.command("inspect")
def inspect_rl_reward(
    reward_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    reward = _read_sealed(reward_path, RewardPack, exit_code=3)
    console.print(
        f"[green]valid-reward-pack[/green] validity={reward.validity} "
        f"band={reward.anchor.scalar_band} mask={reward.training_mask} "
        f"official-score-affected={reward.training_reward_affects_official_score}"
    )


@rl_batch_app.command("validate")
def validate_rl_batch_command(
    batch_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    training_run_path: Annotated[Path, typer.Option("--training-run", exists=True, readable=True)],
    algorithm_path: Annotated[Path, typer.Option("--algorithm", exists=True, readable=True)],
    known_policy_snapshot: Annotated[
        list[str] | None, typer.Option("--known-policy-snapshot")
    ] = None,
    revoked_reward_pack: Annotated[list[str] | None, typer.Option("--revoked-reward-pack")] = None,
) -> None:
    batch = _read_sealed(batch_path, RLBatchManifest, exit_code=10)
    training_run = _read_sealed(training_run_path, TrainingRunSeal, exit_code=10)
    algorithm = _read_sealed(algorithm_path, AlgorithmProfile, exit_code=10)
    known = set(known_policy_snapshot or ())
    known.update(
        {
            batch.target_policy_snapshot_sha256,
            batch.proximal_policy_snapshot_sha256,
        }
    )
    failures = validate_rl_batch(
        batch,
        training_run=training_run,
        algorithm=algorithm,
        known_policy_snapshots=known,
        revoked_reward_packs=set(revoked_reward_pack or ()),
    )
    if failures:
        for failure in failures:
            console.print(f"[red]FAIL[/red] {failure}")
        raise typer.Exit(10)
    console.print(
        f"[green]valid-rl-batch[/green] members={len(batch.members)} "
        f"tokens={batch.valid_token_count} steps={batch.valid_step_count}"
    )


@rl_legacy_app.command("migrate")
def migrate_rl_legacy_experience(
    source_root: Annotated[
        list[Path], typer.Option("--source-root", exists=True, file_okay=False, readable=True)
    ],
    manifest_id: Annotated[str, typer.Option("--manifest-id")],
    output: Annotated[Path, typer.Option("--output")] = Path(
        "legacy-experience-manifest-v0.6.json"
    ),
) -> None:
    try:
        manifest = build_legacy_experience_manifest(
            source_root,
            manifest_id=manifest_id,
        )
    except ValueError as error:
        console.print(f"[red]INVALID[/red] {error}")
        raise typer.Exit(2) from error
    atomic_write_json(output, manifest.model_dump(mode="json"))
    console.print(
        f"[green]migrated-legacy-experience[/green] attempts={manifest.attempted_records} "
        f"valid={manifest.valid_records} invalid={manifest.invalid_records} "
        f"policy-gradient={manifest.policy_gradient_eligible_records} "
        f"output={output.resolve()}"
    )


@rl_fabric_app.command("validate")
def validate_rl_fabric(
    fabric_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    fabric = _read_sealed(fabric_path, RolloutFabricProfile, exit_code=5)
    console.print(
        f"[green]valid-rollout-fabric[/green] id={fabric.fabric_id} "
        f"pools={len(fabric.pools)} isolation={fabric.measurement_isolation}"
    )


@rl_fabric_app.command("preflight")
def preflight_rl_fabric(
    gpu_count: Annotated[int, typer.Option("--gpu-count", min=0)],
    gpu_topology_attested: Annotated[bool, typer.Option("--gpu-topology-attested")] = False,
    rootless_sandbox_enforced: Annotated[bool, typer.Option("--rootless-sandbox-enforced")] = False,
    exact_token_gateway_available: Annotated[
        bool, typer.Option("--exact-token-gateway-available")
    ] = False,
    hosted_policy_exact_tokens_available: Annotated[
        bool, typer.Option("--hosted-policy-exact-tokens-available")
    ] = False,
    trainer_adapter_available: Annotated[bool, typer.Option("--trainer-adapter-available")] = False,
    distributed_gang_enforced: Annotated[bool, typer.Option("--distributed-gang-enforced")] = False,
    output: Annotated[Path, typer.Option("--output")] = Path("runtime-capability-report-v0.6.json"),
) -> None:
    report = build_runtime_capability_report(
        gpu_count=gpu_count,
        gpu_topology_attested=gpu_topology_attested,
        rootless_sandbox_enforced=rootless_sandbox_enforced,
        exact_token_gateway_available=exact_token_gateway_available,
        hosted_policy_exact_tokens_available=hosted_policy_exact_tokens_available,
        trainer_adapter_available=trainer_adapter_available,
        distributed_gang_enforced=distributed_gang_enforced,
    )
    atomic_write_json(output, report.model_dump(mode="json"))
    console.print(
        f"production-ready={str(report.production_ready).lower()} "
        f"gpu-count={report.gpu_count} reasons={','.join(report.unavailable_reasons) or 'none'} "
        f"output={output.resolve()}"
    )
    if not report.production_ready:
        raise typer.Exit(5)


@rl_train_app.command("validate-seal")
def validate_rl_training_seal(
    training_run_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    training_run = _read_sealed(training_run_path, TrainingRunSeal, exit_code=11)
    console.print(
        f"[green]valid-training-run-seal[/green] id={training_run.run_id} "
        f"attempts={training_run.budgets.total_episode_attempts}"
    )


@training_app.command("probe")
def training_probe(
    output: Annotated[Path, typer.Option("--output")] = Path("training-capabilities.json"),
) -> None:
    """Probe installed framework/runtime capabilities without claiming cell certification."""

    manifest = probe_capabilities()
    atomic_write_json(output, manifest.model_dump(mode="json"))
    console.print(f"training-capability-status={manifest.status} output={output.resolve()}")
    if manifest.status in {"protocol_only", "not_ready"}:
        raise typer.Exit(2)


@training_app.command("verify")
def training_verify(
    evidence: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("training-cert.json"),
) -> None:
    """Evaluate TrainingCert hard gates; missing evidence remains unresolved."""

    bundle = TrainingEvidenceBundle.model_validate_json(evidence.read_text(encoding="utf-8"))
    certification = verify_training_evidence(bundle)
    atomic_write_json(output, certification.model_dump(mode="json"))
    console.print(
        f"TrainingCert={certification.status} failures={len(certification.failure_codes)} "
        f"output={output.resolve()}"
    )
    if certification.status != "pass":
        raise typer.Exit(2)


@training_app.command("score")
def training_score(
    evidence: Annotated[Path, typer.Argument(exists=True, readable=True)],
    score_input: Annotated[Path, typer.Option("--score-input", exists=True, readable=True)],
    comparability: Annotated[Path, typer.Option("--comparability", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("training-result.json"),
) -> None:
    """Build the training result through the frozen v0.4 C/U/M scorer."""

    bundle = TrainingEvidenceBundle.model_validate_json(evidence.read_text(encoding="utf-8"))
    certification = verify_training_evidence(bundle)
    result = build_training_result(
        bundle=bundle,
        certification=certification,
        score_input=TrainingScoreInput.model_validate_json(score_input.read_text(encoding="utf-8")),
        comparability=TrainingComparability.model_validate_json(
            comparability.read_text(encoding="utf-8")
        ),
    )
    atomic_write_json(output, result.model_dump(mode="json"))
    deployability = result.v04_score.deployability
    console.print(
        f"TrainingCert={certification.status} "
        f"Deployability={deployability.status if deployability else 'unresolved'} "
        f"output={output.resolve()}"
    )
    if certification.status != "pass" or deployability is None or deployability.score_100 is None:
        raise typer.Exit(2)


@communication_app.command("import-native")
def communication_import_native(
    framework: Annotated[str, typer.Option("--framework")],
    source: Annotated[list[Path], typer.Option("--source", exists=True, readable=True)],
    manifest: Annotated[Path | None, typer.Option("--manifest", exists=True, readable=True)] = None,
    companion: Annotated[
        list[Path] | None, typer.Option("--companion", exists=True, readable=True)
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("native-communication-import.json"),
    trace_output: Annotated[Path | None, typer.Option("--trace-output")] = None,
) -> None:
    """Import native traces without upgrading incomplete timing evidence."""
    from infraswe.telemetry.communication_native import import_native_communication_trace

    try:
        result = import_native_communication_trace(
            framework, source, manifest_path=manifest, companion_paths=companion
        )
    except (OSError, TypeError, ValueError) as error:
        console.print(f"[red]INVALID[/red] {error}")
        raise typer.Exit(2) from error
    atomic_write_json(output, result.model_dump(mode="json"))
    if result.trace_set is not None and trace_output is not None:
        atomic_write_json(trace_output, result.trace_set.model_dump(mode="json"))
    console.print(
        f"status={result.status} records={result.observed_record_count} output={output.resolve()}"
    )
    if result.status != "ready":
        raise typer.Exit(2)


@communication_app.command("phase-regression")
def communication_phase_regression(
    baseline: Annotated[Path, typer.Option("--baseline", exists=True, readable=True)],
    candidate: Annotated[Path, typer.Option("--candidate", exists=True, readable=True)],
    policy: Annotated[Path | None, typer.Option("--policy", exists=True, readable=True)] = None,
    regime: Annotated[str, typer.Option("--regime")] = "normal",
    load_ratio: Annotated[float, typer.Option("--load-ratio", min=0.000001)] = 0.5,
    output: Annotated[Path, typer.Option("--output")] = Path("communication-phase-regression.json"),
) -> None:
    """Compare two traces in one exact cell and emit a load-cell regression result."""

    try:
        baseline_trace = CommunicationPhaseTraceSet.model_validate(_read_mapping(baseline))
        candidate_trace = CommunicationPhaseTraceSet.model_validate(_read_mapping(candidate))
        regression_policy = (
            CommunicationPhaseRegressionPolicy.model_validate(_read_mapping(policy))
            if policy is not None
            else CommunicationPhaseRegressionPolicy()
        )
        result = evaluate_communication_phase_regression(
            baseline_trace,
            candidate_trace,
            regression_policy,
            regime=regime,
            load_ratio=load_ratio,
        )
    except (TypeError, ValueError, ValidationError) as error:
        console.print(f"[red]INVALID[/red] {error}")
        raise typer.Exit(2) from error
    atomic_write_json(output, result.model_dump(mode="json"))
    console.print(
        f"status={result.status} world_size={result.world_size} "
        f"cell={result.cell_identity_sha256} output={output.resolve()}"
    )
    if result.status == "fail":
        raise typer.Exit(1)
    if result.status == "unresolved":
        raise typer.Exit(2)


@app.command("report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    record = TrialRecord.model_validate_json((run_dir / "protocol.json").read_text())
    score = ScoreResult.model_validate_json((run_dir / "score.json").read_text())
    write_reports(run_dir, record, score)
    console.print(f"wrote {run_dir / 'report.md'} and {run_dir / 'index.html'}")


@app.command("serve")
def serve_command(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8080,
) -> None:
    """Serve a run or runs directory; bind loopback by default for SSH forwarding."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory.resolve())
    )
    server = http.server.ThreadingHTTPServer((host, port), handler)
    console.print(f"serving {directory.resolve()} at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command("version")
def version_command() -> None:
    console.print(__version__)


if __name__ == "__main__":
    app()
