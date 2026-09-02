from __future__ import annotations

import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from infraswe.agents import Agent, AgentContext
from infraswe.environments import DockerExecutor, LocalExecutor
from infraswe.environments.hardware_manifest import write_hardware_manifest
from infraswe.io import atomic_write_json, git_provenance, sha256_tree, utc_now
from infraswe.lease import BudgetExceeded, BudgetGuard, LocalLeaseBroker
from infraswe.models.artifact import ArtifactManifest
from infraswe.models.score import ScoreResult
from infraswe.models.task import TaskPackage
from infraswe.models.trial import FailureKind, TrialRecord, TrialState
from infraswe.runner.artifacts import collect_agent_artifacts, initialize_fixture_repository
from infraswe.runner.lifecycle import Lifecycle
from infraswe.scoring.report import write_reports
from infraswe.scoring.score import score_trial
from infraswe.verifier import Verifier


@dataclass(frozen=True)
class TrialRunResult:
    run_dir: Path
    record: TrialRecord
    score: ScoreResult


class TrialRunner:
    def __init__(
        self,
        *,
        task: TaskPackage,
        agent: Agent,
        runs_root: Path,
        executor_kind: str = "docker",
        ttl_minutes: int = 180,
        max_infra_cost_usd: float | None = None,
    ) -> None:
        if executor_kind not in {"docker", "local"}:
            raise ValueError("executor must be 'docker' or 'local'")
        self.task = task
        self.agent = agent
        self.runs_root = runs_root.resolve()
        self.executor_kind = executor_kind
        self.ttl_minutes = ttl_minutes
        self.max_infra_cost_usd = (
            task.budget.max_infra_cost_usd if max_infra_cost_usd is None else max_infra_cost_usd
        )

    def run(self) -> TrialRunResult:
        layout_errors = self.task.validate_layout()
        if layout_errors:
            raise ValueError("invalid task package: " + "; ".join(layout_errors))
        trial_id = self._new_trial_id()
        run_dir = self.runs_root / trial_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for relative in (
            "agent",
            "verifier",
            "evidence/logs",
            "evidence/metrics",
            "evidence/traces",
            "evidence/profiles",
            "evidence/config-diff",
        ):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)

        record = TrialRecord(trial_id=trial_id, task_id=self.task.task.id)
        lifecycle = Lifecycle(record, run_dir)
        broker = LocalLeaseBroker()
        lease = None
        started = time.monotonic()
        guard = BudgetGuard(self.task.budget)
        executor = DockerExecutor() if self.executor_kind == "docker" else LocalExecutor()
        manifest: ArtifactManifest | None = None
        try:
            lifecycle.transition(TrialState.LEASING, self.task.environment.profile)
            lease = broker.acquire(
                profile=self.task.environment.profile,
                ttl_minutes=self.ttl_minutes,
                max_cost_usd=self.max_infra_cost_usd,
                output=run_dir / "lease.json",
            )
            write_hardware_manifest(
                run_dir / "hardware-manifest.json", self.task.environment.profile
            )
            lifecycle.transition(TrialState.SETUP, f"executor={self.executor_kind}")

            with tempfile.TemporaryDirectory(prefix="infraswe-agent-") as temporary:
                workspace = Path(temporary) / "repo"
                shutil.copytree(self.task.resolve(self.task.execution.repo), workspace)
                base_commit = initialize_fixture_repository(workspace)
                task_payload = self.task.model_dump(mode="json")
                task_payload["resolved_base_commit"] = base_commit
                task_payload["agent"] = self.agent.name
                task_payload["executor"] = self.executor_kind
                task_payload["harness_git"] = git_provenance(self.task.package_dir)
                task_payload["digests"] = {
                    "task_package_sha256": sha256_tree(self.task.package_dir),
                    "fixture_sha256": sha256_tree(self.task.resolve(self.task.execution.repo)),
                    "workload_sha256": sha256_tree(self.task.resolve("workload")),
                    "faults_sha256": sha256_tree(self.task.resolve("faults")),
                    "hidden_verifier_sha256": sha256_tree(self.task.resolve("tests")),
                }
                atomic_write_json(run_dir / "task.json", task_payload)
                lifecycle.transition(TrialState.AGENT_RUNNING, self.agent.name)
                agent_result = self.agent.run(
                    AgentContext(
                        task_id=self.task.task.id,
                        workspace=workspace,
                        executor=executor,
                        executor_kind=self.executor_kind,
                        timeout_sec=self.task.budget.agent_timeout_sec,
                        gpu_count=self.task.environment.gpu_count,
                        shm_size=self.task.environment.shm_size,
                        image=self.task.environment.agent_image,
                    )
                )
                record.usage.agent_time_sec = agent_result.duration_sec
                guard.add_gpu_time(agent_result.duration_sec, self.task.environment.gpu_count)
                record.usage.gpu_minutes = guard.gpu_minutes
                lifecycle.transition(TrialState.COLLECTING)
                manifest, agent_policy = collect_agent_artifacts(
                    task=self.task,
                    workspace=workspace,
                    base_commit=base_commit,
                    agent_result=agent_result,
                    usage=record.usage,
                    destination=run_dir / "agent",
                )
            lifecycle.transition(TrialState.AGENT_DESTROYED)

            if agent_result.timed_out:
                record.failure = FailureKind.AGENT_TIMEOUT
                record.failure_detail = "agent command timed out"
                lifecycle.transition(TrialState.FAILED_AGENT, record.failure_detail)
                raise RuntimeError(record.failure_detail)
            if agent_result.exit_code:
                record.failure = FailureKind.ARTIFACT_INVALID
                record.failure_detail = f"agent exited with status {agent_result.exit_code}"
                lifecycle.transition(TrialState.FAILED_AGENT, record.failure_detail)
                raise RuntimeError(record.failure_detail)

            verifier = Verifier(
                task=self.task,
                executor=executor,
                executor_kind=self.executor_kind,
                run_dir=run_dir,
                manifest=manifest,
                agent_policy=agent_policy,
            )
            for index in range(1, self.task.replay.count + 1):
                guard.check()
                lifecycle.transition(
                    TrialState.VERIFYING, f"fresh replay {index}/{self.task.replay.count}"
                )
                replay = verifier.run(index)
                record.replays.append(replay)
                record.usage.verifier_time_sec += replay.duration_sec
                guard.add_gpu_time(replay.duration_sec, self.task.environment.gpu_count)
                record.usage.gpu_minutes = guard.gpu_minutes
                lifecycle.persist()

            lifecycle.transition(TrialState.SCORING)
            record.usage.wall_time_sec = time.monotonic() - started
            score = score_trial(self.task, record, manifest, run_dir)
            atomic_write_json(run_dir / "score.json", score.model_dump(mode="json"))
            if record.replays and not score.stable_resolved_at_1:
                record.failure = next(
                    (replay.failure for replay in record.replays if replay.failure),
                    FailureKind.FLAKY_REPLAY,
                )
                record.failure_detail = "one or more fresh replays failed"
            lifecycle.transition(TrialState.ARCHIVING)
            broker.release(lease, run_dir / "lease.json")
            lease = None
            record.finished_at = utc_now()
            lifecycle.transition(TrialState.COMPLETED)
            write_reports(run_dir, record, score)
            return TrialRunResult(run_dir=run_dir, record=record, score=score)
        except BudgetExceeded as error:
            record.failure = FailureKind.AGENT_BUDGET_EXCEEDED
            record.failure_detail = str(error)
            lifecycle.transition(TrialState.BUDGET_EXCEEDED, str(error))
            raise
        except Exception:
            if record.state not in {
                TrialState.FAILED_AGENT,
                TrialState.BUDGET_EXCEEDED,
                TrialState.COMPLETED,
            }:
                lifecycle.transition(TrialState.FAILED_INFRA, record.failure_detail)
            raise
        finally:
            record.usage.wall_time_sec = time.monotonic() - started
            record.finished_at = record.finished_at or utc_now()
            if lease is not None:
                broker.release(lease, run_dir / "lease.json")
            lifecycle.persist()

    def _new_trial_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{self.task.task.id}-{secrets.token_hex(3)}"
