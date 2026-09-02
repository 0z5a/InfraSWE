from __future__ import annotations

import json
from pathlib import Path

from infraswe.agents import NoopAgent, OracleAgent
from infraswe.models.task import TaskPackage
from infraswe.runner import TrialRunner


def run_trial(task: TaskPackage, agent, root: Path):
    return TrialRunner(
        task=task,
        agent=agent,
        runs_root=root,
        executor_kind="local",
        ttl_minutes=5,
    ).run()


def test_oracle_passes_three_fresh_replays(rollout_task: TaskPackage, tmp_path: Path) -> None:
    result = run_trial(rollout_task, OracleAgent(rollout_task), tmp_path / "runs")

    assert result.score.resolved_at_1
    assert result.score.stable_resolved_at_1
    assert result.score.coverage == 1.0
    assert len(result.record.replays) == 3
    assert all(replay.passed for replay in result.record.replays)
    assert (result.run_dir / "agent" / "model.patch").stat().st_size > 0
    assert (result.run_dir / "agent" / "config-bundle.tar.zst").stat().st_size > 0
    assert (result.run_dir / "report.md").is_file()
    assert (result.run_dir / "index.html").is_file()
    task_record = json.loads((result.run_dir / "task.json").read_text())
    assert len(task_record["digests"]["task_package_sha256"]) == 64
    assert len(task_record["digests"]["hidden_verifier_sha256"]) == 64
    assert not any(path.name.startswith("infraswe-agent-") for path in result.run_dir.rglob("*"))


def test_noop_proves_base_failure_is_not_vacuous(rollout_task: TaskPackage, tmp_path: Path) -> None:
    result = run_trial(rollout_task, NoopAgent(), tmp_path / "runs")

    assert not result.score.resolved_at_1
    assert not result.score.stable_resolved_at_1
    assert result.record.failure is not None
    assert all(not replay.passed for replay in result.record.replays)
    assert any(
        not replay.assertions["functional.zero_rollout_errors"] for replay in result.record.replays
    )


def test_agent_manifest_detects_tampering(rollout_task: TaskPackage, tmp_path: Path) -> None:
    result = run_trial(rollout_task, OracleAgent(rollout_task), tmp_path / "runs")
    manifest = json.loads((result.run_dir / "agent" / "manifest.json").read_text())
    patch = result.run_dir / "agent" / "model.patch"
    patch.write_text(patch.read_text() + "\n# tampered\n", encoding="utf-8")

    from infraswe.models.artifact import ArtifactManifest

    errors = ArtifactManifest.model_validate(manifest).verify(result.run_dir)
    assert errors == ["size mismatch: agent/model.patch"]


def test_agent_context_does_not_expose_hidden_package(
    rollout_task: TaskPackage, tmp_path: Path
) -> None:
    class InspectingAgent:
        name = "context-inspector"

        def run(self, context):
            assert not hasattr(context, "task")
            assert not hasattr(context, "package_dir")
            from infraswe.agents.base import AgentResult

            return AgentResult(exit_code=0, duration_sec=0)

    result = run_trial(rollout_task, InspectingAgent(), tmp_path / "runs")
    assert not result.score.stable_resolved_at_1
