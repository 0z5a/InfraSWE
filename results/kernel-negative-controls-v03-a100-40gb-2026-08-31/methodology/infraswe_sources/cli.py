from __future__ import annotations

import functools
import http.server
import shlex
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from infraswe import __version__
from infraswe.agents import CliAgent, NoopAgent, OracleAgent
from infraswe.environments.hardware_manifest import write_hardware_manifest
from infraswe.io import atomic_write_json
from infraswe.kernel.models import KernelAggregate, RoleResult
from infraswe.kernel.role_graph import RoleGraph
from infraswe.models.artifact import ArtifactManifest
from infraswe.models.hardware import HardwareProfile, validate_hardware_manifest
from infraswe.models.score import ScoreResult
from infraswe.models.task import TaskPackage
from infraswe.models.trial import TrialRecord
from infraswe.runner import TrialRunner
from infraswe.scoring.report import write_reports

app = typer.Typer(no_args_is_help=True, help="Executable benchmark harness for Infra agents.")
task_app = typer.Typer(no_args_is_help=True, help="Inspect and certify task packages.")
schema_app = typer.Typer(no_args_is_help=True, help="Export protocol JSON Schemas.")
lease_app = typer.Typer(no_args_is_help=True, help="Inspect resource leases and hardware.")
app.add_typer(task_app, name="task")
app.add_typer(schema_app, name="schema")
app.add_typer(lease_app, name="lease")
console = Console()


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


@schema_app.command("export")
def export_schemas(
    output: Annotated[Path, typer.Option("--output")] = Path("schemas"),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    schemas = {
        "task.schema.json": TaskPackage.model_json_schema(),
        "artifact.schema.json": ArtifactManifest.model_json_schema(),
        "result.schema.json": ScoreResult.model_json_schema(),
        "kernel-role-result.schema.json": RoleResult.model_json_schema(),
        "kernel-role-graph.schema.json": RoleGraph.model_json_schema(),
        "kernel-aggregate.schema.json": KernelAggregate.model_json_schema(),
    }
    for name, schema in schemas.items():
        atomic_write_json(output / name, schema)
    console.print(f"wrote {len(schemas)} schemas to {output.resolve()}")


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
