from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from infraswe.environments import Executor
from infraswe.io import atomic_write_json
from infraswe.models.artifact import ArtifactManifest
from infraswe.models.task import TaskPackage
from infraswe.models.trial import FailureKind, ReplayResult
from infraswe.runner.artifacts import initialize_fixture_repository
from infraswe.verifier.policy import merge_policy_results


def _read_object(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _flatten_assertions(value: dict[str, Any], prefix: str = "") -> dict[str, bool]:
    flattened: dict[str, bool] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_assertions(item, name))
        elif isinstance(item, bool):
            flattened[name] = item
        else:
            raise TypeError(f"assertion {name} must be boolean, got {type(item).__name__}")
    return flattened


def _numeric_metrics(value: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"metric {key} must be numeric")
        metrics[key] = float(item)
    return metrics


def _failure_for(
    *,
    assertions: dict[str, bool],
    faults: dict[str, Any],
    policy: dict[str, Any],
    timed_out: bool,
    exit_code: int,
) -> FailureKind | None:
    hard = policy.get("hard_failures", [])
    for name in (
        "DATA_CORRUPTION",
        "SILENT_FALLBACK",
        "RESOURCE_LEAK",
        "POLICY_VIOLATION",
    ):
        if name in hard:
            return FailureKind(name)
    if timed_out:
        return FailureKind.DEADLOCK
    for name, passed in assertions.items():
        if passed:
            continue
        if name.startswith("regression."):
            return FailureKind.REGRESSION_FAILED
        if name.startswith("slo."):
            return FailureKind.SLO_FAILED
        if name.startswith("safety.rollback"):
            return FailureKind.ROLLBACK_FAILED
        return FailureKind.FUNCTIONAL_FAILED
    if not faults.get("passed", True):
        return FailureKind.FAULT_RECOVERY_FAILED
    if exit_code:
        return FailureKind.VERIFIER_INFRA_FAILED
    return None


class Verifier:
    def __init__(
        self,
        *,
        task: TaskPackage,
        executor: Executor,
        executor_kind: str,
        run_dir: Path,
        manifest: ArtifactManifest,
        agent_policy: dict[str, Any],
    ) -> None:
        self.task = task
        self.executor = executor
        self.executor_kind = executor_kind
        self.run_dir = run_dir
        self.manifest = manifest
        self.agent_policy = agent_policy

    def run(self, index: int) -> ReplayResult:
        result_dir = self.run_dir / "verifier" / f"replay-{index}"
        result_dir.mkdir(parents=True, exist_ok=False)
        manifest_errors = self.manifest.verify(self.run_dir)
        if manifest_errors:
            policy = {
                "passed": False,
                "hard_failures": ["ARTIFACT_INVALID"],
                "errors": manifest_errors,
            }
            atomic_write_json(result_dir / "policy.json", policy)
            return ReplayResult(
                index=index,
                passed=False,
                exit_code=2,
                duration_sec=0,
                policy=policy,
                failure=FailureKind.ARTIFACT_INVALID,
                message="; ".join(manifest_errors),
            )

        with tempfile.TemporaryDirectory(prefix=f"infraswe-verifier-{index}-") as temporary:
            workspace = Path(temporary) / "repo"
            shutil.copytree(self.task.resolve(self.task.execution.repo), workspace)
            initialize_fixture_repository(workspace)
            patch_failure = self._apply_patch(workspace)
            if patch_failure:
                policy = {"passed": False, "hard_failures": [], "patch_error": patch_failure}
                atomic_write_json(result_dir / "policy.json", policy)
                return ReplayResult(
                    index=index,
                    passed=False,
                    exit_code=2,
                    duration_sec=0,
                    policy=policy,
                    failure=FailureKind.PATCH_APPLY_FAILED,
                    message=patch_failure,
                )

            command, mounts, env = self._verifier_invocation(workspace, result_dir, index)
            command_result = self.executor.run(
                command,
                cwd=workspace,
                timeout_sec=self.task.budget.verifier_timeout_sec,
                env=env,
                mounts=mounts,
                gpu_count=self.task.environment.gpu_count,
                shm_size=self.task.environment.shm_size,
                image=self.task.environment.verifier_image,
            )
            (result_dir / "verifier.stdout.log").write_text(command_result.stdout, encoding="utf-8")
            (result_dir / "verifier.stderr.log").write_text(command_result.stderr, encoding="utf-8")
            try:
                raw_assertions = _read_object(result_dir / "assertions.json", {})
                assertions = _flatten_assertions(raw_assertions)
                metrics = _numeric_metrics(_read_object(result_dir / "metrics.json", {}))
                faults = _read_object(result_dir / "faults.json", {"passed": False})
                verifier_policy = _read_object(
                    result_dir / "policy.json",
                    {"passed": False, "hard_failures": [], "missing": True},
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                assertions, metrics, faults = {}, {}, {"passed": False}
                verifier_policy = {
                    "passed": False,
                    "hard_failures": [],
                    "protocol_error": str(error),
                }
            policy = merge_policy_results(self.task, self.agent_policy, verifier_policy)
            atomic_write_json(result_dir / "policy.json", policy)
            failure = _failure_for(
                assertions=assertions,
                faults=faults,
                policy=policy,
                timed_out=command_result.timed_out,
                exit_code=command_result.exit_code,
            )
            passed = (
                bool(assertions)
                and all(assertions.values())
                and bool(faults.get("passed", True))
                and bool(policy.get("passed", False))
                and command_result.exit_code == 0
            )
            return ReplayResult(
                index=index,
                passed=passed,
                exit_code=command_result.exit_code,
                duration_sec=command_result.duration_sec,
                assertions=assertions,
                metrics=metrics,
                faults=faults,
                policy=policy,
                failure=failure,
                message=command_result.stderr.strip()[-1000:],
            )

    def _apply_patch(self, workspace: Path) -> str:
        patch_path = self.run_dir / "agent" / "model.patch"
        if not patch_path.read_bytes().strip():
            return ""
        for arguments in (
            ["git", "apply", "--check", str(patch_path)],
            ["git", "apply", str(patch_path)],
        ):
            completed = subprocess.run(
                arguments,
                cwd=workspace,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                return completed.stderr.strip() or completed.stdout.strip()
        return ""

    def _verifier_invocation(
        self, workspace: Path, result_dir: Path, index: int
    ) -> tuple[list[str], dict[Path, tuple[str, bool]] | None, dict[str, str]]:
        configured = self.task.execution.verifier_command
        if self.executor_kind == "docker":
            verifier_root = self.task.package_dir
            mounts = {
                self.task.resolve("tests"): ("/verifier/tests", True),
                self.task.resolve("workload"): ("/verifier/workload", True),
                self.task.resolve("faults"): ("/verifier/faults", True),
                result_dir: ("/evidence", False),
            }
            command = [
                token.replace("tests/", "/verifier/tests/", 1)
                if token.startswith("tests/")
                else token
                for token in configured
            ]
            env = {
                "INFRASWE_REPO": "/workspace",
                "INFRASWE_EVIDENCE_DIR": "/evidence",
                "INFRASWE_WORKLOAD_DIR": "/verifier/workload",
                "INFRASWE_FAULTS_DIR": "/verifier/faults",
                "INFRASWE_REPLAY_INDEX": str(index),
                "INFRASWE_CANARY": f"verifier-only-{self.task.task.id}",
            }
            del verifier_root
            return command, mounts, env
        command = [
            str(self.task.resolve(token)) if token.startswith("tests/") else token
            for token in configured
        ]
        env = {
            "INFRASWE_REPO": str(workspace),
            "INFRASWE_EVIDENCE_DIR": str(result_dir),
            "INFRASWE_WORKLOAD_DIR": str(self.task.resolve("workload")),
            "INFRASWE_FAULTS_DIR": str(self.task.resolve("faults")),
            "INFRASWE_REPLAY_INDEX": str(index),
            "INFRASWE_CANARY": f"verifier-only-{self.task.task.id}",
        }
        return command, None, env
