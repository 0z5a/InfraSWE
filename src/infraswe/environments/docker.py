from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .base import CommandResult


class DockerExecutor:
    def __init__(self, default_image: str = "python:3.12-slim") -> None:
        self.default_image = default_image

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_sec: int,
        env: Mapping[str, str] | None = None,
        mounts: Mapping[Path, tuple[str, bool]] | None = None,
        gpu_count: int = 0,
        shm_size: str = "256m",
        image: str | None = None,
    ) -> CommandResult:
        docker_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--shm-size",
            shm_size,
            "--workdir",
            "/workspace",
            "--mount",
            f"type=bind,src={cwd.resolve()},dst=/workspace",
        ]
        for source, (target, read_only) in sorted(
            (mounts or {}).items(), key=lambda item: str(item[0])
        ):
            option = f"type=bind,src={source.resolve()},dst={target}"
            if read_only:
                option += ",readonly"
            docker_command.extend(["--mount", option])
        for key, value in sorted((env or {}).items()):
            docker_command.extend(["--env", f"{key}={value}"])
        if gpu_count:
            docker_command.extend(["--gpus", self._gpu_request(gpu_count)])
        docker_command.append(image or self.default_image)
        docker_command.extend(command)

        started = time.monotonic()
        try:
            completed = subprocess.run(
                docker_command,
                cwd=cwd,
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
            return CommandResult(
                command=tuple(command),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as error:
            return CommandResult(
                command=tuple(command),
                exit_code=124,
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                duration_sec=time.monotonic() - started,
                timed_out=True,
            )

    @staticmethod
    def _gpu_request(gpu_count: int) -> str:
        selected = os.environ.get("INFRASWE_GPU_DEVICES", "").strip()
        if not selected:
            return str(gpu_count)
        devices = [value.strip() for value in selected.split(",")]
        if (
            any(not value.isdecimal() for value in devices)
            or len(devices) != len(set(devices))
            or len(devices) < gpu_count
        ):
            raise ValueError("INFRASWE_GPU_DEVICES must contain enough unique numeric GPU indices")
        return "device=" + ",".join(devices[:gpu_count])
