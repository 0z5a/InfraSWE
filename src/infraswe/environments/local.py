from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .base import CommandResult


class LocalExecutor:
    """Development executor with process-level, rather than container, isolation."""

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
        del mounts, gpu_count, shm_size, image
        process_env = os.environ.copy()
        process_env.update(env or {})
        resolved_command = list(command)
        if resolved_command and resolved_command[0] in {"python", "python3"}:
            resolved_command[0] = sys.executable
        started = time.monotonic()
        try:
            completed = subprocess.run(
                resolved_command,
                cwd=cwd,
                env=process_env,
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
