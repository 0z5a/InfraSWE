from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False


class Executor(Protocol):
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
    ) -> CommandResult: ...
