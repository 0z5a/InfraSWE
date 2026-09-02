from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from infraswe.environments import CommandResult, Executor


@dataclass(frozen=True)
class AgentContext:
    task_id: str
    workspace: Path
    executor: Executor
    executor_kind: str
    timeout_sec: int
    gpu_count: int
    shm_size: str
    image: str


@dataclass(frozen=True)
class AgentResult:
    exit_code: int
    duration_sec: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    events: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_command(cls, result: CommandResult) -> AgentResult:
        return cls(
            exit_code=result.exit_code,
            duration_sec=result.duration_sec,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            events=[
                {
                    "kind": "command",
                    "command": list(result.command),
                    "exit_code": result.exit_code,
                    "duration_sec": result.duration_sec,
                    "timed_out": result.timed_out,
                }
            ],
        )


class Agent(Protocol):
    name: str

    def run(self, context: AgentContext) -> AgentResult: ...
