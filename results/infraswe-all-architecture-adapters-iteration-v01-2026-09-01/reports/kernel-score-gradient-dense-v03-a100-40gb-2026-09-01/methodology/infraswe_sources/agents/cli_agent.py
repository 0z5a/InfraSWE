from __future__ import annotations

from collections.abc import Sequence

from .base import AgentContext, AgentResult


class CliAgent:
    name = "cli"

    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("CLI agent command must be non-empty")
        self.command = list(command)

    def run(self, context: AgentContext) -> AgentResult:
        result = context.executor.run(
            self.command,
            cwd=context.workspace,
            timeout_sec=context.timeout_sec,
            env={"INFRASWE_TASK_ID": context.task_id},
            gpu_count=context.gpu_count,
            shm_size=context.shm_size,
            image=context.image,
        )
        return AgentResult.from_command(result)
