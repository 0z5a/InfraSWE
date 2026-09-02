from __future__ import annotations

from infraswe.models.task import TaskPackage

from .base import AgentContext, AgentResult


class OracleAgent:
    name = "oracle"

    def __init__(self, task: TaskPackage) -> None:
        self.task = task

    def run(self, context: AgentContext) -> AgentResult:
        configured = self.task.execution.solution_command
        solution_dir = self.task.resolve("solution")
        if context.executor_kind == "docker":
            command = [
                token.replace("solution/", "/oracle/", 1)
                if token.startswith("solution/")
                else token
                for token in configured
            ]
            mounts = {solution_dir: ("/oracle", True)}
        else:
            command = [
                str(self.task.resolve(token)) if token.startswith("solution/") else token
                for token in configured
            ]
            mounts = None
        result = context.executor.run(
            command,
            cwd=context.workspace,
            timeout_sec=context.timeout_sec,
            env={"INFRASWE_TASK_ID": context.task_id},
            mounts=mounts,
            gpu_count=context.gpu_count,
            shm_size=context.shm_size,
            image=context.image,
        )
        return AgentResult.from_command(result)
