from __future__ import annotations

from .base import AgentContext, AgentResult


class NoopAgent:
    name = "noop"

    def run(self, context: AgentContext) -> AgentResult:
        del context
        return AgentResult(
            exit_code=0,
            duration_sec=0.0,
            events=[{"kind": "noop", "message": "no changes made"}],
        )
