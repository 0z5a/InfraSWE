from __future__ import annotations

import time
from dataclasses import dataclass, field

from infraswe.models.task import BudgetConfig


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    budget: BudgetConfig
    started_at: float = field(default_factory=time.monotonic)
    gpu_minutes: float = 0.0
    model_cost_usd: float = 0.0
    infra_cost_usd: float = 0.0

    def check(self) -> None:
        elapsed = time.monotonic() - self.started_at
        maximum = self.budget.agent_timeout_sec + (self.budget.verifier_timeout_sec * 10)
        if elapsed > maximum:
            raise BudgetExceeded(f"wall-time budget exceeded: {elapsed:.1f}s > {maximum}s")
        limits = [
            ("gpu minutes", self.gpu_minutes, self.budget.gpu_minutes),
            ("model cost", self.model_cost_usd, self.budget.max_model_cost_usd),
            ("infra cost", self.infra_cost_usd, self.budget.max_infra_cost_usd),
        ]
        for label, actual, limit in limits:
            if limit > 0 and actual > limit:
                raise BudgetExceeded(f"{label} exceeded: {actual:.4f} > {limit:.4f}")

    def add_gpu_time(self, duration_sec: float, gpu_count: int) -> None:
        self.gpu_minutes += duration_sec * gpu_count / 60
        self.check()
