from __future__ import annotations

from dataclasses import dataclass

from infraswe.models.trial import ReplayResult


@dataclass(frozen=True)
class ReplaySummary:
    results: list[ReplayResult]

    @property
    def stable(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def pass_ratio(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.passed for result in self.results) / len(self.results)
