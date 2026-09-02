from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FaultScenarioSet:
    path: Path

    def exists(self) -> bool:
        return self.path.is_file()
