from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadSpec:
    path: Path

    def exists(self) -> bool:
        return self.path.is_file()
