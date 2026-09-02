from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvidenceCollector:
    root: Path

    def ensure_layout(self) -> None:
        for name in ("logs", "metrics", "traces", "profiles", "config-diff"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
