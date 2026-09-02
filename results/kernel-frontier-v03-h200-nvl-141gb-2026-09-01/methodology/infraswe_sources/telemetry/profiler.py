from __future__ import annotations

from pathlib import Path


def profile_present(directory: Path) -> bool:
    return directory.is_dir() and any(path.is_file() for path in directory.rglob("*"))
