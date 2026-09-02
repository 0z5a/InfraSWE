from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from infraswe.io import sha256_file, utc_now


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def from_file(cls, root: Path, path: Path, media_type: str) -> ArtifactEntry:
        return cls(
            path=str(path.relative_to(root)),
            media_type=media_type,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    created_at: datetime = Field(default_factory=utc_now)
    entries: list[ArtifactEntry]

    def verify(self, root: Path) -> list[str]:
        errors: list[str] = []
        for entry in self.entries:
            path = root / entry.path
            if not path.is_file():
                errors.append(f"missing artifact: {entry.path}")
            elif path.stat().st_size != entry.size_bytes:
                errors.append(f"size mismatch: {entry.path}")
            elif sha256_file(path) != entry.sha256:
                errors.append(f"digest mismatch: {entry.path}")
        return errors
