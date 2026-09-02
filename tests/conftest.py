from __future__ import annotations

from pathlib import Path

import pytest

from infraswe.models.task import TaskPackage


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def rollout_task(project_root: Path) -> TaskPackage:
    return TaskPackage.load(project_root / "tasks" / "gpu-service-rollout-regression")
