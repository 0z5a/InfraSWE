from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from infraswe.environments import DockerExecutor


def test_docker_executor_selects_requested_gpu_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def fake_run(command, **kwargs):
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("INFRASWE_GPU_DEVICES", "3,1")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerExecutor().run(
        ["python", "verify.py"], cwd=tmp_path, timeout_sec=10, gpu_count=1
    )
    assert result.exit_code == 0
    assert captured[captured.index("--gpus") + 1] == "device=3"


@pytest.mark.parametrize("selection", ["gpu0", "0,0", "0"])
def test_docker_executor_rejects_invalid_gpu_subset(selection: str) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("INFRASWE_GPU_DEVICES", selection)
        with pytest.raises(ValueError):
            DockerExecutor._gpu_request(2)
