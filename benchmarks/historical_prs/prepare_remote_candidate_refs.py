#!/usr/bin/env python3
"""Fetch exact frozen PR heads into the persistent remote benchmark worktrees."""

from __future__ import annotations

import argparse
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from run_r15_upstream_tests import PROJECT_RUNTIME, _ssh

from infraswe.history.blind import canonical_sha256


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()

    selection = read(args.selection_lock)
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit(f"{args.round} selection digest mismatch")
    cases = material["cases"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["project"], []).append(case)

    def fetch_project(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, str]:
        project, project_cases = item
        worktree = PROJECT_RUNTIME[project][0]
        commands = [f"cd {shlex.quote(worktree)}"]
        for case in project_cases:
            ref = f"refs/{args.round.lower()}/pr-{int(case['pull_number'])}"
            prefix = "GIT_LFS_SKIP_SMUDGE=1 " if project == "tensorrt-llm" else ""
            commands.append(
                f"{prefix}git fetch -q --force origin "
                f"pull/{int(case['pull_number'])}/head:{shlex.quote(ref)}"
            )
            commands.append(
                f"test \"$(git rev-parse {shlex.quote(ref)})\" = "
                f"{shlex.quote(case['head_sha'])}"
            )
        process = _ssh(" && ".join(commands), timeout=max(180, 45 * len(project_cases)))
        if process.returncode != 0:
            raise RuntimeError(
                f"{project} ref preparation failed: {process.stdout}{process.stderr}"
            )
        return project, f"{len(project_cases)} refs verified"

    with ThreadPoolExecutor(max_workers=min(4, len(grouped))) as executor:
        results = list(executor.map(fetch_project, sorted(grouped.items())))
    print(json.dumps(dict(results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
