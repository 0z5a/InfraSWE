#!/usr/bin/env python3
"""Materialize frozen PR refs before a timed remote test matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MISSING_PROMISOR_BLOB = re.compile(r"could not fetch ([0-9a-f]{40}) from promisor remote")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _project_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        project, separator, path = value.partition("=")
        if not separator or not project or not path:
            raise SystemExit("--project must be PROJECT=PATH")
        result[project] = Path(path)
    return result


def _hydrate_raw_missing_blob(
    repo_path: Path,
    case: dict[str, Any],
    ref: str,
    output: str,
) -> tuple[str, str] | None:
    match = MISSING_PROMISOR_BLOB.search(output)
    if match is None:
        return None
    missing_sha = match.group(1)
    tree = subprocess.check_output(
        ["git", "-C", str(repo_path), "ls-tree", "-r", "-z", ref]
    )
    matching_paths: list[str] = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", maxsplit=1)
        _mode, object_type, object_sha = metadata.split()
        if object_type == b"blob" and object_sha.decode() == missing_sha:
            matching_paths.append(path.decode("utf-8", errors="surrogateescape"))
    if not matching_paths:
        return None

    path = matching_paths[0]
    encoded_path = urllib.parse.quote(path, safe="/")
    encoded_repository = urllib.parse.quote(case["repository"], safe="/")
    url = (
        f"https://raw.githubusercontent.com/{encoded_repository}/"
        f"{case['head_sha']}/{encoded_path}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "InfraSWE-source-hydrator"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    written_sha = subprocess.check_output(
        ["git", "-C", str(repo_path), "hash-object", "-w", "--stdin"],
        input=data,
        text=False,
    ).decode().strip()
    if written_sha != missing_sha:
        raise SystemExit(
            f"{case['case_id']} {path}: raw blob hash {written_sha} != {missing_sha}"
        )
    return missing_sha, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--round-label", required=True)
    parser.add_argument("--project", action="append", default=[], required=True)
    parser.add_argument("--checkout-timeout", type=int, default=90)
    parser.add_argument("--hydrate-raw-missing", action="store_true")
    parser.add_argument("--max-hydrations-per-case", type=int, default=8)
    args = parser.parse_args()

    selection = _read(args.selection_lock)["selection_material"]
    projects = _project_map(args.project)
    failures: list[str] = []
    attempted_count = 0
    cases = selection["cases"]
    for index, case in enumerate(cases, 1):
        project = case["project"]
        if project not in projects:
            continue
        attempted_count += 1
        ref = f"refs/{args.round_label.lower()}/pr-{case['pull_number']}"
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment["GIT_NO_LAZY_FETCH"] = "1"
        if project == "tensorrt-llm":
            environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        hydration_count = 0
        switch: subprocess.CompletedProcess[str] | None = None
        while True:
            try:
                switch = subprocess.run(
                    ["git", "-C", str(projects[project]), "switch", "--detach", ref],
                    capture_output=True,
                    text=True,
                    timeout=args.checkout_timeout,
                    env=environment,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{case['case_id']}:timeout")
                print(f"[{index}/{len(cases)}] {case['case_id']}: timeout", flush=True)
                break
            if switch.returncode == 0:
                break
            output = switch.stdout + switch.stderr
            if args.hydrate_raw_missing and hydration_count < args.max_hydrations_per_case:
                hydrated = _hydrate_raw_missing_blob(
                    projects[project], case, ref, output
                )
                if hydrated is not None:
                    hydration_count += 1
                    _sha, path = hydrated
                    print(
                        f"[{index}/{len(cases)}] {case['case_id']}: hydrated {path}",
                        flush=True,
                    )
                    continue
            failures.append(f"{case['case_id']}:rc={switch.returncode}")
            tail = output.strip().splitlines()[-1:]
            print(
                f"[{index}/{len(cases)}] {case['case_id']}: rc={switch.returncode} "
                f"tail={tail}",
                flush=True,
            )
            break
        if switch is None or switch.returncode != 0:
            continue
        actual = subprocess.check_output(
            ["git", "-C", str(projects[project]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual != case["head_sha"]:
            failures.append(f"{case['case_id']}:sha-mismatch")
            print(f"[{index}/{len(cases)}] {case['case_id']}: sha mismatch", flush=True)
            continue
        print(f"[{index}/{len(cases)}] {case['case_id']}: ready", flush=True)

    print(f"prewarmed_case_count={attempted_count - len(failures)}")
    print(f"failure_count={len(failures)}")
    if failures:
        print("failures=" + ",".join(failures))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
