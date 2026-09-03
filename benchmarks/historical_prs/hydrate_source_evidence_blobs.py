#!/usr/bin/env python3
"""Hydrate Git blob objects from a frozen outcome-free source evidence bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _repo_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        repository, separator, path = value.partition("=")
        if not separator or not repository or not path:
            raise SystemExit("--repo must be REPOSITORY=PATH")
        result[repository] = Path(path)
    return result


def _decode(side: dict[str, Any], *, case_id: str, filename: str) -> bytes:
    data = base64.b64decode(side["content_base64"], validate=True)
    if len(data) != side["byte_count"]:
        raise SystemExit(f"{case_id} {filename}: byte count mismatch")
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if digest != side["sha256"]:
        raise SystemExit(f"{case_id} {filename}: content digest mismatch")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", action="append", default=[], required=True)
    args = parser.parse_args()

    bundle = _read(args.bundle)
    repositories = _repo_map(args.repo)
    written: set[tuple[str, str]] = set()
    side_count = 0
    verified_head_blob_count = 0

    for case in bundle["cases"]:
        repository = case["repository"]
        if repository not in repositories:
            continue
        repo_path = repositories[repository]
        for file_evidence in case["files"]:
            for side_name in ("base", "head"):
                side = file_evidence[side_name]
                if not side.get("available") or "content_base64" not in side:
                    continue
                data = _decode(
                    side,
                    case_id=case["case_id"],
                    filename=file_evidence["filename"],
                )
                content_key = (repository, side["sha256"])
                if content_key in written:
                    continue
                process = subprocess.run(
                    ["git", "-C", str(repo_path), "hash-object", "-w", "--stdin"],
                    input=data,
                    capture_output=True,
                    check=True,
                )
                blob_sha = process.stdout.decode().strip()
                expected_head_sha = file_evidence.get("head_blob_sha")
                if side_name == "head" and expected_head_sha:
                    if blob_sha != expected_head_sha:
                        raise SystemExit(
                            f"{case['case_id']} {file_evidence['filename']}: "
                            "Git head blob mismatch"
                        )
                    verified_head_blob_count += 1
                written.add(content_key)
                side_count += 1

    print(f"hydrated_unique_blob_count={side_count}")
    print(f"verified_head_blob_count={verified_head_blob_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
