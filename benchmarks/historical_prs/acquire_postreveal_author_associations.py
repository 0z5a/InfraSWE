#!/usr/bin/env python3
"""Acquire outcome-free author associations for a sealed learning cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _checked(path: Path, digest_field: str, *, material_field: str | None = None) -> dict[str, Any]:
    payload = _read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def _graphql(query: str) -> tuple[dict[str, Any], str]:
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(3):
        process = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    return json.loads(process.stdout)["data"], process.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = _checked(
        args.selection_lock,
        "selection_lock_sha256",
        material_field="selection_material",
    )
    locks = _checked(args.judgment_locks, "lock_set_sha256")
    reveal = _checked(args.reveal, "reveal_sha256")
    if reveal["judgment_lock_set_sha256"] != locks["lock_set_sha256"]:
        raise SystemExit("reveal/judgment binding mismatch")
    if locks["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("judgment/selection binding mismatch")

    locked_ids = [item["material"]["case_id"] for item in locks["locks"]]
    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    if not set(locked_ids) <= selected.keys():
        raise SystemExit("judgment contains an unknown selected case")

    repositories: dict[str, list[dict[str, Any]]] = {}
    for case_id in locked_ids:
        case = selected[case_id]
        repositories.setdefault(case["repository"], []).append(case)
    fields: list[str] = []
    for repo_index, (repository, cases) in enumerate(repositories.items()):
        owner, name = repository.split("/", 1)
        pulls = " ".join(
            f"p{case['pull_number']}:pullRequest(number:{case['pull_number']})"
            "{authorAssociation author{login}}"
            for case in cases
        )
        fields.append(f'r{repo_index}:repository(owner:"{owner}",name:"{name}"){{{pulls}}}')
    query = "query {" + " ".join(fields) + "}"
    data, raw = _graphql(query)

    cases: list[dict[str, Any]] = []
    for repo_index, (_repository, repository_cases) in enumerate(repositories.items()):
        for case in repository_cases:
            pull = data[f"r{repo_index}"][f"p{case['pull_number']}"]
            if pull is None:
                raise SystemExit(f"{case['case_id']}: pull request metadata unavailable")
            cases.append(
                {
                    "case_id": case["case_id"],
                    "author": str((pull.get("author") or {}).get("login") or "unknown"),
                    "author_association": str(pull.get("authorAssociation") or "UNKNOWN"),
                }
            )

    cases_by_id = {item["case_id"]: item for item in cases}
    material = {
        "schema_version": "0.1",
        "protocol_id": "postreveal-outcome-free-author-association-v0.1",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "judgment_lock_set_sha256": locks["lock_set_sha256"],
        "reveal_sha256": reveal["reveal_sha256"],
        "acquired_at": datetime.now(UTC).isoformat(),
        "acquired_after_reveal": True,
        "learning_only": True,
        "state_or_merge_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "raw_response_sha256": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(),
        "cases": [cases_by_id[case_id] for case_id in locked_ids],
    }
    payload = {**material, "metadata_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "association_counts": {
                    association: sum(item["author_association"] == association for item in cases)
                    for association in sorted({item["author_association"] for item in cases})
                },
                "metadata_sha256": payload["metadata_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
