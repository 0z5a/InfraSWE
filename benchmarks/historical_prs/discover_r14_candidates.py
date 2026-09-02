#!/usr/bin/env python3
"""Acquire outcome-free GitHub metadata for the R14 communication round."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

QUERY = """
query($queryString: String!, $cursor: String) {
  search(query: $queryString, type: ISSUE, first: 50, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        title
        createdAt
        baseRefName
        baseRefOid
        headRefOid
        changedFiles
        additions
        deletions
        files(first: 20) { totalCount nodes { path } }
        commits(first: 1) {
          nodes { commit { oid parents(first: 1) { nodes { oid } } } }
        }
      }
    }
  }
}
"""

DEPENDENCY_NAMES = {
    "cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}


def _query(query_string: str, cursor: str | None) -> dict[str, Any]:
    command = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={QUERY}",
        "-F",
        f"queryString={query_string}",
    ]
    if cursor:
        command.extend(["-F", f"cursor={cursor}"])
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode == 0:
            break
        if attempt < 3:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    payload = json.loads(process.stdout)
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    return payload["data"]["search"]


def _paths(node: dict[str, Any]) -> list[str]:
    return [str(item["path"]) for item in node["files"]["nodes"]]


def _first_commit(node: dict[str, Any]) -> dict[str, Any] | None:
    commits = node["commits"]["nodes"]
    return commits[0]["commit"] if commits else None


def _first_parent(node: dict[str, Any]) -> str | None:
    commit = _first_commit(node)
    parents = commit["parents"]["nodes"] if commit else []
    return str(parents[0]["oid"]) if parents else None


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith(("_test.py", "_test.cpp", "_test.cu", ".test.ts"))
    )


def _roughly_eligible(
    node: dict[str, Any], project: dict[str, Any], policy: dict[str, Any]
) -> bool:
    rules = policy["eligibility"]
    paths = _paths(node)
    changed_files = int(node.get("changedFiles") or 0)
    changed_lines = int(node.get("additions") or 0) + int(node.get("deletions") or 0)
    if not int(rules["changed_files_min"]) <= changed_files <= int(
        rules["changed_files_max"]
    ):
        return False
    if changed_lines > int(rules["changed_lines_max"]):
        return False
    if node["files"]["totalCount"] != len(paths):
        return False
    if not _first_parent(node) or not node.get("headRefOid") or not node.get("baseRefOid"):
        return False
    if not any(path.startswith(tuple(project["source_prefixes"])) for path in paths):
        return False
    if all(path.endswith(".md") or path.startswith(("docs/", "doc/")) for path in paths):
        return False
    if all(_is_test_path(path) for path in paths):
        return False
    signal_policy = policy["communication_signal_policy"]
    title = str(node["title"]).lower()
    joined_paths = " ".join(paths).lower()
    if not (
        any(term in title for term in signal_policy["strong_terms"])
        or any(term in title for term in signal_policy["topology_terms"])
        or any(term in joined_paths for term in signal_policy["path_terms"])
    ):
        return False
    return not all(
        path.rsplit("/", 1)[-1].lower() in DEPENDENCY_NAMES
        or path.endswith((".lock", ".sum", ".min.js"))
        or path.startswith(("third_party/", "vendor/"))
        for path in paths
    )


def _project_node(node: dict[str, Any], band: str) -> dict[str, Any]:
    commit = _first_commit(node)
    paths = _paths(node)
    return {
        "number": int(node["number"]),
        "title": node["title"],
        "created_at": node["createdAt"],
        "temporal_band": band,
        "base_ref": node["baseRefName"],
        "base_ref_oid": node["baseRefOid"],
        "base_sha": _first_parent(node),
        "head_sha": node["headRefOid"],
        "first_pr_commit_sha": commit["oid"] if commit else None,
        "changed_files": int(node["changedFiles"]),
        "additions": int(node["additions"]),
        "deletions": int(node["deletions"]),
        "paths": paths,
        "path_list_complete": node["files"]["totalCount"] == len(paths),
    }


def _acquire_band(
    repository: str,
    project: dict[str, Any],
    policy: dict[str, Any],
    *,
    band: str,
    start: str,
    end: str,
    eligible_target: int,
    page_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_string = f"repo:{repository} is:pr created:{start}..{end} sort:created-desc"
    cursor: str | None = None
    nodes: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    issue_count = 0
    while True:
        search = _query(query_string, cursor)
        issue_count = int(search["issueCount"])
        page_nodes = [node for node in search["nodes"] if node is not None]
        nodes.extend(page_nodes)
        pages.append(
            {
                "returned": len(page_nodes),
                "end_cursor": search["pageInfo"]["endCursor"],
            }
        )
        eligible_count = sum(
            _roughly_eligible(node, project, policy) for node in nodes
        )
        page_info = search["pageInfo"]
        if eligible_count >= eligible_target:
            break
        if not page_info["hasNextPage"] or len(pages) >= page_limit:
            break
        cursor = page_info["endCursor"]
    projected = [_project_node(node, band) for node in nodes]
    metadata = {
        "band": band,
        "query": query_string,
        "issue_count": issue_count,
        "page_count": len(pages),
        "pages": pages,
        "returned_count": len(projected),
        "rough_eligible_count": sum(
            _roughly_eligible(node, project, policy) for node in nodes
        ),
    }
    return projected, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit("R14 policy digest mismatch")
    blindness = policy["blindness"]
    forbidden = (
        blindness["candidate_body_visible"],
        blindness["diff_content_visible"],
        blindness["state_or_merge_visible"],
        blindness["review_or_comment_visible"],
        blindness["ci_or_label_visible"],
    )
    if any(value is not False for value in forbidden):
        raise SystemExit("R14 policy exposes forbidden evidence")

    cutoff = policy["created_at_window"]["prospective_check_cutoff"][:10]
    discoveries: dict[str, Any] = {}
    for project_name in policy["projects_in_order"]:
        project = policy["projects"][project_name]
        repository = project["repository"]
        recent, recent_metadata = _acquire_band(
            repository,
            project,
            policy,
            band="recent",
            start=cutoff,
            end=policy["created_at_window"]["end"][:10],
            eligible_target=5,
            page_limit=5,
        )
        mature, mature_metadata = _acquire_band(
            repository,
            project,
            policy,
            band="mature",
            start=policy["created_at_window"]["start"][:10],
            end="2026-08-02",
            eligible_target=20,
            page_limit=10,
        )
        by_number = {item["number"]: item for item in mature}
        by_number.update({item["number"]: item for item in recent})
        discoveries[project_name] = {
            "repository": repository,
            "queries": [recent_metadata, mature_metadata],
            "candidates": list(by_number.values()),
        }

    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "allowed_graphql_fields": blindness["allowed_selection_fields"],
        "outcome_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "candidate_body_requested": False,
        "diff_content_requested": False,
        "discoveries": discoveries,
    }
    payload = {**material, "discovery_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    counts = {
        project: len(item["candidates"]) for project, item in discoveries.items()
    }
    print(json.dumps(counts, sort_keys=True))
    print(f"discovery_sha256={payload['discovery_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
