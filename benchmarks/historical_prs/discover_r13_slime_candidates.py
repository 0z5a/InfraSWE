#!/usr/bin/env python3
"""Freeze outcome-free metadata for the R13 slime training extension."""

from __future__ import annotations

import argparse
import json
import subprocess
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
        files(first: 100) { totalCount nodes { path } }
        commits(first: 1) {
          nodes { commit { oid parents(first: 1) { nodes { oid } } } }
        }
      }
    }
  }
}
"""


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
    if cursor is not None:
        command.extend(["-F", f"cursor={cursor}"])
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    payload = json.loads(process.stdout)
    if "errors" in payload:
        raise RuntimeError(str(payload["errors"]))
    return payload["data"]["search"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    hidden = (
        policy["merge_outcomes_visible_to_machine_judge"],
        policy["review_text_visible_to_machine_judge"],
        policy["ci_fields_visible_to_machine_judge"],
        policy["candidate_body_visible_during_policy_freeze"],
        policy["diff_content_visible_during_policy_freeze"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("extension policy exposes forbidden evidence")

    query_string = policy["slime_discovery_query"]
    cursor: str | None = None
    pages: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    selected_numbers = {
        int(case_id.rsplit("-", 1)[1])
        for case_id in policy["selected_case_ids"]
        if case_id.startswith("slime-pr-")
    }
    while True:
        search = _query(query_string, cursor)
        pages.append(
            {
                "issue_count": search["issueCount"],
                "end_cursor": search["pageInfo"]["endCursor"],
            }
        )
        nodes.extend(node for node in search["nodes"] if node is not None)
        returned_numbers = {node["number"] for node in nodes}
        if selected_numbers <= returned_numbers:
            break
        page_info = search["pageInfo"]
        if not page_info["hasNextPage"] or len(pages) >= 5:
            break
        cursor = page_info["endCursor"]
    if not selected_numbers <= {node["number"] for node in nodes}:
        raise SystemExit("selected slime candidates were not discovered")

    candidates = []
    for node in nodes:
        commits = node["commits"]["nodes"]
        commit = commits[0]["commit"] if commits else None
        parents = commit["parents"]["nodes"] if commit else []
        paths = [entry["path"] for entry in node["files"]["nodes"]]
        path_list_complete = node["files"]["totalCount"] == len(paths)
        candidates.append(
            {
                "number": node["number"],
                "title": node["title"],
                "created_at": node["createdAt"],
                "base_ref": node["baseRefName"],
                "base_ref_oid": node["baseRefOid"],
                "base_sha": parents[0]["oid"] if parents else node["baseRefOid"],
                "base_derivation": (
                    "first-pr-commit-first-parent" if parents else "base-ref-oid-fallback"
                ),
                "head_sha": node["headRefOid"],
                "first_pr_commit_sha": commit["oid"] if commit else None,
                "changed_files": node["changedFiles"],
                "additions": node["additions"],
                "deletions": node["deletions"],
                "paths": paths,
                "path_list_complete": path_list_complete,
            }
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "preselection_policy_sha256": canonical_sha256(policy),
        "repository": "THUDM/slime",
        "query": query_string,
        "page_metadata": pages,
        "review_text_requested": False,
        "outcome_fields_requested": False,
        "ci_fields_requested": False,
        "candidate_body_requested": False,
        "allowed_graphql_fields": [
            "number",
            "title",
            "createdAt",
            "baseRefName",
            "baseRefOid",
            "headRefOid",
            "changedFiles",
            "additions",
            "deletions",
            "files.path",
            "commits.first.commit.oid",
            "commits.first.commit.parents.first.oid",
        ],
        "candidates": candidates,
    }
    payload = {**material, "discovery_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"candidate_count={len(candidates)}")
    print(f"discovery_sha256={payload['discovery_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
