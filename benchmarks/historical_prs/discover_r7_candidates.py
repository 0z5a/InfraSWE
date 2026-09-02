#!/usr/bin/env python3
"""Discover R7 candidates using outcome-free GitHub metadata only."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
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
        author { login }
        baseRefName
        baseRefOid
        headRefOid
        changedFiles
        additions
        deletions
        files(first: 10) { nodes { path } }
        commits(first: 1) {
          nodes {
            commit {
              oid
              parents(first: 1) { nodes { oid } }
            }
          }
        }
      }
    }
  }
}
"""

PROJECTS = {
    "cutlass-cute": {
        "repository": "NVIDIA/cutlass",
        "excluded": {2275},
        "source_prefixes": ("include/cutlass/", "include/cute/", "examples/", "python/CuTeDSL/"),
    },
    "liger-kernel": {
        "repository": "linkedin/Liger-Kernel",
        "excluded": {804},
        "source_prefixes": (
            "src/liger_kernel/ops/",
            "src/liger_kernel/chunked_loss/",
            "src/liger_kernel/transformers/",
        ),
    },
    "deepgemm": {
        "repository": "deepseek-ai/DeepGEMM",
        "excluded": {55},
        "source_prefixes": ("deep_gemm/",),
    },
    "megatron-core": {
        "repository": "NVIDIA/Megatron-LM",
        "excluded": {5608},
        "source_prefixes": ("megatron/core/",),
    },
    "torchtitan": {
        "repository": "pytorch/torchtitan",
        "excluded": {2717},
        "source_prefixes": ("torchtitan/",),
    },
    "verl": {
        "repository": "verl-project/verl",
        "excluded": {1688},
        "source_prefixes": ("verl/",),
    },
}

TITLE_TOKENS = (
    "fix",
    "bug",
    "perf",
    "performance",
    "kernel",
    "triton",
    "cuda",
    "compile",
    "graph",
    "dtype",
    "memory",
    "fuse",
    "optimize",
    "optimise",
)

EXCLUDED_FILE_NAMES = {
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
    if cursor is not None:
        command.extend(["-F", f"cursor={cursor}"])
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    payload = json.loads(process.stdout)
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]["search"]


def _paths(node: dict[str, Any]) -> list[str]:
    return [item["path"] for item in node["files"]["nodes"]]


def _is_generated_or_dependency_only(paths: list[str]) -> bool:
    dependency_prefixes = (".github/", "docker/", "requirements/", "third_party/")
    return all(
        path.rsplit("/", 1)[-1] in EXCLUDED_FILE_NAMES
        or path.startswith(dependency_prefixes)
        or path.endswith((".lock", ".sum"))
        for path in paths
    )


def _eligible(project: str, node: dict[str, Any]) -> tuple[bool, list[str]]:
    config = PROJECTS[project]
    paths = _paths(node)
    reasons: list[str] = []
    changed_files = int(node["changedFiles"] or 0)
    changed_lines = int(node["additions"] or 0) + int(node["deletions"] or 0)
    if node["number"] in config["excluded"]:
        reasons.append("previously-tested")
    if not 1 <= changed_files <= 6:
        reasons.append("changed-file-count-out-of-range")
    if changed_lines > 400:
        reasons.append("changed-lines-over-400")
    if not any(token in node["title"].lower() for token in TITLE_TOKENS):
        reasons.append("no-high-signal-title-token")
    if not any(path.startswith(config["source_prefixes"]) for path in paths):
        reasons.append("no-profile-source-path")
    if all(path.startswith(("docs/", "doc/")) or path.endswith(".md") for path in paths):
        reasons.append("docs-only")
    if all(
        path.startswith(("test/", "tests/")) or "/test/" in path or "/tests/" in path
        for path in paths
    ):
        reasons.append("tests-only")
    if _is_generated_or_dependency_only(paths):
        reasons.append("dependency-or-generated-only")
    commits = node["commits"]["nodes"]
    if not commits or not commits[0]["commit"]["parents"]["nodes"]:
        reasons.append("first-parent-unavailable")
    if not node.get("headRefOid"):
        reasons.append("head-sha-unavailable")
    return not reasons, reasons


def _rank_key(node: dict[str, Any]) -> tuple[float, int, int, int]:
    created = datetime.fromisoformat(node["createdAt"].replace("Z", "+00:00")).timestamp()
    changed_lines = int(node["additions"] or 0) + int(node["deletions"] or 0)
    return (-created, int(node["changedFiles"]), changed_lines, int(node["number"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if policy["review_text_requested"] is not False:
        raise SystemExit("R7 discovery policy does not keep review text hidden")
    if policy["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 discovery policy does not keep outcomes hidden")

    discoveries: dict[str, Any] = {}
    for project in policy["projects_in_order"]:
        repository = PROJECTS[project]["repository"]
        query_string = f"repo:{repository} is:pr created:2024-01-01..2026-07-31 sort:created-desc"
        cursor: str | None = None
        nodes: list[dict[str, Any]] = []
        issue_count = 0
        page_count = 0
        while True:
            search = _query(query_string, cursor)
            page_count += 1
            issue_count = search["issueCount"]
            nodes.extend(search["nodes"])
            if any(_eligible(project, node)[0] for node in nodes):
                break
            page_info = search["pageInfo"]
            if not page_info["hasNextPage"] or page_count >= 10:
                break
            cursor = page_info["endCursor"]
        candidates: list[dict[str, Any]] = []
        for node in nodes:
            eligible, exclusion_reasons = _eligible(project, node)
            commit = node["commits"]["nodes"][0]["commit"] if node["commits"]["nodes"] else None
            first_parent = (
                commit["parents"]["nodes"][0]["oid"]
                if commit and commit["parents"]["nodes"]
                else None
            )
            candidates.append(
                {
                    "number": node["number"],
                    "title": node["title"],
                    "created_at": node["createdAt"],
                    "author": (node.get("author") or {}).get("login"),
                    "base_ref": node.get("baseRefName"),
                    "base_ref_oid": node.get("baseRefOid"),
                    "head_sha": node.get("headRefOid"),
                    "first_pr_commit_sha": commit["oid"] if commit else None,
                    "first_pr_commit_first_parent_sha": first_parent,
                    "changed_files": node.get("changedFiles"),
                    "additions": node.get("additions"),
                    "deletions": node.get("deletions"),
                    "paths": _paths(node),
                    "eligible": eligible,
                    "exclusion_reasons": exclusion_reasons,
                }
            )
        eligible_nodes = [node for node in nodes if _eligible(project, node)[0]]
        eligible_nodes.sort(key=_rank_key)
        if not eligible_nodes:
            raise SystemExit(f"no eligible R7 candidate for {project}")
        selected_number = eligible_nodes[0]["number"]
        discoveries[project] = {
            "repository": repository,
            "query": query_string,
            "issue_count": issue_count,
            "page_count": page_count,
            "returned_count": len(nodes),
            "eligible_count": len(eligible_nodes),
            "selected_pull_number": selected_number,
            "candidates": candidates,
        }

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "preselection_policy_sha256": canonical_sha256(policy),
        "review_text_visible_during_discovery": False,
        "merge_outcomes_visible_during_discovery": False,
        "discoveries": discoveries,
    }
    payload = {**material, "discovery_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                project: {
                    "selected_pull_number": item["selected_pull_number"],
                    "eligible_count": item["eligible_count"],
                }
                for project, item in discoveries.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"discovery_sha256={payload['discovery_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
