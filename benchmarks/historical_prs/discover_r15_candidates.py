#!/usr/bin/env python3
"""Acquire outcome-free GitHub metadata for the mixed 30-case R15 group."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from discover_r14_candidates import (
    DEPENDENCY_NAMES,
    _first_parent,
    _is_test_path,
    _paths,
    _project_node,
    _query,
)

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def domain_score(node: dict[str, Any], domain: str, policy: dict[str, Any]) -> int:
    signals = policy["domain_signals"][domain]
    weights = policy["domain_signals"]["weights"]
    title = str(node["title"]).lower()
    paths = _paths(node)
    joined_paths = " ".join(paths).lower()
    score = 0
    if any(term in title for term in signals["strong_title_terms"]):
        score += int(weights["strong_title"])
    if any(term in title for term in signals["secondary_title_terms"]):
        score += int(weights["secondary_title"])
    if any(term in joined_paths for term in signals["path_terms"]):
        score += int(weights["domain_path"])
    if any(_is_test_path(path) for path in paths):
        score += int(weights["candidate_test_path"])
    return score


def roughly_eligible(
    node: dict[str, Any], domain: str, project: dict[str, Any], policy: dict[str, Any]
) -> bool:
    rules = policy["eligibility"]
    if not isinstance(node.get("files"), dict):
        return False
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
    if domain_score(node, domain, policy) < int(rules["domain_score_min"]):
        return False
    return not all(
        path.rsplit("/", 1)[-1].lower() in DEPENDENCY_NAMES
        or path.endswith((".lock", ".sum", ".min.js"))
        or path.startswith(("third_party/", "vendor/"))
        for path in paths
    )


def acquire_band(
    repository: str,
    domain: str,
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
            roughly_eligible(node, domain, project, policy) for node in nodes
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
            roughly_eligible(node, domain, project, policy) for node in nodes
        ),
    }
    return projected, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(material):
        raise SystemExit("R15 policy digest mismatch")
    if any(
        policy["blindness"][key] is not False
        for key in (
            "candidate_body_visible",
            "diff_content_visible",
            "state_or_merge_visible",
            "review_or_comment_visible",
            "ci_or_label_visible",
        )
    ):
        raise SystemExit("R15 policy exposes forbidden evidence")

    window = policy["created_at_window"]
    discoveries: dict[str, Any] = {
        domain: {} for domain in policy["domains_in_order"]
    }
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for domain in policy["domains_in_order"]:
        for project_name, project in policy["projects"][domain].items():
            tasks.append((domain, project_name, project))

    def acquire_slice(
        task: tuple[str, str, dict[str, Any]],
    ) -> tuple[str, str, dict[str, Any]]:
        domain, project_name, project = task
        repository = project["repository"]
        recent, recent_metadata = acquire_band(
            repository,
            domain,
            project,
            policy,
            band="recent",
            start=window["recent_start"][:10],
            end=window["observation_cutoff"][:10],
            eligible_target=max(4, int(project["count"])),
            page_limit=4,
        )
        mature, mature_metadata = acquire_band(
            repository,
            domain,
            project,
            policy,
            band="mature",
            start=window["start"][:10],
            end=window["mature_end"][:10],
            eligible_target=max(8, int(project["count"]) * 2),
            page_limit=6,
        )
        by_number = {item["number"]: item for item in mature}
        by_number.update({item["number"]: item for item in recent})
        result = {
            "repository": repository,
            "queries": [recent_metadata, mature_metadata],
            "candidates": list(by_number.values()),
        }
        return domain, project_name, result

    with ThreadPoolExecutor(max_workers=4) as executor:
        acquired = list(executor.map(acquire_slice, tasks))
    for domain, project_name, result in acquired:
        discoveries[domain][project_name] = result

    output_material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "allowed_graphql_fields": policy["blindness"]["allowed_selection_fields"],
        "outcome_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "candidate_body_requested": False,
        "diff_content_requested": False,
        "excluded_resolution_gray_zone_queried": False,
        "discoveries": discoveries,
    }
    payload = {
        **output_material,
        "discovery_sha256": canonical_sha256(output_material),
    }
    atomic_write_json(args.output, payload)
    counts = {
        f"{domain}/{project}": len(item["candidates"])
        for domain, projects in discoveries.items()
        for project, item in projects.items()
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"discovery_sha256={payload['discovery_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
