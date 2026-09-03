#!/usr/bin/env python3
"""Freeze the metadata-only 30-case R16 training selection."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from discover_r14_candidates import DEPENDENCY_NAMES, _is_test_path
from freeze_r15_selection import (
    MODEL_PROJECT_ALIASES,
    domain_score,
    prior_identities,
    rank,
    select_project,
    training_family,
)

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    prefix = r"(?<![a-z0-9])" if term[0].isalnum() else ""
    suffix = r"(?![a-z0-9])" if term[-1].isalnum() else ""
    return re.search(prefix + escaped + suffix, text.lower()) is not None


def has_training_anchor(item: dict[str, Any], policy: dict[str, Any]) -> bool:
    anchor = policy["domain_anchor"]
    title = item["title"].strip().lower()
    if anchor["title_docs_prefix_is_excluded"] and re.match(r"^(docs?\b|docs?\(|\[docs?\])", title):
        return False
    source_paths = " ".join(path for path in item["paths"] if not _is_test_path(path)).lower()
    return any(contains_term(title, term) for term in anchor["direct_terms"]) or any(
        term in source_paths for term in anchor["source_path_terms"]
    )


def eligibility_reasons(
    item: dict[str, Any],
    project: dict[str, Any],
    policy: dict[str, Any],
    amendment: dict[str, Any] | None,
) -> list[str]:
    rules = policy["eligibility"]
    paths = item["paths"]
    reasons: list[str] = []
    changed_lines = int(item["additions"]) + int(item["deletions"])
    if (
        not int(rules["changed_files_min"])
        <= int(item["changed_files"])
        <= int(rules["changed_files_max"])
    ):
        reasons.append("changed-file-count-out-of-range")
    if changed_lines > int(rules["changed_lines_max"]):
        reasons.append("changed-lines-over-limit")
    if not item["path_list_complete"]:
        reasons.append("incomplete-path-list")
    if not item.get("base_sha") or not item.get("head_sha") or not item.get("base_ref_oid"):
        reasons.append("sha-metadata-unavailable")
    if not any(path.startswith(tuple(project["source_prefixes"])) for path in paths):
        reasons.append("no-runtime-source-path")
    if all(path.endswith(".md") or path.startswith(("docs/", "doc/")) for path in paths):
        reasons.append("docs-only")
    if all(_is_test_path(path) for path in paths):
        reasons.append("tests-only")
    if all(
        path.rsplit("/", 1)[-1].lower() in DEPENDENCY_NAMES
        or path.endswith((".lock", ".sum", ".min.js"))
        or path.startswith(("third_party/", "vendor/"))
        for path in paths
    ):
        reasons.append("dependency-or-generated-only")
    if domain_score(item, "training", policy) < int(rules["domain_score_min"]):
        reasons.append("domain-score-below-minimum")
    if not has_training_anchor(item, policy):
        reasons.append("no-explicit-training-anchor")
    if amendment is not None:
        inference_rule = amendment["rules"]["exclude_inference_only"]
        runtime_paths = [path.lower() for path in paths if not _is_test_path(path)]
        inference_only = bool(runtime_paths) and all(
            any(fragment in path for fragment in inference_rule["inference_path_fragments"])
            for path in runtime_paths
        )
        has_training_override = any(
            contains_term(item["title"], term)
            for term in inference_rule["training_override_title_terms"]
        )
        if inference_only and not has_training_override:
            reasons.append("inference-only-runtime")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
    parser.add_argument("--metadata-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = read(args.policy)
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit("R16 policy digest mismatch")
    discovery = read(args.discovery)
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit("R16 discovery digest mismatch")
    if discovery["policy_sha256"] != policy["policy_sha256"]:
        raise SystemExit("R16 discovery/policy binding mismatch")
    hidden = (
        discovery["outcome_fields_requested"],
        discovery["review_or_comment_fields_requested"],
        discovery["ci_or_label_fields_requested"],
        discovery["candidate_body_requested"],
        discovery["diff_content_requested"],
        discovery["excluded_resolution_gray_zone_queried"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R16 discovery exposes forbidden evidence")

    amendment = read(args.metadata_amendment) if args.metadata_amendment else None
    if amendment is not None:
        amendment_material = {
            key: value for key, value in amendment.items() if key != "amendment_sha256"
        }
        if amendment["amendment_sha256"] != canonical_sha256(amendment_material):
            raise SystemExit("R16 amendment digest mismatch")
        if amendment["policy_sha256"] != policy["policy_sha256"]:
            raise SystemExit("R16 amendment/policy binding mismatch")
        if amendment["discovery_sha256"] != discovery["discovery_sha256"]:
            raise SystemExit("R16 amendment/discovery binding mismatch")
        forbidden = (
            amendment["outcome_or_state_used"],
            amendment["review_or_comment_used"],
            amendment["ci_or_label_used"],
            amendment["candidate_body_used"],
            amendment["diff_content_used"],
            amendment["identity_specific_exception_used"],
        )
        if any(value is not False for value in forbidden):
            raise SystemExit("R16 amendment used forbidden evidence")

    prior, prior_bindings = prior_identities(args.prior_lock)
    chosen: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for project_name, project in policy["projects"]["training"].items():
        repository = project["repository"]
        repository_key = repository.lower()
        eligible: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        candidates = discovery["discoveries"]["training"][project_name]["candidates"]
        for item in candidates:
            reasons = eligibility_reasons(item, project, policy, amendment)
            if (repository_key, int(item["number"])) in prior:
                reasons.append("previously-scored")
            enriched = {
                **item,
                "domain_score": domain_score(item, "training", policy),
                "risk_family": training_family(item),
            }
            if reasons:
                exclusions.append({**enriched, "exclusion_reasons": reasons})
            else:
                eligible.append(enriched)
        if amendment is not None:
            deduplicated: list[dict[str, Any]] = []
            signatures: set[tuple[str, tuple[str, ...]]] = set()
            for item in sorted(eligible, key=lambda value: rank(value, "training", policy)):
                signature = (
                    " ".join(item["title"].lower().split()),
                    tuple(path for path in item["paths"] if not _is_test_path(path)),
                )
                if signature in signatures:
                    exclusions.append(
                        {
                            **item,
                            "exclusion_reasons": ["duplicate-metadata-signature"],
                        }
                    )
                else:
                    signatures.add(signature)
                    deduplicated.append(item)
            eligible = deduplicated
        required = int(project["count"])
        selected = select_project(
            eligible,
            "training",
            policy,
            required,
            allow_recent=True,
        )
        queries = [
            item["query"] for item in discovery["discoveries"]["training"][project_name]["queries"]
        ]
        diagnostics[project_name] = {
            "repository": repository,
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "excluded_count": len(exclusions),
            "selected_numbers": [int(item["number"]) for item in selected],
            "selected_recent_count": sum(item["temporal_band"] == "recent" for item in selected),
            "selected_risk_families": {
                family: sum(item["risk_family"] == family for item in selected)
                for family in policy["domain_signals"]["training"]["risk_families"]
            },
            "excluded_identity_count": sum(
                "previously-scored" in item["exclusion_reasons"] for item in exclusions
            ),
        }
        for item in selected:
            chosen.append(
                {
                    "schema_version": "0.5",
                    "case_id": f"{project_name}-pr-{item['number']}",
                    "project": MODEL_PROJECT_ALIASES.get(project_name, project_name),
                    "repository": repository,
                    "pull_number": item["number"],
                    "title": item["title"],
                    "created_at": item["created_at"],
                    "base_ref": item["base_ref"],
                    "base_tip_sha": item["base_ref_oid"],
                    "base_sha": item["base_sha"],
                    "base_derivation": "first-pr-commit-first-parent-path-parity",
                    "head_sha": item["head_sha"],
                    "pr_commit_shas": [item["head_sha"]],
                    "changed_files": item["changed_files"],
                    "additions": item["additions"],
                    "deletions": item["deletions"],
                    "paths": item["paths"],
                    "acquisition_query": " | ".join(queries),
                    "selection_policy_id": policy["protocol_id"],
                    "outcome_fields_requested": False,
                    "benchmark_domain": "training",
                    "temporal_band": item["temporal_band"],
                    "domain_score": item["domain_score"],
                    "risk_family": item["risk_family"],
                }
            )

    if len(chosen) != 30 or len({item["case_id"] for item in chosen}) != 30:
        raise SystemExit("R16 selection is not 30 unique cases")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        "r15_policy_iteration_sha256": policy["r15_policy_iteration_sha256"],
        "metadata_amendment_sha256": (
            amendment["amendment_sha256"] if amendment is not None else None
        ),
        "supersedes_selection_lock_sha256": (
            amendment["superseded_selection_lock_sha256"] if amendment is not None else None
        ),
        "prior_lock_bindings": prior_bindings,
        "prior_identity_count": len(prior),
        "review_or_comment_visible": False,
        "merge_outcomes_visible": False,
        "ci_or_label_visible": False,
        "candidate_body_visible": False,
        "diff_content_visible": False,
        "excluded_resolution_gray_zone_used": False,
        "selection_basis": "frozen title/path/size/time/SHA/training-family ranking only",
        "machine_policy_id": policy["machine_policy"]["policy_id"],
        "domain_allocation": {"training": 30},
        "frozen_at": datetime.now(UTC).isoformat(),
        "selection_diagnostics": diagnostics,
        "cases": chosen,
    }
    payload = {
        "selection_material": material,
        "selection_lock_sha256": canonical_sha256(material),
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            [
                {
                    "case_id": item["case_id"],
                    "band": item["temporal_band"],
                    "risk_family": item["risk_family"],
                    "changed_files": item["changed_files"],
                }
                for item in chosen
            ],
            indent=2,
        )
    )
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
