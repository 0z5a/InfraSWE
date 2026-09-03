#!/usr/bin/env python3
"""Freeze the metadata-only 10-training/20-inference R17 selection."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from discover_r14_candidates import DEPENDENCY_NAMES, _is_test_path
from freeze_r15_selection import domain_score, prior_identities, rank, training_family

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

PROJECT_ALIASES = {
    "megatron": "megatron-core",
    "liger": "liger-kernel",
    "tensorrt_llm": "tensorrt-llm",
}


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


def has_domain_anchor(item: dict[str, Any], domain: str, policy: dict[str, Any]) -> bool:
    anchor = policy["domain_anchor"]
    title = item["title"].strip().lower()
    if anchor["title_docs_prefix_is_excluded"] and re.match(r"^(docs?\b|docs?\(|\[docs?\])", title):
        return False
    source_paths = " ".join(path for path in item["paths"] if not _is_test_path(path)).lower()
    rule = anchor[domain]
    return any(contains_term(title, term) for term in rule["direct_terms"]) or any(
        term in source_paths for term in rule["source_path_terms"]
    )


def inference_family(item: dict[str, Any]) -> str:
    text = (item["title"] + " " + " ".join(item["paths"])).lower()
    if any(term in text for term in ("scheduler", "request", "batch", "queue", "preempt")):
        return "scheduler-progress"
    if any(term in text for term in ("kv_cache", "kv cache", "prefix cache", "cache_manager")):
        return "cache-state-layout"
    if any(term in text for term in ("attention", "decode", "prefill", "sampling")):
        return "attention-numerics"
    if any(term in text for term in ("runner", "engine", "executor", "worker", "lora", "adapter")):
        return "model-runtime-integration"
    return "memory-performance"


def risk_family(item: dict[str, Any], domain: str) -> str:
    return training_family(item) if domain == "training" else inference_family(item)


def eligibility_reasons(
    item: dict[str, Any], domain: str, project: dict[str, Any], policy: dict[str, Any]
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
    if domain_score(item, domain, policy) < int(rules["domain_score_min"]):
        reasons.append("domain-score-below-minimum")
    if not has_domain_anchor(item, domain, policy):
        reasons.append("no-explicit-domain-anchor")
    return reasons


def normalized_title_prefix(title: str, count: int) -> tuple[str, ...]:
    cleaned = re.sub(r"\[[^]]+\]|\([^)]*\)", " ", title.lower())
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    ignored = {"fix", "feat", "feature", "bug", "add", "update", "support", "refactor"}
    meaningful = [token for token in tokens if token not in ignored]
    return tuple(meaningful[:count])


def non_test_paths(item: dict[str, Any]) -> set[str]:
    return {path for path in item["paths"] if not _is_test_path(path)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def stack_allowed(
    candidate: dict[str, Any], selected: list[dict[str, Any]], policy: dict[str, Any]
) -> bool:
    rule = policy["selection_algorithm"]["stacked_series_diversity"]
    prefix = normalized_title_prefix(
        candidate["title"], int(rule["normalized_title_prefix_token_count"])
    )
    paths = non_test_paths(candidate)
    overlaps = sum(
        normalized_title_prefix(item["title"], len(prefix)) == prefix
        and jaccard(non_test_paths(item), paths) >= float(rule["non_test_path_jaccard_min"])
        for item in selected
    )
    return overlaps < int(rule["max_per_overlap_cluster_per_project"])


def select_project(
    items: list[dict[str, Any]], domain: str, policy: dict[str, Any], required: int
) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: rank(item, domain, policy))
    selected: list[dict[str, Any]] = []
    recent = [item for item in ranked if item["temporal_band"] == "recent"]
    if recent and int(policy["created_at_window"]["per_project_recent_target"]) > 0:
        selected.append(recent[0])
    mature = [item for item in ranked if item["temporal_band"] == "mature"]
    covered = {risk_family(item, domain) for item in selected}
    for family in policy["domain_signals"][domain]["risk_families"]:
        if len(selected) >= required:
            break
        if family in covered:
            continue
        candidate = next(
            (
                item
                for item in mature
                if item not in selected
                and risk_family(item, domain) == family
                and stack_allowed(item, selected, policy)
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            covered.add(family)
    for item in mature:
        if len(selected) >= required:
            break
        if item not in selected and stack_allowed(item, selected, policy):
            selected.append(item)
    if len(selected) != required:
        raise SystemExit(f"{domain}: only {len(selected)} eligible diverse cases, need {required}")
    return selected


def main(
    round_label: str = "R17",
    iteration_binding_field: str = "r16_policy_iteration_sha256",
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = read(args.policy)
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit(f"{round_label} policy digest mismatch")
    discovery = read(args.discovery)
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit(f"{round_label} discovery digest mismatch")
    if discovery["policy_sha256"] != policy["policy_sha256"]:
        raise SystemExit(f"{round_label} discovery/policy binding mismatch")
    hidden = (
        discovery["outcome_fields_requested"],
        discovery["review_or_comment_fields_requested"],
        discovery["ci_or_label_fields_requested"],
        discovery["candidate_body_requested"],
        discovery["diff_content_requested"],
        discovery["excluded_resolution_gray_zone_queried"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit(f"{round_label} discovery exposes forbidden evidence")

    prior, prior_bindings = prior_identities(args.prior_lock)
    reserved = set(prior)
    chosen: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for domain in policy["domains_in_order"]:
        diagnostics[domain] = {}
        for project_name, project in policy["projects"][domain].items():
            repository = project["repository"]
            repository_key = repository.lower()
            eligible: list[dict[str, Any]] = []
            exclusions: list[dict[str, Any]] = []
            candidates = discovery["discoveries"][domain][project_name]["candidates"]
            for item in candidates:
                reasons = eligibility_reasons(item, domain, project, policy)
                candidate_identity = (repository_key, int(item["number"]))
                if candidate_identity in prior:
                    reasons.append("previously-scored")
                elif candidate_identity in reserved:
                    reasons.append(f"selected-in-earlier-{round_label.lower()}-domain")
                enriched = {
                    **item,
                    "domain_score": domain_score(item, domain, policy),
                    "risk_family": risk_family(item, domain),
                }
                if reasons:
                    exclusions.append({**enriched, "exclusion_reasons": reasons})
                else:
                    eligible.append(enriched)
            required = int(project["count"])
            selected = select_project(eligible, domain, policy, required)
            query_strings = [
                item["query"] for item in discovery["discoveries"][domain][project_name]["queries"]
            ]
            diagnostics[domain][project_name] = {
                "repository": repository,
                "candidate_count": len(candidates),
                "eligible_count": len(eligible),
                "excluded_count": len(exclusions),
                "selected_numbers": [int(item["number"]) for item in selected],
                "selected_recent_count": sum(
                    item["temporal_band"] == "recent" for item in selected
                ),
                "selected_risk_families": {
                    family: sum(item["risk_family"] == family for item in selected)
                    for family in policy["domain_signals"][domain]["risk_families"]
                },
                "stack_diversity_rule_applied": True,
                "excluded_identity_count": sum(
                    "previously-scored" in item["exclusion_reasons"] for item in exclusions
                ),
            }
            for item in selected:
                reserved.add((repository_key, int(item["number"])))
                chosen.append(
                    {
                        "schema_version": "0.5",
                        "case_id": f"{project_name}-pr-{item['number']}",
                        "project": PROJECT_ALIASES.get(project_name, project_name),
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
                        "acquisition_query": " | ".join(query_strings),
                        "selection_policy_id": policy["protocol_id"],
                        "outcome_fields_requested": False,
                        "benchmark_domain": domain,
                        "temporal_band": item["temporal_band"],
                        "domain_score": item["domain_score"],
                        "risk_family": item["risk_family"],
                    }
                )

    expected_count = int(policy["case_count"])
    if len(chosen) != expected_count or len({item["case_id"] for item in chosen}) != expected_count:
        raise SystemExit(f"{round_label} selection is not {expected_count} unique cases")
    allocation = {
        domain: sum(item["benchmark_domain"] == domain for item in chosen)
        for domain in policy["domains_in_order"]
    }
    if allocation != policy["domain_allocation"]:
        raise SystemExit(f"{round_label} domain allocation changed")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        iteration_binding_field: policy[iteration_binding_field],
        "prior_lock_bindings": prior_bindings,
        "prior_identity_count": len(prior),
        "review_or_comment_visible": False,
        "merge_outcomes_visible": False,
        "ci_or_label_visible": False,
        "candidate_body_visible": False,
        "diff_content_visible": False,
        "excluded_resolution_gray_zone_used": False,
        "selection_basis": "frozen title/path/size/time/SHA/domain/stack ranking only",
        "machine_policy_id": policy["machine_policy"]["policy_id"],
        "domain_allocation": allocation,
        "frozen_at": datetime.now(UTC).isoformat(),
        "selection_diagnostics": diagnostics,
        "cases": chosen,
    }
    payload = {"selection_material": material, "selection_lock_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            [
                {
                    "case_id": item["case_id"],
                    "domain": item["benchmark_domain"],
                    "band": item["temporal_band"],
                    "risk_family": item["risk_family"],
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
