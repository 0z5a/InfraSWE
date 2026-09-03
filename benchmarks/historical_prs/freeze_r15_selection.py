#!/usr/bin/env python3
"""Deterministically freeze the mixed 20-communication/10-training R15 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from discover_r14_candidates import DEPENDENCY_NAMES, _is_test_path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

REPOSITORY_ALIASES = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
    "flashinfer": "flashinfer-ai/flashinfer",
    "flashattention": "Dao-AILab/flash-attention",
    "megatron": "NVIDIA/Megatron-LM",
    "torchtitan": "pytorch/torchtitan",
    "verl": "verl-project/verl",
    "slime": "THUDM/slime",
    "liger": "linkedin/Liger-Kernel",
    "cutlass": "NVIDIA/cutlass",
    "deepgemm": "deepseek-ai/DeepGEMM",
    "tensorrt_llm": "NVIDIA/TensorRT-LLM",
    "tensorrt-llm": "NVIDIA/TensorRT-LLM",
}
MODEL_PROJECT_ALIASES = {"megatron": "megatron-core", "liger": "liger-kernel"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_repository(repository: str) -> str:
    value = repository.removeprefix("https://github.com/").removesuffix(".git")
    return value.strip("/").lower()


def identity(item: Any) -> tuple[str, int] | None:
    if isinstance(item, str):
        case_id = item
        repository = None
        number = None
    elif isinstance(item, dict):
        material = item.get("material")
        if isinstance(material, dict):
            return identity(material)
        repository = item.get("repository")
        number = item.get("pull_number", item.get("number"))
        if isinstance(repository, str) and isinstance(number, int):
            return normalize_repository(repository), number
        case_id = item.get("case_id")
    else:
        return None
    if not isinstance(case_id, str):
        return None
    prefix, separator, raw_number = case_id.rpartition("-pr-")
    if not separator:
        prefix, separator, raw_number = case_id.rpartition("#")
    if not separator or not raw_number.isdigit() or prefix not in REPOSITORY_ALIASES:
        return None
    return REPOSITORY_ALIASES[prefix].lower(), int(raw_number)


def collections(payload: Any) -> list[list[Any]]:
    if isinstance(payload, list):
        return [payload]
    if not isinstance(payload, dict):
        return []
    found: list[list[Any]] = []
    for key in ("cases", "selected_cases", "locks"):
        if isinstance(payload.get(key), list):
            found.append(payload[key])
    material = payload.get("selection_material")
    if isinstance(material, dict) and isinstance(material.get("cases"), list):
        found.append(material["cases"])
    return found


def prior_identities(
    paths: list[Path],
) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    identities: set[tuple[str, int]] = set()
    bindings: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=str):
        payload = read(path)
        before = len(identities)
        for group in collections(payload):
            for item in group:
                resolved = identity(item)
                if resolved is not None:
                    identities.add(resolved)
        bindings.append(
            {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "new_identity_count": len(identities) - before,
            }
        )
    return identities, bindings


def domain_score(item: dict[str, Any], domain: str, policy: dict[str, Any]) -> int:
    signals = policy["domain_signals"][domain]
    weights = policy["domain_signals"]["weights"]
    title = item["title"].lower()
    paths = " ".join(item["paths"]).lower()
    score = 0
    if any(term in title for term in signals["strong_title_terms"]):
        score += int(weights["strong_title"])
    if any(term in title for term in signals["secondary_title_terms"]):
        score += int(weights["secondary_title"])
    if any(term in paths for term in signals["path_terms"]):
        score += int(weights["domain_path"])
    if any(_is_test_path(path) for path in item["paths"]):
        score += int(weights["candidate_test_path"])
    return score


def contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    prefix = r"(?<![a-z0-9])" if term[0].isalnum() else ""
    suffix = r"(?![a-z0-9])" if term[-1].isalnum() else ""
    return re.search(prefix + escaped + suffix, text.lower()) is not None


def has_domain_anchor(
    item: dict[str, Any], domain: str, amendment: dict[str, Any]
) -> bool:
    rule = amendment["rule"]
    title = item["title"].strip().lower()
    if rule["title_docs_prefix_is_excluded"] and re.match(
        r"^(docs?\b|docs?\(|\[docs?\])", title
    ):
        return False
    source_paths = " ".join(
        path for path in item["paths"] if not _is_test_path(path)
    ).lower()
    domain_rule = rule[domain]
    if domain == "communication":
        if any(
            contains_term(title, term) or contains_term(source_paths, term)
            for term in domain_rule["direct_terms"]
        ):
            return True
        title_has_topology = any(
            contains_term(title, term)
            for term in domain_rule["topology_title_terms"]
        )
        path_has_runtime = any(
            term in source_paths for term in domain_rule["runtime_path_terms"]
        )
        return title_has_topology and path_has_runtime
    return any(
        contains_term(title, term) for term in domain_rule["direct_terms"]
    ) or any(term in source_paths for term in domain_rule["source_path_terms"])


def communication_family(item: dict[str, Any]) -> str:
    text = (item["title"] + " " + " ".join(item["paths"])).lower()
    if any(
        term in text
        for term in (
            "all_reduce", "all-reduce", "allreduce", "all_gather", "all-gather",
            "reduce_scatter", "all_to_all", "all-to-all", "broadcast", "collective",
        )
    ):
        return "collective-numerics"
    if any(
        term in text
        for term in (
            "p2p", "send/recv", "send_recv", "cuda ipc", "cuda_ipc",
            "weight transfer", "weight_transfer", "weight sync", "weight_sync",
        )
    ):
        return "p2p-transfer"
    if any(term in text for term in ("overlap", "async", "stream", "deadlock", "hang")):
        return "overlap-ordering"
    if any(
        term in text
        for term in (
            "communicator", "process_group", "process group", "initialize",
            "destroy", "teardown", "lifecycle", "ipc cache",
        )
    ):
        return "resource-lifecycle"
    return "rank-topology-sharding"


def training_family(item: dict[str, Any]) -> str:
    text = (item["title"] + " " + " ".join(item["paths"])).lower()
    if any(
        term in text
        for term in ("checkpoint", "resume", "state_dict", "state dict", "save", "load")
    ):
        return "checkpoint-resume"
    if any(term in text for term in ("optimizer", "adam", "lion", "muon", "learning rate", "lr_")):
        return "optimizer-state"
    if any(
        term in text
        for term in ("loss", "gradient", "backward", "reward", "ppo", "grpo", "dpo")
    ):
        return "loss-gradient"
    if any(term in text for term in ("pipeline", "schedule", "microbatch", "micro-batch")):
        return "scheduling-pipeline"
    return "memory-performance"


def risk_family(item: dict[str, Any], domain: str) -> str:
    return communication_family(item) if domain == "communication" else training_family(item)


def eligibility_reasons(
    item: dict[str, Any],
    domain: str,
    project: dict[str, Any],
    policy: dict[str, Any],
    amendment: dict[str, Any],
) -> list[str]:
    rules = policy["eligibility"]
    paths = item["paths"]
    reasons: list[str] = []
    changed_lines = int(item["additions"]) + int(item["deletions"])
    if not int(rules["changed_files_min"]) <= int(item["changed_files"]) <= int(
        rules["changed_files_max"]
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
    if not has_domain_anchor(item, domain, amendment):
        reasons.append("no-explicit-domain-anchor")
    return reasons


def rank(item: dict[str, Any], domain: str, policy: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if any(_is_test_path(path) for path in item["paths"]) else 1,
        -domain_score(item, domain, policy),
        -datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")).timestamp(),
        int(item["changed_files"]),
        int(item["additions"]) + int(item["deletions"]),
        int(item["number"]),
    )


def select_project(
    items: list[dict[str, Any]],
    domain: str,
    policy: dict[str, Any],
    required: int,
    *,
    allow_recent: bool,
) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: rank(item, domain, policy))
    selected: list[dict[str, Any]] = []
    numbers: set[int] = set()
    recent = [item for item in ranked if item["temporal_band"] == "recent"]
    if (
        allow_recent
        and recent
        and int(policy["created_at_window"]["per_project_recent_target"]) > 0
    ):
        selected.append(recent[0])
        numbers.add(int(recent[0]["number"]))
    remaining = [item for item in ranked if item["temporal_band"] == "mature"]
    covered = {risk_family(item, domain) for item in selected}
    for family in policy["domain_signals"][domain]["risk_families"]:
        if len(selected) >= required:
            break
        if family in covered:
            continue
        candidate = next(
            (
                item
                for item in remaining
                if int(item["number"]) not in numbers
                and risk_family(item, domain) == family
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            numbers.add(int(candidate["number"]))
            covered.add(family)
    for item in remaining:
        if len(selected) >= required:
            break
        if int(item["number"]) not in numbers:
            selected.append(item)
            numbers.add(int(item["number"]))
    if len(selected) != required:
        raise SystemExit(
            f"{domain}: only {len(selected)} eligible cases, need {required}"
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
    parser.add_argument("--domain-amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = read(args.policy)
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit("R15 policy digest mismatch")
    discovery = read(args.discovery)
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit("R15 discovery digest mismatch")
    if discovery["policy_sha256"] != policy["policy_sha256"]:
        raise SystemExit("R15 discovery/policy binding mismatch")
    amendment = read(args.domain_amendment)
    amendment_material = {
        key: value for key, value in amendment.items() if key != "amendment_sha256"
    }
    if amendment["amendment_sha256"] != canonical_sha256(amendment_material):
        raise SystemExit("R15 amendment digest mismatch")
    if amendment["policy_sha256"] != policy["policy_sha256"]:
        raise SystemExit("R15 amendment/policy binding mismatch")
    if amendment["discovery_sha256"] != discovery["discovery_sha256"]:
        raise SystemExit("R15 amendment/discovery binding mismatch")
    forbidden_amendment = (
        amendment["outcome_or_state_used"],
        amendment["review_or_comment_used"],
        amendment["ci_or_label_used"],
        amendment["candidate_body_used"],
        amendment["diff_content_used"],
        amendment["identity_specific_exception_used"],
    )
    if any(value is not False for value in forbidden_amendment):
        raise SystemExit("R15 amendment used forbidden evidence")
    hidden = (
        discovery["outcome_fields_requested"],
        discovery["review_or_comment_fields_requested"],
        discovery["ci_or_label_fields_requested"],
        discovery["candidate_body_requested"],
        discovery["diff_content_requested"],
        discovery["excluded_resolution_gray_zone_queried"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit("R15 discovery exposes forbidden evidence")

    prior, prior_bindings = prior_identities(args.prior_lock)
    reserved = set(prior)
    recent_repositories: set[str] = set()
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
                reasons = eligibility_reasons(
                    item, domain, project, policy, amendment
                )
                candidate_identity = (repository_key, int(item["number"]))
                if candidate_identity in prior:
                    reasons.append("previously-scored")
                elif candidate_identity in reserved:
                    reasons.append("selected-in-earlier-r15-domain")
                enriched = {
                    **item,
                    "domain_score": domain_score(item, domain, policy),
                    "risk_family": risk_family(item, domain),
                }
                if reasons:
                    exclusions.append({**enriched, "exclusion_reasons": reasons})
                else:
                    eligible.append(enriched)
            required = int(
                amendment.get("project_count_override", {})
                .get(domain, {})
                .get(project_name, project["count"])
            )
            selected = select_project(
                eligible,
                domain,
                policy,
                required,
                allow_recent=repository_key not in recent_repositories,
            )
            if any(item["temporal_band"] == "recent" for item in selected):
                recent_repositories.add(repository_key)
            query_strings = [
                item["query"]
                for item in discovery["discoveries"][domain][project_name]["queries"]
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
                "excluded_identity_count": sum(
                    "previously-scored" in item["exclusion_reasons"]
                    for item in exclusions
                ),
            }
            for item in selected:
                reserved.add((repository_key, int(item["number"])))
                candidate = {
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
                    "acquisition_query": " | ".join(query_strings),
                    "selection_policy_id": policy["protocol_id"],
                    "outcome_fields_requested": False,
                    "benchmark_domain": domain,
                    "temporal_band": item["temporal_band"],
                    "domain_score": item["domain_score"],
                    "risk_family": item["risk_family"],
                }
                chosen.append(candidate)

    if len(chosen) != 30 or len({item["case_id"] for item in chosen}) != 30:
        raise SystemExit("R15 selection is not 30 unique cases")
    allocation = {
        domain: sum(item["benchmark_domain"] == domain for item in chosen)
        for domain in policy["domains_in_order"]
    }
    if allocation != policy["domain_allocation"]:
        raise SystemExit("R15 domain allocation changed")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        "r14_policy_iteration_sha256": policy["r14_policy_iteration_sha256"],
        "domain_amendment_sha256": amendment["amendment_sha256"],
        "supersedes_invalid_selection_lock_sha256": amendment[
            "superseded_selection_lock_sha256"
        ],
        "supersession_reason": amendment["reason"],
        "prior_lock_bindings": prior_bindings,
        "prior_identity_count": len(prior),
        "review_or_comment_visible": False,
        "merge_outcomes_visible": False,
        "ci_or_label_visible": False,
        "candidate_body_visible": False,
        "diff_content_visible": False,
        "excluded_resolution_gray_zone_used": False,
        "selection_basis": "frozen title/path/size/time/SHA/domain ranking only",
        "machine_policy_id": policy["machine_policy"]["policy_id"],
        "domain_allocation": allocation,
        "frozen_at": datetime.now(UTC).isoformat(),
        "selection_diagnostics": diagnostics,
        "cases": chosen,
    }
    payload = {
        "selection_material": material,
        "selection_lock_sha256": canonical_sha256(material),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps([
        {
            "case_id": item["case_id"],
            "domain": item["benchmark_domain"],
            "band": item["temporal_band"],
            "risk_family": item["risk_family"],
        }
        for item in chosen
    ], indent=2))
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
