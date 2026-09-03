#!/usr/bin/env python3
"""Deterministically freeze 30 R14 communication cases from blind metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

REPOSITORY_ALIASES = {
    "vllm": "vllm-project/vllm",
    "sglang": "sgl-project/sglang",
    "flashinfer": "flashinfer-ai/flashinfer",
    "flashattention": "Dao-AILab/flash-attention",
    "megatron": "NVIDIA/Megatron-LM",
    "megatron-core": "NVIDIA/Megatron-LM",
    "torchtitan": "pytorch/torchtitan",
    "verl": "verl-project/verl",
    "slime": "THUDM/slime",
    "liger": "linkedin/Liger-Kernel",
    "liger-kernel": "linkedin/Liger-Kernel",
    "cutlass": "NVIDIA/cutlass",
    "cutlass-cute": "NVIDIA/cutlass",
    "deepgemm": "deepseek-ai/DeepGEMM",
}
MODEL_PROJECT_ALIASES = {"megatron": "megatron-core"}
DEPENDENCY_NAMES = {
    "cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _case_collections(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    collections: list[list[dict[str, Any]]] = []
    for key in ("cases", "selected_cases"):
        value = payload.get(key)
        if isinstance(value, list):
            collections.append([item for item in value if isinstance(item, dict)])
    material = payload.get("selection_material")
    if isinstance(material, dict) and isinstance(material.get("cases"), list):
        collections.append([item for item in material["cases"] if isinstance(item, dict)])
    return collections


def _normalize_repository(repository: str) -> str:
    normalized = repository.removeprefix("https://github.com/").removesuffix(".git")
    return normalized.strip("/")


def _identity(item: dict[str, Any]) -> tuple[str, int] | None:
    number = item.get("pull_number", item.get("number"))
    repository = item.get("repository")
    if isinstance(repository, str) and isinstance(number, int):
        return _normalize_repository(repository).lower(), number
    case_id = item.get("case_id")
    if not isinstance(case_id, str):
        candidate_ref = item.get("candidate_ref")
        if isinstance(candidate_ref, str) and "#" in candidate_ref:
            prefix, raw_number = candidate_ref.rsplit("#", 1)
            if raw_number.isdigit() and prefix in REPOSITORY_ALIASES:
                return REPOSITORY_ALIASES[prefix].lower(), int(raw_number)
        return None
    prefix, separator, raw_number = case_id.rpartition("-pr-")
    if not separator or not raw_number.isdigit() or prefix not in REPOSITORY_ALIASES:
        return None
    return REPOSITORY_ALIASES[prefix].lower(), int(raw_number)


def _prior_identities(paths: list[Path]) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    identities: set[tuple[str, int]] = set()
    bindings: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item)):
        payload = _read(path)
        before = len(identities)
        for collection in _case_collections(payload):
            for item in collection:
                identity = _identity(item)
                if identity is not None:
                    identities.add(identity)
        bindings.append(
            {
                "path": str(path),
                "file_sha256": _file_sha256(path),
                "new_identity_count": len(identities) - before,
            }
        )
    return identities, bindings


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith(("_test.py", "_test.cpp", "_test.cu", ".test.ts"))
    )


def _communication_score(item: dict[str, Any], policy: dict[str, Any]) -> int:
    signals = policy["communication_signal_policy"]
    weights = signals["weights"]
    title = item["title"].lower()
    paths = " ".join(item["paths"]).lower()
    score = 0
    if any(term in title for term in signals["strong_terms"]):
        score += int(weights["strong_title"])
    if any(term in title for term in signals["topology_terms"]):
        score += int(weights["topology_title"])
    if any(term in paths for term in signals["path_terms"]):
        score += int(weights["communication_path"])
    topology_path_terms = [term.replace(" ", "_") for term in signals["topology_terms"]]
    if any(term in paths for term in topology_path_terms):
        score += int(weights["topology_path"])
    if any(_is_test_path(path) for path in item["paths"]):
        score += int(weights["candidate_test_path"])
    return score


def _risk_family(item: dict[str, Any]) -> str:
    text = (item["title"] + " " + " ".join(item["paths"])).lower()
    if any(
        term in text
        for term in (
            "all_reduce",
            "all-reduce",
            "allreduce",
            "all_gather",
            "all-gather",
            "reduce_scatter",
            "all_to_all",
            "all-to-all",
            "broadcast",
            "collective",
        )
    ):
        return "collective-numerics"
    if any(
        term in text
        for term in (
            "p2p",
            "send/recv",
            "send_recv",
            "cuda ipc",
            "cuda_ipc",
            "weight transfer",
            "weight_transfer",
            "weight sync",
            "weight_sync",
        )
    ):
        return "p2p-transfer"
    if any(term in text for term in ("overlap", "async", "stream", "deadlock", "hang")):
        return "overlap-ordering"
    if any(
        term in text
        for term in (
            "communicator",
            "process_group",
            "process group",
            "initialize",
            "destroy",
            "teardown",
            "lifecycle",
            "ipc cache",
        )
    ):
        return "resource-lifecycle"
    return "rank-topology-sharding"


def _has_domain_anchor(item: dict[str, Any], amendment: dict[str, Any]) -> bool:
    rule = amendment["rule"]
    title = item["title"].lower()
    paths = " ".join(path for path in item["paths"] if not _is_test_path(path)).lower()
    if any(term in title or term in paths for term in rule["direct_anchor_terms"]):
        return True
    title_has_topology = any(term in title for term in rule["coupled_topology_title_terms"])
    path_has_runtime = any(term in paths for term in rule["coupled_runtime_path_terms"])
    return title_has_topology and path_has_runtime


def _eligibility_reasons(
    item: dict[str, Any], project: dict[str, Any], policy: dict[str, Any]
) -> list[str]:
    rules = policy["eligibility"]
    paths = item["paths"]
    changed_lines = int(item["additions"]) + int(item["deletions"])
    reasons: list[str] = []
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
    if not item.get("base_sha"):
        reasons.append("first-commit-parent-unavailable")
    if not item.get("head_sha") or not item.get("base_ref_oid"):
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
    if _communication_score(item, policy) < int(rules["communication_score_min"]):
        reasons.append("communication-score-below-minimum")
    return reasons


def _rank(item: dict[str, Any], policy: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if any(_is_test_path(path) for path in item["paths"]) else 1,
        -_communication_score(item, policy),
        -datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")).timestamp(),
        int(item["changed_files"]),
        int(item["additions"]) + int(item["deletions"]),
        int(item["number"]),
    )


def _select_project(
    items: list[dict[str, Any]], policy: dict[str, Any], required: int
) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: _rank(item, policy))
    selected: list[dict[str, Any]] = []
    selected_numbers: set[int] = set()
    recent = [item for item in ranked if item["temporal_band"] == "recent"]
    if recent and int(policy["created_at_window"]["per_project_recent_target"]) > 0:
        selected.append(recent[0])
        selected_numbers.add(int(recent[0]["number"]))
    covered_families = {_risk_family(item) for item in selected}
    for family in policy["risk_families_in_order"]:
        if len(selected) >= required:
            break
        if family in covered_families:
            continue
        candidate = next(
            (
                item
                for item in ranked
                if int(item["number"]) not in selected_numbers and _risk_family(item) == family
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_numbers.add(int(candidate["number"]))
            covered_families.add(family)
    for item in ranked:
        if len(selected) >= required:
            break
        if int(item["number"]) not in selected_numbers:
            selected.append(item)
            selected_numbers.add(int(item["number"]))
    if len(selected) != required:
        raise SystemExit(f"only {len(selected)} eligible cases, need {required}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
    parser.add_argument("--domain-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = _read(args.policy)
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit("R14 policy digest mismatch")
    discovery = _read(args.discovery)
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit("R14 discovery digest mismatch")
    if discovery["policy_sha256"] != policy["policy_sha256"]:
        raise SystemExit("R14 discovery/policy binding mismatch")
    hidden_flags = (
        discovery["outcome_fields_requested"],
        discovery["review_or_comment_fields_requested"],
        discovery["ci_or_label_fields_requested"],
        discovery["candidate_body_requested"],
        discovery["diff_content_requested"],
    )
    if any(value is not False for value in hidden_flags):
        raise SystemExit("R14 discovery exposes forbidden evidence")

    amendment: dict[str, Any] | None = None
    if args.domain_amendment is not None:
        amendment = _read(args.domain_amendment)
        amendment_material = {
            key: value for key, value in amendment.items() if key != "amendment_sha256"
        }
        if amendment["amendment_sha256"] != canonical_sha256(amendment_material):
            raise SystemExit("R14 domain amendment digest mismatch")
        if amendment["policy_sha256"] != policy["policy_sha256"]:
            raise SystemExit("R14 amendment/policy binding mismatch")
        if amendment["discovery_sha256"] != discovery["discovery_sha256"]:
            raise SystemExit("R14 amendment/discovery binding mismatch")
        forbidden_amendment_inputs = (
            amendment["outcome_or_state_used"],
            amendment["review_or_comment_used"],
            amendment["ci_or_label_used"],
            amendment["candidate_body_used"],
            amendment["diff_content_used"],
            amendment["identity_specific_exception_used"],
        )
        if any(value is not False for value in forbidden_amendment_inputs):
            raise SystemExit("R14 domain amendment used forbidden or identity-specific input")

    prior, prior_bindings = _prior_identities(args.prior_lock)
    chosen: list[HistoricalPRCandidate] = []
    selection_diagnostics: dict[str, Any] = {}
    for project_name in policy["projects_in_order"]:
        project = policy["projects"][project_name]
        repository = project["repository"]
        repository_key = repository.lower()
        candidates = discovery["discoveries"][project_name]["candidates"]
        eligible: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for item in candidates:
            reasons = _eligibility_reasons(item, project, policy)
            if amendment is not None and not _has_domain_anchor(item, amendment):
                reasons.append("no-explicit-communication-domain-anchor")
            if (repository_key, int(item["number"])) in prior:
                reasons.append("previously-scored")
            enriched = {
                **item,
                "communication_score": _communication_score(item, policy),
                "risk_family": _risk_family(item),
            }
            if reasons:
                exclusions.append({**enriched, "exclusion_reasons": reasons})
            else:
                eligible.append(enriched)
        selected = _select_project(eligible, policy, int(project["count"]))
        selected_numbers = {int(item["number"]) for item in selected}
        selection_diagnostics[project_name] = {
            "repository": repository,
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "excluded_count": len(exclusions),
            "selected_numbers": sorted(selected_numbers),
            "selected_risk_families": {
                family: sum(item["risk_family"] == family for item in selected)
                for family in policy["risk_families_in_order"]
            },
            "selected_recent_count": sum(item["temporal_band"] == "recent" for item in selected),
            "excluded_identity_count": sum(
                "previously-scored" in item["exclusion_reasons"] for item in exclusions
            ),
        }
        query_strings = [
            item["query"] for item in discovery["discoveries"][project_name]["queries"]
        ]
        for item in selected:
            chosen.append(
                HistoricalPRCandidate(
                    case_id=f"{project_name}-pr-{item['number']}",
                    project=MODEL_PROJECT_ALIASES.get(project_name, project_name),
                    repository=repository,
                    pull_number=item["number"],
                    title=item["title"],
                    created_at=item["created_at"],
                    base_ref=item["base_ref"],
                    base_tip_sha=item["base_ref_oid"],
                    base_sha=item["base_sha"],
                    base_derivation="first-pr-commit-first-parent-path-parity",
                    head_sha=item["head_sha"],
                    pr_commit_shas=[item["head_sha"]],
                    changed_files=item["changed_files"],
                    additions=item["additions"],
                    deletions=item["deletions"],
                    paths=item["paths"],
                    acquisition_query=" | ".join(query_strings),
                    selection_policy_id=policy["protocol_id"],
                    outcome_fields_requested=False,
                )
            )

    if len(chosen) != int(policy["case_count"]):
        raise SystemExit("R14 allocation must contain exactly 30 cases")
    if len({item.case_id for item in chosen}) != len(chosen):
        raise SystemExit("R14 case IDs are not unique")
    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        "domain_amendment_sha256": (
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
        "selection_basis": "frozen title/path/size/time/SHA ranking only",
        "machine_policy_id": policy["machine_policy"]["policy_id"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "selection_diagnostics": selection_diagnostics,
        "cases": [item.model_dump(mode="json") for item in chosen],
    }
    payload = {"selection_material": material, "selection_lock_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps([item.case_id for item in chosen], indent=2))
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
