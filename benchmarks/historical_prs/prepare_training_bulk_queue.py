#!/usr/bin/env python3
"""Freeze a deterministic, outcome-blind queue of historical training PRs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

PROFILES = {
    "training": {
        "repositories": {
            "megatron-core": "NVIDIA/Megatron-LM",
            "slime": "THUDM/slime",
            "verl": "verl-project/verl",
            "verl-omni": "verl-project/verl-omni",
        },
        "seed": "infraswe-training-bulk-v0.1-20260903",
        "protocol_id": "training-bulk-outcome-blind-queue-v0.1",
    },
    "inference": {
        "repositories": {
            "flashinfer": "flashinfer-ai/flashinfer",
            "sglang": "sgl-project/sglang",
            "tensorrt-llm": "NVIDIA/TensorRT-LLM",
            "vllm": "vllm-project/vllm",
        },
        "seed": "infraswe-inference-bulk-v0.1-20260903",
        "protocol_id": "inference-bulk-outcome-blind-queue-v0.1",
    },
    "communication": {
        "repositories": {
            "nccl": "NVIDIA/nccl",
            "rccl": "ROCm/rccl",
            "nvshmem": "NVIDIA/nvshmem",
            "uccl": "uccl-project/uccl",
            "ucx": "openucx/ucx",
            "ucc": "openucx/ucc",
            "pytorch": "pytorch/pytorch",
            "vllm": "vllm-project/vllm",
            "sglang": "sgl-project/sglang",
            "megatron-core": "NVIDIA/Megatron-LM",
        },
        "seed": "infraswe-communication-bulk-v0.1-20260904",
        "protocol_id": "communication-bulk-outcome-blind-queue-v0.1",
    },
}
QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: 100
      after: $cursor
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { number createdAt }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""


def _run_graphql(repository: str, cursor: str | None) -> dict[str, Any]:
    owner, name = repository.split("/", maxsplit=1)
    command = [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={QUERY}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={name}",
    ]
    if cursor is not None:
        command.extend(["-f", f"cursor={cursor}"])
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(8):
        try:
            process = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            process = None
        if process is not None and process.returncode == 0:
            payload = json.loads(process.stdout)
            if not payload.get("errors"):
                return payload
            errors = json.dumps(payload.get("errors", [])).lower()
            if "rate limit" in errors or "rate_limit" in errors:
                probe = subprocess.run(
                    ["gh", "api", "rate_limit"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if probe.returncode == 0:
                    resource = json.loads(probe.stdout)["resources"]["graphql"]
                    delay = max(5, int(resource["reset"]) - int(time.time()) + 5)
                    print(
                        f"graphql budget exhausted; waiting {delay}s before resume",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
        if attempt < 7:
            time.sleep(min(30, 2**attempt))
    detail = "timed out" if process is None else (process.stderr.strip() or process.stdout.strip())
    raise RuntimeError(f"{repository}: GraphQL failed after retries: {detail}")


def _acquire_repository(
    item: tuple[str, str],
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    project, repository = item
    cursor: str | None = None
    identities: list[dict[str, Any]] = []
    page_digests: list[str] = []
    expected_total: int | None = None
    observed_totals: list[int] = []
    final_rate_limit: dict[str, Any] = {}
    while True:
        payload = _run_graphql(repository, cursor)
        connection = payload["data"]["repository"]["pullRequests"]
        if expected_total is None:
            expected_total = int(connection["totalCount"])
        observed_totals.append(int(connection["totalCount"]))
        nodes = connection["nodes"]
        identities.extend(
            {
                "number": int(node["number"]),
                "created_at": node["createdAt"],
            }
            for node in nodes
        )
        page_digests.append(canonical_sha256(nodes))
        if len(page_digests) % 20 == 0:
            print(
                f"repository={repository} pages={len(page_digests)} "
                f"identities={len(identities)} observed_total={observed_totals[-1]}",
                flush=True,
            )
        final_rate_limit = payload["data"]["rateLimit"]
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
    if len({item["number"] for item in identities}) != len(identities):
        raise RuntimeError(f"{repository}: duplicate pull request number")
    acquisition = {
        "queried_count": len(identities),
        "initial_total_count": expected_total,
        "observed_total_count_min": min(observed_totals),
        "observed_total_count_max": max(observed_totals),
        "snapshot_count": len(identities),
        "page_count": len(page_digests),
        "page_digests": page_digests,
        "final_rate_limit": final_rate_limit,
    }
    return project, repository, identities, acquisition


def _acquire_repository_git_refs(
    item: tuple[str, str],
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    project, repository = item
    command = [
        "git",
        "ls-remote",
        f"https://github.com/{repository}.git",
        "refs/pull/*/head",
    ]
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    identities: list[dict[str, Any]] = []
    for line in process.stdout.splitlines():
        sha, reference = line.split("\t", maxsplit=1)
        parts = reference.split("/")
        if len(parts) != 4 or parts[:2] != ["refs", "pull"] or parts[3] != "head":
            continue
        identities.append(
            {
                "number": int(parts[2]),
                "created_at": None,
                "selected_ref_sha": sha,
            }
        )
    identities.sort(key=lambda value: value["number"], reverse=True)
    if not identities:
        raise RuntimeError(f"{repository}: no pull refs acquired")
    if len({item["number"] for item in identities}) != len(identities):
        raise RuntimeError(f"{repository}: duplicate pull ref number")
    print(
        f"repository={repository} git_pull_refs={len(identities)}",
        flush=True,
    )
    acquisition = {
        "identity_source": "git-pull-head-refs",
        "queried_count": len(identities),
        "snapshot_count": len(identities),
        "response_sha256": "sha256:" + hashlib.sha256(process.stdout.encode()).hexdigest(),
    }
    return project, repository, identities, acquisition


def _find_identities(value: Any) -> set[tuple[str, int]]:
    found: set[tuple[str, int]] = set()
    if isinstance(value, dict):
        repository = value.get("repository")
        number = value.get("pull_number", value.get("number"))
        if isinstance(repository, str) and isinstance(number, int):
            found.add((repository.lower(), number))
        for child in value.values():
            found.update(_find_identities(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_identities(child))
    return found


def _prior_identities(paths: list[Path]) -> tuple[set[tuple[str, int]], list[str]]:
    identities: set[tuple[str, int]] = set()
    digests: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identities.update(_find_identities(payload))
        digests.append(canonical_sha256(payload))
    return identities, digests


def _rank(seed: str, repository: str, number: int, purpose: str) -> str:
    value = f"{seed}\0{purpose}\0{repository.lower()}\0{number}".encode()
    return hashlib.sha256(value).hexdigest()


def _largest_remainder_quotas(counts: dict[str, int], target_count: int) -> dict[str, int]:
    total = sum(counts.values())
    if target_count > total:
        raise ValueError(f"target {target_count} exceeds available count {total}")
    exact = {project: target_count * count / total for project, count in counts.items()}
    quotas = {project: int(value) for project, value in exact.items()}
    remaining = target_count - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda project: (exact[project] - quotas[project], project),
        reverse=True,
    )
    for project in order[:remaining]:
        quotas[project] += 1
    return quotas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="training")
    parser.add_argument("--identity-source", choices=("graphql", "git-refs"), default="graphql")
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--target-fraction", type=float)
    parser.add_argument("--group-size", type=int, default=30)
    parser.add_argument("--prior-lock", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_count is not None and args.target_fraction is not None:
        raise SystemExit("choose either --target-count or --target-fraction")
    if args.target_count is not None and args.target_count <= 0:
        raise SystemExit("target count must be positive")
    if args.target_fraction is not None and not 0 < args.target_fraction <= 1:
        raise SystemExit("target fraction must be in (0, 1]")
    if args.group_size <= 0:
        raise SystemExit("group size must be positive")

    profile = PROFILES[args.profile]
    repositories_config = profile["repositories"]
    seed = str(profile["seed"])

    prior, prior_digests = _prior_identities(args.prior_lock)
    acquire = (
        _acquire_repository_git_refs if args.identity_source == "git-refs" else _acquire_repository
    )
    with ThreadPoolExecutor(max_workers=len(repositories_config)) as executor:
        acquired = list(executor.map(acquire, repositories_config.items()))

    pools: dict[str, list[dict[str, Any]]] = {}
    acquisitions: dict[str, Any] = {}
    repositories: dict[str, str] = {}
    for project, repository, identities, acquisition in acquired:
        repositories[project] = repository
        pools[project] = [
            item for item in identities if (repository.lower(), item["number"]) not in prior
        ]
        acquisitions[project] = {
            **acquisition,
            "repository": repository,
            "prior_scored_excluded_count": len(identities) - len(pools[project]),
            "candidate_pool_count": len(pools[project]),
        }

    available_count = sum(len(pool) for pool in pools.values())
    if args.target_fraction is not None:
        target_count = int(available_count * args.target_fraction)
    elif args.target_count is not None:
        target_count = args.target_count
    else:
        target_count = 8000
    quotas = _largest_remainder_quotas(
        {project: len(pool) for project, pool in pools.items()}, target_count
    )
    selected: list[dict[str, Any]] = []
    for project, repository in repositories.items():
        ranked = sorted(
            pools[project],
            key=lambda item: _rank(seed, repository, item["number"], "sample"),
        )
        if quotas[project] > len(ranked):
            raise SystemExit(f"{project}: quota exceeds candidate pool")
        for item in ranked[: quotas[project]]:
            selected.append(
                {
                    "case_id": f"{project}-pr-{item['number']}",
                    "project": project,
                    "repository": repository,
                    "pull_number": item["number"],
                    "created_at": item["created_at"],
                    **(
                        {"selected_ref_sha": item["selected_ref_sha"]}
                        if item.get("selected_ref_sha")
                        else {}
                    ),
                }
            )
    selected.sort(key=lambda item: _rank(seed, item["repository"], item["pull_number"], "order"))
    for index, item in enumerate(selected):
        item["queue_index"] = index
        item["group_index"] = index // args.group_size
        item["group_offset"] = index % args.group_size

    group_count = (len(selected) + args.group_size - 1) // args.group_size
    material = {
        "schema_version": "0.1",
        "protocol_id": profile["protocol_id"],
        "profile": args.profile,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "identity_source": args.identity_source,
        "target_count": target_count,
        "target_fraction": args.target_fraction,
        "available_count_after_prior_exclusions": available_count,
        "group_size": args.group_size,
        "group_count": group_count,
        "last_group_size": len(selected) - (group_count - 1) * args.group_size,
        "allowed_graphql_fields": (
            ["number", "createdAt"] if args.identity_source == "graphql" else []
        ),
        "allowed_git_ref_fields": (
            ["pull_number", "head_sha"] if args.identity_source == "git-refs" else []
        ),
        "outcome_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "diff_or_body_fields_requested": False,
        "prior_lock_sha256s": prior_digests,
        "prior_identity_count": len(prior),
        "repositories": repositories,
        "acquisitions": acquisitions,
        "project_quotas": quotas,
        "cases": selected,
    }
    payload = {**material, "queue_lock_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "target_count": len(selected),
                "group_count": group_count,
                "last_group_size": material["last_group_size"],
                "project_quotas": quotas,
                "queue_lock_sha256": payload["queue_lock_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
