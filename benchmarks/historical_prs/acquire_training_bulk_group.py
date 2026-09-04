#!/usr/bin/env python3
"""Acquire outcome-free metadata for one frozen training bulk group."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      title
      createdAt
      baseRefName
      baseRefOid
      headRefOid
      changedFiles
      additions
      deletions
      author { login }
      authorAssociation
      files(first: 100) {
        totalCount
        nodes { path additions deletions changeType }
      }
      commits(first: 1) {
        nodes { commit { oid parents(first: 1) { nodes { oid } } } }
      }
      reviews(first: 100) {
        totalCount
        nodes {
          state
          submittedAt
          commit { oid }
          author { login }
          authorAssociation
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

BATCH_PULL_FIELDS = """
  number
  title
  createdAt
  baseRefName
  baseRefOid
  headRefOid
  changedFiles
  additions
  deletions
  author { login }
  authorAssociation
  files(first: 100) {
    totalCount
    nodes { path additions deletions changeType }
  }
  commits(first: 1) {
    nodes { commit { oid parents(first: 1) { nodes { oid } } } }
  }
  reviews(first: 100) {
    totalCount
    nodes {
      state
      submittedAt
      commit { oid }
      author { login }
      authorAssociation
    }
  }
"""

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("INFRASWE_GITHUB_REQUEST_TIMEOUT", "20"))
BATCH_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("INFRASWE_GITHUB_BATCH_REQUEST_TIMEOUT", "90"))
MAX_REQUEST_ATTEMPTS = int(os.environ.get("INFRASWE_GITHUB_REQUEST_ATTEMPTS", "3"))


class MetadataUnavailable(RuntimeError):
    """One frozen PR cannot satisfy the outcome-blind metadata contract."""

    def __init__(self, failure_code: str, detail: str) -> None:
        super().__init__(detail)
        self.failure_code = failure_code
        self.detail = detail[-1000:]


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _is_bot(login: str) -> bool:
    lowered = login.lower()
    return lowered.endswith("[bot]") or lowered.endswith("-bot") or lowered == "github-actions"


def _query(repository: str, number: int) -> dict[str, Any]:
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
        "-F",
        f"number={number}",
    ]
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            process = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            process = None
            if attempt + 1 < MAX_REQUEST_ATTEMPTS:
                time.sleep(min(5, 2**attempt))
            continue
        payload: dict[str, Any] | None = None
        if process.stdout:
            try:
                payload = json.loads(process.stdout)
            except json.JSONDecodeError:
                payload = None
        if payload is not None and payload.get("errors"):
            errors = json.dumps(payload.get("errors", []), sort_keys=True)
            lowered_errors = errors.lower()
            repository_payload = (payload.get("data") or {}).get("repository") or {}
            if repository_payload.get("pullRequest") is None and (
                "not_found" in lowered_errors
                or "could not resolve to a pullrequest" in lowered_errors
            ):
                raise MetadataUnavailable("PULL_REQUEST_NOT_FOUND", errors)
            if "rate limit" in lowered_errors or "rate_limit" in lowered_errors:
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
        if process.returncode == 0 and payload is not None and not payload.get("errors"):
            return payload
        if attempt + 1 < MAX_REQUEST_ATTEMPTS:
            time.sleep(min(5, 2**attempt))
    detail = (
        "request timed out" if process is None else process.stderr.strip() or process.stdout.strip()
    )
    raise MetadataUnavailable(
        "GITHUB_METADATA_TIMEOUT" if process is None else "GITHUB_METADATA_UNAVAILABLE",
        f"{repository}#{number}: {detail}",
    )


def _query_batch(repository: str, numbers: list[int]) -> dict[int, dict[str, Any]]:
    if not numbers:
        return {}
    owner, name = repository.split("/", maxsplit=1)
    aliases = "\n".join(
        f"p{index}: pullRequest(number: {number}) {{ {BATCH_PULL_FIELDS} }}"
        for index, number in enumerate(numbers)
    )
    query = (
        "query {\n"
        f"repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{\n{aliases}\n}}\n"
        "rateLimit { cost remaining resetAt }\n}"
    )
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            process = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=BATCH_REQUEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            process = None
            if attempt + 1 < MAX_REQUEST_ATTEMPTS:
                time.sleep(min(5, 2**attempt))
            continue
        payload: dict[str, Any] | None = None
        if process.stdout:
            try:
                payload = json.loads(process.stdout)
            except json.JSONDecodeError:
                payload = None
        errors = json.dumps((payload or {}).get("errors", []), sort_keys=True)
        lowered_errors = errors.lower()
        if "rate limit" in lowered_errors or "rate_limit" in lowered_errors:
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
                    f"graphql budget exhausted; waiting {delay}s before batch resume",
                    flush=True,
                )
                time.sleep(delay)
                continue
        repository_payload = ((payload or {}).get("data") or {}).get("repository")
        rate_limit = ((payload or {}).get("data") or {}).get("rateLimit")
        if (
            process.returncode == 0
            and isinstance(repository_payload, dict)
            and isinstance(rate_limit, dict)
            and not (payload or {}).get("errors")
        ):
            return {
                number: {
                    "data": {
                        "repository": {"pullRequest": repository_payload.get(f"p{index}")},
                        "rateLimit": rate_limit,
                    }
                }
                for index, number in enumerate(numbers)
            }
        if attempt + 1 < MAX_REQUEST_ATTEMPTS:
            time.sleep(min(5, 2**attempt))
    detail = (
        "request timed out" if process is None else process.stderr.strip() or process.stdout.strip()
    )
    raise MetadataUnavailable(
        "GITHUB_BATCH_TIMEOUT" if process is None else "GITHUB_BATCH_UNAVAILABLE",
        f"{repository} batch({len(numbers)}): {detail}",
    )


def _first_commit(pull: dict[str, Any]) -> tuple[str | None, str | None]:
    commits = pull["commits"]["nodes"]
    if not commits:
        return None, None
    commit = commits[0]["commit"]
    parents = commit["parents"]["nodes"]
    return commit["oid"], parents[0]["oid"] if parents else None


def _review_projection(pull: dict[str, Any]) -> dict[str, Any]:
    pr_author = str((pull.get("author") or {}).get("login") or "")
    head_sha = pull.get("headRefOid")
    reviews: list[dict[str, Any]] = []
    for review in pull["reviews"]["nodes"]:
        login = str((review.get("author") or {}).get("login") or "")
        if not login or login == pr_author or _is_bot(login):
            continue
        commit = review.get("commit") or {}
        reviews.append(
            {
                "state": review["state"],
                "submitted_at": review["submittedAt"],
                "commit_oid": commit.get("oid"),
                "author_association": review.get("authorAssociation"),
                "is_final_head": bool(head_sha and commit.get("oid") == head_sha),
                "text_stored": False,
            }
        )
    states = ("APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING")
    return {
        "review_total_count": int(pull["reviews"]["totalCount"]),
        "review_list_complete": int(pull["reviews"]["totalCount"]) == len(pull["reviews"]["nodes"]),
        "human_non_author_reviews": reviews,
        "human_non_author_review_state_counts": {
            state: sum(review["state"] == state for review in reviews) for state in states
        },
        "final_head_human_non_author_review_state_counts": {
            state: sum(review["state"] == state and review["is_final_head"] for review in reviews)
            for state in states
        },
    }


def _invalid_case(
    case: dict[str, Any], error: MetadataUnavailable
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    review_states = (
        "APPROVED",
        "CHANGES_REQUESTED",
        "COMMENTED",
        "DISMISSED",
        "PENDING",
    )
    projected = {
        **case,
        "acquisition_status": "invalid",
        "acquisition_failure_code": error.failure_code,
        "title": "[metadata unavailable]",
        "base_ref": "",
        "base_ref_oid": None,
        "head_sha": case.get("selected_ref_sha"),
        "first_pr_commit_sha": None,
        "base_sha": None,
        "changed_files": 0,
        "additions": 0,
        "deletions": 0,
        "pr_author": "",
        "pr_author_association": "NONE",
        "files": [],
        "path_list_complete": False,
        "review_total_count": 0,
        "review_list_complete": False,
        "human_non_author_reviews": [],
        "human_non_author_review_state_counts": {state: 0 for state in review_states},
        "final_head_human_non_author_review_state_counts": {state: 0 for state in review_states},
    }
    failure_projection = {
        "case_id": case["case_id"],
        "repository": case["repository"],
        "pull_number": case["pull_number"],
        "failure_code": error.failure_code,
        "detail": error.detail,
    }
    return (
        projected,
        canonical_sha256(failure_projection),
        {"cost": 0, "remaining": 0, "resetAt": None},
    )


def _project_case(
    case: dict[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    pull = payload["data"]["repository"]["pullRequest"]
    if pull is None:
        return _invalid_case(
            case,
            MetadataUnavailable("PULL_REQUEST_NOT_FOUND", case["case_id"]),
        )
    if not all(isinstance(pull.get(field), dict) for field in ("files", "commits", "reviews")):
        return _invalid_case(
            case,
            MetadataUnavailable(
                "GITHUB_METADATA_INCOMPLETE",
                f"{case['case_id']}: files/commits/reviews connection is unavailable",
            ),
        )
    if int(pull["number"]) != int(case["pull_number"]):
        raise RuntimeError(f"{case['case_id']}: pull request number changed")
    if case.get("created_at") is not None and pull["createdAt"] != case["created_at"]:
        raise RuntimeError(f"{case['case_id']}: createdAt changed")
    first_commit_sha, first_parent_sha = _first_commit(pull)
    files = pull["files"]
    projected = {
        **case,
        "acquisition_status": "acquired",
        "acquisition_failure_code": None,
        "created_at": pull["createdAt"],
        "title": pull["title"],
        "base_ref": pull["baseRefName"],
        "base_ref_oid": pull["baseRefOid"],
        "head_sha": pull["headRefOid"],
        "first_pr_commit_sha": first_commit_sha,
        "base_sha": first_parent_sha,
        "changed_files": int(pull["changedFiles"]),
        "additions": int(pull["additions"]),
        "deletions": int(pull["deletions"]),
        "pr_author": str((pull.get("author") or {}).get("login") or ""),
        "pr_author_association": pull.get("authorAssociation"),
        "files": [
            {
                "path": item["path"],
                "additions": int(item["additions"]),
                "deletions": int(item["deletions"]),
                "change_type": str(item["changeType"]).lower(),
            }
            for item in files["nodes"]
        ],
        "path_list_complete": int(files["totalCount"]) == len(files["nodes"]),
        **_review_projection(pull),
    }
    response_projection = {
        "pull": pull,
        "rate_limit": payload["data"]["rateLimit"],
    }
    return projected, canonical_sha256(response_projection), payload["data"]["rateLimit"]


def _acquire_case(case: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        payload = _query(case["repository"], case["pull_number"])
    except MetadataUnavailable as error:
        return _invalid_case(case, error)
    return _project_case(case, payload)


def _acquire_batch(
    cases: list[dict[str, Any]],
) -> list[tuple[str, tuple[dict[str, Any], str, dict[str, Any]]]]:
    if len(cases) == 1:
        case = cases[0]
        return [(case["case_id"], _acquire_case(case))]
    repository = cases[0]["repository"]
    if any(case["repository"] != repository for case in cases):
        raise ValueError("a GraphQL batch cannot span repositories")
    try:
        payloads = _query_batch(repository, [int(case["pull_number"]) for case in cases])
    except MetadataUnavailable:
        return [(case["case_id"], _acquire_case(case)) for case in cases]
    return [
        (case["case_id"], _project_case(case, payloads[int(case["pull_number"])])) for case in cases
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-lock", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("INFRASWE_GITHUB_BATCH_SIZE", "1"))
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        raise SystemExit("workers and batch size must be positive")

    queue = _read(args.queue_lock)
    material = {key: value for key, value in queue.items() if key != "queue_lock_sha256"}
    if queue["queue_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("queue lock digest mismatch")
    group_index_base = int(queue.get("group_index_base", 0))
    group_indexes = sorted({int(case["group_index"]) for case in queue["cases"]})
    if args.group_index not in group_indexes:
        raise SystemExit("group index outside queue")
    cases = [case for case in queue["cases"] if case["group_index"] == args.group_index]
    expected = (
        queue["last_group_size"]
        if args.group_index == group_indexes[-1]
        else queue["group_size"]
        if args.group_index >= group_index_base
        else len(cases)
    )
    if len(cases) != expected:
        raise SystemExit(f"group has {len(cases)} cases, expected {expected}")

    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    acquired_by_id: dict[str, tuple[dict[str, Any], str, dict[str, Any]]] = {}
    if partial_path.exists():
        checkpoint = _read(partial_path)
        if (
            checkpoint.get("queue_lock_sha256") != queue["queue_lock_sha256"]
            or int(checkpoint.get("group_index", -1)) != args.group_index
        ):
            raise SystemExit("input acquisition checkpoint binding mismatch")
        for item in checkpoint.get("acquired", []):
            item["case"].setdefault("acquisition_status", "acquired")
            item["case"].setdefault("acquisition_failure_code", None)
            acquired_by_id[item["case"]["case_id"]] = (
                item["case"],
                item["response_digest"],
                item["rate_limit"],
            )

    pending = [case for case in cases if case["case_id"] not in acquired_by_id]
    pending_by_repository: dict[str, list[dict[str, Any]]] = {}
    for case in pending:
        pending_by_repository.setdefault(case["repository"], []).append(case)
    batches = [
        repository_cases[offset : offset + args.batch_size]
        for repository_cases in pending_by_repository.values()
        for offset in range(0, len(repository_cases), args.batch_size)
    ]
    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_acquire_batch, batch) for batch in batches]
        for future in as_completed(futures):
            results = future.result()
            for case_id, result in results:
                acquired_by_id[case_id] = result
            completed_since_checkpoint += len(results)
            if completed_since_checkpoint >= 25 or any(
                result[0]["acquisition_status"] == "invalid" for _, result in results
            ):
                checkpoint = {
                    "schema_version": "0.1",
                    "protocol_id": "bulk-group-input-acquisition-checkpoint-v0.1",
                    "queue_lock_sha256": queue["queue_lock_sha256"],
                    "group_index": args.group_index,
                    "acquired": [
                        {
                            "case": value[0],
                            "response_digest": value[1],
                            "rate_limit": value[2],
                        }
                        for _, value in sorted(acquired_by_id.items())
                    ],
                }
                atomic_write_json(partial_path, checkpoint)
                print(
                    f"input checkpoint acquired={len(acquired_by_id)}/{len(cases)}",
                    flush=True,
                )
                completed_since_checkpoint = 0

    acquired = [acquired_by_id[case["case_id"]] for case in cases]
    projected_cases = [item[0] for item in acquired]
    response_digests = [item[1] for item in acquired]
    rate_limits = [item[2] for item in acquired]
    output_material = {
        "schema_version": "0.1",
        "protocol_id": f"{queue.get('profile', 'training')}-bulk-group-input-v0.1",
        "profile": queue.get("profile", "training"),
        "queue_lock_sha256": queue["queue_lock_sha256"],
        "group_index": args.group_index,
        "case_count": len(projected_cases),
        "valid_case_count": sum(
            case["acquisition_status"] == "acquired" for case in projected_cases
        ),
        "invalid_case_count": sum(
            case["acquisition_status"] == "invalid" for case in projected_cases
        ),
        "acquired_at": datetime.now(UTC).isoformat(),
        "allowed_pull_fields": [
            "number",
            "title",
            "createdAt",
            "baseRefName",
            "baseRefOid",
            "headRefOid",
            "changedFiles",
            "additions",
            "deletions",
            "author.login",
            "authorAssociation",
            "files",
            "commits.first-parent",
            "reviews.state-and-commit-without-text",
        ],
        "pull_state_or_merge_fields_requested": False,
        "review_text_requested": False,
        "ci_or_label_fields_requested": False,
        "candidate_body_requested": False,
        "diff_content_requested": False,
        "response_digests": response_digests,
        "acquisition_attempts": [
            {
                "case_id": case["case_id"],
                "status": case["acquisition_status"],
                "failure_code": case["acquisition_failure_code"],
                "response_digest": response_digest,
            }
            for case, response_digest in zip(projected_cases, response_digests, strict=True)
        ],
        "minimum_remaining_rate_limit": min(int(item["remaining"]) for item in rate_limits),
        "cases": projected_cases,
    }
    payload = {
        **output_material,
        "group_input_sha256": canonical_sha256(output_material),
    }
    atomic_write_json(args.output, payload)
    partial_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "group_index": args.group_index,
                "case_count": len(projected_cases),
                "project_counts": {
                    project: sum(case["project"] == project for case in projected_cases)
                    for project in sorted({case["project"] for case in projected_cases})
                },
                "complete_path_lists": sum(case["path_list_complete"] for case in projected_cases),
                "reviewed_cases": sum(
                    bool(case["human_non_author_reviews"]) for case in projected_cases
                ),
                "group_input_sha256": payload["group_input_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
