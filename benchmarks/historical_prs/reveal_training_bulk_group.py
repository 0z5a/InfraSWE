#!/usr/bin/env python3
"""Reveal outcomes after one training bulk judgment lock is frozen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      state
      merged
      mergedAt
      closedAt
      headRefOid
      url
    }
  }
  rateLimit { cost remaining resetAt }
}
"""

BATCH_OUTCOME_FIELDS = """
  number
  state
  merged
  mergedAt
  closedAt
  headRefOid
  url
"""

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("INFRASWE_GITHUB_REQUEST_TIMEOUT", "20"))
BATCH_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("INFRASWE_GITHUB_BATCH_REQUEST_TIMEOUT", "90"))
MAX_REQUEST_ATTEMPTS = int(os.environ.get("INFRASWE_GITHUB_REQUEST_ATTEMPTS", "3"))


class OutcomeUnavailable(RuntimeError):
    """A frozen PR can no longer provide a reveal-time oracle."""

    def __init__(self, failure_code: str, detail: str) -> None:
        super().__init__(detail)
        self.failure_code = failure_code
        self.detail = detail[-1000:]


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _checked(path: Path, digest_field: str) -> dict[str, Any]:
    payload = _read(path)
    material = {key: value for key, value in payload.items() if key != digest_field}
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path}: digest mismatch")
    return payload


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
                raise OutcomeUnavailable("PULL_REQUEST_NOT_FOUND", errors)
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
    raise OutcomeUnavailable(
        "GITHUB_OUTCOME_TIMEOUT" if process is None else "GITHUB_OUTCOME_UNAVAILABLE",
        f"{repository}#{number}: {detail}",
    )


def _query_batch(repository: str, numbers: list[int]) -> dict[int, dict[str, Any]]:
    if not numbers:
        return {}
    owner, name = repository.split("/", maxsplit=1)
    aliases = "\n".join(
        f"p{index}: pullRequest(number: {number}) {{ {BATCH_OUTCOME_FIELDS} }}"
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
    raise OutcomeUnavailable(
        "GITHUB_BATCH_TIMEOUT" if process is None else "GITHUB_BATCH_UNAVAILABLE",
        f"{repository} batch({len(numbers)}): {detail}",
    )


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _oracle(case: dict[str, Any], outcome: dict[str, Any], frozen_at: datetime) -> tuple[str, str]:
    if outcome.get("availability") != "available":
        return "unresolved", str(outcome.get("failure_code") or "GITHUB_OUTCOME_UNAVAILABLE")
    if outcome["merged"]:
        return "accept", "MERGED_PR_ACCEPT_ORACLE"
    if outcome["state"].lower() == "closed":
        return "reject", "CLOSED_UNMERGED_REJECT_ORACLE"
    age_days = (frozen_at - _time(case["created_at"])).total_seconds() / 86_400
    final_reviews = [
        review
        for review in case["human_non_author_reviews"]
        if review["is_final_head"] and _time(review["submitted_at"]) <= frozen_at
    ]
    last_review_at = max(
        (_time(review["submitted_at"]) for review in final_reviews),
        default=None,
    )
    review_idle_days = (
        (frozen_at - last_review_at).total_seconds() / 86_400
        if last_review_at is not None
        else None
    )
    active = (
        age_days <= 30
        and final_reviews
        and review_idle_days is not None
        and review_idle_days <= 14
        and outcome["current_head_sha"] == case.get("head_sha")
    )
    if active:
        return "check", "ACTIVE_NEW_FINAL_HEAD_REVIEW_CHECK_ORACLE"
    return "reject", "OPEN_NOT_ACTIVE_NEW_REVIEW_REJECT_ORACLE"


def _unavailable_outcome(
    case: dict[str, Any], failure_code: str, detail: str | None = None
) -> tuple[str, dict[str, Any], str]:
    outcome = {
        "availability": "invalid",
        "failure_code": failure_code,
        "state": "unavailable",
        "merged": False,
        "merged_at": None,
        "closed_at": None,
        "locked_head_sha": case.get("head_sha"),
        "current_head_sha": None,
        "head_matches_lock": False,
        "html_url": None,
    }
    digest_material = {**outcome, **({"detail": detail} if detail else {})}
    return case["case_id"], outcome, canonical_sha256(digest_material)


def _project_outcome(
    case: dict[str, Any], payload: dict[str, Any]
) -> tuple[str, dict[str, Any], str]:
    pull = payload["data"]["repository"]["pullRequest"]
    if pull is None:
        return _unavailable_outcome(case, "PULL_REQUEST_NOT_FOUND")
    if int(pull["number"]) != int(case["pull_number"]):
        raise RuntimeError(f"{case['case_id']}: pull request number changed")
    outcome = {
        "availability": "available",
        "failure_code": None,
        "state": str(pull["state"]).lower(),
        "merged": bool(pull["merged"]),
        "merged_at": pull["mergedAt"],
        "closed_at": pull["closedAt"],
        "locked_head_sha": case.get("head_sha"),
        "current_head_sha": pull["headRefOid"],
        "head_matches_lock": pull["headRefOid"] == case.get("head_sha"),
        "html_url": pull["url"],
    }
    return case["case_id"], outcome, canonical_sha256(payload["data"])


def _acquire(case: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    if case.get("acquisition_status", "acquired") != "acquired":
        failure_code = str(case.get("acquisition_failure_code") or "GITHUB_METADATA_UNAVAILABLE")
        return _unavailable_outcome(case, failure_code)
    try:
        payload = _query(case["repository"], case["pull_number"])
    except OutcomeUnavailable as error:
        return _unavailable_outcome(case, error.failure_code, error.detail)
    return _project_outcome(case, payload)


def _acquire_batch(cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    if len(cases) == 1:
        return [_acquire(cases[0])]
    repository = cases[0]["repository"]
    if any(case["repository"] != repository for case in cases):
        raise ValueError("a GraphQL batch cannot span repositories")
    try:
        payloads = _query_batch(repository, [int(case["pull_number"]) for case in cases])
    except OutcomeUnavailable:
        return [_acquire(case) for case in cases]
    return [_project_outcome(case, payloads[int(case["pull_number"])]) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("INFRASWE_GITHUB_BATCH_SIZE", "1"))
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        raise SystemExit("workers and batch size must be positive")

    input_lock = _checked(args.input_lock, "group_input_sha256")
    judgment = _checked(args.judgment_locks, "lock_set_sha256")
    if judgment["group_input_sha256"] != input_lock["group_input_sha256"]:
        raise SystemExit("judgment/input binding mismatch")
    locked = {item["material"]["case_id"]: item for item in judgment["locks"]}
    for case_id, lock in locked.items():
        if lock["lock_sha256"] != canonical_sha256(lock["material"]):
            raise SystemExit(f"{case_id}: judgment lock digest mismatch")

    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    acquired_by_id: dict[str, tuple[str, dict[str, Any], str]] = {}
    if partial_path.exists():
        checkpoint = _read(partial_path)
        if (
            checkpoint.get("group_input_sha256") != input_lock["group_input_sha256"]
            or checkpoint.get("judgment_lock_set_sha256") != judgment["lock_set_sha256"]
        ):
            raise SystemExit("outcome reveal checkpoint binding mismatch")
        for item in checkpoint.get("acquired", []):
            acquired_by_id[item["case_id"]] = (
                item["case_id"],
                item["outcome"],
                item["response_digest"],
            )

    pending = [case for case in input_lock["cases"] if case["case_id"] not in acquired_by_id]
    immediately_unavailable = [
        case for case in pending if case.get("acquisition_status", "acquired") != "acquired"
    ]
    for case in immediately_unavailable:
        result = _acquire(case)
        acquired_by_id[result[0]] = result
    pending_by_repository: dict[str, list[dict[str, Any]]] = {}
    for case in pending:
        if case.get("acquisition_status", "acquired") == "acquired":
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
            for result in results:
                acquired_by_id[result[0]] = result
            completed_since_checkpoint += len(results)
            if completed_since_checkpoint >= 25 or any(
                result[1]["availability"] == "invalid" for result in results
            ):
                checkpoint = {
                    "schema_version": "0.1",
                    "protocol_id": "bulk-group-outcome-reveal-checkpoint-v0.1",
                    "group_input_sha256": input_lock["group_input_sha256"],
                    "judgment_lock_set_sha256": judgment["lock_set_sha256"],
                    "acquired": [
                        {
                            "case_id": value[0],
                            "outcome": value[1],
                            "response_digest": value[2],
                        }
                        for _, value in sorted(acquired_by_id.items())
                    ],
                }
                atomic_write_json(partial_path, checkpoint)
                print(
                    f"reveal checkpoint acquired={len(acquired_by_id)}/{len(input_lock['cases'])}",
                    flush=True,
                )
                completed_since_checkpoint = 0
    acquired = [acquired_by_id[case["case_id"]] for case in input_lock["cases"]]
    by_id = {case_id: (outcome, digest) for case_id, outcome, digest in acquired}
    frozen_at = _time(judgment["frozen_at"])
    cases: list[dict[str, Any]] = []
    for case in input_lock["cases"]:
        case_id = case["case_id"]
        outcome, response_digest = by_id[case_id]
        oracle, reason = _oracle(case, outcome, frozen_at)
        lock = locked[case_id]
        cases.append(
            {
                "case_id": case_id,
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "title": case["title"],
                "machine_decision": lock["material"]["decision"],
                "legacy_decision": lock["material"]["legacy_decision"],
                "machine_rationale_codes": lock["material"]["rationale_codes"],
                "technical_contract": lock["material"]["technical_contract"],
                "judgment_lock_sha256": lock["lock_sha256"],
                "outcome": outcome,
                "outcome_sha256": canonical_sha256(outcome),
                "oracle_decision": oracle,
                "oracle_reason": reason,
                "api_response_digest": response_digest,
            }
        )
    material = {
        "schema_version": "0.1",
        "protocol_id": (f"{input_lock.get('profile', 'training')}-bulk-group-outcome-reveal-v0.1"),
        "group_index": input_lock["group_index"],
        "group_input_sha256": input_lock["group_input_sha256"],
        "judgment_lock_set_sha256": judgment["lock_set_sha256"],
        "revealed_after_lock": True,
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "valid_oracle_cases": sum(
                item["outcome"]["availability"] == "available" for item in cases
            ),
            "invalid_oracle_cases": sum(
                item["outcome"]["availability"] != "available" for item in cases
            ),
            "merged": sum(item["outcome"]["merged"] for item in cases),
            "closed_unmerged": sum(
                item["outcome"]["state"] == "closed" and not item["outcome"]["merged"]
                for item in cases
            ),
            "open": sum(item["outcome"]["state"] == "open" for item in cases),
            "oracle_counts": {
                decision: sum(item["oracle_decision"] == decision for item in cases)
                for decision in ("accept", "check", "reject", "unresolved")
            },
        },
    }
    payload = {**material, "reveal_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"reveal_sha256={payload['reveal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
