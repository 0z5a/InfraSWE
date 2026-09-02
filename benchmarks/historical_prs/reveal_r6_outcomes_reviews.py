#!/usr/bin/env python3
"""Reveal R6 pull outcomes and reviewer text after machine judgments are locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import audit_explainable_judgment_lock
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalExplainableJudgmentLock


def _api(endpoint: str, *, paginate: bool = False) -> tuple[Any, str]:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    command.append(endpoint)
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(3):
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(f"GitHub API failed for {endpoint}: {process.stderr.strip()}")
    payload = json.loads(process.stdout)
    if paginate:
        payload = [item for page in payload for item in page]
    return payload, process.stdout


def _digest(raw: str) -> str:
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _is_bot(item: dict[str, Any]) -> bool:
    user = item.get("user") or {}
    login = str(user.get("login") or "")
    return user.get("type") == "Bot" or login.endswith("[bot]")


def _feedback(source: str, item: dict[str, Any]) -> dict[str, Any]:
    user = item.get("user") or {}
    return {
        "feedback_id": f"{source}:{item['id']}",
        "source": source,
        "author": str(user.get("login") or "unknown"),
        "author_association": str(item.get("author_association") or "UNKNOWN"),
        "is_bot": _is_bot(item),
        "review_state": item.get("state"),
        "commit_id": item.get("commit_id") or item.get("original_commit_id"),
        "path": item.get("path"),
        "line": item.get("line") or item.get("original_line"),
        "body": str(item.get("body") or "").strip(),
        "html_url": str(item.get("html_url") or ""),
        "created_at": item.get("submitted_at") or item.get("created_at"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock_payload = json.loads(args.judgment_locks.read_text(encoding="utf-8"))
    lock_set_digest = lock_payload.pop("lock_set_sha256")
    if canonical_sha256(lock_payload) != lock_set_digest:
        raise SystemExit("R6 lock-set digest mismatch")
    if lock_payload["review_text_visible_during_machine_judgment"] is not False:
        raise SystemExit("R6 machine lock does not assert hidden review text")
    if lock_payload["merge_outcomes_visible_during_machine_judgment"] is not False:
        raise SystemExit("R6 machine lock does not assert hidden merge outcomes")
    locks = {
        item.material.case_id: item
        for item in map(
            HistoricalExplainableJudgmentLock.model_validate,
            lock_payload["locks"],
        )
    }
    if not all(audit_explainable_judgment_lock(item) for item in locks.values()):
        raise SystemExit("R6 judgment-lock audit failed")

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    plan = json.loads(args.test_plan.read_text(encoding="utf-8"))
    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    planned = {item["case_id"]: item for item in plan["cases"]}
    if locks.keys() != selected.keys() or locks.keys() != planned.keys():
        raise SystemExit("R6 selection, plan, and judgment-lock case sets differ")

    cases: list[dict[str, Any]] = []
    for case_id in sorted(locks):
        case = selected[case_id]
        planned_case = planned[case_id]
        lock = locks[case_id]
        prefix = f"repos/{case['repository']}"
        pull, raw_pull = _api(f"{prefix}/pulls/{case['pull_number']}")
        reviews, raw_reviews = _api(f"{prefix}/pulls/{case['pull_number']}/reviews", paginate=True)
        inline, raw_inline = _api(f"{prefix}/pulls/{case['pull_number']}/comments", paginate=True)
        issue, raw_issue = _api(f"{prefix}/issues/{case['pull_number']}/comments", paginate=True)
        observed_at = datetime.now(UTC)
        if observed_at < lock.material.frozen_at:
            raise SystemExit(f"{case_id} reveal predates the machine lock")
        if planned_case["head_sha"] != case["head_sha"]:
            raise SystemExit(f"{case_id} selected and planned head SHAs differ")

        author = str((pull.get("user") or {}).get("login") or "unknown")
        feedback = [
            *(_feedback("review", item) for item in reviews if item.get("body")),
            *(_feedback("review-comment", item) for item in inline if item.get("body")),
            *(_feedback("issue-comment", item) for item in issue if item.get("body")),
        ]
        human_non_author_reviews = [
            item
            for item in reviews
            if not _is_bot(item)
            and str((item.get("user") or {}).get("login") or "unknown") != author
        ]
        explicit_human_non_author = [
            item
            for item in feedback
            if not item["is_bot"]
            and item["author"] != author
            and len(item["body"]) >= 10
            and (
                item["source"] in {"review", "review-comment"}
                or item["author_association"] in {"COLLABORATOR", "MEMBER", "OWNER"}
            )
        ]
        outcome = {
            "state": pull["state"],
            "merged": bool(pull.get("merged")),
            "merged_at": pull.get("merged_at"),
            "closed_at": pull.get("closed_at"),
            "merge_commit_sha": pull.get("merge_commit_sha"),
            "html_url": pull["html_url"],
            "locked_head_sha": case["head_sha"],
            "current_head_sha": (pull.get("head") or {}).get("sha"),
            "head_matches_lock": (pull.get("head") or {}).get("sha") == case["head_sha"],
        }
        cases.append(
            {
                "case_id": case_id,
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "title": case["title"],
                "machine_decision": lock.material.decision,
                "machine_rationale_codes": lock.material.rationale_codes,
                "judgment_lock_sha256": lock.lock_sha256,
                "observed_at": observed_at.isoformat(),
                "pr_author": author,
                "outcome": outcome,
                "outcome_sha256": canonical_sha256(outcome),
                "feedback": feedback,
                "human_non_author_review_count": len(human_non_author_reviews),
                "explicit_human_non_author_feedback_count": len(explicit_human_non_author),
                "strict_feedback_audit_eligible": bool(
                    human_non_author_reviews and explicit_human_non_author
                ),
                "api_response_digests": [
                    _digest(raw_pull),
                    _digest(raw_reviews),
                    _digest(raw_inline),
                    _digest(raw_issue),
                ],
            }
        )

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6-reveal",
        "judgment_lock_set_sha256": lock_set_digest,
        "revealed_after_lock": True,
        "revealed_at": datetime.now(UTC).isoformat(),
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "merged": sum(item["outcome"]["merged"] for item in cases),
            "closed_unmerged": sum(
                item["outcome"]["state"] == "closed" and not item["outcome"]["merged"]
                for item in cases
            ),
            "open": sum(item["outcome"]["state"] == "open" for item in cases),
            "strict_feedback_audit_eligible": sum(
                item["strict_feedback_audit_eligible"] for item in cases
            ),
        },
    }
    payload = {**material, "reveal_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    for item in cases:
        print(
            f"{item['case_id']}: decision={item['machine_decision']} "
            f"state={item['outcome']['state']} merged={item['outcome']['merged']} "
            f"human_feedback={item['explicit_human_non_author_feedback_count']}"
        )
    print(f"reveal_sha256={payload['reveal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
