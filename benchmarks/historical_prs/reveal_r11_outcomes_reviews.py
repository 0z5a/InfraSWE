#!/usr/bin/env python3
"""Reveal outcomes and review text after custom repairability locks are frozen."""

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
from infraswe.io import atomic_write_json


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


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _digest(raw: str) -> str:
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_bot(item: dict[str, Any]) -> bool:
    user = item.get("user") or {}
    login = str(user.get("login") or "").lower()
    return (
        user.get("type") == "Bot"
        or login.endswith("[bot]")
        or login.endswith("-bot")
        or login.endswith("_bot")
    )


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


def _is_substantive_feedback(item: dict[str, Any]) -> bool:
    body = item["body"].strip()
    return len(body) >= 10 and not body.startswith("/")


def _validate_lock_payload(
    payload: dict[str, Any], *, round_label: str
) -> dict[str, dict[str, Any]]:
    material = {key: value for key, value in payload.items() if key != "lock_set_sha256"}
    if payload["lock_set_sha256"] != canonical_sha256(material):
        raise SystemExit(f"{round_label} machine lock-set digest mismatch")
    hidden = (
        payload["review_text_visible_during_machine_judgment"],
        payload["merge_outcomes_visible_during_machine_judgment"],
        payload["ci_fields_visible_during_machine_judgment"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit(f"{round_label} machine lock does not preserve the blind boundary")
    if payload["weighted_score_used"] is not False:
        raise SystemExit(f"{round_label} unexpectedly enables weighted scoring")
    if payload["forced_polarization_used"] is not False:
        raise SystemExit(f"{round_label} unexpectedly enables forced polarization")
    locks: dict[str, dict[str, Any]] = {}
    for lock in payload["locks"]:
        lock_material = lock["material"]
        if lock["lock_sha256"] != canonical_sha256(lock_material):
            raise SystemExit(
                f"{round_label} judgment digest mismatch for {lock_material['case_id']}"
            )
        case_id = lock_material["case_id"]
        if case_id in locks:
            raise SystemExit(f"{round_label} duplicate judgment for {case_id}")
        locks[case_id] = lock
    return locks


def main(*, round_label: str = "R11") -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock_payload = _read(args.judgment_locks)
    locks = _validate_lock_payload(lock_payload, round_label=round_label)
    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    if lock_payload["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit(f"{round_label} machine lock is not bound to selection")
    if lock_payload["test_plan_sha256"] != plan["test_plan_sha256"]:
        raise SystemExit(f"{round_label} machine lock is not bound to test plan")
    selected = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    planned = {item["case_id"]: item for item in plan["cases"]}
    expected_case_ids = set(lock_payload.get("selected_case_ids", selected))
    if (
        locks.keys() != expected_case_ids
        or not expected_case_ids <= selected.keys()
        or not expected_case_ids <= planned.keys()
    ):
        raise SystemExit(f"{round_label} selection, plan, and judgment case sets differ")

    cases: list[dict[str, Any]] = []
    for case_id in sorted(locks):
        case = selected[case_id]
        planned_case = planned[case_id]
        lock = locks[case_id]
        lock_material = lock["material"]
        prefix = f"repos/{case['repository']}"
        pull, raw_pull = _api(f"{prefix}/pulls/{case['pull_number']}")
        head, raw_head = _api(f"{prefix}/commits/{case['head_sha']}")
        reviews, raw_reviews = _api(f"{prefix}/pulls/{case['pull_number']}/reviews", paginate=True)
        inline, raw_inline = _api(f"{prefix}/pulls/{case['pull_number']}/comments", paginate=True)
        issue, raw_issue = _api(f"{prefix}/issues/{case['pull_number']}/comments", paginate=True)
        observed_at = datetime.now(UTC)
        if observed_at < _time(lock_material["frozen_at"]):
            raise SystemExit(f"{case_id}: reveal predates frozen judgment")
        if planned_case["head_sha"] != case["head_sha"]:
            raise SystemExit(f"{case_id}: selection and plan head SHAs differ")

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
            and _is_substantive_feedback(item)
            and (
                item["source"] in {"review", "review-comment"}
                or item["author_association"] in {"COLLABORATOR", "MEMBER", "OWNER"}
            )
        ]
        head_committed_at = _time(head["commit"]["committer"]["date"])
        final_head_explicit = [
            item
            for item in explicit_human_non_author
            if item["commit_id"] == case["head_sha"]
            or (item["created_at"] is not None and _time(item["created_at"]) >= head_committed_at)
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
                "machine_decision": lock_material["decision"],
                "legacy_r10_style_decision": lock_material["legacy_r10_style_decision"],
                # R18 locks the full technical findings and residual contract but
                # predates this presentational field.  Derive only a display code
                # from the already-locked decision so the original lock and its
                # digest remain unchanged during reveal.
                "machine_rationale_codes": lock_material.get(
                    "rationale_codes",
                    [f"PRELOCKED_{lock_material['decision'].upper()}_DECISION"],
                ),
                "judgment_lock_sha256": lock["lock_sha256"],
                "observed_at": observed_at.isoformat(),
                "head_committed_at": head_committed_at.isoformat(),
                "pr_author": author,
                "outcome": outcome,
                "outcome_sha256": canonical_sha256(outcome),
                "feedback": feedback,
                "human_non_author_review_count": len(human_non_author_reviews),
                "explicit_human_non_author_feedback_count": len(explicit_human_non_author),
                "final_head_explicit_human_feedback_count": len(final_head_explicit),
                "strict_feedback_audit_eligible": bool(
                    human_non_author_reviews and final_head_explicit
                ),
                "api_response_digests": [
                    _digest(raw_pull),
                    _digest(raw_head),
                    _digest(raw_reviews),
                    _digest(raw_inline),
                    _digest(raw_issue),
                ],
            }
        )

    material = {
        "schema_version": lock_payload["schema_version"],
        "protocol_id": f"{lock_payload['protocol_id']}-reveal",
        "judgment_lock_set_sha256": lock_payload["lock_set_sha256"],
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
            f"legacy={item['legacy_r10_style_decision']} "
            f"state={item['outcome']['state']} merged={item['outcome']['merged']} "
            f"human_feedback={item['explicit_human_non_author_feedback_count']}"
        )
    print(f"reveal_sha256={payload['reveal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
