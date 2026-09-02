#!/usr/bin/env python3
"""Audit whether reviewer feedback survives the final head and explains closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalReviewFinalityEvidence


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


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _login(item: dict[str, Any], key: str = "user") -> str:
    return str((item.get(key) or {}).get("login") or "unknown")


def _is_bot(item: dict[str, Any], key: str = "user") -> bool:
    user = item.get(key) or {}
    login = str(user.get("login") or "")
    return user.get("type") == "Bot" or login.endswith("[bot]")


def _is_human_owner(item: dict[str, Any], author: str) -> bool:
    return not _is_bot(item) and (
        _login(item) == author
        or str(item.get("author_association") or "") in {"COLLABORATOR", "MEMBER", "OWNER"}
    )


def _explicit_close_language(body: str) -> bool:
    normalized = " ".join(body.lower().split())
    phrases = (
        "close this",
        "closing this",
        "closed as",
        "supersed",
        "duplicate",
        "prefer to merge",
        "only optimizes",
        "only optimises",
        "no longer needed",
        "better pr",
        "will not merge",
    )
    return any(phrase in normalized for phrase in phrases)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    review_payload = json.loads(args.reviews.read_text(encoding="utf-8"))
    plan = json.loads(args.test_plan.read_text(encoding="utf-8"))
    planned = {item["case_id"]: item for item in plan["cases"]}
    revealed = {item["case_id"]: item for item in review_payload["cases"]}
    if planned.keys() != revealed.keys():
        raise SystemExit("finality audit plan and review cases differ")

    results: list[HistoricalReviewFinalityEvidence] = []
    for case_id in sorted(planned):
        case = planned[case_id]
        revealed_case = revealed[case_id]
        prefix = f"repos/{case['repository']}"
        pull, raw_pull = _api(f"{prefix}/pulls/{case['pull_number']}")
        head, raw_head = _api(f"{prefix}/commits/{case['head_sha']}")
        reviews, raw_reviews = _api(f"{prefix}/pulls/{case['pull_number']}/reviews", paginate=True)
        inline, raw_inline = _api(f"{prefix}/pulls/{case['pull_number']}/comments", paginate=True)
        issue, raw_issue = _api(f"{prefix}/issues/{case['pull_number']}/comments", paginate=True)
        timeline, raw_timeline = _api(
            f"{prefix}/issues/{case['pull_number']}/timeline", paginate=True
        )
        if pull.get("state") != "closed" or pull.get("merged") is not False:
            raise SystemExit(f"{case_id} is not closed and unmerged")
        if pull.get("head", {}).get("sha") != case["head_sha"]:
            raise SystemExit(f"{case_id} head changed after judgment lock")

        author = _login(pull)
        head_time = _time(head["commit"]["committer"]["date"])
        closed_at = _time(pull["closed_at"])
        close_events = [item for item in timeline if item.get("event") == "closed"]
        if not close_events:
            raise SystemExit(f"{case_id} has no close event")
        close_event = min(
            close_events,
            key=lambda item: abs((_time(item["created_at"]) - closed_at).total_seconds()),
        )
        closed_by = _login(close_event, key="actor")

        final_inline = [
            f"review-comment:{item['id']}"
            for item in inline
            if not _is_bot(item)
            and _login(item) != author
            and len(str(item.get("body") or "").strip()) >= 10
            and str(item.get("original_commit_id") or item.get("commit_id") or "")
            == case["head_sha"]
        ]
        post_head: list[str] = []
        for source, items, date_key in (
            ("review", reviews, "submitted_at"),
            ("review-comment", inline, "created_at"),
            ("issue-comment", issue, "created_at"),
        ):
            for item in items:
                value = item.get(date_key)
                if (
                    value
                    and not _is_bot(item)
                    and _login(item) != author
                    and len(str(item.get("body") or "").strip()) >= 10
                    and _time(value) >= head_time
                ):
                    post_head.append(f"{source}:{item['id']}")

        changes_requested = [
            f"review:{item['id']}"
            for item in reviews
            if not _is_bot(item)
            and _login(item) != author
            and str(item.get("state") or "").upper() == "CHANGES_REQUESTED"
            and (
                item.get("commit_id") == case["head_sha"]
                or (item.get("submitted_at") and _time(item["submitted_at"]) >= head_time)
            )
        ]
        close_window_start = closed_at - timedelta(days=30)
        explicit_close = [
            f"issue-comment:{item['id']}"
            for item in issue
            if item.get("created_at")
            and close_window_start <= _time(item["created_at"]) <= closed_at
            and _is_human_owner(item, author)
            and _explicit_close_language(str(item.get("body") or ""))
        ]
        stale_close = [
            f"issue-comment:{item['id']}"
            for item in issue
            if item.get("created_at")
            and abs((_time(item["created_at"]) - closed_at).total_seconds()) <= 120
            and _is_bot(item)
            and "closed due to inactivity" in str(item.get("body") or "").lower()
        ]

        final_eligible = bool(final_inline or post_head or changes_requested)
        if changes_requested:
            classification = "final-head-changes-requested"
        elif explicit_close:
            classification = "explicit-human-technical-close"
        elif stale_close:
            classification = "stale-inactivity"
        elif closed_by == author:
            classification = "author-close-without-explicit-reason"
        else:
            classification = "other-unattributed-close"
        attributable = classification in {
            "final-head-changes-requested",
            "explicit-human-technical-close",
        }
        calibration_eligible = final_eligible and attributable
        reasons = (
            ["FINAL_HEAD_FEEDBACK_PRESENT", "TECHNICAL_CLOSURE_ATTRIBUTABLE"]
            if calibration_eligible
            else []
        )
        results.append(
            HistoricalReviewFinalityEvidence(
                case_id=case_id,
                repository=case["repository"],
                pull_number=case["pull_number"],
                judgment_lock_sha256=revealed_case["judgment_lock_sha256"],
                head_sha=case["head_sha"],
                head_committed_at=head_time,
                closed_at=closed_at,
                closed_by=closed_by,
                final_head_inline_feedback_ids=sorted(set(final_inline)),
                post_head_human_feedback_ids=sorted(set(post_head)),
                final_head_changes_requested_review_ids=sorted(set(changes_requested)),
                explicit_close_feedback_ids=sorted(set(explicit_close)),
                stale_close_feedback_ids=sorted(set(stale_close)),
                final_head_feedback_eligible=final_eligible,
                closure_reason_attributable=attributable,
                close_classification=classification,
                calibration_eligible=calibration_eligible,
                eligibility_reasons=reasons,
                api_response_digests=[
                    _digest(raw_pull),
                    _digest(raw_head),
                    _digest(raw_reviews),
                    _digest(raw_inline),
                    _digest(raw_issue),
                    _digest(raw_timeline),
                ],
            )
        )

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-review-finality-audit-v0.5-r1",
        "review_reveal_sha256": review_payload["reveal_sha256"],
        "audited_at": datetime.now(UTC).isoformat(),
        "cases": [item.model_dump(mode="json") for item in results],
        "summary": {
            "cases": len(results),
            "final_head_feedback_eligible": sum(
                item.final_head_feedback_eligible for item in results
            ),
            "closure_reason_attributable": sum(
                item.closure_reason_attributable for item in results
            ),
            "calibration_eligible": sum(item.calibration_eligible for item in results),
        },
    }
    payload = {**material, "audit_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    for item in results:
        print(f"{item.case_id}: {item.close_classification} eligible={item.calibration_eligible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
