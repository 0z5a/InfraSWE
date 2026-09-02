#!/usr/bin/env python3
"""Reveal strict human review evidence after supplemental machine locks exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import audit_explainable_judgment_lock
from infraswe.io import atomic_write_json
from infraswe.models.history import (
    HistoricalExplainableJudgmentLock,
    HistoricalJudgmentReviewEvidence,
    HistoricalReviewFeedbackItem,
)


def _api(endpoint: str, *, paginate: bool = False) -> tuple[Any, str]:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    command.append(endpoint)
    process = subprocess.run(command, check=True, text=True, capture_output=True)
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


def _feedback(source: str, item: dict[str, Any]) -> HistoricalReviewFeedbackItem:
    user = item.get("user") or {}
    return HistoricalReviewFeedbackItem(
        feedback_id=f"{source}:{item['id']}",
        source=source,
        author=str(user.get("login") or "unknown"),
        author_association=str(item.get("author_association") or "UNKNOWN"),
        is_bot=_is_bot(item),
        review_state=item.get("state"),
        path=item.get("path"),
        line=item.get("line") or item.get("original_line"),
        body=str(item.get("body") or "").strip(),
        html_url=str(item.get("html_url") or ""),
        created_at=item.get("submitted_at") or item.get("created_at"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument(
        "--protocol-id",
        choices=(
            "historical-reviewed-failure-alignment-v0.5-r2",
            "historical-reviewed-failure-alignment-v0.5-r3",
            "historical-reviewed-failure-alignment-v0.5-r4",
        ),
        default="historical-reviewed-failure-alignment-v0.5-r2",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock_payload = json.loads(args.judgment_locks.read_text(encoding="utf-8"))
    lock_set_digest = lock_payload.pop("lock_set_sha256")
    if canonical_sha256(lock_payload) != lock_set_digest:
        raise SystemExit("supplemental lock-set digest mismatch")
    if lock_payload["review_text_visible_during_machine_judgment"] is not False:
        raise SystemExit("machine lock does not assert hidden reviewer text")
    locks = {
        item.material.case_id: item
        for item in map(
            HistoricalExplainableJudgmentLock.model_validate,
            lock_payload["locks"],
        )
    }
    if not all(audit_explainable_judgment_lock(item) for item in locks.values()):
        raise SystemExit("supplemental judgment lock audit failed")
    plan = json.loads(args.test_plan.read_text(encoding="utf-8"))
    cases = {item["case_id"]: item for item in plan["cases"]}
    if locks.keys() != cases.keys():
        raise SystemExit("test-plan cases do not match judgment locks")

    revealed: list[HistoricalJudgmentReviewEvidence] = []
    for case_id in sorted(locks):
        case = cases[case_id]
        lock = locks[case_id]
        prefix = f"repos/{case['repository']}"
        pull, raw_pull = _api(f"{prefix}/pulls/{case['pull_number']}")
        reviews, raw_reviews = _api(f"{prefix}/pulls/{case['pull_number']}/reviews", paginate=True)
        inline, raw_inline = _api(f"{prefix}/pulls/{case['pull_number']}/comments", paginate=True)
        issue, raw_issue = _api(f"{prefix}/issues/{case['pull_number']}/comments", paginate=True)
        if pull.get("state") != "closed" or pull.get("merged") is not False:
            raise SystemExit(f"{case_id} is not currently closed and unmerged")
        if pull.get("head", {}).get("sha") != case["head_sha"]:
            raise SystemExit(f"{case_id} head SHA changed from the locked test plan")
        observed_at = datetime.now(UTC)
        if observed_at < lock.material.frozen_at:
            raise SystemExit(f"{case_id} review reveal predates machine lock")
        author = str((pull.get("user") or {}).get("login") or "unknown")
        feedback = [
            *(_feedback("review", item) for item in reviews if item.get("body")),
            *(_feedback("review-comment", item) for item in inline if item.get("body")),
            *(_feedback("issue-comment", item) for item in issue if item.get("body")),
        ]
        human_reviews = [
            item
            for item in reviews
            if not _is_bot(item)
            and str((item.get("user") or {}).get("login") or "unknown") != author
        ]
        explicit = []
        reasons: set[str] = set()
        for item in feedback:
            if item.is_bot or item.author == author or len(item.body) < 10:
                continue
            if item.source == "review-comment" and item.path:
                explicit.append(item)
                reasons.add("HUMAN_NON_AUTHOR_INLINE_TECHNICAL_FEEDBACK")
            elif item.source == "review":
                explicit.append(item)
                reasons.add("HUMAN_NON_AUTHOR_REVIEW_BODY")
            elif item.source == "issue-comment" and item.author_association in {
                "COLLABORATOR",
                "MEMBER",
                "OWNER",
            }:
                explicit.append(item)
                reasons.add("HUMAN_NON_AUTHOR_MAINTAINER_FEEDBACK")
        eligible = bool(human_reviews and explicit)
        result = HistoricalJudgmentReviewEvidence(
            case_id=case_id,
            repository=case["repository"],
            pull_number=case["pull_number"],
            judgment_lock_sha256=lock.lock_sha256,
            observed_at=observed_at,
            pr_author=author,
            feedback=feedback,
            human_non_author_review_count=len(human_reviews),
            explicit_human_non_author_feedback_count=len(explicit),
            strict_feedback_audit_eligible=eligible,
            eligibility_reasons=sorted(reasons) if eligible else [],
            api_response_digests=[
                _digest(raw_pull),
                _digest(raw_reviews),
                _digest(raw_inline),
                _digest(raw_issue),
            ],
        )
        if not result.strict_feedback_audit_eligible:
            raise SystemExit(f"{case_id} failed strict reviewed-feedback eligibility")
        revealed.append(result)

    material = {
        "schema_version": "0.5",
        "protocol_id": args.protocol_id,
        "judgment_lock_set_sha256": lock_set_digest,
        "revealed_after_lock": True,
        "cases": [item.model_dump(mode="json") for item in revealed],
    }
    payload = {**material, "reveal_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"strict_eligible={len(revealed)}/{len(locks)} reveal_sha256={payload['reveal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
