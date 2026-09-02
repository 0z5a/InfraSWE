#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import audit_prediction_lock, canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import (
    HistoricalGroundTruth,
    HistoricalPredictionLock,
    HistoricalReviewEvidence,
    HistoricalReviewFeedbackItem,
)


def api_sha256(payload: str) -> str:
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fetch_pages(endpoint: str) -> tuple[list[dict[str, Any]], str]:
    process = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        check=True,
        text=True,
        capture_output=True,
    )
    pages = json.loads(process.stdout)
    return [item for page in pages for item in page], process.stdout


def is_bot(item: dict[str, Any]) -> bool:
    user = item.get("user") or {}
    login = str(user.get("login") or "")
    return user.get("type") == "Bot" or login.endswith("[bot]")


def feedback_item(source: str, item: dict[str, Any]) -> HistoricalReviewFeedbackItem:
    user = item.get("user") or {}
    created_at = item.get("submitted_at") or item.get("created_at")
    return HistoricalReviewFeedbackItem(
        feedback_id=f"{source}:{item['id']}",
        source=source,
        author=str(user.get("login") or "unknown"),
        author_association=str(item.get("author_association") or "UNKNOWN"),
        is_bot=is_bot(item),
        review_state=item.get("state"),
        path=item.get("path"),
        line=item.get("line") or item.get("original_line"),
        body=str(item.get("body") or "").strip(),
        html_url=str(item.get("html_url") or ""),
        created_at=created_at,
    )


def eligibility_reasons(items: list[HistoricalReviewFeedbackItem]) -> list[str]:
    reasons: set[str] = set()
    for item in items:
        if item.is_bot or len(item.body) < 10:
            continue
        if item.source == "review-comment" and item.path:
            reasons.add("HUMAN_INLINE_TECHNICAL_REVIEW")
        elif item.source == "review" and item.review_state == "CHANGES_REQUESTED":
            reasons.add("HUMAN_CHANGES_REQUESTED_WITH_BODY")
        elif item.source == "issue-comment" and item.author_association in {
            "COLLABORATOR",
            "MEMBER",
            "OWNER",
        }:
            reasons.add("HUMAN_MAINTAINER_PR_FEEDBACK")
    return sorted(reasons)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-locks", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    locks = {
        item.material.case_id: item
        for item in map(
            HistoricalPredictionLock.model_validate,
            json.loads(options.prediction_locks.read_text(encoding="utf-8")),
        )
    }
    truths = {
        item.case_id: item
        for item in map(
            HistoricalGroundTruth.model_validate,
            json.loads(options.ground_truth.read_text(encoding="utf-8")),
        )
    }
    if locks.keys() != truths.keys():
        raise SystemExit("review reveal requires the complete locked ground-truth set")
    if not all(audit_prediction_lock(lock) for lock in locks.values()):
        raise SystemExit("review reveal refused: prediction lock audit failed")

    results: list[HistoricalReviewEvidence] = []
    for case_id in sorted(truths):
        truth = truths[case_id]
        lock = locks[case_id]
        if truth.prediction_lock_sha256 != lock.lock_sha256:
            raise SystemExit(f"review reveal refused: lock mismatch for {case_id}")
        if truth.observed_at < lock.material.frozen_at:
            raise SystemExit(f"review reveal refused: early ground truth for {case_id}")
        if truth.state != "closed" or truth.merged:
            continue

        prefix = f"repos/{truth.repository}"
        reviews, raw_reviews = fetch_pages(f"{prefix}/pulls/{truth.pull_number}/reviews")
        review_comments, raw_review_comments = fetch_pages(
            f"{prefix}/pulls/{truth.pull_number}/comments"
        )
        issue_comments, raw_issue_comments = fetch_pages(
            f"{prefix}/issues/{truth.pull_number}/comments"
        )
        items = [
            *(feedback_item("review", item) for item in reviews if item.get("body")),
            *(
                feedback_item("review-comment", item)
                for item in review_comments
                if item.get("body")
            ),
            *(feedback_item("issue-comment", item) for item in issue_comments if item.get("body")),
        ]
        reasons = eligibility_reasons(items)
        results.append(
            HistoricalReviewEvidence(
                case_id=case_id,
                repository=truth.repository,
                pull_number=truth.pull_number,
                prediction_lock_sha256=lock.lock_sha256,
                ground_truth_sha256=canonical_sha256(truth),
                observed_at=datetime.now(UTC),
                feedback=items,
                human_feedback_count=sum(not item.is_bot for item in items),
                machine_eligible_for_feedback_audit=bool(reasons),
                eligibility_reasons=reasons,
                api_response_digests=[
                    api_sha256(raw_reviews),
                    api_sha256(raw_review_comments),
                    api_sha256(raw_issue_comments),
                ],
            )
        )

    atomic_write_json(
        options.output,
        [result.model_dump(mode="json") for result in results],
    )
    eligible = sum(result.machine_eligible_for_feedback_audit for result in results)
    print(f"closed_unmerged={len(results)} feedback_audit_eligible={eligible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
