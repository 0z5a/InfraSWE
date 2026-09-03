#!/usr/bin/env python3
"""Project recent review activity without exposing text or PR outcomes to judgment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

PULL_IDENTITY_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
    repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      author { login __typename }
      authorAssociation
      headRefOid
      commits(last: 1) {
        nodes { commit { oid committedDate } }
      }
    }
  }
}
"""


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def checked(path: Path, digest_field: str, *, material_field: str | None = None) -> dict[str, Any]:
    payload = read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    if payload[digest_field] != canonical_sha256(material):
        raise SystemExit(f"{path.name} digest mismatch")
    return payload


def api(arguments: list[str], *, paginate: bool = False) -> Any:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    command.extend(arguments)
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode == 0:
            break
        if attempt < 3:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip())
    payload = json.loads(process.stdout)
    if paginate:
        return [item for page in payload for item in page]
    return payload


def is_bot(item: dict[str, Any]) -> bool:
    user = item.get("user") or item.get("author") or {}
    login = str(user.get("login") or "").lower()
    return (
        user.get("type") == "Bot"
        or user.get("__typename") == "Bot"
        or login.endswith("[bot]")
        or login.endswith("-bot")
        or login.endswith("_bot")
    )


def projected_event(source: str, item: dict[str, Any]) -> dict[str, Any]:
    user = item.get("user") or {}
    body = str(item.get("body") or "").strip()
    return {
        "event_id": f"{source}:{item['id']}",
        "source": source,
        "author": str(user.get("login") or "unknown"),
        "author_association": str(item.get("author_association") or "UNKNOWN"),
        "is_bot": is_bot(item),
        "commit_id": item.get("commit_id") or item.get("original_commit_id"),
        "created_at": item.get("submitted_at") or item.get("created_at"),
        "body_byte_count": len(body.encode("utf-8")),
        "is_slash_command": body.startswith("/"),
        "is_substantive": len(body) >= 10 and not body.startswith("/"),
        "text_stored": False,
    }


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = checked(
        args.selection_lock,
        "selection_lock_sha256",
        material_field="selection_material",
    )
    plan = checked(args.test_plan, "test_plan_sha256")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("review activity plan/selection mismatch")
    if not plan.get("review_activity_metadata_projection_requested", False):
        raise SystemExit("test plan did not authorize review activity metadata")
    if plan.get("review_activity_text_requested") is not False:
        raise SystemExit("test plan unexpectedly requests review text")

    cases: list[dict[str, Any]] = []
    for case in selection["selection_material"]["cases"]:
        if case["temporal_band"] != "recent":
            continue
        owner, name = case["repository"].split("/", 1)
        identity_payload = api(
            [
                "graphql",
                "-f",
                f"query={PULL_IDENTITY_QUERY}",
                "-F",
                f"owner={owner}",
                "-F",
                f"name={name}",
                "-F",
                f"number={case['pull_number']}",
            ]
        )
        pull = identity_payload["data"]["repository"]["pullRequest"]
        if pull is None or pull["headRefOid"] != case["head_sha"]:
            raise SystemExit(f"{case['case_id']}: activity projection head mismatch")
        author = str((pull.get("author") or {}).get("login") or "unknown")
        commits = pull["commits"]["nodes"]
        if not commits or commits[-1]["commit"]["oid"] != case["head_sha"]:
            raise SystemExit(f"{case['case_id']}: head commit metadata unavailable")
        head_committed_at = commits[-1]["commit"]["committedDate"]
        prefix = f"repos/{case['repository']}"
        reviews = api(
            [f"{prefix}/pulls/{case['pull_number']}/reviews"], paginate=True
        )
        inline = api(
            [f"{prefix}/pulls/{case['pull_number']}/comments"], paginate=True
        )
        issue = api(
            [f"{prefix}/issues/{case['pull_number']}/comments"], paginate=True
        )
        events = [
            *(projected_event("review", item) for item in reviews if item.get("body")),
            *(
                projected_event("review-comment", item)
                for item in inline
                if item.get("body")
            ),
            *(
                projected_event("issue-comment", item)
                for item in issue
                if item.get("body")
            ),
        ]
        explicit = [
            event
            for event in events
            if not event["is_bot"]
            and event["author"] != author
            and event["is_substantive"]
            and (
                event["source"] in {"review", "review-comment"}
                or event["author_association"] in {"COLLABORATOR", "MEMBER", "OWNER"}
            )
        ]
        head_time = parse_time(head_committed_at)
        final_head = [
            event
            for event in explicit
            if event["commit_id"] == case["head_sha"]
            or (
                event["created_at"] is not None
                and parse_time(event["created_at"]) >= head_time
            )
        ]
        cases.append(
            {
                "case_id": case["case_id"],
                "repository": case["repository"],
                "pull_number": case["pull_number"],
                "locked_head_sha": case["head_sha"],
                "head_committed_at": head_committed_at,
                "pr_author": author,
                "pr_author_association": str(
                    pull.get("authorAssociation") or "UNKNOWN"
                ),
                "projected_events": events,
                "explicit_human_non_author_activity_count": len(explicit),
                "final_head_explicit_human_non_author_activity_count": len(final_head),
                "check_activity_gate": bool(final_head),
            }
        )

    material = {
        "schema_version": "0.1",
        "protocol_id": "outcome-free-recent-review-activity-projection-v0.1",
        "round": args.round.upper(),
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "acquired_at": datetime.now(UTC).isoformat(),
        "recent_cases_only": True,
        "state_or_merge_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "review_or_comment_text_stored": False,
        "raw_api_responses_stored": False,
        "judgment_consumes_derived_activity_only": True,
        "cases": cases,
    }
    payload = {**material, "activity_projection_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "round": args.round.upper(),
                "recent_cases": len(cases),
                "check_activity_cases": sum(case["check_activity_gate"] for case in cases),
                "activity_projection_sha256": payload["activity_projection_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
