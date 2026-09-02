#!/usr/bin/env python3
"""Run the staged, outcome-blind R8 cohort of thirty historical PRs."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import (
    audit_prediction_lock,
    canonical_sha256,
    freeze_prediction,
)
from infraswe.history.oracle import (
    compile_polarized_oracle,
    polarized_oracle_matches_machine,
)
from infraswe.history.static_cohort import assess_static_change, is_source_path, is_test_path
from infraswe.io import atomic_write_json
from infraswe.models.history import (
    HistoricalGroundTruth,
    HistoricalPRCandidate,
    HistoricalPredictionLock,
    HistoricalPredictionMaterial,
    HistoricalReviewActivitySnapshot,
)

QUERY = """
query($queryString: String!, $cursor: String) {
  search(query: $queryString, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        title
        createdAt
        author { login }
        baseRefName
        baseRefOid
        headRefOid
        changedFiles
        additions
        deletions
        files(first: 10) { nodes { path } }
        commits(first: 1) {
          nodes {
            commit {
              oid
              parents(first: 1) { nodes { oid } }
            }
          }
        }
      }
    }
  }
}
"""

DEPENDENCY_NAMES = {
    "cargo.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "yarn.lock",
}
GENERATED_SUFFIXES = (".lock", ".min.js", ".pb.cc", ".pb.h")


def _run_gh(arguments: list[str]) -> tuple[Any, str]:
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        process = subprocess.run(
            ["gh", "api", *arguments], check=False, capture_output=True, text=True
        )
        if process.returncode == 0:
            break
        if attempt < 3:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(f"GitHub API failed for {' '.join(arguments)}: {process.stderr.strip()}")
    return json.loads(process.stdout), process.stdout


def _graphql(query_string: str, cursor: str | None) -> dict[str, Any]:
    arguments = ["graphql", "-f", f"query={QUERY}", "-F", f"queryString={query_string}"]
    if cursor:
        arguments.extend(["-F", f"cursor={cursor}"])
    payload, _ = _run_gh(arguments)
    if payload.get("errors"):
        raise RuntimeError(str(payload["errors"]))
    return payload["data"]["search"]


def _rest(endpoint: str, *, paginate: bool = False) -> tuple[Any, str]:
    arguments: list[str] = []
    if paginate:
        arguments.extend(["--paginate", "--slurp"])
    arguments.append(endpoint)
    payload, raw = _run_gh(arguments)
    if paginate:
        payload = [item for page in payload for item in page]
    return payload, raw


def _digest_raw(raw: str) -> str:
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _material(payload: dict[str, Any], digest_key: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != digest_key}


def _validate_digest(payload: dict[str, Any], digest_key: str) -> None:
    if payload.get(digest_key) != canonical_sha256(_material(payload, digest_key)):
        raise SystemExit(f"digest mismatch: {digest_key}")


def _paths(node: dict[str, Any]) -> list[str]:
    return [str(item["path"]) for item in node["files"]["nodes"]]


def _first_parent(node: dict[str, Any]) -> str | None:
    commits = node["commits"]["nodes"]
    if not commits:
        return None
    parents = commits[0]["commit"]["parents"]["nodes"]
    return str(parents[0]["oid"]) if parents else None


def _eligible(
    node: dict[str, Any], project_policy: dict[str, Any], policy: dict[str, Any]
) -> tuple[bool, list[str]]:
    paths = _paths(node)
    changed_files = int(node.get("changedFiles") or 0)
    changed_lines = int(node.get("additions") or 0) + int(node.get("deletions") or 0)
    rules = policy["eligibility"]
    reasons: list[str] = []
    if int(node["number"]) in set(project_policy["excluded"]):
        reasons.append("previously-scored")
    if not int(rules["changed_files_min"]) <= changed_files <= int(rules["changed_files_max"]):
        reasons.append("changed-file-count-out-of-range")
    if changed_lines > int(rules["changed_lines_max"]):
        reasons.append("changed-lines-over-limit")
    if not any(token in str(node["title"]).lower() for token in policy["title_tokens"]):
        reasons.append("no-title-signal")
    if not any(path.startswith(tuple(project_policy["source_prefixes"])) for path in paths):
        reasons.append("no-profile-source-path")
    if all(path.endswith(".md") or path.startswith(("docs/", "doc/")) for path in paths):
        reasons.append("docs-only")
    if all(is_test_path(path) for path in paths):
        reasons.append("tests-only")
    if all(
        path.rsplit("/", 1)[-1].lower() in DEPENDENCY_NAMES
        or path.endswith(GENERATED_SUFFIXES)
        or path.startswith(("third_party/", "vendor/"))
        for path in paths
    ):
        reasons.append("dependency-or-generated-only")
    if _first_parent(node) is None:
        reasons.append("first-commit-parent-unavailable")
    if not node.get("headRefOid"):
        reasons.append("head-sha-unavailable")
    if not node.get("baseRefOid") or not node.get("baseRefName"):
        reasons.append("base-reference-unavailable")
    return not reasons, reasons


def _rank(node: dict[str, Any]) -> tuple[int, float, int, int, int]:
    created = datetime.fromisoformat(str(node["createdAt"]).replace("Z", "+00:00"))
    lines = int(node.get("additions") or 0) + int(node.get("deletions") or 0)
    has_test = any(is_test_path(path) for path in _paths(node))
    return (
        0 if has_test else 1,
        -created.timestamp(),
        int(node.get("changedFiles") or 0),
        lines,
        int(node["number"]),
    )


def discover(policy_path: Path, output: Path) -> None:
    policy = _read_object(policy_path)
    if policy["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R8 policy exposes review text")
    if policy["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R8 policy exposes outcomes")
    if sum(int(item["count"]) for item in policy["projects"].values()) != 30:
        raise SystemExit("R8 allocation must contain exactly thirty cases")

    discoveries: dict[str, Any] = {}
    selected_cases: list[HistoricalPRCandidate] = []
    window = policy["created_at_window"]
    start = str(window["start"])[:10]
    end = str(window["end"])[:10]
    for project, project_policy in policy["projects"].items():
        repository = project_policy["repository"]
        query_string = f"repo:{repository} is:pr created:{start}..{end} sort:created-desc"
        cursor: str | None = None
        nodes: list[dict[str, Any]] = []
        issue_count = 0
        page_count = 0
        required = int(project_policy["count"])
        while True:
            search = _graphql(query_string, cursor)
            issue_count = int(search["issueCount"])
            page_count += 1
            nodes.extend(search["nodes"])
            eligible = [node for node in nodes if _eligible(node, project_policy, policy)[0]]
            if len(eligible) >= required:
                break
            page_info = search["pageInfo"]
            if not page_info["hasNextPage"] or page_count >= 10:
                break
            cursor = page_info["endCursor"]
        eligible = [node for node in nodes if _eligible(node, project_policy, policy)[0]]
        eligible.sort(key=_rank)
        if len(eligible) < required:
            raise SystemExit(
                f"only {len(eligible)} eligible R8 cases for {project}, need {required}"
            )
        chosen = eligible[:required]
        candidates: list[dict[str, Any]] = []
        for node in nodes:
            ok, exclusion_reasons = _eligible(node, project_policy, policy)
            candidates.append(
                {
                    "number": int(node["number"]),
                    "title": node["title"],
                    "created_at": node["createdAt"],
                    "author": (node.get("author") or {}).get("login"),
                    "base_ref": node.get("baseRefName"),
                    "base_ref_oid": node.get("baseRefOid"),
                    "base_sha": _first_parent(node),
                    "head_sha": node.get("headRefOid"),
                    "changed_files": node.get("changedFiles"),
                    "additions": node.get("additions"),
                    "deletions": node.get("deletions"),
                    "paths": _paths(node),
                    "eligible": ok,
                    "exclusion_reasons": exclusion_reasons,
                }
            )
        for node in chosen:
            case_id = f"{project.replace('-attention', 'attention')}-pr-{node['number']}"
            candidate = HistoricalPRCandidate(
                case_id=case_id,
                project=project,
                repository=repository,
                pull_number=node["number"],
                title=node["title"],
                created_at=node["createdAt"],
                base_ref=node["baseRefName"],
                base_tip_sha=node["baseRefOid"],
                base_sha=_first_parent(node),
                base_derivation="first-pr-commit-first-parent-path-parity",
                head_sha=node["headRefOid"],
                pr_commit_shas=[node["headRefOid"]],
                changed_files=node["changedFiles"],
                additions=node["additions"],
                deletions=node["deletions"],
                paths=_paths(node),
                acquisition_query=query_string,
                selection_policy_id=policy["protocol_id"],
                outcome_fields_requested=False,
            )
            selected_cases.append(candidate)
        discoveries[project] = {
            "repository": repository,
            "query": query_string,
            "issue_count": issue_count,
            "page_count": page_count,
            "returned_count": len(nodes),
            "eligible_count": len(eligible),
            "selected_pull_numbers": [int(node["number"]) for node in chosen],
            "candidates": candidates,
        }

    material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "preselection_policy_sha256": canonical_sha256(policy),
        "review_fields_requested": False,
        "outcome_fields_requested": False,
        "ci_fields_requested": False,
        "discovered_at": datetime.now(UTC).isoformat(),
        "discoveries": discoveries,
        "selected_cases": [item.model_dump(mode="json") for item in selected_cases],
    }
    if len(material["selected_cases"]) != 30:
        raise SystemExit("R8 discovery did not select exactly thirty cases")
    payload = {**material, "selection_lock_sha256": canonical_sha256(material)}
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {project: data["selected_pull_numbers"] for project, data in discoveries.items()},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")


def freeze_plan(selection_path: Path, output: Path) -> None:
    selection = _read_object(selection_path)
    _validate_digest(selection, "selection_lock_sha256")
    cases = [HistoricalPRCandidate.model_validate(item) for item in selection["selected_cases"]]
    material = {
        "schema_version": "0.1",
        "protocol_id": selection["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_diff_content_or_commit_message_inspection": True,
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "execution_tier": "offline-exact-sha-static-contract",
        "authority_limits": [
            "No runtime, GPU, integration, performance, or mergeability correctness is inferred.",
            "A high score means only that the preregistered static contract found no hard defect.",
        ],
        "checks": [
            "exact changed-path parity",
            "head content retrieval by locked blob SHA",
            "Python/JSON/TOML syntax where a standard-library parser exists",
            "merge conflict marker rejection",
            "silent broad exception fallback without changed test rejection",
            "removed validation without replacement or changed test rejection",
            "removed synchronization without replacement or changed test rejection",
        ],
        "polarized_scoring": {
            "hard_static_failure": 35,
            "static_pass_without_changed_test": 88,
            "static_pass_with_changed_test": 94,
            "acquisition_or_parity_failure": None,
            "accept_floor": 85,
            "middle_band_60_to_84_emitted": False,
            "machine_revise_emitted": False,
        },
        "cases": [
            {
                "case_id": item.case_id,
                "candidate_sha256": canonical_sha256(item),
                "base_sha": item.base_sha,
                "head_sha": item.head_sha,
                "paths": item.paths,
            }
            for item in cases
        ],
    }
    payload = {**material, "test_plan_sha256": canonical_sha256(material)}
    atomic_write_json(output, payload)
    print(f"case_count={len(cases)}")
    print(f"test_plan_sha256={payload['test_plan_sha256']}")


def _decode_blob(repository: str, blob_sha: str) -> tuple[bytes, str]:
    payload, raw = _rest(f"repos/{repository}/git/blobs/{blob_sha}")
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unsupported GitHub blob encoding for {blob_sha}")
    return base64.b64decode(payload["content"]), raw


def score(selection_path: Path, plan_path: Path, evidence_output: Path, lock_output: Path) -> None:
    selection = _read_object(selection_path)
    plan = _read_object(plan_path)
    _validate_digest(selection, "selection_lock_sha256")
    _validate_digest(plan, "test_plan_sha256")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R8 test plan is not bound to selection")
    candidates = [
        HistoricalPRCandidate.model_validate(item) for item in selection["selected_cases"]
    ]
    planned = {item["case_id"]: item for item in plan["cases"]}
    if {item.case_id for item in candidates} != set(planned):
        raise SystemExit("R8 selection and plan case sets differ")

    evidence_cases: list[dict[str, Any]] = []
    locks: list[HistoricalPredictionLock] = []
    for index, candidate in enumerate(candidates, start=1):
        files, raw_files = _rest(
            f"repos/{candidate.repository}/pulls/{candidate.pull_number}/files?per_page=100",
            paginate=True,
        )
        head_sources: dict[str, bytes] = {}
        source_digests: dict[str, str] = {}
        api_digests = [_digest_raw(raw_files)]
        for item in files:
            path = str(item["filename"])
            if is_source_path(path) and not is_test_path(path) and item.get("status") != "removed":
                content, raw_blob = _decode_blob(candidate.repository, str(item["sha"]))
                head_sources[path] = content
                source_digests[path] = "sha256:" + hashlib.sha256(content).hexdigest()
                api_digests.append(_digest_raw(raw_blob))
        assessment = assess_static_change(
            expected_paths=candidate.paths,
            changed_files=files,
            head_sources=head_sources,
        )
        evidence_material = {
            "schema_version": "0.1",
            "protocol_id": selection["protocol_id"],
            "case_id": candidate.case_id,
            "candidate_sha256": canonical_sha256(candidate),
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "base_sha": candidate.base_sha,
            "head_sha": candidate.head_sha,
            "observed_paths": sorted(str(item["filename"]) for item in files),
            "head_source_digests": source_digests,
            "patch_digests": {
                str(item["filename"]): canonical_sha256(item.get("patch") or "") for item in files
            },
            "api_response_digests": api_digests,
            "assessment": assessment.as_dict(),
            "outcome_fields_requested": False,
            "review_fields_requested": False,
            "ci_fields_requested": False,
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        evidence = {
            **evidence_material,
            "evidence_sha256": canonical_sha256(evidence_material),
        }
        evidence_cases.append(evidence)
        if assessment.decision == "accept_with_scope":
            predicted = "merged"
            confidence = "medium"
        elif assessment.decision == "reject":
            predicted = "not-merged"
            confidence = "high"
        else:
            predicted = "abstain"
            confidence = "not-applicable"
        prediction = HistoricalPredictionMaterial(
            case_id=candidate.case_id,
            candidate_sha256=canonical_sha256(candidate),
            evidence_sha256=evidence["evidence_sha256"],
            prediction_policy_id="historical-merge-prediction-v0.5-r2-polarized",
            predicted_outcome=predicted,
            mergeability_decision=assessment.decision,
            score_100=assessment.score_100,
            confidence=confidence,
            rationale_codes=list(assessment.rationale_codes),
            frozen_at=datetime.now(UTC),
        )
        locks.append(freeze_prediction(prediction))
        print(
            f"[{index:02d}/{len(candidates)}] {candidate.case_id}: "
            f"{assessment.decision} score={assessment.score_100}"
        )

    evidence_material = {
        "schema_version": "0.1",
        "protocol_id": selection["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "outcomes_visible_during_scoring": False,
        "review_text_visible_during_scoring": False,
        "cases": evidence_cases,
    }
    evidence_payload = {
        **evidence_material,
        "evidence_set_sha256": canonical_sha256(evidence_material),
    }
    lock_material = {
        "schema_version": "0.1",
        "protocol_id": selection["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "evidence_set_sha256": evidence_payload["evidence_set_sha256"],
        "review_text_visible_during_machine_judgment": False,
        "merge_outcomes_visible_during_machine_judgment": False,
        "locks": [item.model_dump(mode="json") for item in locks],
        "summary": {
            "cases": len(locks),
            "accept_with_scope": sum(
                item.material.mergeability_decision == "accept_with_scope" for item in locks
            ),
            "reject": sum(item.material.mergeability_decision == "reject" for item in locks),
            "unresolved": sum(
                item.material.mergeability_decision == "unresolved" for item in locks
            ),
            "revise": 0,
            "scores_60_to_84": sum(
                item.material.score_100 is not None and 60 <= item.material.score_100 < 85
                for item in locks
            ),
        },
    }
    lock_payload = {**lock_material, "lock_set_sha256": canonical_sha256(lock_material)}
    atomic_write_json(evidence_output, evidence_payload)
    atomic_write_json(lock_output, lock_payload)
    print(json.dumps(lock_material["summary"], indent=2, sort_keys=True))
    print(f"lock_set_sha256={lock_payload['lock_set_sha256']}")


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _is_bot(user: dict[str, Any] | None) -> bool:
    user = user or {}
    login = str(user.get("login") or "").lower()
    return user.get("type") == "Bot" or login.endswith(("[bot]", "-bot", "_bot"))


def reveal(selection_path: Path, lock_path: Path, output: Path) -> None:
    selection = _read_object(selection_path)
    lock_payload = _read_object(lock_path)
    _validate_digest(selection, "selection_lock_sha256")
    _validate_digest(lock_payload, "lock_set_sha256")
    if lock_payload["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R8 machine locks are not bound to selection")
    locks = {
        item.material.case_id: item
        for item in map(HistoricalPredictionLock.model_validate, lock_payload["locks"])
    }
    if not all(audit_prediction_lock(item) for item in locks.values()):
        raise SystemExit("R8 prediction lock audit failed")
    candidates = {
        item.case_id: item
        for item in map(HistoricalPRCandidate.model_validate, selection["selected_cases"])
    }
    if set(candidates) != set(locks):
        raise SystemExit("R8 selection and lock case sets differ")

    revealed_cases: list[dict[str, Any]] = []
    for index, case_id in enumerate(sorted(candidates), start=1):
        candidate = candidates[case_id]
        lock = locks[case_id]
        prefix = f"repos/{candidate.repository}"
        pull, raw_pull = _rest(f"{prefix}/pulls/{candidate.pull_number}")
        reviews, raw_reviews = _rest(
            f"{prefix}/pulls/{candidate.pull_number}/reviews?per_page=100", paginate=True
        )
        issue_comments, raw_comments = _rest(
            f"{prefix}/issues/{candidate.pull_number}/comments?per_page=100", paginate=True
        )
        observed_at = datetime.now(UTC)
        if observed_at < lock.material.frozen_at:
            raise SystemExit(f"{case_id} reveal predates prediction lock")
        author = str((pull.get("user") or {}).get("login") or "unknown")
        human_reviews = [
            item
            for item in reviews
            if not _is_bot(item.get("user"))
            and str((item.get("user") or {}).get("login") or "unknown") != author
            and item.get("submitted_at")
        ]
        current_head_reviews = [
            item for item in human_reviews if item.get("commit_id") == candidate.head_sha
        ]
        last_human_review = max(
            (_parse_time(item.get("submitted_at")) for item in human_reviews),
            default=None,
        )
        activity_times = [
            candidate.created_at,
            *(
                parsed
                for parsed in (
                    _parse_time(pull.get("updated_at")),
                    *(_parse_time(item.get("submitted_at")) for item in reviews),
                    *(_parse_time(item.get("updated_at")) for item in issue_comments),
                )
                if parsed is not None
            ),
        ]
        last_activity = min(max(activity_times), observed_at)
        truth = HistoricalGroundTruth(
            case_id=case_id,
            repository=candidate.repository,
            pull_number=candidate.pull_number,
            state=pull["state"],
            merged=bool(pull.get("merged")),
            merged_at=pull.get("merged_at"),
            closed_at=pull.get("closed_at"),
            merge_commit_sha=pull.get("merge_commit_sha") if pull.get("merged") else None,
            html_url=pull["html_url"],
            observed_at=observed_at,
            prediction_lock_sha256=lock.lock_sha256,
            api_response_sha256=_digest_raw(raw_pull),
        )
        review = HistoricalReviewActivitySnapshot(
            case_id=case_id,
            observed_at=observed_at,
            last_activity_at=last_activity,
            last_human_review_at=last_human_review,
            current_head_human_non_author_review_count=len(current_head_reviews),
            total_human_non_author_review_count=len(human_reviews),
            pending_human_review_request=bool(
                pull.get("requested_reviewers") or pull.get("requested_teams")
            ),
            api_response_digests=[_digest_raw(raw_reviews), _digest_raw(raw_comments)],
        )
        oracle = compile_polarized_oracle(
            candidate,
            truth,
            review,
            machine_score_100=lock.material.score_100,
        )
        matches = polarized_oracle_matches_machine(oracle, lock.material.mergeability_decision)
        revealed_cases.append(
            {
                "case_id": case_id,
                "project": candidate.project,
                "repository": candidate.repository,
                "pull_number": candidate.pull_number,
                "title": candidate.title,
                "machine_decision": lock.material.mergeability_decision,
                "machine_score_100": lock.material.score_100,
                "machine_rationale_codes": lock.material.rationale_codes,
                "prediction_lock_sha256": lock.lock_sha256,
                "ground_truth": truth.model_dump(mode="json"),
                "review_activity": review.model_dump(mode="json"),
                "oracle": oracle.model_dump(mode="json"),
                "oracle_matches_machine": matches,
            }
        )
        print(
            f"[{index:02d}/{len(candidates)}] {case_id}: "
            f"machine={lock.material.mergeability_decision}/{lock.material.score_100} "
            f"oracle={oracle.decision} match={matches}"
        )

    summary = {
        "cases": len(revealed_cases),
        "merged": sum(item["ground_truth"]["merged"] for item in revealed_cases),
        "closed_unmerged": sum(
            item["ground_truth"]["state"] == "closed" and not item["ground_truth"]["merged"]
            for item in revealed_cases
        ),
        "open": sum(item["ground_truth"]["state"] == "open" for item in revealed_cases),
        "oracle_accept": sum(item["oracle"]["decision"] == "accept" for item in revealed_cases),
        "oracle_revise": sum(item["oracle"]["decision"] == "revise" for item in revealed_cases),
        "oracle_reject": sum(item["oracle"]["decision"] == "reject" for item in revealed_cases),
        "machine_matches_oracle": sum(item["oracle_matches_machine"] for item in revealed_cases),
        "merged_score_floor_failures": sum(
            item["oracle"]["decision"] == "accept"
            and item["oracle"]["merged_score_floor_satisfied"] is not True
            for item in revealed_cases
        ),
    }
    material = {
        "schema_version": "0.1",
        "protocol_id": f"{selection['protocol_id']}-reveal",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "judgment_lock_set_sha256": lock_payload["lock_set_sha256"],
        "revealed_after_all_machine_scores_locked": True,
        "revealed_at": datetime.now(UTC).isoformat(),
        "cases": revealed_cases,
        "summary": summary,
    }
    payload = {**material, "reveal_sha256": canonical_sha256(material)}
    atomic_write_json(output, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"reveal_sha256={payload['reveal_sha256']}")


def report(reveal_path: Path, output: Path) -> None:
    reveal_payload = _read_object(reveal_path)
    _validate_digest(reveal_payload, "reveal_sha256")
    summary = reveal_payload["summary"]
    rows: list[str] = []
    for item in sorted(
        reveal_payload["cases"], key=lambda value: (value["project"], value["pull_number"])
    ):
        truth = item["ground_truth"]
        oracle = item["oracle"]
        state = "merged" if truth["merged"] else truth["state"]
        score = "—" if item["machine_score_100"] is None else f"{item['machine_score_100']:.0f}"
        rows.append(
            "| {project} | [#{number}]({url}) | {score} | {machine} | {state} | {oracle} | {match} |".format(
                project=item["project"],
                number=item["pull_number"],
                url=truth["html_url"],
                score=score,
                machine=item["machine_decision"],
                state=state,
                oracle=oracle["decision"],
                match="是" if item["oracle_matches_machine"] else "否",
            )
        )
    accuracy = summary["machine_matches_oracle"] / summary["cases"]
    markdown = f"""# InfraSWE 历史 PR R8 离线补测报告

日期：{reveal_payload["revealed_at"][:10]}  
协议：`historical-pr-blind-cross-project-v0.1-r8-30`

## 结论

本轮新增补测 **{summary["cases"]} 个**此前未正式评分的历史 PR，覆盖全部 10 个默认 Draft 项目，其中 SGLang 5 个。所有样本先用不含状态、合并结果、CI、review 的元数据完成选择，随后冻结测试计划与机器评分；最后才读取 GitHub 状态编译 polarized oracle。

- 机器评分：只输出 `94/88` 的 scoped accept、`35` 的 hard reject，取证失败才 unresolved；没有 60–84 分，也不输出机器 `revise`。
- Oracle：accept {summary["oracle_accept"]}、revise {summary["oracle_revise"]}、reject {summary["oracle_reject"]}。只有 30 天内且当前 head 有近期真人评审活动的新 PR 才可判 revise；本轮旧样本不满足时按 reject。
- 命中：{summary["machine_matches_oracle"]}/{summary["cases"]}（{accuracy:.1%}）。这是静态 tier 的历史校准结果，不冒充 GPU/runtime/performance 验证。
- 已合并 PR 的 85 分硬下限违规数：**{summary["merged_score_floor_failures"]}**。

## 明细

| Project | PR | 机器分 | 机器判断 | GitHub 状态 | Oracle | 命中 |
|---|---:|---:|---|---|---|---:|
{"".join(row + chr(10) for row in rows)}
## 边界

本轮验证 exact-SHA 路径一致性、按 blob SHA 获取源码、可支持语言的语法、冲突标记、静默异常 fallback、验证边界与同步原语的明显删除。它不证明运行时正确性、GPU 机制、性能或最终可合并性；高分明确是 `accept_with_scope`。

完整不可变材料见同目录的 `selection-lock.json`、`test-plan.json`、`blind-static-evidence.json`、`machine-prediction-locks.json` 与 `revealed-polarized-oracles.json`。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--policy", type=Path, required=True)
    discover_parser.add_argument("--output", type=Path, required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--selection", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--selection", type=Path, required=True)
    score_parser.add_argument("--plan", type=Path, required=True)
    score_parser.add_argument("--evidence-output", type=Path, required=True)
    score_parser.add_argument("--lock-output", type=Path, required=True)

    reveal_parser = subparsers.add_parser("reveal")
    reveal_parser.add_argument("--selection", type=Path, required=True)
    reveal_parser.add_argument("--locks", type=Path, required=True)
    reveal_parser.add_argument("--output", type=Path, required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--reveal", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "discover":
        discover(args.policy, args.output)
    elif args.command == "plan":
        freeze_plan(args.selection, args.output)
    elif args.command == "score":
        score(args.selection, args.plan, args.evidence_output, args.lock_output)
    elif args.command == "reveal":
        reveal(args.selection, args.locks, args.output)
    else:
        report(args.reveal, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
