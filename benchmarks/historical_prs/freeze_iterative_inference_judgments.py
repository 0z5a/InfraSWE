#!/usr/bin/env python3
"""Freeze scalable outcome-blind judgments for iterative inference cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from build_iterative_pr_precedents import STOP_TOKENS
from discover_r14_candidates import _is_test_path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

DECISIONS = ("accept_with_scope", "check", "reject", "unresolved")
SUMMARY_RE = re.compile(r"(?P<count>\d+)\s+(?P<label>passed|failed|skipped)")
EXACT_FAILURE_MARKERS = (
    "AssertionError",
    "Failed: DID NOT RAISE",
    "IndexError:",
    "assert ",
)
ENVIRONMENT_GAP_MARKERS = (
    "ModuleNotFoundError",
    "ImportError while importing",
    "cannot import name",
    "FileNotFoundError",
    "ConnectionError",
    "Connection refused",
    "fixture '",
    'fixture "',
    "Failed to infer device type",
    "Device string must not be empty",
    "not supported in this architecture",
    "requires sm",
    "no tests ran",
    "no tests collected",
    "unrecognized arguments",
)
NON_NVIDIA_PATH_MARKERS = (
    "/amd/",
    "/ascend/",
    "/npu/",
    "mi300",
    "mi325",
    "mi35",
    "rocm",
)
SM90_PATH_MARKERS = ("sm90", "h100", "h200", "hopper")
SM100_PATH_MARKERS = ("sm100", "b200", "gb200")
SM120_PATH_MARKERS = ("sm120",)
BLACKWELL_PATH_MARKERS = ("blackwell",)
SELF_DECLARED_INCOMPLETE_MARKERS = (
    "[wip]",
    "work in progress",
    "do not merge",
    "not ready for review",
    "todo before merge",
)
TITLE_READINESS_RE = re.compile(
    r"(?:^\s*\[draft\]|^\s*draft\s*[-:]|\bwip\b)", re.IGNORECASE
)


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def checked_payload(
    path: Path, digest_field: str, *, material_field: str | None = None
) -> dict[str, Any]:
    payload = read(path)
    material = (
        payload[material_field]
        if material_field is not None
        else {key: value for key, value in payload.items() if key != digest_field}
    )
    require(
        payload[digest_field] == canonical_sha256(material),
        f"{path.name} embedded digest mismatch",
    )
    return payload


def summary_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for match in SUMMARY_RE.finditer(output):
        counts[match.group("label")] = max(counts[match.group("label")], int(match.group("count")))
    return counts


def targets_foreign_accelerator(path: str, executor_architecture: str) -> bool:
    lowered = path.lower()
    if any(marker in lowered for marker in NON_NVIDIA_PATH_MARKERS):
        return True
    if any(marker in lowered for marker in SM90_PATH_MARKERS):
        return executor_architecture != "sm90"
    if any(marker in lowered for marker in SM100_PATH_MARKERS):
        return executor_architecture != "sm100"
    if any(marker in lowered for marker in SM120_PATH_MARKERS):
        return executor_architecture != "sm120"
    if any(marker in lowered for marker in BLACKWELL_PATH_MARKERS):
        return executor_architecture not in {"sm100", "sm120"}
    return False


def exact_candidate_failure(
    record: dict[str, Any], test_names: list[str], executor_architecture: str
) -> bool:
    if record.get("returncode") != 1 or not test_names:
        return False
    test_paths = [str(path).lower() for path in record.get("test_paths", [])]
    if any(targets_foreign_accelerator(path, executor_architecture) for path in test_paths):
        return False
    output = str(record.get("output_tail") or "")
    if not any(marker in output for marker in EXACT_FAILURE_MARKERS):
        return False
    explicit_failure_lines = "\n".join(
        line for line in output.splitlines() if line.startswith("FAILED ")
    )
    if any(name in explicit_failure_lines for name in test_names):
        return True
    if any(marker in output for marker in ENVIRONMENT_GAP_MARKERS):
        return False
    failure_lines = "\n".join(line for line in output.splitlines() if line.startswith("E   "))
    return any(
        name in failure_lines or name in output or f"::{name}" in output for name in test_names
    )


def technical_contract(
    record: dict[str, Any], test_names: list[str], executor_architecture: str
) -> tuple[str, dict[str, int], bool]:
    output = str(record.get("output_tail") or "")
    counts = summary_counts(output)
    exact_failure = exact_candidate_failure(record, test_names, executor_architecture)
    if exact_failure:
        return "fail", counts, True
    if record.get("returncode") == 0 and counts["passed"] > 0:
        return "pass", counts, False
    return "bounded-gap", counts, False


def source_complete(static: dict[str, Any]) -> bool:
    critical_suffixes = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py")

    def exact_sides_available(item: dict[str, Any]) -> bool:
        change_type = str(item.get("change_type") or "modified")
        if change_type == "added":
            return bool(item.get("head_available"))
        if change_type == "removed":
            return bool(item.get("base_available"))
        return bool(item.get("base_available")) and bool(item.get("head_available"))

    critical_source_missing = any(
        not exact_sides_available(item)
        and (item["is_test"] or str(item["path"]).lower().endswith(critical_suffixes))
        for item in static["files"]
    )
    return (
        bool(static["head_matches_selection_at_acquisition"])
        and not critical_source_missing
        and not static["head_conflict_marker_files"]
        and not static["head_python_syntax_failures"]
        and static["source_file_count"] > 0
    )


def mature_contract_closed(
    selected: dict[str, Any], static: dict[str, Any], technical: str
) -> bool:
    if not source_complete(static):
        return False
    tests = static["candidate_test_functions_added"]
    body = static["body_evidence"]
    if technical == "pass":
        return True
    if tests and static["candidate_test_path_present"]:
        return True
    if static["candidate_test_path_present"] and static["test_file_count"] > 0:
        return True
    if body["mentions_test"] and (
        body["mentions_benchmark"]
        or static["test_file_count"] > 0
        or selected["additions"] + selected["deletions"] <= 800
    ):
        return True
    if body["mentions_benchmark"] and static["source_file_count"] > 0:
        return True
    boundary_signatures = sum(static["aggregate_added_signatures"].values())
    return (
        selected["changed_files"] <= 3
        and selected["additions"] + selected["deletions"] <= 120
        and boundary_signatures > 0
    )


def body_text(bundle_case: dict[str, Any]) -> str:
    projection = bundle_case.get("body_projection") or {}
    return str(projection.get("body") or "")


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _title_tokens(title: str) -> set[str]:
    text = re.sub(r"#\d+", " ", title.lower())
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 1 and token not in STOP_TOKENS
    }


def _source_paths(paths: list[str]) -> set[str]:
    return {
        path
        for path in paths
        if not _is_test_path(path) and not path.endswith(".md")
    }


def _directory_prefixes(paths: set[str]) -> set[str]:
    prefixes: set[str] = set()
    for path in paths:
        parts = path.split("/")
        for depth in range(1, min(4, len(parts))):
            prefixes.add("/".join(parts[:depth]))
    return prefixes


def precedent_consensus(
    selected: dict[str, Any],
    precedents: list[dict[str, Any]],
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if not precedents or policy is None:
        return {"negative": False, "matches": []}
    title = _title_tokens(selected["title"])
    paths = _source_paths(list(selected["paths"]))
    directories = _directory_prefixes(paths)
    matches: list[dict[str, Any]] = []
    for precedent in precedents:
        if (
            precedent["project"] != selected["project"]
            or precedent["risk_family"] != selected["risk_family"]
        ):
            continue
        title_similarity = _jaccard(title, set(precedent["title_tokens"]))
        path_similarity = _jaccard(paths, set(precedent["source_paths"]))
        directory_similarity = _jaccard(
            directories, set(precedent["source_directory_prefixes"])
        )
        if (
            title_similarity >= float(policy["title_token_jaccard_min"])
            or path_similarity >= float(policy["source_path_jaccard_min"])
            or directory_similarity
            >= float(policy["source_directory_prefix_jaccard_min"])
        ):
            matches.append(
                {
                    "precedent_id": precedent["precedent_id"],
                    "oracle_decision": precedent["oracle_decision"],
                    "title_token_jaccard": title_similarity,
                    "source_path_jaccard": path_similarity,
                    "source_directory_prefix_jaccard": directory_similarity,
                }
            )
    dispositions = {item["oracle_decision"] for item in matches}
    negative = (
        len(matches) >= int(policy["negative_consensus_minimum"])
        and dispositions == {"reject"}
    )
    return {
        "negative": negative,
        "matches": sorted(matches, key=lambda item: item["precedent_id"]),
    }


def decision_for(
    selected: dict[str, Any],
    static: dict[str, Any],
    bundle_case: dict[str, Any],
    record: dict[str, Any],
    *,
    precedents: list[dict[str, Any]],
    precedent_policy: dict[str, Any] | None,
    review_activity: dict[str, Any] | None,
    executor_architecture: str,
) -> dict[str, Any]:
    test_names = list(static["candidate_test_functions_added"])
    technical, counts, exact_failure = technical_contract(
        record, test_names, executor_architecture
    )
    body = body_text(bundle_case)
    body_lower = body.lower()
    incomplete = any(marker in body_lower for marker in SELF_DECLARED_INCOMPLETE_MARKERS)
    closed = mature_contract_closed(selected, static, technical)
    precedent = precedent_consensus(selected, precedents, precedent_policy)
    explicit_readiness = TITLE_READINESS_RE.search(selected["title"]) is not None
    review_state_counts = (
        review_activity.get("human_non_author_review_state_counts", {})
        if review_activity
        else {}
    )
    review_state_metadata_available = bool(
        review_activity
        and "human_non_author_review_state_counts" in review_activity
    )
    approved_reviews = int(review_state_counts.get("APPROVED", 0))

    if exact_failure:
        decision = "reject"
        code = "EXACT_CANDIDATE_FAILURE"
        residual = "Repair the candidate-owned failing boundary and rerun its exact test."
    elif not source_complete(static):
        decision = "reject"
        code = "SOURCE_INTEGRITY_FAILURE"
        residual = "Restore a complete, parseable exact-head source boundary."
    elif selected["temporal_band"] == "recent":
        if review_activity and review_activity["check_activity_gate"]:
            decision = "check"
            code = "ACTIVE_FINAL_HEAD_REVIEW_ACTIVITY"
            residual = (
                "Complete the active final-head review handoff and freeze its "
                "terminal disposition."
            )
        elif (
            technical == "pass"
            and review_activity
            and review_activity.get("pr_author_association")
            in {"COLLABORATOR", "MEMBER", "OWNER"}
            and selected["additions"] + selected["deletions"] <= 120
        ):
            decision = "accept_with_scope"
            code = "RECENT_MAINTAINER_EXACT_TEST_PASS"
            residual = None
        else:
            decision = "reject"
            code = "RECENT_WITHOUT_ACTIVE_REVIEW"
            residual = (
                "Obtain named non-author final-head review activity before using check, "
                "or wait for a terminal disposition."
            )
    elif explicit_readiness and technical != "pass":
        decision = "reject"
        code = "EXPLICIT_TITLE_READINESS_VETO"
        residual = "Remove the explicit draft/WIP marker after closing the title-scoped contract."
    elif incomplete and technical != "pass":
        decision = "reject"
        code = "SELF_DECLARED_INCOMPLETE"
        residual = "Close the author-declared incomplete boundary and rerun the frozen plan."
    elif approved_reviews > 0:
        decision = "accept_with_scope"
        code = "MATURE_HUMAN_APPROVAL"
        residual = None
    elif (
        review_state_metadata_available
        and review_activity
        and review_activity.get("human_non_author_review_count", 0) > 0
    ):
        decision = "reject"
        code = "MATURE_REVIEW_WITHOUT_APPROVAL"
        residual = (
            "Obtain a non-author approval on the candidate before treating review "
            "activity as a merge-readiness receipt."
        )
    elif review_activity and review_activity.get("pr_author_association") == "NONE":
        decision = "reject"
        code = "MATURE_UNAFFILIATED_AUTHOR"
        residual = "Obtain a non-NONE repository association or a terminal disposition."
    elif (
        review_activity
        and review_activity.get("human_non_author_review_count", 0) == 0
        and not (
            selected["project"] == "sglang"
            and review_activity.get("pr_author_association")
            in {"COLLABORATOR", "MEMBER", "OWNER"}
        )
    ):
        decision = "reject"
        code = "MATURE_WITHOUT_HUMAN_REVIEW"
        residual = "Obtain a named non-author human review or a terminal disposition."
    elif precedent["negative"]:
        decision = "reject"
        code = "UNANIMOUS_NEGATIVE_PRECEDENT_CLUSTER"
        residual = (
            "Demonstrate a title-scoped distinction from the prior rejected cluster "
            "with an exact evaluator-owned closure."
        )
    elif closed:
        decision = "accept_with_scope"
        code = "MATURE_CONTRACT_CLOSED"
        residual = None
    else:
        decision = "accept_with_scope"
        code = "MATURE_SOURCE_COMPLETE_RUNTIME_GAP"
        residual = "The exact runtime closure remains unavailable; source integrity is closed."

    status = str(record.get("status") or "unknown")
    returncode = record.get("returncode")
    if exact_failure:
        execution_finding = (
            "The exact-head run reached a candidate-owned test and reproduced a "
            "candidate-boundary assertion failure."
        )
    elif counts["passed"]:
        execution_finding = (
            f"The preferred exact-head run completed {counts['passed']} passing test(s); "
            f"returncode={returncode}, status={status}."
        )
    elif counts["skipped"]:
        execution_finding = (
            f"The exact-head run reported {counts['skipped']} capability-gated skip(s); "
            f"returncode={returncode}, status={status}."
        )
    else:
        execution_finding = (
            "The exact-head runtime path did not produce a candidate assertion receipt; "
            f"returncode={returncode}, status={status}, so the gap is neutral."
        )
    static_finding = (
        f"The frozen source has {static['source_file_count']} production file(s), "
        f"{static['test_file_count']} test file(s), and "
        f"{len(test_names)} newly detected test function(s); source integrity "
        f"is {'closed' if source_complete(static) else 'not closed'}."
    )
    if selected["temporal_band"] == "recent" and review_activity is not None:
        disposition_finding = (
            "The PR is in the recent band; the metadata-only projection found "
            f"{review_activity['final_head_explicit_human_non_author_activity_count']} "
            "qualifying final-head event(s), with all review text and outcomes hidden."
        )
        check_observability = "outcome-free-activity-projection"
    elif selected["temporal_band"] == "recent":
        disposition_finding = (
            "The PR is in the recent band, while review/comment evidence is prohibited "
            "before lock; check eligibility is therefore unobservable."
        )
        check_observability = "unavailable-by-blind-policy"
    elif review_activity is not None:
        disposition_finding = (
            "The PR is in the mature band; the metadata-only projection found "
            f"author association {review_activity.get('pr_author_association', 'UNKNOWN')} "
            "and "
            f"{review_activity.get('human_non_author_review_count', 0)} non-author human "
            f"review(s), including {approved_reviews} approval(s), with all review text "
            "and outcomes hidden."
        )
        check_observability = "outcome-free-activity-projection-mature"
    else:
        disposition_finding = (
            "The PR is in the mature band; the decision follows the frozen technical "
            f"closure cascade with {len(precedent['matches'])} close prior precedent(s) "
            "and without current outcome, state, review text, CI, or label evidence."
        )
        check_observability = "not-applicable-mature"
    return {
        "decision": decision,
        "technical_contract": technical,
        "rationale_code": code,
        "residual_contract": residual,
        "technical_findings": [
            execution_finding,
            static_finding,
            disposition_finding,
        ],
        "check_observability": check_observability,
        "execution_summary": counts,
        "mature_contract_closed": closed,
        "precedent_consensus": precedent,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-label", required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--static-evidence", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--precedents", type=Path)
    parser.add_argument("--review-activity", type=Path)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    round_label = args.round_label.upper()
    selection = checked_payload(
        args.selection_lock,
        "selection_lock_sha256",
        material_field="selection_material",
    )
    plan = checked_payload(args.test_plan, "test_plan_sha256")
    manifest = checked_payload(args.manifest, "evidence_manifest_sha256")
    static_payload = checked_payload(args.static_evidence, "evidence_sha256")
    bundle = checked_payload(args.source_bundle, "source_bundle_sha256")
    evidence_payloads = [checked_payload(path, "evidence_sha256") for path in args.evidence]
    precedent_payload = (
        checked_payload(args.precedents, "precedent_set_sha256")
        if args.precedents
        else None
    )
    activity_payload = (
        checked_payload(args.review_activity, "activity_projection_sha256")
        if args.review_activity
        else None
    )

    selection_sha = selection["selection_lock_sha256"]
    plan_sha = plan["test_plan_sha256"]
    source_sha = bundle["source_bundle_sha256"]
    require(plan["selection_lock_sha256"] == selection_sha, "plan/selection mismatch")
    require(manifest["selection_lock_sha256"] == selection_sha, "manifest/selection mismatch")
    require(static_payload["selection_lock_sha256"] == selection_sha, "static/selection mismatch")
    require(bundle["selection_lock_sha256"] == selection_sha, "bundle/selection mismatch")
    require(manifest["source_bundle_sha256"] == source_sha, "manifest/bundle mismatch")
    require(static_payload["source_bundle_sha256"] == source_sha, "static/bundle mismatch")
    for payload in evidence_payloads:
        require(payload["selection_lock_sha256"] == selection_sha, "test/selection mismatch")
        require(payload["test_plan_sha256"] == plan_sha, "test/plan mismatch")
    if precedent_payload is not None:
        require(
            precedent_payload["target_round"] == round_label,
            "precedent target round mismatch",
        )
    if activity_payload is not None:
        require(
            activity_payload["round"] == round_label,
            "review activity target round mismatch",
        )
        require(
            activity_payload["selection_lock_sha256"] == selection_sha,
            "review activity/selection mismatch",
        )
        require(
            activity_payload["test_plan_sha256"] == plan_sha,
            "review activity/plan mismatch",
        )
        require(
            activity_payload["review_or_comment_text_stored"] is False,
            "review activity projection stores text",
        )
        require(
            activity_payload["state_or_merge_fields_requested"] is False,
            "review activity projection exposes outcomes",
        )

    hidden = (
        selection["selection_material"]["review_or_comment_visible"],
        selection["selection_material"]["merge_outcomes_visible"],
        selection["selection_material"]["ci_or_label_visible"],
        plan["review_or_comment_requested"],
        plan["merge_outcome_or_state_requested"],
        plan["ci_or_label_requested"],
        bundle["review_or_comment_fields_requested"],
        bundle["state_or_merge_fields_requested"],
        bundle["ci_or_label_fields_requested"],
        static_payload["outcome_review_ci_fields_used"],
    )
    require(all(value is False for value in hidden), "blind boundary is not intact")

    all_selected = {
        item["case_id"]: item for item in selection["selection_material"]["cases"]
    }
    planned = {item["case_id"]: item for item in plan["cases"]}
    statics = {item["case_id"]: item for item in static_payload["cases"]}
    bundles = {item["case_id"]: item for item in bundle["cases"]}
    require(
        all_selected.keys() == planned.keys() == statics.keys() == bundles.keys(),
        "selection, plan, static, and bundle case sets differ",
    )
    requested_projects = set(args.project)
    known_projects = {item["project"] for item in all_selected.values()}
    require(
        requested_projects <= known_projects,
        f"unknown --project values: {sorted(requested_projects - known_projects)}",
    )
    selected = {
        case_id: item
        for case_id, item in all_selected.items()
        if not requested_projects or item["project"] in requested_projects
    }
    require(bool(selected), "project filter selected no cases")
    precedents = precedent_payload["records"] if precedent_payload is not None else []
    precedent_policy = (
        precedent_payload["retrieval_policy"] if precedent_payload is not None else None
    )
    target_ids = set(selected)
    precedent_ids = {item["precedent_id"] for item in precedents}
    require(not (target_ids & precedent_ids), "current target appears in prior precedent set")
    all_activities = (
        {item["case_id"]: item for item in activity_payload["cases"]}
        if activity_payload is not None
        else {}
    )
    all_recent_ids = {
        case_id
        for case_id, item in all_selected.items()
        if item["temporal_band"] == "recent"
    }
    if activity_payload is not None:
        if activity_payload.get("recent_cases_only", True):
            expected_activity_ids = all_recent_ids
        else:
            activity_projects = set(activity_payload.get("case_filter_projects", []))
            expected_activity_ids = {
                case_id
                for case_id, item in all_selected.items()
                if not activity_projects or item["project"] in activity_projects
            }
        require(all_activities.keys() == expected_activity_ids, "review activity case set differs")
    activities = {
        case_id: item
        for case_id, item in all_activities.items()
        if case_id in selected
    }

    bindings: dict[str, dict[str, Any]] = {
        "manifest": {
            "path": args.manifest.name,
            "file_sha256": file_sha256(args.manifest),
            "evidence_sha256": manifest["evidence_manifest_sha256"],
        },
        "static": {
            "path": args.static_evidence.name,
            "file_sha256": file_sha256(args.static_evidence),
            "evidence_sha256": static_payload["evidence_sha256"],
        },
    }
    if args.precedents is not None and precedent_payload is not None:
        bindings["precedents"] = {
            "path": args.precedents.name,
            "file_sha256": file_sha256(args.precedents),
            "evidence_sha256": precedent_payload["precedent_set_sha256"],
        }
    if args.review_activity is not None and activity_payload is not None:
        bindings["review_activity"] = {
            "path": args.review_activity.name,
            "file_sha256": file_sha256(args.review_activity),
            "evidence_sha256": activity_payload["activity_projection_sha256"],
        }
    records_by_id: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in selected}
    for index, (path, payload) in enumerate(zip(args.evidence, evidence_payloads, strict=True)):
        name = f"runtime_{index:02d}"
        bindings[name] = {
            "path": path.name,
            "file_sha256": file_sha256(path),
            "evidence_sha256": payload["evidence_sha256"],
        }
        for record_index, record in enumerate(payload["records"]):
            case_id = record["case_id"]
            require(case_id in selected, f"unknown runtime case {case_id}")
            records_by_id[case_id].append(
                {
                    "artifact": name,
                    "record_index": record_index,
                    "executor_architecture": payload.get("remote", {}).get(
                        "executor_architecture", "unknown"
                    ),
                    "record": record,
                }
            )

    frozen_at = datetime.now(UTC).isoformat()
    locks: list[dict[str, Any]] = []
    for case_id, selected_case in selected.items():
        records = records_by_id[case_id]
        require(records, f"{case_id}: no runtime record")
        preferred = records[-1]
        assessment = decision_for(
            selected_case,
            statics[case_id],
            bundles[case_id],
            preferred["record"],
            precedents=precedents,
            precedent_policy=precedent_policy,
            review_activity=activities.get(case_id),
            executor_architecture=preferred["executor_architecture"],
        )
        record_bindings = [
            {
                "artifact": item["artifact"],
                "record_index": item["record_index"],
                "returncode": item["record"].get("returncode"),
                "status": item["record"].get("status"),
                "output_sha256": item["record"].get("output_sha256"),
            }
            for item in records
        ]
        material = {
            "schema_version": "0.1",
            "policy_id": f"inference-contract-disposition-cascade-v0.1-{round_label.lower()}",
            "case_id": case_id,
            "candidate_sha256": canonical_sha256(
                {"selection": selected_case, "test_plan": planned[case_id]}
            ),
            "selection_lock_sha256": selection_sha,
            "test_plan_sha256": plan_sha,
            "source_bundle_sha256": source_sha,
            "common_evidence_binding_sha256": canonical_sha256(bindings),
            "supplemental_evidence_binding_sha256": canonical_sha256(record_bindings),
            "preferred_runtime_artifact": preferred["artifact"],
            "technical_contract": assessment["technical_contract"],
            "executor_architecture": preferred["executor_architecture"],
            "decision": assessment["decision"],
            "rationale_codes": [assessment["rationale_code"]],
            "technical_findings": assessment["technical_findings"],
            "residual_contract": assessment["residual_contract"],
            "check_observability": assessment["check_observability"],
            "hot_window_check_eligible": assessment["decision"] == "check",
            "precedent_consensus": assessment["precedent_consensus"],
            "legacy_r10_style_decision": (
                "accept_with_scope" if assessment["decision"] == "accept_with_scope" else "check"
            ),
            "frozen_at": frozen_at,
        }
        locks.append({"material": material, "lock_sha256": canonical_sha256(material)})

    counts = {
        decision: sum(lock["material"]["decision"] == decision for lock in locks)
        for decision in DECISIONS
    }
    output_material = {
        "schema_version": "0.1",
        "protocol_id": plan["protocol_id"],
        "policy_id": f"inference-contract-disposition-cascade-v0.1-{round_label.lower()}",
        "review_text_visible_during_machine_judgment": False,
        "review_activity_metadata_visible_during_machine_judgment": (
            activity_payload is not None
        ),
        "merge_outcomes_visible_during_machine_judgment": False,
        "ci_fields_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "forced_polarization_used": False,
        "terminology": "check",
        "long_running_case_gate": {
            "disposition": "abandon-time-budget-neutral",
            "candidate_failure_inferred_from_timeout": False,
        },
        "selection_lock_file_sha256": file_sha256(args.selection_lock),
        "selection_lock_sha256": selection_sha,
        "selected_case_ids": list(selected),
        "case_filter_projects": sorted(requested_projects),
        "test_plan_file_sha256": file_sha256(args.test_plan),
        "test_plan_sha256": plan_sha,
        "source_bundle_sha256": source_sha,
        "candidate_body_integrity_note": (
            "Bodies were acquired only after the plan lock and outcome-bearing blocks "
            "were removed before storage."
        ),
        "common_evidence_bindings": bindings,
        "frozen_at": frozen_at,
        "decision_counts": counts,
        "legacy_r10_style_decision_counts": {
            "accept_with_scope": sum(
                lock["material"]["legacy_r10_style_decision"] == "accept_with_scope"
                for lock in locks
            ),
            "check": sum(
                lock["material"]["legacy_r10_style_decision"] == "check" for lock in locks
            ),
        },
        "locks": locks,
    }
    output = {
        **output_material,
        "lock_set_sha256": canonical_sha256(output_material),
    }
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "round": round_label,
                "lock_set_sha256": output["lock_set_sha256"],
                "decision_counts": counts,
                "technical_contract_counts": {
                    status: sum(lock["material"]["technical_contract"] == status for lock in locks)
                    for status in ("pass", "bounded-gap", "fail")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
