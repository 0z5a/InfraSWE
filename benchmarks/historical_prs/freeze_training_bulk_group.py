#!/usr/bin/env python3
"""Freeze decisions for one outcome-blind training bulk group."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.policy import (
    OVERALL_SCORE_ACCEPT_ABOVE_100,
    OVERALL_SCORE_REJECT_BELOW_100,
    overall_score_decision_band,
)

DEFAULT_POLICY = {
    "policy_id": "training-bulk-disposition-v0.1-g0000",
    "recent_pr_max_age_days": 30,
    "active_review_max_idle_days": 14,
    "maintainer_associations": ["COLLABORATOR", "MEMBER", "OWNER"],
    "small_change_max_lines": 120,
    "small_compile_accept_enabled": True,
    "explicit_revert_accept_enabled": False,
    "active_final_head_review_priority": False,
    "maintainer_precedes_review_without_approval": False,
    "maintainer_requires_runtime_source": False,
    "mature_review_without_approval_is_reject": True,
    "uncertain_disposition": "reject",
    "check_requires_recent_final_head_human_review": True,
    "merged_recall_guard_projects": [],
    "merged_recall_guard_max_changed_lines": None,
    "merged_recall_guard_author_associations": [],
    "merged_recall_guard_review_modes": [],
    "structural_reject_rules": [],
}
READINESS_RE = re.compile(
    r"(?:^\s*\[draft\]|^\s*draft\s*[-:]|\bwip\b|work in progress|do not merge)",
    re.IGNORECASE,
)
REVERT_RE = re.compile(r"^\s*(?:revert\b|\[revert\])", re.IGNORECASE)
PYTHON_FAILURE_MARKERS = (
    "SyntaxError:",
    "IndentationError:",
    "TabError:",
    "PyCompileError",
)
ENVIRONMENT_MARKERS = (
    "ModuleNotFoundError",
    "ImportError while importing",
    "cannot import name",
    "No such file or directory: '/opt/data/",
    "world size (1) is not divisible",
    "ConnectionError",
    "Connection refused",
    "LocalEntryNotFoundError",
    "OfflineModeIsEnabled",
    "ProxyError",
    "MaxRetryError",
    "outgoing traffic has been disabled",
    "not found in the cached files",
    "CUDA out of memory",
)
SUMMARY_RE = re.compile(r"(?P<count>\d+) (?P<label>passed|failed|errors?)")
RATIONALE_SCORE_100 = {
    "SELF_DECLARED_NOT_READY": 10.0,
    "EXACT_CANDIDATE_CONTRACT_FAILED": 5.0,
    "ACTIVE_RECENT_FINAL_HEAD_REVIEW": 57.5,
    "EXPLICIT_REVERT_WITHOUT_HARD_FAILURE": 74.0,
    "HUMAN_NON_AUTHOR_APPROVAL": 92.0,
    "MAINTAINER_RUNTIME_SOURCE_CHANGE": 84.0,
    "CUMULATIVE_PROJECT_AUTHOR_REJECT_GUARD": 35.0,
    "MERGED_RECALL_GUARD_PROJECT_SCOPE": 66.0,
    "REVIEW_WITHOUT_APPROVAL": 45.0,
    "MAINTAINER_AUTHORED_NO_HARD_FAILURE": 78.0,
    "CANDIDATE_TEST_OR_COMPILE_CONTRACT_CLOSED": 82.0,
    "SMALL_COMPILE_CLOSED_SOURCE_CHANGE": 70.0,
    "UNRESOLVED_MATURE_OR_UNREVIEWED_GAP": {
        "accept": 66.0,
        "reject": 40.0,
    },
}


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


def _usable(record: dict[str, Any]) -> bool:
    return record.get("returncode") not in {None, 255} and record.get("status") not in {
        "prewarm_failed",
        "transport_timeout",
        "checkout_failed",
        "checkout_timeout",
    }


def _merge_evidence(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for record in payload["records"]:
            prior = records.get(record["case_id"])
            if prior is None or (not _usable(prior) and _usable(record)):
                records[record["case_id"]] = record
    return records


def _technical(record: dict[str, Any]) -> tuple[str, list[str]]:
    output = str(record.get("output_tail") or "")
    returncode = record.get("returncode")
    test_paths = record.get("test_paths") or []
    compile_paths = record.get("compile_paths") or []
    if not _usable(record):
        return "bounded-gap", ["INFRASTRUCTURE_EVIDENCE_UNAVAILABLE"]
    if returncode == 0:
        summaries = {
            match.group("label"): int(match.group("count")) for match in SUMMARY_RE.finditer(output)
        }
        if test_paths and summaries.get("passed", 0) > 0:
            return "test-pass", ["CANDIDATE_TESTS_PASSED"]
        if compile_paths:
            return "compile-pass", ["CHANGED_PYTHON_COMPILED"]
        return "bounded-gap", ["NO_CHANGED_PYTHON_OR_CANDIDATE_TEST"]
    if any(marker in output for marker in PYTHON_FAILURE_MARKERS):
        return "fail", ["CHANGED_PYTHON_SYNTAX_FAILED"]
    if (
        returncode == 1
        and "AssertionError" in output
        and "1 failed" in output
        and not any(marker in output for marker in ENVIRONMENT_MARKERS)
    ):
        return "fail", ["CANDIDATE_TEST_ASSERTION_FAILED"]
    return "bounded-gap", ["TEST_ENVIRONMENT_OR_COLLECTION_GAP"]


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".py") and (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith("_test.py")
    )


def _source_paths(case: dict[str, Any]) -> list[str]:
    return [
        item["path"]
        for item in case["files"]
        if item["change_type"] != "deleted"
        and not _is_test_path(item["path"])
        and not item["path"].lower().endswith((".md", ".rst", ".txt"))
        and not item["path"].startswith(("docs/", "doc/"))
    ]


def _runtime_source_paths(case: dict[str, Any]) -> list[str]:
    prefixes = {
        "megatron-core": ("megatron/", "experimental/lite/megatron/"),
        "slime": ("slime/",),
        "verl": ("verl/",),
        "verl-omni": ("verl/",),
        "flashinfer": ("flashinfer/", "python/flashinfer/", "include/", "csrc/"),
        "sglang": ("python/sglang/", "sgl-kernel/"),
        "tensorrt-llm": ("tensorrt_llm/", "cpp/", "include/"),
        "vllm": ("vllm/", "csrc/"),
        "nccl": ("src/", "include/", "device/"),
        "rccl": ("src/", "include/", "tools/"),
        "nvshmem": ("src/", "include/"),
        "uccl": ("src/", "include/", "collectives/"),
        "ucx": ("src/",),
        "ucc": ("src/",),
        "pytorch": ("torch/", "aten/", "c10/"),
    }[case["project"]]
    return [path for path in _source_paths(case) if path.startswith(prefixes)]


def _review_times(case: dict[str, Any], *, final_head: bool) -> list[datetime]:
    return [
        datetime.fromisoformat(review["submitted_at"].replace("Z", "+00:00"))
        for review in case["human_non_author_reviews"]
        if review["submitted_at"] and (not final_head or review["is_final_head"])
    ]


def _merged_recall_guard_applies(
    case: dict[str, Any],
    policy: dict[str, Any],
    *,
    changed_lines: int,
) -> bool:
    """Return whether a narrow, outcome-blind recall repair covers this PR."""
    projects = set(policy.get("merged_recall_guard_projects") or [])
    associations = set(policy.get("merged_recall_guard_author_associations") or [])
    review_modes = set(policy.get("merged_recall_guard_review_modes") or [])
    review_mode = "reviewed" if case["human_non_author_reviews"] else "unreviewed"
    max_changed_lines = policy.get("merged_recall_guard_max_changed_lines")
    return (
        case["project"] in projects
        and case["pr_author_association"] in associations
        and review_mode in review_modes
        and (max_changed_lines is None or changed_lines <= int(max_changed_lines))
    )


def _structural_reject_guard_applies(
    case: dict[str, Any],
    policy: dict[str, Any],
) -> bool:
    """Apply an outcome-blind project/author rule learned on prior sealed groups."""

    for rule in policy.get("structural_reject_rules") or []:
        projects = set(rule.get("projects") or [])
        associations = set(rule.get("author_associations") or [])
        if (
            projects
            and associations
            and case["project"] in projects
            and case["pr_author_association"] in associations
        ):
            return True
    return False


def _current_decision(value: str) -> str:
    """Normalize legacy scoped-accept labels into the current three-class vocabulary."""

    return "accept" if value == "accept_with_scope" else value


def _rule_decision(
    case: dict[str, Any],
    technical: str,
    policy: dict[str, Any],
    frozen_at: datetime,
) -> tuple[str, list[str]]:
    if case.get("acquisition_status", "acquired") != "acquired":
        failure_code = case.get("acquisition_failure_code", "GITHUB_METADATA_UNAVAILABLE")
        return "unresolved", [f"INVALID_INPUT_{failure_code}"]
    title = str(case["title"])
    age_days = (
        frozen_at - datetime.fromisoformat(str(case["created_at"]).replace("Z", "+00:00"))
    ).total_seconds() / 86_400
    final_head_times = _review_times(case, final_head=True)
    final_counts = case["final_head_human_non_author_review_state_counts"]
    all_counts = case["human_non_author_review_state_counts"]
    final_head_activity = sum(final_counts.values()) > 0
    final_head_unapproved_activity = final_head_activity and final_counts["APPROVED"] == 0
    review_idle_days = (
        (frozen_at - max(final_head_times)).total_seconds() / 86_400 if final_head_times else None
    )
    source_paths = _source_paths(case)
    runtime_source_paths = _runtime_source_paths(case)
    candidate_test_path = any(_is_test_path(item["path"]) for item in case["files"])
    changed_lines = int(case["additions"]) + int(case["deletions"])

    if READINESS_RE.search(title):
        return "reject", ["SELF_DECLARED_NOT_READY"]
    if technical == "fail":
        return "reject", ["EXACT_CANDIDATE_CONTRACT_FAILED"]
    if (
        age_days <= int(policy["recent_pr_max_age_days"])
        and (
            final_head_activity
            if policy.get("active_final_head_review_priority", False)
            else final_head_unapproved_activity
        )
        and review_idle_days is not None
        and review_idle_days <= int(policy["active_review_max_idle_days"])
    ):
        return "check", ["ACTIVE_RECENT_FINAL_HEAD_REVIEW"]
    if policy.get("explicit_revert_accept_enabled", False) and REVERT_RE.search(title):
        return "accept", ["EXPLICIT_REVERT_WITHOUT_HARD_FAILURE"]
    if all_counts["APPROVED"] > 0:
        return "accept", ["HUMAN_NON_AUTHOR_APPROVAL"]
    maintainer_authored = case["pr_author_association"] in policy["maintainer_associations"]
    maintainer_scope_closed = maintainer_authored and (
        runtime_source_paths or not policy.get("maintainer_requires_runtime_source", False)
    )
    if policy.get("maintainer_precedes_review_without_approval", False) and maintainer_scope_closed:
        return "accept", ["MAINTAINER_RUNTIME_SOURCE_CHANGE"]
    if _structural_reject_guard_applies(case, policy):
        return "reject", ["CUMULATIVE_PROJECT_AUTHOR_REJECT_GUARD"]
    if _merged_recall_guard_applies(case, policy, changed_lines=changed_lines):
        return "accept", ["MERGED_RECALL_GUARD_PROJECT_SCOPE"]
    if case["human_non_author_reviews"]:
        return "reject", ["REVIEW_WITHOUT_APPROVAL"]
    if maintainer_scope_closed:
        return "accept", ["MAINTAINER_AUTHORED_NO_HARD_FAILURE"]
    if technical in {"test-pass", "compile-pass"} and candidate_test_path:
        return "accept", ["CANDIDATE_TEST_OR_COMPILE_CONTRACT_CLOSED"]
    if (
        policy.get("small_compile_accept_enabled", True)
        and technical == "compile-pass"
        and source_paths
        and changed_lines <= int(policy["small_change_max_lines"])
    ):
        return "accept", ["SMALL_COMPILE_CLOSED_SOURCE_CHANGE"]
    return _current_decision(str(policy["uncertain_disposition"])), [
        "UNRESOLVED_MATURE_OR_UNREVIEWED_GAP"
    ]


def _assessment(
    case: dict[str, Any],
    technical: str,
    policy: dict[str, Any],
    frozen_at: datetime,
) -> tuple[str, float | None, list[str]]:
    """Score outcome-free evidence, then derive the disposition from fixed bands."""

    expected_decision, rationale = _rule_decision(case, technical, policy, frozen_at)
    if expected_decision == "unresolved":
        return expected_decision, None, rationale
    score_spec = RATIONALE_SCORE_100[rationale[0]]
    score = (
        float(score_spec[expected_decision]) if isinstance(score_spec, dict) else float(score_spec)
    )
    band = overall_score_decision_band(score)
    decision = band
    if decision != expected_decision:
        raise ValueError(
            f"score/disposition invariant failed for {rationale[0]}: "
            f"score={score} expected={expected_decision} derived={decision}"
        )
    return decision, score, rationale


def _decision(
    case: dict[str, Any],
    technical: str,
    policy: dict[str, Any],
    frozen_at: datetime,
) -> tuple[str, list[str]]:
    decision, _, rationale = _assessment(case, technical, policy, frozen_at)
    return decision, rationale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_lock = _checked(args.input_lock, "group_input_sha256")
    evidence_payloads = [_checked(path, "evidence_sha256") for path in args.evidence]
    if any(
        payload["group_input_sha256"] != input_lock["group_input_sha256"]
        for payload in evidence_payloads
    ):
        raise SystemExit("evidence/input lock binding mismatch")
    policy = _checked(args.policy, "policy_sha256") if args.policy else DEFAULT_POLICY
    evidence = _merge_evidence(evidence_payloads)
    frozen_at = datetime.now(UTC)
    locks: list[dict[str, Any]] = []
    for case in input_lock["cases"]:
        record = evidence[case["case_id"]]
        technical, technical_reasons = _technical(record)
        decision, overall_score_100, rationale = _assessment(case, technical, policy, frozen_at)
        if case.get("acquisition_status", "acquired") != "acquired":
            legacy = "unresolved"
        else:
            legacy = (
                "accept_with_scope"
                if case["human_non_author_review_state_counts"]["APPROVED"] > 0
                or case["pr_author_association"] in policy["maintainer_associations"]
                or technical == "test-pass"
                else "check"
            )
        lock_material = {
            "schema_version": "0.3",
            "case_id": case["case_id"],
            "group_input_sha256": input_lock["group_input_sha256"],
            "policy_id": policy["policy_id"],
            "decision": decision,
            "acceptance_scope": "limited" if decision == "accept" else "not-applicable",
            "overall_score_100": overall_score_100,
            "overall_score_role": "historical-offline-evaluation-with-fixed-disposition-bands",
            "formal_infraswe_result_issued": False,
            "official_microscores_issued": False,
            "legacy_decision": legacy,
            "technical_contract": technical,
            "technical_reasons": technical_reasons,
            "rationale_codes": rationale,
            "evidence_case_output_sha256": record.get("output_sha256"),
            "evidence_case_status": record.get("status"),
            "evidence_case_returncode": record.get("returncode"),
            "frozen_at": frozen_at.isoformat(),
        }
        locks.append(
            {
                "material": lock_material,
                "lock_sha256": canonical_sha256(lock_material),
            }
        )
    output_material = {
        "schema_version": "0.3",
        "protocol_id": (f"{input_lock.get('profile', 'training')}-bulk-group-judgment-lock-v0.3"),
        "group_index": input_lock["group_index"],
        "group_input_sha256": input_lock["group_input_sha256"],
        "evidence_file_sha256s": [
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() for path in args.evidence
        ],
        "policy": policy,
        "merge_outcomes_visible_during_machine_judgment": False,
        "review_text_visible_during_machine_judgment": False,
        "ci_or_label_visible_during_machine_judgment": False,
        "forced_polarization_used": False,
        "formal_infraswe_result_issued": False,
        "historical_score_limitation": (
            "coarse outcome-blind historical evidence cannot issue official ProjectFit or "
            "BenchmarkTrust microscores"
        ),
        "overall_score_band_policy": {
            "reject_below_100": OVERALL_SCORE_REJECT_BELOW_100,
            "check_minimum_100": OVERALL_SCORE_REJECT_BELOW_100,
            "check_maximum_100": OVERALL_SCORE_ACCEPT_ABOVE_100,
            "accept_above_100": OVERALL_SCORE_ACCEPT_ABOVE_100,
            "accept_band_score_role": "evaluation-only",
        },
        "frozen_at": frozen_at.isoformat(),
        "decision_counts": {
            decision: sum(lock["material"]["decision"] == decision for lock in locks)
            for decision in ("accept", "check", "reject", "unresolved")
        },
        "technical_contract_counts": {
            technical: sum(lock["material"]["technical_contract"] == technical for lock in locks)
            for technical in ("test-pass", "compile-pass", "fail", "bounded-gap")
        },
        "locks": locks,
    }
    payload = {**output_material, "lock_set_sha256": canonical_sha256(output_material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "decision_counts": payload["decision_counts"],
                "technical_contract_counts": payload["technical_contract_counts"],
                "lock_set_sha256": payload["lock_set_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
