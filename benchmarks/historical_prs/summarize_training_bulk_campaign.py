#!/usr/bin/env python3
"""Verify and summarize all completed groups in a training PR campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from historical_bulk_quality_gates import (
    EXACT_ACCURACY_MINIMUM,
    MERGED_ACCEPT_RECALL_MINIMUM,
    exact_accuracy_gate_satisfied,
    merged_accept_recall_gate_satisfied,
    minimum_successes,
    release_quality_gate_satisfied,
)

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _checked(path: Path, digest_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    material = {key: value for key, value in payload.items() if key != digest_field}
    if payload.get(digest_field) != canonical_sha256(material):
        raise SystemExit(f"{path}: {digest_field} mismatch")
    return payload


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _case_ids(payload: dict[str, Any], field: str = "cases") -> list[str]:
    return [str(item["case_id"]) for item in payload[field]]


def _validate_group(
    *,
    group_dir: Path,
    expected_case_ids: list[str],
    allowed_queue_digests: set[str],
    expected_policy_digest: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    artifacts = {
        "input": _checked(group_dir / "input-lock.json", "group_input_sha256"),
        "evidence": _checked(group_dir / "exact-head-evidence.json", "evidence_sha256"),
        "judgment": _checked(group_dir / "judgment-locks.json", "lock_set_sha256"),
        "reveal": _checked(group_dir / "outcome-reveal.json", "reveal_sha256"),
        "audit": _checked(group_dir / "oracle-audit.json", "audit_sha256"),
        "next_policy": _checked(group_dir / "next-policy.json", "policy_sha256"),
    }
    group_index = int(artifacts["audit"]["group_index"])
    expected_directory = f"group-{group_index:04d}"
    if group_dir.name != expected_directory:
        raise SystemExit(f"{group_dir}: group index/directory mismatch")
    for name, payload in artifacts.items():
        if name == "next_policy":
            actual_group = int(payload["derived_from_group_index"])
        else:
            actual_group = int(payload["group_index"])
        if actual_group != group_index:
            raise SystemExit(f"{group_dir}/{name}: group index mismatch")

    input_lock = artifacts["input"]
    evidence = artifacts["evidence"]
    judgment = artifacts["judgment"]
    reveal = artifacts["reveal"]
    audit = artifacts["audit"]
    next_policy = artifacts["next_policy"]
    input_digest = input_lock["group_input_sha256"]
    judgment_digest = judgment["lock_set_sha256"]
    reveal_digest = reveal["reveal_sha256"]
    audit_digest = audit["audit_sha256"]

    if input_lock["queue_lock_sha256"] not in allowed_queue_digests:
        raise SystemExit(f"{group_dir}: input references an unknown queue lock")
    if evidence["group_input_sha256"] != input_digest:
        raise SystemExit(f"{group_dir}: evidence/input binding mismatch")
    if judgment["group_input_sha256"] != input_digest:
        raise SystemExit(f"{group_dir}: judgment/input binding mismatch")
    if reveal["group_input_sha256"] != input_digest:
        raise SystemExit(f"{group_dir}: reveal/input binding mismatch")
    if reveal["judgment_lock_set_sha256"] != judgment_digest:
        raise SystemExit(f"{group_dir}: reveal/judgment binding mismatch")
    if audit["judgment_lock_set_sha256"] != judgment_digest:
        raise SystemExit(f"{group_dir}: audit/judgment binding mismatch")
    if audit["reveal_sha256"] != reveal_digest:
        raise SystemExit(f"{group_dir}: audit/reveal binding mismatch")
    if next_policy["source_audit_sha256"] != audit_digest:
        raise SystemExit(f"{group_dir}: next-policy/audit binding mismatch")
    if next_policy["source_reveal_sha256"] != reveal_digest:
        raise SystemExit(f"{group_dir}: next-policy/reveal binding mismatch")

    evidence_paths = [group_dir / "exact-head-evidence.json"]
    infra_rerun = group_dir / "exact-head-infra-rerun.json"
    if infra_rerun.exists():
        rerun = _checked(infra_rerun, "evidence_sha256")
        if rerun["group_input_sha256"] != input_digest:
            raise SystemExit(f"{group_dir}: rerun/input binding mismatch")
        evidence_paths.append(infra_rerun)
    evidence_file_digests = [_file_sha256(path) for path in evidence_paths]
    if judgment["evidence_file_sha256s"] != evidence_file_digests:
        raise SystemExit(f"{group_dir}: judgment/evidence file binding mismatch")

    lock_case_ids: list[str] = []
    for lock in judgment["locks"]:
        if lock["lock_sha256"] != canonical_sha256(lock["material"]):
            raise SystemExit(f"{group_dir}: case judgment digest mismatch")
        lock_case_ids.append(str(lock["material"]["case_id"]))
    projection_case_ids = _case_ids(next_policy["retrospective_projection"])
    artifact_case_ids = {
        "input": _case_ids(input_lock),
        "evidence": _case_ids(evidence, "records"),
        "judgment": lock_case_ids,
        "reveal": _case_ids(reveal),
        "audit": _case_ids(audit),
        "next_policy": projection_case_ids,
    }
    for name, case_ids in artifact_case_ids.items():
        if case_ids != expected_case_ids:
            raise SystemExit(f"{group_dir}/{name}: case ordering/coverage mismatch")

    if expected_policy_digest is not None:
        embedded_policy = judgment["policy"]
        if embedded_policy.get("policy_sha256") != expected_policy_digest:
            raise SystemExit(f"{group_dir}: policy chain digest mismatch")
        embedded_material = {
            key: value for key, value in embedded_policy.items() if key != "policy_sha256"
        }
        if canonical_sha256(embedded_material) != expected_policy_digest:
            raise SystemExit(f"{group_dir}: embedded policy digest mismatch")
    if input_lock["pull_state_or_merge_fields_requested"] is not False:
        raise SystemExit(f"{group_dir}: input outcome-blind boundary violated")
    if judgment["merge_outcomes_visible_during_machine_judgment"] is not False:
        raise SystemExit(f"{group_dir}: judgment outcome-blind boundary violated")
    if reveal["revealed_after_lock"] is not True:
        raise SystemExit(f"{group_dir}: reveal ordering boundary violated")

    chain = {
        "group_index": group_index,
        "case_count": len(expected_case_ids),
        "queue_lock_sha256": input_lock["queue_lock_sha256"],
        "group_input_sha256": input_digest,
        "evidence_sha256": evidence["evidence_sha256"],
        "lock_set_sha256": judgment_digest,
        "reveal_sha256": reveal_digest,
        "audit_sha256": audit_digest,
        "next_policy_sha256": next_policy["policy_sha256"],
    }
    return audit, next_policy["policy_sha256"], chain


def main() -> int:
    default_coverage_queue = os.environ.get(
        "INFRASWE_TRAINING_QUEUE_LOCK", os.environ.get("INFRASWE_INFERENCE_QUEUE_LOCK")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument(
        "--coverage-queue",
        type=Path,
        default=Path(default_coverage_queue) if default_coverage_queue else None,
        required=default_coverage_queue is None,
    )
    parser.add_argument("--queue-lock", type=Path, action="append", default=[])
    parser.add_argument("--initial-policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.coverage_queue is None:
        raise SystemExit("coverage queue is required")
    coverage_queue = _checked(args.coverage_queue, "queue_lock_sha256")
    queue_lock_paths = args.queue_lock or sorted(args.result_root.glob("queue-lock*.json"))
    queue_locks = [coverage_queue] + [
        _checked(path, "queue_lock_sha256") for path in queue_lock_paths
    ]
    allowed_queue_digests = {queue["queue_lock_sha256"] for queue in queue_locks}
    coverage_case_ids = _case_ids(coverage_queue)
    if len(coverage_case_ids) != len(set(coverage_case_ids)):
        raise SystemExit("duplicate cases in coverage queue")
    if len(coverage_case_ids) != args.expected_cases:
        raise SystemExit(
            f"coverage queue has {len(coverage_case_ids)} cases, expected {args.expected_cases}"
        )
    coverage_by_group: dict[int, list[str]] = {}
    for case in coverage_queue["cases"]:
        coverage_by_group.setdefault(int(case["group_index"]), []).append(str(case["case_id"]))

    audit_paths = sorted((args.result_root / "groups").glob("group-*/oracle-audit.json"))
    audit_group_indexes = [int(path.parent.name.removeprefix("group-")) for path in audit_paths]
    if audit_group_indexes != sorted(coverage_by_group):
        raise SystemExit("completed audit groups do not exactly cover the frozen queue")
    initial_policy_path = args.initial_policy
    if initial_policy_path is None and (args.result_root / "seed-policy.json").exists():
        initial_policy_path = args.result_root / "seed-policy.json"
    expected_policy_digest = (
        _checked(initial_policy_path, "policy_sha256")["policy_sha256"]
        if initial_policy_path
        else None
    )
    audits: list[dict[str, Any]] = []
    artifact_chains: list[dict[str, Any]] = []
    for path in audit_paths:
        group_index = int(path.parent.name.removeprefix("group-"))
        audit, expected_policy_digest, chain = _validate_group(
            group_dir=path.parent,
            expected_case_ids=coverage_by_group[group_index],
            allowed_queue_digests=allowed_queue_digests,
            expected_policy_digest=expected_policy_digest,
        )
        audits.append(audit)
        artifact_chains.append(chain)
    case_ids = [case["case_id"] for audit in audits for case in audit["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("duplicate cases across campaign audits")
    if len(case_ids) != args.expected_cases:
        raise SystemExit(
            f"campaign has {len(case_ids)} audited cases, expected {args.expected_cases}"
        )

    eligible = sum(
        int(audit["summary"].get("eligible_cases", audit["summary"]["cases"])) for audit in audits
    )
    invalid = len(case_ids) - eligible
    exact = sum(int(audit["summary"]["exact_label_matches"]) for audit in audits)
    legacy_exact = sum(int(audit["summary"]["legacy_exact_label_matches"]) for audit in audits)
    binary = sum(int(audit["summary"]["binary_direction_matches"]) for audit in audits)
    legacy_binary = sum(
        int(audit["summary"]["legacy_binary_direction_matches"]) for audit in audits
    )
    merged = sum(int(audit["summary"]["merged_cases"]) for audit in audits)
    merged_accepts = sum(int(audit["summary"]["merged_machine_accepts"]) for audit in audits)
    reject_predictions = sum(
        int(audit["summary"]["machine_reject_predictions"]) for audit in audits
    )
    reject_correct = sum(int(audit["summary"]["machine_reject_correct"]) for audit in audits)
    exact_gate_satisfied = exact_accuracy_gate_satisfied(
        exact_matches=exact,
        eligible_cases=eligible,
    )
    merged_recall_gate_satisfied = merged_accept_recall_gate_satisfied(
        merged_accepts=merged_accepts,
        merged_cases=merged,
    )
    release_gate_satisfied = release_quality_gate_satisfied(
        exact_matches=exact,
        eligible_cases=eligible,
        merged_accepts=merged_accepts,
        merged_cases=merged,
    )
    material = {
        "schema_version": "0.1",
        "protocol_id": (
            "inference-bulk-campaign-summary-v0.1"
            if audits and audits[0]["protocol_id"].startswith("inference-")
            else "training-bulk-campaign-summary-v0.1"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "group_count": len(audits),
        "case_count": len(case_ids),
        "eligible_case_count": eligible,
        "invalid_case_count": invalid,
        "exact_matches": exact,
        "exact_accuracy": exact / eligible if eligible else None,
        "exact_accuracy_minimum": EXACT_ACCURACY_MINIMUM,
        "exact_accuracy_required_matches": minimum_successes(
            eligible, EXACT_ACCURACY_MINIMUM
        ),
        "exact_accuracy_gate_satisfied": exact_gate_satisfied,
        "legacy_exact_matches": legacy_exact,
        "legacy_exact_accuracy": legacy_exact / eligible if eligible else None,
        "exact_accuracy_gain": (exact - legacy_exact) / eligible if eligible else None,
        "binary_matches": binary,
        "binary_accuracy": binary / eligible if eligible else None,
        "legacy_binary_matches": legacy_binary,
        "legacy_binary_accuracy": legacy_binary / eligible if eligible else None,
        "merged_cases": merged,
        "merged_machine_accepts": merged_accepts,
        "merged_accept_recall": merged_accepts / merged if merged else None,
        "merged_accept_recall_minimum": MERGED_ACCEPT_RECALL_MINIMUM,
        "merged_accept_recall_required_accepts": minimum_successes(
            merged, MERGED_ACCEPT_RECALL_MINIMUM
        ),
        "merged_accept_recall_gate_satisfied": merged_recall_gate_satisfied,
        "release_quality_gate_satisfied": release_gate_satisfied,
        "machine_reject_predictions": reject_predictions,
        "machine_reject_correct": reject_correct,
        "machine_reject_precision": (
            reject_correct / reject_predictions if reject_predictions else None
        ),
        "target_metric_improved": exact > legacy_exact,
        "full_artifact_chain_verified": True,
        "outcome_blind_boundary_verified": True,
        "policy_chain_verified": True,
        "coverage_queue_lock_sha256": coverage_queue["queue_lock_sha256"],
        "allowed_queue_lock_sha256s": sorted(allowed_queue_digests),
        "audit_sha256s": [audit["audit_sha256"] for audit in audits],
        "group_artifact_chains": artifact_chains,
    }
    payload = {**material, "summary_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
