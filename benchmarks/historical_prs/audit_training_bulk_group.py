#!/usr/bin/env python3
"""Audit one frozen training bulk group against its revealed oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


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


def _label(value: str) -> str:
    return "accept" if value == "accept_with_scope" else value


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment-locks", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    judgment = _checked(args.judgment_locks, "lock_set_sha256")
    reveal = _checked(args.reveal, "reveal_sha256")
    if reveal["judgment_lock_set_sha256"] != judgment["lock_set_sha256"]:
        raise SystemExit("reveal/judgment binding mismatch")
    rows: list[dict[str, Any]] = []
    for case in reveal["cases"]:
        machine = _label(case["machine_decision"])
        legacy = _label(case["legacy_decision"])
        oracle = case["oracle_decision"]
        oracle_eligible = case.get("outcome", {}).get(
            "availability", "available"
        ) == "available" and oracle in {"accept", "check", "reject"}
        machine_binary = "accept" if machine == "accept" else "reject"
        legacy_binary = "accept" if legacy == "accept" else "reject"
        oracle_binary = "accept" if oracle == "accept" else "reject"
        rows.append(
            {
                "case_id": case["case_id"],
                "machine_decision": machine,
                "legacy_decision": legacy,
                "oracle_decision": oracle,
                "oracle_eligible": oracle_eligible,
                "oracle_failure_code": case.get("outcome", {}).get("failure_code"),
                "exact_match": oracle_eligible and machine == oracle,
                "legacy_exact_match": oracle_eligible and legacy == oracle,
                "binary_match": oracle_eligible and machine_binary == oracle_binary,
                "legacy_binary_match": oracle_eligible and legacy_binary == oracle_binary,
                "machine_rationale_codes": case["machine_rationale_codes"],
                "technical_contract": case["technical_contract"],
            }
        )
    case_count = len(rows)
    eligible_count = sum(row["oracle_eligible"] for row in rows)
    exact = sum(row["exact_match"] for row in rows)
    legacy_exact = sum(row["legacy_exact_match"] for row in rows)
    binary = sum(row["binary_match"] for row in rows)
    legacy_binary = sum(row["legacy_binary_match"] for row in rows)
    machine_rejects = [
        row for row in rows if row["oracle_eligible"] and row["machine_decision"] == "reject"
    ]
    machine_checks = [
        row for row in rows if row["oracle_eligible"] and row["machine_decision"] == "check"
    ]
    merged = [case for case in reveal["cases"] if case["outcome"]["merged"]]
    material = {
        "schema_version": "0.1",
        "protocol_id": (
            f"{judgment.get('policy', {}).get('domain', 'training')}-bulk-group-oracle-audit-v0.1"
        ),
        "group_index": judgment["group_index"],
        "judgment_lock_set_sha256": judgment["lock_set_sha256"],
        "reveal_sha256": reveal["reveal_sha256"],
        "summary": {
            "cases": case_count,
            "eligible_cases": eligible_count,
            "invalid_cases": case_count - eligible_count,
            "exact_label_matches": exact,
            "exact_accuracy": _ratio(exact, eligible_count),
            "binary_direction_matches": binary,
            "binary_accuracy": _ratio(binary, eligible_count),
            "legacy_exact_label_matches": legacy_exact,
            "legacy_exact_accuracy": _ratio(legacy_exact, eligible_count),
            "legacy_binary_direction_matches": legacy_binary,
            "legacy_binary_accuracy": _ratio(legacy_binary, eligible_count),
            "same_cohort_exact_accuracy_gain": _ratio(exact - legacy_exact, eligible_count),
            "machine_reject_predictions": len(machine_rejects),
            "machine_reject_correct": sum(
                row["oracle_decision"] == "reject" for row in machine_rejects
            ),
            "machine_reject_precision": _ratio(
                sum(row["oracle_decision"] == "reject" for row in machine_rejects),
                len(machine_rejects),
            ),
            "machine_check_predictions": len(machine_checks),
            "machine_check_correct": sum(
                row["oracle_decision"] == "check" for row in machine_checks
            ),
            "machine_check_precision": _ratio(
                sum(row["oracle_decision"] == "check" for row in machine_checks),
                len(machine_checks),
            ),
            "merged_cases": len(merged),
            "merged_machine_accepts": sum(
                _label(case["machine_decision"]) == "accept" for case in merged
            ),
            "target_metric_improved": exact > legacy_exact,
        },
        "errors": [row for row in rows if row["oracle_eligible"] and not row["exact_match"]],
        "invalid_cases": [row for row in rows if not row["oracle_eligible"]],
        "cases": rows,
    }
    payload = {**material, "audit_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"audit_sha256={payload['audit_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
