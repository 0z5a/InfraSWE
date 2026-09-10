#!/usr/bin/env python3
"""Versioned, non-official E1 shadow adapter; never replaces a bulk judgment.

Live mode freezes before reveal. Replay is explicitly diagnostic and cannot
qualify a policy for release. Labels and eligibility come from the existing
sealed six-artifact chain; neither is selected by the candidate policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from freeze_training_bulk_group import DEFAULT_POLICY, _assessment, _checked, _merge_evidence
from summarize_training_bulk_campaign import _validate_group

from infraswe.pr_decision.cascade import count_accept_corrections
from infraswe.pr_decision.contracts import STRICT_95_99_99_CONTRACT, canonical_sha256
from infraswe.pr_decision.release_gate import DecisionEvaluationCase, evaluate_release_gate

PROFILE = {
    "profile_id": "v061-e1-final-head-approval-shadow-1",
    "mode": "shadow-only",
    "approval_guard": "complete-list-exact-head-no-later-negative-review",
    "otherwise": "same-whitelisted-prequential-rule-parameters",
    "score_status": "non-official",
    "formal_microscores": "unavailable",
    "calibration": "not-fitted",
    "label_kind": "upstream-disposition-and-activity-contract",
    "release_authorized": False,
}


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    result = [row["case_id"] for row in rows]
    if len(result) != len(set(result)):
        raise ValueError("duplicate case identities")
    return result


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("review time must be timezone-aware")
    return parsed


def guarded_case(case: dict[str, Any], prediction_at: datetime) -> dict[str, Any]:
    """Counterfactual E1 only; no inference about missing reviewer identities.

    The historical collector omits reviewer IDs, so withdrawal attribution is
    not provable. Conservatively decline this shortcut if any later negative
    review exists; this is why the candidate stays in shadow, not production.
    """
    result = copy.deepcopy(case)
    if case.get("acquisition_status", "acquired") != "acquired":
        return result
    reviews = case["human_non_author_reviews"]
    usable = [
        review
        for review in reviews
        if review.get("submitted_at") and _time(review["submitted_at"]) <= prediction_at
    ]
    approved = [
        review
        for review in usable
        if review["state"] == "APPROVED"
        and review.get("commit_oid") == case["head_sha"]
        and review.get("is_final_head") is True
    ]
    negatives = [
        _time(review["submitted_at"])
        for review in usable
        if review["state"] in {"CHANGES_REQUESTED", "DISMISSED"}
    ]
    approved = [
        review
        for review in approved
        if not negatives or _time(review["submitted_at"]) > max(negatives)
    ]
    allowed = bool(case.get("review_list_complete") is True and approved)
    result["human_non_author_review_state_counts"]["APPROVED"] = int(allowed)
    result["final_head_human_non_author_review_state_counts"]["APPROVED"] = int(allowed)
    return result


def candidate_prediction(
    case: dict[str, Any], baseline: dict[str, Any], policy: dict[str, Any], at: datetime
) -> tuple[str, float | None, list[str]]:
    # Never pass retrospective_projection (contains revealed labels) to the rule.
    rules = {key: copy.deepcopy(policy.get(key, value)) for key, value in DEFAULT_POLICY.items()}
    return _assessment(guarded_case(case, at), baseline["technical_contract"], rules, at)


def _write_once(path: Path, material: dict[str, Any], digest_field: str) -> dict[str, Any]:
    payload = {**material, digest_field: canonical_sha256(material)}
    if path.exists():
        if _checked(path, digest_field) != payload:
            raise ValueError(f"refusing to overwrite different sealed sidecar: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic create-if-absent, including concurrent retries. A check followed
        # by os.replace could silently overwrite another process's sealed result.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            try:
                os.link(temporary.name, path)
            except FileExistsError:
                if _checked(path, digest_field) != payload:
                    raise ValueError(f"refusing concurrent overwrite: {path}") from None
    return payload


def activate(group: Path, output: Path) -> dict[str, Any]:
    return _write_once(
        output / "activation.json",
        {
            "profile": PROFILE,
            "profile_sha256": canonical_sha256(PROFILE),
            "requested_contract": STRICT_95_99_99_CONTRACT.model_dump(mode="json"),
            "group": group.name,
            "adapter_file_sha256": "sha256:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "release_authorized": False,
        },
        "activation_sha256",
    )


def freeze(group: Path, queue_path: Path, output: Path, *, replay: bool) -> dict[str, Any]:
    input_lock = _checked(group / "input-lock.json", "group_input_sha256")
    baseline = _checked(group / "judgment-locks.json", "lock_set_sha256")
    queue = _checked(queue_path, "queue_lock_sha256")
    expected = [
        row["case_id"] for row in queue["cases"] if row["group_index"] == input_lock["group_index"]
    ]
    if not expected or _ids(input_lock["cases"]) != expected:
        raise ValueError("frozen population coverage mismatch")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate frozen population cases")
    if input_lock["queue_lock_sha256"] != queue["queue_lock_sha256"]:
        raise ValueError("queue binding mismatch")
    if baseline["group_input_sha256"] != input_lock["group_input_sha256"]:
        raise ValueError("baseline/input binding mismatch")
    locks = [row["material"] for row in baseline["locks"]]
    if _ids(locks) != expected or any(
        row["lock_sha256"] != canonical_sha256(row["material"]) for row in baseline["locks"]
    ):
        raise ValueError("baseline case lock mismatch")
    if input_lock["pull_state_or_merge_fields_requested"] is not False or any(
        baseline[key] is not False
        for key in (
            "merge_outcomes_visible_during_machine_judgment",
            "review_text_visible_during_machine_judgment",
            "ci_or_label_visible_during_machine_judgment",
        )
    ):
        raise ValueError("historical blind boundary violated")
    paths = [group / "exact-head-evidence.json"]
    if (group / "exact-head-infra-rerun.json").exists():
        paths.append(group / "exact-head-infra-rerun.json")
    if baseline["evidence_file_sha256s"] != [
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ]:
        raise ValueError("baseline/evidence binding mismatch")
    evidence_sets = [_checked(path, "evidence_sha256") for path in paths]
    if any(row["group_input_sha256"] != input_lock["group_input_sha256"] for row in evidence_sets):
        raise ValueError("evidence/input binding mismatch")
    evidence = _merge_evidence(evidence_sets)
    if set(evidence) != set(expected):
        raise ValueError("evidence coverage mismatch")
    destination = output / "shadow-lock.json"
    if (
        not replay
        and not destination.exists()
        and ((group / "outcome-reveal.json").exists() or (group / "oracle-audit.json").exists())
    ):
        raise ValueError("live shadow must freeze before reveal; use explicit diagnostic replay")
    records = []
    at = _time(baseline["frozen_at"])
    for case, lock in zip(input_lock["cases"], locks, strict=True):
        label, score, reasons = candidate_prediction(case, lock, baseline["policy"], at)
        records.append(
            {
                "case_id": case["case_id"],
                "repository": case["repository"],
                "pr_number": case["pull_number"],
                "base_sha": case["base_sha"],
                "head_sha": case["head_sha"],
                "baseline_label": lock["decision"],
                "candidate_label": label,
                "overall_score_100": score,
                "score_status": "non-official",
                "microscores": None,
                "rationale_codes": reasons,
                "neutral_abandon": evidence[case["case_id"]].get("status") == "timed_out",
            }
        )
    return _write_once(
        destination,
        {
            "schema_version": "0.6.1",
            "profile": PROFILE,
            "profile_sha256": canonical_sha256(PROFILE),
            "evaluation_track": (
                "historical_diagnostic_replay_result" if replay else "prequential_campaign_result"
            ),
            "group_index": input_lock["group_index"],
            "queue_lock_sha256": queue["queue_lock_sha256"],
            "group_input_sha256": input_lock["group_input_sha256"],
            "baseline_lock_set_sha256": baseline["lock_set_sha256"],
            "rules_sha256": canonical_sha256(
                {key: baseline["policy"].get(key, value) for key, value in DEFAULT_POLICY.items()}
            ),
            "adapter_file_sha256": "sha256:"
            + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_rule_file_sha256": "sha256:"
            + hashlib.sha256(
                Path(__file__).with_name("freeze_training_bulk_group.py").read_bytes()
            ).hexdigest(),
            "rule_evaluation_at": at.isoformat(),
            "provenance_authenticated": False,
            "release_authorized": False,
            "cases": records,
        },
        "shadow_sha256",
    )


def audit(group: Path, queue_path: Path, output: Path) -> dict[str, Any]:
    frozen = _checked(output / "shadow-lock.json", "shadow_sha256")
    queue = _checked(queue_path, "queue_lock_sha256")
    index = frozen["group_index"]
    expected = [row["case_id"] for row in queue["cases"] if row["group_index"] == index]
    previous = (
        group.parent.parent / "seed-policy.json"
        if index == 0
        else (group.parent / f"group-{index - 1:04d}" / "next-policy.json")
    )
    sealed, _, chain = _validate_group(
        group_dir=group,
        expected_case_ids=expected,
        allowed_queue_digests={queue["queue_lock_sha256"]},
        expected_policy_digest=_checked(previous, "policy_sha256")["policy_sha256"],
    )
    if _ids(frozen["cases"]) != expected or (
        frozen["baseline_lock_set_sha256"] != sealed["judgment_lock_set_sha256"]
        or frozen["group_input_sha256"] != chain["group_input_sha256"]
        or frozen["queue_lock_sha256"] != queue["queue_lock_sha256"]
    ):
        raise ValueError("shadow/sealed chain mismatch")
    contract = STRICT_95_99_99_CONTRACT.model_copy(
        update={
            "evaluation_track": frozen["evaluation_track"],
        }
    )
    eligible = [row for row in sealed["cases"] if row["oracle_eligible"]]
    predictions = {row["case_id"]: row for row in frozen["cases"]}
    gates = {}
    for name in ("baseline", "candidate"):
        rows = []
        missing = []
        for row in eligible:
            label = predictions[row["case_id"]][f"{name}_label"]
            if label not in {"accept", "check", "reject"}:
                missing.append(row["case_id"])
                continue
            rows.append(
                DecisionEvaluationCase(
                    case_id=row["case_id"],
                    predicted_label=label,
                    oracle_label=row["oracle_decision"],
                )
            )
        gate = evaluate_release_gate(rows, contract).model_dump(mode="json")
        gate["missing_prediction_case_ids"] = missing
        gate["frozen_oracle_invalid_cases"] = len(expected) - len(eligible)
        gate["complete_population_numerical_passed"] = gate["passed"] and not missing
        gates[name] = gate
    oracle = {row["case_id"]: row["oracle_decision"] for row in eligible}
    common = {
        key
        for key in oracle
        if all(
            predictions[key][f"{name}_label"] in {"accept", "check", "reject"}
            for name in ("baseline", "candidate")
        )
    }
    delta = count_accept_corrections(
        oracle_by_case={key: oracle[key] for key in common},
        old_by_case={key: predictions[key]["baseline_label"] for key in common},
        new_by_case={key: predictions[key]["candidate_label"] for key in common},
    )
    return _write_once(
        output / "shadow-audit.json",
        {
            "schema_version": "0.6.1",
            "shadow_sha256": frozen["shadow_sha256"],
            "source_audit_sha256": sealed["audit_sha256"],
            "group_index": index,
            "contract": contract.model_dump(mode="json"),
            "gates": gates,
            "accept_corrections": delta.model_dump(mode="json"),
            "paired_eligible_cases": len(common),
            "population_cases": len(expected),
            "eligibility_source": "unchanged-sealed-oracle-audit-not-policy-selected",
            "release_authorized": False,
            "promotion_status": "no-go-independent-calibration-holdout-and-provenance-required",
        },
        "shadow_audit_sha256",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["activate", "freeze", "audit"])
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--queue-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-replay", action="store_true")
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.group_dir.resolve()):
        raise SystemExit("sidecars must be outside immutable group artifacts")
    if args.action == "activate":
        print(json.dumps(activate(args.group_dir, args.output_dir)))
        return 0
    result = (
        freeze(args.group_dir, args.queue_lock, args.output_dir, replay=args.diagnostic_replay)
        if args.action == "freeze"
        else audit(args.group_dir, args.queue_lock, args.output_dir)
    )
    print(
        json.dumps({key: value for key, value in result.items() if key not in {"cases", "gates"}})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
