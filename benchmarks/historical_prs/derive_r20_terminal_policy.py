#!/usr/bin/env python3
# ruff: noqa: E501
"""Derive the terminal policy recommendation from the locked R20 audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_AUDIT_SHA256 = "sha256:8c261986bb3aef08ab6ca3d8c94f38c0ba7b1b272c144263b672eea77c9bd414"
EXPECTED_LOCK_SHA256 = "sha256:cf8deab1e41bf825f98d23ee0715927b42fff5128a25a7a7f2f5ef6a3b939bb0"


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = read(args.audit)
    if audit["audit_sha256"] != canonical_sha256({key: value for key, value in audit.items() if key != "audit_sha256"}):
        raise SystemExit("R20 audit digest mismatch")
    if audit["audit_sha256"] != EXPECTED_AUDIT_SHA256:
        raise SystemExit("R20 audit identity changed")
    locks = read(args.audit.parent / "machine-judgment-locks.json")
    if audit["source_digests"]["machine_judgment_locks"] != canonical_sha256(locks):
        raise SystemExit("R20 audit/lock artifact binding mismatch")
    if locks["lock_set_sha256"] != EXPECTED_LOCK_SHA256:
        raise SystemExit("R20 lock-set identity changed")
    summary = audit["summary"]
    if not summary["target_check_reject_metric_improved"]:
        raise SystemExit("R20 did not meet the user-gated improvement condition")

    recommendation = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-terminal-policy-after-r20-v0.1",
        "derived_after_r20_reveal": True,
        "retrospective_r20_locks_changed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_audit_sha256": audit["audit_sha256"],
        "source_lock_set_sha256": locks["lock_set_sha256"],
        "observed_metrics": {
            key: summary[key]
            for key in (
                "cases",
                "exact_label_matches",
                "legacy_exact_label_matches",
                "same_cohort_exact_accuracy_gain",
                "binary_direction_matches",
                "frozen_nonaccept_exact_matches",
                "legacy_frozen_nonaccept_exact_matches",
                "machine_reject_precision",
                "machine_check_precision",
                "merged_cases",
                "merged_machine_accepts",
                "oracle_decisions",
            )
        },
        "workload_ledger": {
            "requested_after_r13": {"communication": 50, "training": 50, "inference": 100},
            "completed_through_r20": {"communication": 50, "training": 50, "inference": 100},
            "remaining_after_r20": {"communication": 0, "training": 0, "inference": 0},
            "inference_project_totals": {"vllm": 25, "sglang": 25, "tensorrt_llm": 25, "flashinfer": 25},
        },
        "terminal_rules": [
            {
                "id": "POST-R20-CHECK-NEEDS-INDEPENDENT-HUMAN-ACTIVITY",
                "rule": "Use check only when pre-lock allowed evidence itself contains a named, non-author human review or handoff event tied to the final head. Bot-generated QA summaries, author-written reviewer notes, and generic `needs follow-up` prose are not check evidence.",
                "evidence": [
                    "FlashInfer #4879 was the only oracle-check and had ten post-lock human feedback items, which the blind evidence could not see.",
                    "TensorRT-LLM #18600 contained a bot-generated QA follow-up in its body but had zero revealed human feedback and was oracle-reject.",
                ],
            },
            {
                "id": "POST-R20-RECENT-BLIND-DEFAULT-REJECT",
                "rule": "When independent human activity is unavailable under the blind protocol, classify a recent technically complete PR as reject for historical disposition and retain its technical pass separately; do not synthesize check from quality alone.",
                "evidence": [
                    "Three of four recent R20 cases were oracle-reject; the remaining check case was distinguishable only by hidden review activity.",
                ],
            },
            {
                "id": "POST-R20-EXACT-CANDIDATE-FAILURE-DOMINATES",
                "rule": "A reproducible candidate-owned assertion failure at the changed boundary rejects immediately, even if sibling cases pass or the implementation looks locally plausible.",
                "evidence": [
                    "SGLang #27257 passed four selected cases but its own parallel-expand case raised an exact IndexError and was correctly predicted reject.",
                ],
            },
            {
                "id": "POST-R20-INDEPENDENT-NARROW-PROBE",
                "rule": "For a mature PR without candidate test changes, require either a target integration route or an independently executable narrow invariant before technical accept; store author receipts only as corroboration.",
                "evidence": [
                    "Source-extracted TensorRT-LLM #14869 matched the strided tensor reference and merged; FlashInfer #3461 passed 18 existing alignment cases and merged.",
                    "Technically passing FlashInfer #3506 and vLLM #44475/#44526 still remained inactive-open, proving that technical closure must not be presented as disposition certainty.",
                ],
            },
            {
                "id": "POST-R20-TEST-LANGUAGE-AND-ROUTE-NEUTRAL",
                "rule": "Continue treating C++, integration-list, waiver removal, API-schema, and existing-matrix evidence as first-class. Local generated-binding failures are capability gaps unless a candidate assertion is reached.",
                "evidence": [
                    "All four mature TensorRT-LLM predictions accepted under this rule merged, despite local generated-binding collection gaps.",
                ],
            },
            {
                "id": "POST-R20-MERGED-RECALL-GUARD",
                "rule": "Preserve the final-head contract rule that accepts a mature, title-scoped implementation with represented positive and negative controls. Audit merged recall explicitly before release.",
                "evidence": [
                    "R20 accepted all 11 merged PRs while improving exact labels by 10 percentage points over the same-cohort legacy baseline.",
                ],
            },
            {
                "id": "POST-R20-DUAL-OUTPUT-REQUIRED",
                "rule": "Always emit technical_contract and historical_disposition separately. If review activity is prohibited, also mark check_observability as unavailable instead of implying that a technically passing open PR is review-active.",
                "evidence": [
                    "R20 binary accuracy was unchanged at 16/20 while exact non-accept labeling improved from one to three; the remaining errors were governance outcomes, not kernel-contract mistakes.",
                ],
            },
        ],
        "release_gate": {
            "same_cohort_target_improved": True,
            "exact_accuracy": summary["exact_accuracy"],
            "legacy_exact_accuracy": summary["legacy_exact_accuracy"],
            "merged_accept_recall": summary["merged_machine_accepts"] / summary["merged_cases"],
            "eligible_for_commit": True,
            "reason": "R20 improved exact and frozen non-accept classification while accepting every merged case; R18 and R19 had already independently improved their target metric.",
        },
        "known_limitations": [
            "The blind evidence policy excludes the non-author review activity that defines the check oracle, so perfect check recall is structurally impossible without a new allowed evidence channel.",
            "Technically excellent mature PRs may remain open for ownership, priority, duplication, or product reasons that source and tests cannot identify.",
            "The 200-PR extension is historical and project-stratified but is not a random sample of every infrastructure repository or PR type.",
            "No weighted score was used, so the historical decision labels must not be interpreted as calibrated code-quality scores.",
        ],
    }
    payload = {**recommendation, "recommendation_sha256": canonical_sha256(recommendation)}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
