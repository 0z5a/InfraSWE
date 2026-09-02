#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze 30 title/path-derived R17 contracts before body or diff access."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

FAMILY_CONTRACTS: dict[str, dict[str, Any]] = {
    "optimizer-state": {
        "matrix": ["one/two steps", "fresh/resumed state", "mixed/full precision", "enabled/disabled path"],
        "runtime": "candidate exact tests plus parameter, gradient, optimizer-state, and continuous-versus-resumed base/head oracle",
        "closure": "each trainable parameter and state slot has one owner and the head trajectory matches the independent reference",
    },
    "loss-gradient": {
        "matrix": ["forward/backward", "boundary/normal shape", "FP32/mixed precision", "feature on/off"],
        "runtime": "candidate exact tests plus independent value/gradient base-head comparison on available accelerator",
        "closure": "loss or operator values and every reachable gradient meet the frozen reference tolerances",
    },
    "checkpoint-resume": {
        "matrix": ["save/load", "fresh/resumed", "complete/partial state", "single/distributed owner"],
        "runtime": "candidate exact tests plus key, tensor, shard/index, and resumed-trajectory parity checks",
        "closure": "all state is uniquely represented, atomically finalized, reloadable, and equivalent after resume",
    },
    "scheduling-pipeline": {
        "matrix": ["one/multiple microbatches", "first/steady/last", "single/reduced topology", "normal/delayed peer"],
        "runtime": "candidate exact tests plus timeout-protected progress, ordering, cardinality, and payload oracle",
        "closure": "the schedule completes with balanced ownership and reference-equivalent outputs or gradients",
    },
    "scheduler-progress": {
        "matrix": ["empty/one/many requests", "prefill/decode", "normal/preempted/aborted", "single/repeated batch"],
        "runtime": "candidate exact tests plus request-state transition, progress, output, and cleanup base/head oracle",
        "closure": "every request reaches exactly one legal terminal or continuation state without starvation, leak, or changed output",
    },
    "cache-state-layout": {
        "matrix": ["empty/partial/full cache", "allocate/free/reuse", "contiguous/paged layout", "single/multiple owners"],
        "runtime": "candidate exact tests plus offset, ownership, eviction, reuse, and byte/value reconstruction oracle",
        "closure": "cache regions are complete, in bounds, non-overlapping, correctly owned, and reconstruct reference values",
    },
    "attention-numerics": {
        "matrix": ["prefill/decode", "short/boundary/long sequence", "one/multiple heads", "supported dtype/backend"],
        "runtime": "candidate exact tests plus independent attention or sampling value oracle and target-kernel execution where available",
        "closure": "outputs and declared metadata match the reference across boundary shapes and supported numeric modes",
    },
    "model-runtime-integration": {
        "matrix": ["feature on/off", "one/repeated request", "supported model/config", "success/invalid input"],
        "runtime": "candidate exact tests plus import/configuration, production reachability, state ownership, and output oracle",
        "closure": "the production runtime reaches the new path, preserves the control path, and fails invalid configurations closed",
    },
    "memory-performance": {
        "matrix": ["small/large shape", "cold/warm run", "feature on/off", "forward/repeated execution"],
        "runtime": "candidate exact tests plus value parity and measured allocator, workspace, latency, or throughput comparison matching the claim",
        "closure": "reference values are preserved and the named resource metric improves without hidden lifetime or synchronization debt",
    },
}


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main(
    round_label: str = "R17",
    iteration_option: str = "--r16-iteration",
    iteration_binding_field: str = "r16_policy_iteration_sha256",
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument(iteration_option, dest="iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = read(args.selection_lock)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit(f"{round_label} selection lock digest mismatch")
    iteration = read(args.iteration)
    iteration_material = {
        key: value for key, value in iteration.items() if key != "iteration_sha256"
    }
    if iteration["iteration_sha256"] != canonical_sha256(iteration_material):
        raise SystemExit(f"{round_label} prior iteration digest mismatch")
    if selection_material[iteration_binding_field] != iteration["iteration_sha256"]:
        raise SystemExit(f"{round_label} selection/iteration binding mismatch")
    hidden = (
        selection_material["review_or_comment_visible"],
        selection_material["merge_outcomes_visible"],
        selection_material["ci_or_label_visible"],
        selection_material["candidate_body_visible"],
        selection_material["diff_content_visible"],
        selection_material["excluded_resolution_gray_zone_used"],
    )
    if any(value is not False for value in hidden):
        raise SystemExit(f"{round_label} selection exposes hidden evidence")
    cases = selection_material["cases"]
    expected_count = int(selection_material["domain_allocation"].get("training", 0)) + int(
        selection_material["domain_allocation"].get("inference", 0)
    )
    if len(cases) != expected_count:
        raise SystemExit(f"{round_label} test plan requires {expected_count} cases")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": selection_material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        iteration_binding_field: iteration["iteration_sha256"],
        "machine_policy_id": selection_material["machine_policy_id"],
        "domain_allocation": selection_material["domain_allocation"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_candidate_body_access": True,
        "frozen_before_source_diff_content_access": True,
        "review_or_comment_requested": False,
        "merge_outcome_or_state_requested": False,
        "ci_or_label_requested": False,
        "evaluation_layers": {
            "technical_contract": ["pass", "bounded-gap", "fail", "unresolved"],
            "disposition_prediction": ["accept", "check", "reject", "unresolved"],
            "governance_gap_recorded_separately": True,
        },
        "disposition_policy": {
            "accept": "title-scoped contract has candidate or evaluator exact closure and no reachable blocker",
            "check": "<=7 days, <=8 files, candidate-owned exact core test, no explicit not-ready body, and exactly one executable residual",
            "reject": "exact failure, explicit not-ready body, mature algorithm without tests/exception, or unresolved runtime resource claim",
            "unresolved": "required backend or topology is unavailable and neither a bounded structural nor artifact exception closes it",
            "prospective_created_at_cutoff": "2026-08-27T00:00:00Z",
            "mature_created_at_cutoff": "2026-06-04T23:59:59Z",
            "resolution_gray_zone_excluded": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "prospective_rules": iteration["prospective_rules"],
        "ordered_reachability_gate": [
            "configuration is legal and title-scoped",
            "a changed production call site reaches the behavior",
            "shape/state/owner/lifetime invariant is explicit",
            "an exact candidate test or independent oracle distinguishes base and head",
            "the remaining remediation is bounded to the candidate direction",
        ],
        "cross_case_controls": [
            "Syntax, conflict-marker, import, or candidate exact failure rejects first.",
            "A resource claim requires the actual scheduler, allocator, or runtime unless it is exact algebraic data motion.",
            "Structural migration and artifact-boundary exceptions remain narrow and require complete inventories.",
            "Technical correctness and historical disposition are frozen separately.",
            "No outcome, state, review, comment, CI, or label evidence is visible before judgment lock.",
        ],
        "cases": [],
    }
    for item in cases:
        contract = FAMILY_CONTRACTS[item["risk_family"]]
        plan_material["cases"].append({
            "case_id": item["case_id"],
            "project": item["project"],
            "repository": item["repository"],
            "pull_number": item["pull_number"],
            "title": item["title"],
            "created_at": item["created_at"],
            "temporal_band": item["temporal_band"],
            "benchmark_domain": item["benchmark_domain"],
            "base_sha": item["base_sha"],
            "head_sha": item["head_sha"],
            "changed_paths": item["paths"],
            "risk_family": item["risk_family"],
            "claim": (
                f"The production change titled {item['title']!r} preserves all unchanged "
                f"{item['risk_family']} behavior while satisfying its named direction."
            ),
            **contract,
        })

    payload = {**plan_material, "test_plan_sha256": canonical_sha256(plan_material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(cases)}")
    print(f"test_plan_sha256={payload['test_plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
