#!/usr/bin/env python3
"""Freeze r3 machine judgments before revealing selected reviewer text."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.heuristics import (
    R3_POLICY_ID,
    compile_explainable_judgment,
    freeze_explainable_judgment,
)
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalHeuristicObservation


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _pass(
    rule_id: str,
    question: str,
    conclusion: str,
    *evidence: str,
    counterevidence: list[str] | None = None,
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="pass",
        blocking=True,
        evidence=list(evidence),
        counterevidence=counterevidence or [],
        conclusion=conclusion,
    )


def _failure(
    rule_id: str,
    question: str,
    conclusion: str,
    failure_code: str,
    *evidence: str,
    counterevidence: list[str] | None = None,
) -> HistoricalHeuristicObservation:
    return HistoricalHeuristicObservation(
        rule_id=rule_id,
        question=question,
        status="fail",
        blocking=True,
        evidence=list(evidence),
        counterevidence=counterevidence or [],
        conclusion=conclusion,
        failure_code=failure_code,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = _read(args.test_plan)
    stored_plan_sha = plan.pop("test_plan_sha256")
    if canonical_sha256(plan) != stored_plan_sha:
        raise SystemExit("r3 test-plan digest mismatch")
    selection = _read(args.selection_lock)
    if canonical_sha256(selection["selection_material"]) != selection["selection_lock_sha256"]:
        raise SystemExit("r3 selection-lock digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("r3 plan is not bound to the selection lock")
    if plan["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("r3 plan does not assert hidden reviewer text")

    evidence_root = args.result_root / "supplemental-r3/probes"
    paths = {
        "sglang-pr-3136": [evidence_root / "sglang-pr-3136-contract.json"],
        "vllm-pr-11531": [
            evidence_root / "vllm-pr-11531-swap.json",
            evidence_root / "vllm-pr-11531-swap-cache-stable.json",
        ],
        "vllm-pr-12111": [evidence_root / "vllm-pr-12111-contract.json"],
    }
    evidence = {
        case_id: [(path, _read(path)) for path in case_paths]
        for case_id, case_paths in paths.items()
    }
    for case_id, items in evidence.items():
        if any(payload["case_id"] != case_id for _, payload in items):
            raise SystemExit(f"evidence case mismatch for {case_id}")

    sglang = evidence["sglang-pr-3136"][0][1]
    sglang_observations = [
        _failure(
            "submodule-pointer-does-not-move-backward",
            "Is the proposed submodule commit a descendant of the pinned base commit?",
            (
                "The proposed FlashInfer pointer is an ancestor, not a descendant: it "
                f"moves backward by {sglang['facts']['reverse_commit_distance']} commits and "
                f"implicitly changes {sglang['facts']['submodule_changed_file_count']} files."
            ),
            "SUBMODULE_POINTER_REGRESSION",
            sglang["evidence_sha256"],
        ),
        _failure(
            "test-poke-adds-observable-coverage",
            "Does a test-only CI poke add an assertion, input case, or useful diagnostic?",
            (
                "The only test edit changes an unreachable error message from 'p not "
                "recognized' to 'pp not recognized'; it adds no coverage and degrades the "
                "diagnostic."
            ),
            "TEST_POKE_WITHOUT_COVERAGE",
            sglang["source_identity"]["test_diff_sha256"],
        ),
        _pass(
            "early-exit-avoids-unnecessary-build",
            "Can a decisive contract failure stop before compiling the regressed submodule?",
            "The ancestry and test contracts are decisive; compilation was not required.",
            "steady_state_compile_seconds=0",
        ),
    ]

    swap_first = evidence["vllm-pr-11531"][0][1]
    swap_stable = evidence["vllm-pr-11531"][1][1]
    failed_pages = [item for item in swap_stable["correctness"] if not item["matches_oracle"]]
    one_mapping = next(item for item in swap_stable["performance"] if item["mapping_count"] == 1)
    many_mapping = next(item for item in swap_stable["performance"] if item["mapping_count"] == 256)
    swap_observations = [
        _failure(
            "page-copy-preserves-every-byte",
            "Does the fused page copy preserve tensor pages not divisible by eight bytes?",
            (
                "The kernel divides byte size by eight after reinterpreting every tensor as "
                f"int64; {len(failed_pages)} odd-byte cases disagree with the byte oracle."
            ),
            "PAGE_COPY_DROPS_NON_INT64_TAIL_BYTES",
            swap_stable["evidence_sha256"],
            counterevidence=["Aligned 256-byte FP16 pages match the oracle."],
        ),
        _failure(
            "asynchronous-metadata-outlives-kernel-consumption",
            "Can reusable pinned mapping storage remain unchanged until the kernel consumes it?",
            (
                "A delayed current-stream launch consumed the mapping written by the next "
                "request, while the old per-copy path retained the intended mapping."
            ),
            "ASYNC_PINNED_MAPPING_BUFFER_LIFETIME_RACE",
            swap_stable["evidence_sha256"],
        ),
        _failure(
            "zero-work-launch-is-guarded",
            "Does an empty block mapping return before a zero-grid CUDA launch?",
            "The exact head source has no num_blocks == 0 guard.",
            "ZERO_BLOCK_KERNEL_LAUNCH_UNGUARDED",
            swap_stable["source_identity"]["head_cache_kernels_sha256"],
        ),
        _pass(
            "performance-direction-is-measured-not-assumed",
            "Is the fused direction measured across both small and large mapping counts?",
            (
                f"At one mapping the candidate speedup is {one_mapping['speedup']:.3f}x; "
                f"at 256 mappings it is {many_mapping['speedup']:.2f}x. The optimization "
                "direction is valuable but does not override correctness failures."
            ),
            swap_stable["evidence_sha256"],
        ),
        _pass(
            "compilation-is-outside-steady-state",
            "Is unavoidable CUDA compilation excluded from benchmark timing?",
            (
                f"The first precompile took {swap_first['phases']['precompile']['seconds']:.2f}s; "
                "the stable cache load took "
                f"{swap_stable['phases']['precompile']['seconds']:.3f}s and steady-state "
                "compile time was zero."
            ),
            swap_first["evidence_sha256"],
            swap_stable["evidence_sha256"],
        ),
    ]

    activation = evidence["vllm-pr-12111"][0][1]
    affected_classes = activation["facts"]["classes_with_uninitialized_cpu_op"]
    activation_observations = [
        _failure(
            "backend-dispatch-initializes-every-loaded-attribute",
            "Does the CPU dispatch path initialize every attribute loaded downstream?",
            (
                "CustomOp.forward_cpu still delegates to forward_cuda, which loads self.op, "
                f"but the head no longer initializes it on CPU for {', '.join(affected_classes)}."
            ),
            "CPU_CUSTOM_OP_ATTRIBUTE_UNINITIALIZED",
            activation["evidence_sha256"],
        ),
        _failure(
            "test-process-environment-is-restored",
            "Does a test fixture restore each process-global environment mutation?",
            (
                "The module-scoped fixture deletes CUDA_VISIBLE_DEVICES before yield without "
                "a monkeypatch or finally restoration."
            ),
            "PROCESS_ENV_MUTATION_NOT_RESTORED",
            activation["evidence_sha256"],
        ),
        _pass(
            "compile-free-call-graph-probe",
            "Can the affected CPU and fixture contracts be resolved without CUDA compilation?",
            "AST call-graph and lifecycle checks completed in milliseconds with no compilation.",
            "steady_state_compile_seconds=0",
        ),
    ]

    observations = {
        "sglang-pr-3136": sglang_observations,
        "vllm-pr-11531": swap_observations,
        "vllm-pr-12111": activation_observations,
    }
    cases = {item["case_id"]: item for item in plan["cases"]}
    selected_case_ids = selection["selection_material"]["case_ids"]
    if set(cases) != set(selected_case_ids) or set(observations) != set(cases):
        raise SystemExit("r3 cases, observations, and selection do not match")

    frozen_at = datetime.now(UTC)
    locks: list[dict[str, Any]] = []
    bindings_by_case: dict[str, dict[str, str]] = {}
    for case_id in selected_case_ids:
        case = cases[case_id]
        bindings = {
            str(path.relative_to(args.result_root)): canonical_sha256(payload)
            for path, payload in evidence[case_id]
        }
        candidate_sha256 = canonical_sha256(
            {
                key: case[key]
                for key in (
                    "case_id",
                    "repository",
                    "pull_number",
                    "base_sha",
                    "head_sha",
                    "paths",
                )
            }
        )
        material = compile_explainable_judgment(
            case_id=case_id,
            candidate_sha256=candidate_sha256,
            test_plan_sha256=stored_plan_sha,
            evidence_sha256=canonical_sha256(bindings),
            observations=observations[case_id],
            frozen_at=frozen_at,
            policy_id=R3_POLICY_ID,
        )
        locks.append(freeze_explainable_judgment(material).model_dump(mode="json"))
        bindings_by_case[case_id] = bindings

    lock_material = {
        "schema_version": "0.5",
        "protocol_id": "historical-reviewed-failure-alignment-v0.5-r3",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": stored_plan_sha,
        "review_text_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "frozen_at": frozen_at.isoformat(),
        "evidence_bindings": bindings_by_case,
        "locks": locks,
    }
    payload = {**lock_material, "lock_set_sha256": canonical_sha256(lock_material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "lock_set_sha256": payload["lock_set_sha256"],
                "cases": [
                    {
                        "case_id": item["material"]["case_id"],
                        "decision": item["material"]["decision"],
                        "rationale_codes": item["material"]["rationale_codes"],
                        "lock_sha256": item["lock_sha256"],
                    }
                    for item in locks
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
