#!/usr/bin/env python3
"""Freeze R7 test questions before inspecting selected source diffs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

CASE_PLANS: dict[str, dict[str, Any]] = {
    "cutlass-pr-3427": {
        "claim": (
            "CuTe ScaledBasis equality distinguishes mismatched bases and the SM120 sparse "
            "collective rejects an incompatible metadata K dimension."
        ),
        "execution_tier": "host-compile-contract-plus-sm120-aot",
        "predeclared_questions": [
            "Do equal ScaledBasis values remain equal while mismatched bases compare unequal?",
            "Does the equality expression compile for the exact representative basis types?",
            "Does an invalid SM120 sparse metadata K fail at the intended compile-time boundary?",
            "Does a valid neighboring SM120 sparse collective continue to instantiate?",
            "Are direct regression tests present for both the equality and metadata-K changes?",
        ],
        "decision_rule": (
            "Accept with scope only if equal/mismatched basis behavior is correct, invalid K is "
            "rejected, a valid neighbor instantiates, and both changes have direct tests. A "
            "valid-path compile regression or incorrect equality requires revise; SM120 runtime "
            "performance remains unresolved without SM120 hardware."
        ),
    },
    "liger-pr-1328": {
        "claim": (
            "Fused linear cross entropy remains correct under torch.compile without selecting "
            "the unsupported aten.addmm.dtype_out overload."
        ),
        "execution_tier": "h100-torch-compile-forward-backward",
        "predeclared_questions": [
            "Does fullgraph torch.compile complete without the dtype_out overload failure?",
            "Do compiled and eager loss, logits-derived behavior, and gradients match an oracle?",
            "Do representative and boundary shapes and supported training dtypes pass?",
            "After precompile, do repeated calls avoid graph breaks and new compile artifacts?",
            "Does eager/common-path steady latency remain within 3%?",
            "Does the changed test directly reproduce the compile failure?",
        ],
        "decision_rule": (
            "Accept with scope only if compiled forward/backward matches eager/reference behavior, "
            "the target failure is reproduced on base and fixed on head, steady compilation is "
            "zero, common latency stays within 3%, and a direct regression test exists."
        ),
    },
    "deepgemm-pr-389": {
        "claim": (
            "SM100 Mega-MoE combine-buffer reuse is fenced before the next producer/consumer "
            "phase in both BF16 and FP8/FP4 implementations."
        ),
        "execution_tier": "exact-source-ordering-plus-sm100-aot",
        "predeclared_questions": [
            "Is the new fence placed before every affected combine-buffer reuse transition?",
            "Do BF16 and FP8/FP4 implementations use the same intended fence primitive?",
            "Does exact SM100 AOT compilation retain valid PTX/SASS ordering?",
            "On SM100, do repeated and concurrent Mega-MoE launches avoid stale data or races?",
            "Does the valid common path remain within 3% steady latency with zero steady compile?",
            "Is direct race/reuse regression coverage included?",
        ],
        "decision_rule": (
            "Accept with scope only if both implementations are fenced, exact SM100 compilation "
            "succeeds, dynamic repeated/concurrent correctness and latency pass, and a direct "
            "regression exists. Without SM100 hardware, dynamic race and performance properties "
            "remain unresolved rather than inferred from source."
        ),
    },
    "megatron-pr-6174": {
        "claim": (
            "Megatron FSDP forward prefetch follows the combined-1F1B model-chunk schedule without "
            "missing, duplicating, or prematurely prefetching work."
        ),
        "execution_tier": "schedule-oracle-plus-multirank-fsdp-if-available",
        "predeclared_questions": [
            "Does prefetch follow the next valid forward chunk in the combined-1F1B plan?",
            (
                "Are warmup, steady, cooldown, first, and final boundaries handled without bad "
                "indices?"
            ),
            "Do multiple chunk/stage/microbatch configurations match a schedule oracle?",
            "Does multi-rank FSDP complete without hang and preserve outputs/gradients?",
            "Does prefetch improve or retain step latency and peak memory after warmup?",
            "Do changed tests directly exercise the failing combined schedule?",
        ],
        "decision_rule": (
            "Accept with scope only if schedule-oracle boundaries pass, multi-rank FSDP preserves "
            "correctness and liveness, performance does not regress over 3%, and direct coverage "
            "exists. Multi-rank lifecycle and performance remain unresolved if fewer than two "
            "suitable GPUs are available."
        ),
    },
    "torchtitan-pr-4032": {
        "claim": (
            "The graph-trainer hardware-queue helper sets GPU_MAX_HW_QUEUES correctly on ROCm "
            "without changing CUDA, CPU, or explicit-user configurations."
        ),
        "execution_tier": "exact-python-helper-with-backend-mocks",
        "predeclared_questions": [
            "Does the helper derive and set the intended ROCm hardware-queue value?",
            "Is an explicit user value preserved rather than overwritten?",
            "Are CUDA and CPU paths strict no-ops?",
            (
                "Do missing tools, malformed output, and unsupported devices fail or fall back "
                "explicitly?"
            ),
            "Is helper state evaluated per launch without leaking across trainer invocations?",
            "Do direct tests cover ROCm, no-op, override, and error branches?",
        ],
        "decision_rule": (
            "Accept with scope only if the exact helper passes ROCm parsing, override, no-op, "
            "error, and per-launch-state tests with direct coverage. Real ROCm scheduling impact "
            "remains unresolved without a ROCm GPU and rocminfo-equivalent runtime."
        ),
    },
    "verl-pr-7220": {
        "claim": (
            "Both RL trainer entrypoints construct tokenizer/processor state from HFModelConfig "
            "without diverging text and multimodal initialization semantics."
        ),
        "execution_tier": "exact-ast-call-contract-plus-mocked-config-runtime",
        "predeclared_questions": [
            "Do both trainer entrypoints construct and consume the same HFModelConfig contract?",
            "Are text-only tokenizer and multimodal processor branches both preserved?",
            "Are revision, trust_remote_code, tokenizer path, and processor kwargs propagated?",
            "Does initialization avoid duplicate loads and stale cross-run state?",
            "Are legacy configuration inputs either compatible or rejected explicitly?",
            "Do direct tests cover both entrypoints and text/multimodal branches?",
        ],
        "decision_rule": (
            "Accept with scope only if both entrypoints and both modality branches preserve the "
            "same configuration semantics, avoid duplicate/stale initialization, retain an "
            "explicit compatibility policy, and have direct two-entrypoint coverage. Missing "
            "branch or compatibility evidence requires revise."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(selection_material):
        raise SystemExit("R7 selection lock digest mismatch")
    if selection_material["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 selection does not keep review text hidden")
    if selection_material["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R7 selection does not keep outcomes hidden")

    cases: list[dict[str, Any]] = []
    for selected in selection_material["cases"]:
        case_id = selected["case_id"]
        plan = CASE_PLANS[case_id]
        cases.append(
            {
                "case_id": case_id,
                "project": selected["project"],
                "repository": selected["repository"],
                "pull_number": selected["pull_number"],
                "base_sha": selected["base_sha"],
                "head_sha": selected["head_sha"],
                "changed_paths": selected["paths"],
                "bound_contracts": [
                    f"{selected['default_profile_id']}:api-abi",
                    f"{selected['default_profile_id']}:lifecycle",
                    f"{selected['default_profile_id']}:build-test-matrix",
                    f"{selected['default_profile_id']}:fallback-policy",
                    f"{selected['default_profile_id']}:performance-acceptance-targets",
                ],
                **plan,
            }
        )

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "review_text_requested": False,
        "frozen_before_source_diff_content_inspection": True,
        "execution_policy": {
            "comparison": "exact first-parent base SHA versus exact head SHA",
            "primary_gpu_cell": "existing NVIDIA H100 PCIe 80GB, SM90",
            "environment_fallback": (
                "Use exact changed-source extraction or AOT compilation when the pinned full "
                "stack cannot run; mark untested hardware/integration properties unresolved."
            ),
            "timing": {
                "precompile_all_variants": True,
                "steady_state_compilation_allowed": False,
                "paired_order": "interleave base/head and alternate first position",
                "minimum_pairs": 7,
                "common_path_regression_limit": 1.03,
            },
            "machine_decisions": [
                "accept_with_scope",
                "revise",
                "reject",
                "unresolved",
            ],
        },
        "cases": cases,
    }
    payload = {**material, "test_plan_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "test_plan_sha256": payload["test_plan_sha256"],
                "case_count": len(cases),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
