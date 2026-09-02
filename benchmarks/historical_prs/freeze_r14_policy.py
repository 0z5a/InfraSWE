#!/usr/bin/env python3
"""Freeze the outcome-free policy for the first 30-case iterative round."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

PROJECTS = {
    "vllm": {
        "repository": "vllm-project/vllm",
        "count": 5,
        "source_prefixes": ["vllm/", "csrc/"],
    },
    "sglang": {
        "repository": "sgl-project/sglang",
        "count": 5,
        "source_prefixes": ["python/sglang/", "sgl-kernel/"],
    },
    "flashinfer": {
        "repository": "flashinfer-ai/flashinfer",
        "count": 5,
        "source_prefixes": ["flashinfer/", "python/flashinfer/", "include/", "csrc/"],
    },
    "megatron": {
        "repository": "NVIDIA/Megatron-LM",
        "count": 5,
        "source_prefixes": ["megatron/core/"],
    },
    "torchtitan": {
        "repository": "pytorch/torchtitan",
        "count": 5,
        "source_prefixes": ["torchtitan/"],
    },
    "verl": {
        "repository": "verl-project/verl",
        "count": 5,
        "source_prefixes": ["verl/"],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    material = {
        "schema_version": "0.1",
        "protocol_id": "communication-iterative-contract-v0.2-r14-30",
        "round": "R14",
        "case_count": 30,
        "domain": "communication",
        "grouping_policy": {
            "current_r13_case_count_remains": 29,
            "new_round_size": 30,
            "finish_reveal_audit_and_iterate_before_next_round": True,
            "future_rounds_are_not_preselected": True,
        },
        "created_at_window": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2026-08-31T23:59:59Z",
            "prospective_check_cutoff": "2026-08-03T00:00:00Z",
            "per_project_recent_target": 1,
            "recent_shortfall_fallback": "fill from the mature ranking",
        },
        "projects_in_order": list(PROJECTS),
        "projects": PROJECTS,
        "eligibility": {
            "changed_files_min": 1,
            "changed_files_max": 12,
            "changed_lines_max": 1200,
            "require_complete_path_list": True,
            "require_first_commit_parent": True,
            "require_runtime_source_path": True,
            "exclude_docs_only": True,
            "exclude_tests_only": True,
            "exclude_dependency_or_generated_only": True,
            "exclude_all_previously_scored_pr_identities": True,
            "communication_score_min": 4,
        },
        "communication_signal_policy": {
            "strong_terms": [
                "all_reduce",
                "all-reduce",
                "allreduce",
                "all_gather",
                "all-gather",
                "allgather",
                "reduce_scatter",
                "reduce-scatter",
                "reducescatter",
                "all_to_all",
                "all-to-all",
                "alltoall",
                "broadcast",
                "collective",
                "communicator",
                "communication",
                "nccl",
                "xccl",
                "deepep",
                "p2p",
                "send/recv",
                "cuda ipc",
                "cuda_ipc",
                "weight sync",
                "weight transfer",
            ],
            "topology_terms": [
                "tensor parallel",
                "expert parallel",
                "data parallel",
                "context parallel",
                "sequence parallel",
                "pipeline parallel",
                "distributed",
                "sharding",
                "overlap",
                "remote",
                "moe",
            ],
            "path_terms": [
                "distributed",
                "communication",
                "communicator",
                "collective",
                "all_reduce",
                "all_gather",
                "reduce_scatter",
                "all_to_all",
                "nccl",
                "xccl",
                "deepep",
                "p2p",
                "cuda_ipc",
                "parallel_state",
                "tensor_parallel",
                "expert_parallel",
                "context_parallel",
                "pipeline_parallel",
                "weight_transfer",
                "weight_sync",
            ],
            "weights": {
                "strong_title": 5,
                "topology_title": 3,
                "communication_path": 4,
                "topology_path": 2,
                "candidate_test_path": 1,
            },
        },
        "risk_families_in_order": [
            "collective-numerics",
            "rank-topology-sharding",
            "p2p-transfer",
            "overlap-ordering",
            "resource-lifecycle",
        ],
        "selection_algorithm": {
            "per_project_count": 5,
            "recent_first": "select the best eligible recent case when one exists",
            "family_diversity": (
                "then greedily select the best unselected case from each risk family in order"
            ),
            "fill": "fill remaining slots from the global deterministic ranking",
            "rank_key": [
                "candidate-owned-test-first",
                "communication-score-descending",
                "created-at-descending",
                "changed-files-ascending",
                "changed-lines-ascending",
                "pull-number-ascending",
            ],
        },
        "blindness": {
            "allowed_selection_fields": [
                "number",
                "title",
                "createdAt",
                "baseRefName",
                "baseRefOid",
                "headRefOid",
                "changedFiles",
                "additions",
                "deletions",
                "files.path",
                "commits.first.commit.oid",
                "commits.first.commit.parents.first.oid",
            ],
            "candidate_body_visible": False,
            "diff_content_visible": False,
            "state_or_merge_visible": False,
            "review_or_comment_visible": False,
            "ci_or_label_visible": False,
        },
        "machine_policy": {
            "policy_id": "communication-contract-disposition-split-v0.2-r14",
            "technical_contract_and_disposition_are_separate": True,
            "disposition_check_requires_prospective_eligibility": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "policy_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(f"policy_sha256={payload['policy_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
