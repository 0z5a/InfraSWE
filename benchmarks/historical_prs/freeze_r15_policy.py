#!/usr/bin/env python3
"""Freeze the outcome-blind mixed-domain policy for the 30-case R15 group."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_ITERATION_SHA256 = (
    "sha256:b4326c71480c9c468b1cecf354c337f26a18bd43ba053dca3541df76c3f2a027"
)

COMMUNICATION_PROJECTS = {
    "vllm": {
        "repository": "vllm-project/vllm",
        "count": 4,
        "source_prefixes": ["vllm/", "csrc/"],
    },
    "sglang": {
        "repository": "sgl-project/sglang",
        "count": 4,
        "source_prefixes": ["python/sglang/", "sgl-kernel/"],
    },
    "flashinfer": {
        "repository": "flashinfer-ai/flashinfer",
        "count": 3,
        "source_prefixes": ["flashinfer/", "python/flashinfer/", "include/", "csrc/"],
    },
    "megatron": {
        "repository": "NVIDIA/Megatron-LM",
        "count": 3,
        "source_prefixes": ["megatron/core/"],
    },
    "torchtitan": {
        "repository": "pytorch/torchtitan",
        "count": 3,
        "source_prefixes": ["torchtitan/"],
    },
    "verl": {
        "repository": "verl-project/verl",
        "count": 3,
        "source_prefixes": ["verl/"],
    },
}

TRAINING_PROJECTS = {
    "megatron": {
        "repository": "NVIDIA/Megatron-LM",
        "count": 2,
        "source_prefixes": ["megatron/core/"],
    },
    "torchtitan": {
        "repository": "pytorch/torchtitan",
        "count": 2,
        "source_prefixes": ["torchtitan/"],
    },
    "verl": {
        "repository": "verl-project/verl",
        "count": 2,
        "source_prefixes": ["verl/"],
    },
    "slime": {
        "repository": "THUDM/slime",
        "count": 2,
        "source_prefixes": ["slime/"],
    },
    "liger": {
        "repository": "linkedin/Liger-Kernel",
        "count": 2,
        "source_prefixes": ["src/liger_kernel/"],
    },
}


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r14-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iteration = read(args.r14_iteration)
    material = {key: value for key, value in iteration.items() if key != "iteration_sha256"}
    if iteration["iteration_sha256"] != canonical_sha256(material):
        raise SystemExit("R14 iteration digest mismatch")
    if iteration["iteration_sha256"] != EXPECTED_ITERATION_SHA256:
        raise SystemExit("R14 iteration identity changed")
    if iteration["r15_group"]["allocation"] != {
        "communication": 20,
        "training": 10,
    }:
        raise SystemExit("R15 allocation changed")

    policy = {
        "schema_version": "0.1",
        "protocol_id": "mixed-iterative-contract-v0.1-r15-30",
        "round": "R15",
        "case_count": 30,
        "domain_allocation": {"communication": 20, "training": 10},
        "r14_policy_iteration_sha256": iteration["iteration_sha256"],
        "grouping_policy": {
            "round_size": 30,
            "finish_reveal_audit_and_iterate_before_next_round": True,
            "future_rounds_are_not_preselected": True,
        },
        "created_at_window": {
            "observation_cutoff": "2026-09-02T23:59:59Z",
            "start": "2025-01-01T00:00:00Z",
            "recent_start": "2026-08-04T00:00:00Z",
            "mature_end": "2026-06-04T23:59:59Z",
            "excluded_resolution_gray_zone": [
                "2026-06-05T00:00:00Z",
                "2026-08-03T23:59:59Z",
            ],
            "per_project_recent_target": 1,
            "recent_shortfall_fallback": "fill from mature ranking",
        },
        "domains_in_order": ["communication", "training"],
        "projects": {
            "communication": COMMUNICATION_PROJECTS,
            "training": TRAINING_PROJECTS,
        },
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
            "domain_score_min": 4,
        },
        "domain_signals": {
            "communication": {
                "strong_title_terms": [
                    "all_reduce",
                    "all-reduce",
                    "allreduce",
                    "all_gather",
                    "all-gather",
                    "allgather",
                    "reduce_scatter",
                    "reduce-scatter",
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
                    "deadlock",
                ],
                "secondary_title_terms": [
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
                "risk_families": [
                    "collective-numerics",
                    "rank-topology-sharding",
                    "p2p-transfer",
                    "overlap-ordering",
                    "resource-lifecycle",
                ],
            },
            "training": {
                "strong_title_terms": [
                    "train",
                    "training",
                    "optimizer",
                    "loss",
                    "gradient",
                    "backward",
                    "checkpoint",
                    "resume",
                    "activation checkpoint",
                    "mixed precision",
                    "fp8",
                    "reward",
                    "rollout",
                    "actor",
                    "critic",
                    "grpo",
                    "ppo",
                    "dpo",
                    "distill",
                    "finetun",
                    "learning rate",
                    "lr scheduler",
                ],
                "secondary_title_terms": [
                    "fsdp",
                    "ddp",
                    "pipeline",
                    "microbatch",
                    "micro-batch",
                    "lora",
                    "moe",
                    "memory",
                    "offload",
                    "recompute",
                    "freeze",
                    "frozen",
                ],
                "path_terms": [
                    "optimizer",
                    "training",
                    "trainer",
                    "loss",
                    "checkpoint",
                    "scheduler",
                    "backward",
                    "gradient",
                    "fsdp",
                    "ppo",
                    "rollout",
                    "reward",
                    "actor",
                    "critic",
                    "finetun",
                    "distill",
                    "lora",
                ],
                "risk_families": [
                    "optimizer-state",
                    "loss-gradient",
                    "checkpoint-resume",
                    "scheduling-pipeline",
                    "memory-performance",
                ],
            },
            "weights": {
                "strong_title": 5,
                "secondary_title": 3,
                "domain_path": 4,
                "candidate_test_path": 1,
            },
        },
        "selection_algorithm": {
            "recent_first": "one best eligible recent case per project when available",
            "family_diversity": "greedy family coverage within each domain/project slice",
            "fill": "deterministic rank among remaining eligible cases",
            "rank_key": [
                "candidate-owned-test-first",
                "domain-score-descending",
                "created-at-descending-within-band",
                "changed-files-ascending",
                "changed-lines-ascending",
                "pull-number-ascending",
            ],
        },
        "r15_disposition_rules": [rule["id"] for rule in iteration["prospective_rules"]],
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
            "policy_id": "mixed-contract-disposition-split-v0.1-r15",
            "technical_contract_and_disposition_are_separate": True,
            "disposition_check_requires_prospective_eligibility": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    payload = {**policy, "policy_sha256": canonical_sha256(policy)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "policy_sha256": payload["policy_sha256"],
                "domain_allocation": payload["domain_allocation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
