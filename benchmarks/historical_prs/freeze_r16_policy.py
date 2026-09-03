#!/usr/bin/env python3
"""Freeze the outcome-blind policy for the 30-case R16 training group."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_ITERATION_SHA256 = (
    "sha256:80a7d5e7b2fb2b94d1a3f099dcbc381baa8a8e08d66f0e98a11171d630f5fd06"
)

TRAINING_PROJECTS = {
    "megatron": {
        "repository": "NVIDIA/Megatron-LM",
        "count": 6,
        "source_prefixes": ["megatron/core/", "megatron/training/"],
    },
    "torchtitan": {
        "repository": "pytorch/torchtitan",
        "count": 6,
        "source_prefixes": ["torchtitan/"],
    },
    "verl": {
        "repository": "verl-project/verl",
        "count": 6,
        "source_prefixes": ["verl/"],
    },
    "slime": {
        "repository": "THUDM/slime",
        "count": 6,
        "source_prefixes": ["slime/"],
    },
    "liger": {
        "repository": "linkedin/Liger-Kernel",
        "count": 6,
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
    parser.add_argument("--r15-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iteration = read(args.r15_iteration)
    material = {key: value for key, value in iteration.items() if key != "iteration_sha256"}
    if iteration["iteration_sha256"] != canonical_sha256(material):
        raise SystemExit("R15 iteration digest mismatch")
    if iteration["iteration_sha256"] != EXPECTED_ITERATION_SHA256:
        raise SystemExit("R15 iteration identity changed")
    if iteration["r16_group"]["allocation"] != {"training": 30}:
        raise SystemExit("R16 allocation changed")

    policy = {
        "schema_version": "0.1",
        "protocol_id": "training-iterative-contract-v0.1-r16-30",
        "round": "R16",
        "case_count": 30,
        "domain_allocation": {"training": 30},
        "r15_policy_iteration_sha256": iteration["iteration_sha256"],
        "grouping_policy": {
            "round_size": 30,
            "finish_reveal_audit_and_iterate_before_next_round": True,
            "future_rounds_are_not_preselected": True,
        },
        "created_at_window": {
            "observation_cutoff": "2026-09-02T23:59:59Z",
            "start": "2025-01-01T00:00:00Z",
            "recent_start": "2026-08-27T00:00:00Z",
            "mature_end": "2026-06-04T23:59:59Z",
            "excluded_resolution_gray_zone": [
                "2026-06-05T00:00:00Z",
                "2026-08-26T23:59:59Z",
            ],
            "per_project_recent_target": 1,
            "recent_shortfall_fallback": "fill from mature ranking",
        },
        "domains_in_order": ["training"],
        "projects": {"training": TRAINING_PROJECTS},
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
                    "offload",
                    "recompute",
                    "log_prob",
                    "logprob",
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
                    "logprob",
                    "log_prob",
                    "memory_policy",
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
        "domain_anchor": {
            "title_docs_prefix_is_excluded": True,
            "direct_terms": [
                "train",
                "training",
                "trainer",
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
                "offload",
                "recompute",
                "log_prob",
                "logprob",
            ],
            "source_path_terms": [
                "/optimizer",
                "training/",
                "/trainer",
                "/loss",
                "checkpoint",
                "/scheduler",
                "backward",
                "gradient",
                "/fsdp",
                "/ppo",
                "/rollout",
                "/reward",
                "/actor",
                "/critic",
                "finetun",
                "distill",
                "logprob",
                "log_prob",
                "cpu_offload",
                "memory_policy",
            ],
        },
        "selection_algorithm": {
            "recent_first": "one best eligible <=7-day case per project when available",
            "family_diversity": "greedy family coverage within each project slice",
            "fill": "deterministic rank among remaining mature eligible cases",
            "rank_key": [
                "candidate-owned-test-first",
                "domain-score-descending",
                "created-at-descending-within-band",
                "changed-files-ascending",
                "changed-lines-ascending",
                "pull-number-ascending",
            ],
        },
        "r16_disposition_rules": [rule["id"] for rule in iteration["prospective_rules"]],
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
            "policy_id": "training-contract-disposition-split-v0.1-r16",
            "technical_contract_and_disposition_are_separate": True,
            "disposition_check_hot_window_days": 7,
            "check_coherent_risk_changed_files_max": 8,
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
