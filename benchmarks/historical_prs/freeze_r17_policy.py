#!/usr/bin/env python3
"""Freeze the outcome-blind policy for the 30-case mixed R17 group."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_ITERATION_SHA256 = (
    "sha256:9a91111429959388bf11331e499c8eb28713c5ef5091bf83e86fb14e759d9b4f"
)

TRAINING_PROJECTS = {
    "megatron": {
        "repository": "NVIDIA/Megatron-LM",
        "count": 2,
        "source_prefixes": ["megatron/core/", "megatron/training/"],
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

INFERENCE_PROJECTS = {
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
    "tensorrt_llm": {
        "repository": "NVIDIA/TensorRT-LLM",
        "count": 5,
        "source_prefixes": ["tensorrt_llm/", "cpp/"],
    },
    "flashinfer": {
        "repository": "flashinfer-ai/flashinfer",
        "count": 5,
        "source_prefixes": ["flashinfer/", "python/flashinfer/", "include/", "csrc/"],
    },
}


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r16-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    iteration = read(args.r16_iteration)
    material = {key: value for key, value in iteration.items() if key != "iteration_sha256"}
    if iteration["iteration_sha256"] != canonical_sha256(material):
        raise SystemExit("R16 iteration digest mismatch")
    if iteration["iteration_sha256"] != EXPECTED_ITERATION_SHA256:
        raise SystemExit("R16 iteration identity changed")
    if iteration["r17_group"]["allocation"] != {"training": 10, "inference": 20}:
        raise SystemExit("R17 allocation changed")

    policy = {
        "schema_version": "0.1",
        "protocol_id": "mixed-iterative-contract-v0.1-r17-30",
        "round": "R17",
        "case_count": 30,
        "domain_allocation": {"training": 10, "inference": 20},
        "r16_policy_iteration_sha256": iteration["iteration_sha256"],
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
        "domains_in_order": ["training", "inference"],
        "projects": {"training": TRAINING_PROJECTS, "inference": INFERENCE_PROJECTS},
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
                    "train", "training", "optimizer", "loss", "gradient", "backward",
                    "checkpoint", "resume", "activation checkpoint", "mixed precision",
                    "fp8", "reward", "rollout", "actor", "critic", "grpo", "ppo",
                    "dpo", "distill", "finetun", "learning rate", "lr scheduler",
                    "offload", "recompute", "log_prob", "logprob",
                ],
                "secondary_title_terms": [
                    "fsdp", "ddp", "pipeline", "microbatch", "micro-batch", "lora",
                    "moe", "memory", "freeze", "frozen",
                ],
                "path_terms": [
                    "optimizer", "training", "trainer", "loss", "checkpoint",
                    "scheduler", "backward", "gradient", "fsdp", "ppo", "rollout",
                    "reward", "actor", "critic", "finetun", "distill", "lora",
                    "logprob", "log_prob", "memory_policy",
                ],
                "risk_families": [
                    "optimizer-state", "loss-gradient", "checkpoint-resume",
                    "scheduling-pipeline", "memory-performance",
                ],
            },
            "inference": {
                "strong_title_terms": [
                    "inference", "serving", "scheduler", "kv cache", "kv-cache",
                    "prefix cache", "decode", "prefill", "speculative", "draft",
                    "cuda graph", "cudagraph", "attention", "quantization", "lora",
                    "adapter", "batching", "request", "engine", "executor", "kernel",
                ],
                "secondary_title_terms": [
                    "latency", "throughput", "memory", "cache", "sampling", "token",
                    "model runner", "worker", "backend", "flashinfer", "trtllm",
                ],
                "path_terms": [
                    "scheduler", "engine", "executor", "worker", "model_runner",
                    "kv_cache", "attention", "spec_decode", "speculative", "sampling",
                    "lora", "adapter", "quant", "serve", "runtime", "batch",
                    "kernel", "cache_manager",
                ],
                "risk_families": [
                    "scheduler-progress", "cache-state-layout", "attention-numerics",
                    "model-runtime-integration", "memory-performance",
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
            "training": {
                "direct_terms": [
                    "train", "training", "trainer", "optimizer", "loss", "gradient",
                    "backward", "checkpoint", "resume", "activation checkpoint",
                    "mixed precision", "fp8", "reward", "rollout", "actor", "critic",
                    "grpo", "ppo", "dpo", "distill", "finetun", "learning rate",
                    "lr scheduler", "offload", "recompute", "log_prob", "logprob",
                ],
                "source_path_terms": [
                    "/optimizer", "training/", "/trainer", "/loss", "checkpoint",
                    "/scheduler", "backward", "gradient", "/fsdp", "/ppo",
                    "/rollout", "/reward", "/actor", "/critic", "finetun",
                    "distill", "logprob", "log_prob", "cpu_offload", "memory_policy",
                ],
            },
            "inference": {
                "direct_terms": [
                    "inference", "serving", "serve", "scheduler", "kv cache",
                    "kv-cache", "prefix cache", "decode", "prefill", "speculative",
                    "draft", "cuda graph", "cudagraph", "attention", "quantization",
                    "lora", "adapter", "batching", "request", "engine", "executor",
                    "model runner", "model_runner", "sampling",
                ],
                "source_path_terms": [
                    "/scheduler", "/engine", "/executor", "/worker", "model_runner",
                    "kv_cache", "/attention", "spec_decode", "speculative", "/sampling",
                    "/lora", "/adapter", "/quant", "/serve", "/runtime", "/batch",
                    "cache_manager",
                ],
            },
        },
        "selection_algorithm": {
            "recent_first": "one best eligible <=7-day case per project when available",
            "family_diversity": "greedy family coverage within each project slice",
            "fill": "deterministic rank among remaining mature eligible cases",
            "rank_key": [
                "candidate-owned-test-first", "domain-score-descending",
                "created-at-descending-within-band", "changed-files-ascending",
                "changed-lines-ascending", "pull-number-ascending",
            ],
            "stacked_series_diversity": {
                "normalized_title_prefix_token_count": 3,
                "non_test_path_jaccard_min": 0.5,
                "max_per_overlap_cluster_per_project": 2,
            },
        },
        "r17_disposition_rules": [rule["id"] for rule in iteration["prospective_rules"]],
        "blindness": {
            "allowed_selection_fields": [
                "number", "title", "createdAt", "baseRefName", "baseRefOid",
                "headRefOid", "changedFiles", "additions", "deletions", "files.path",
                "commits.first.commit.oid", "commits.first.commit.parents.first.oid",
            ],
            "candidate_body_visible": False,
            "diff_content_visible": False,
            "state_or_merge_visible": False,
            "review_or_comment_visible": False,
            "ci_or_label_visible": False,
        },
        "machine_policy": {
            "policy_id": "mixed-contract-disposition-split-v0.1-r17",
            "technical_contract_and_disposition_are_separate": True,
            "explicit_not_ready_body_is_disposition_veto": True,
            "candidate_test_required_for_check": True,
            "weighted_score_used": False,
            "forced_polarization_used": False,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    payload = {**policy, "policy_sha256": canonical_sha256(policy)}
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "policy_sha256": payload["policy_sha256"],
        "domain_allocation": payload["domain_allocation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
