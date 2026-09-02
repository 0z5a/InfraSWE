#!/usr/bin/env python3
"""Freeze the metadata-only R15 domain-anchor amendment before source access."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

SUPERSEDED_SELECTION_LOCKS = [
    "sha256:5e990d6f3260f5dd9e6fe7e42fbac45dcb2fa0ebf42e09495c68789042719529",
    "sha256:dfc4394657d8df3b1f7e14b85433bfd2b4bdbd33e1b06df91ac353062ce13c77",
]


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = read(args.policy)
    discovery = read(args.discovery)
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit("R15 policy digest mismatch")
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit("R15 discovery digest mismatch")
    if discovery["policy_sha256"] != policy["policy_sha256"]:
        raise SystemExit("R15 discovery/policy binding mismatch")

    material = {
        "schema_version": "0.1",
        "protocol_id": "mixed-iterative-contract-v0.1.1-r15-domain-amendment",
        "stage": "after title/path selection audit and before body/diff/outcome/review access",
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        "superseded_selection_lock_sha256": SUPERSEDED_SELECTION_LOCKS,
        "canonical_replacement": "selection-lock-amended.json",
        "failed_initial_case_ids": [
            "flashinfer-pr-3444",
            "megatron-pr-5144",
            "liger-pr-1424",
            "megatron-pr-5135",
        ],
        "failed_initial_case_count": 4,
        "project_count_override": {
            "communication": {"flashinfer": 2, "torchtitan": 4}
        },
        "project_count_override_reason": (
            "only two unscored FlashInfer cases retain a strict production communication "
            "anchor in the frozen time bands; move one slot to TorchTitan"
        ),
        "reason": (
            "substring scoring admitted an explicit docs-only change, a test-config p2p "
            "substring, ppo inside support, and a generic MoE overlap without a training owner"
        ),
        "evidence_used_for_amendment": ["title", "changed paths"],
        "outcome_or_state_used": False,
        "review_or_comment_used": False,
        "ci_or_label_used": False,
        "candidate_body_used": False,
        "diff_content_used": False,
        "identity_specific_exception_used": False,
        "rule": {
            "rule_id": "explicit-mixed-domain-anchor-v0.1",
            "title_docs_prefix_is_excluded": True,
            "test_path_anchor_is_sufficient": False,
            "short_terms_require_token_boundaries": ["p2p", "ppo", "dpo", "tp", "ep"],
            "communication": {
                "direct_terms": [
                    "all_reduce", "all-reduce", "allreduce", "all_gather", "all-gather",
                    "reduce_scatter", "reduce-scatter", "all_to_all", "all-to-all",
                    "broadcast", "collective", "communicator", "communication", "nccl",
                    "xccl", "deepep", "nixl", "p2p", "send/recv", "send_recv",
                    "cuda ipc", "cuda_ipc", "weight transfer", "weight_transfer",
                    "weight sync", "weight_sync", "kv transfer", "kv_transfer",
                    "ec transfer", "ec_transfer", "process group", "process_group",
                    "parallel_state", "reshard", "moe_ep", "async_ep", "eplb",
                    "dtensor", "mnnvl", "shm_broadcast", "token_dispatcher",
                ],
                "topology_title_terms": [
                    "tensor parallel", "expert parallel", "data parallel",
                    "context parallel", "sequence parallel", "pipeline parallel",
                    "distributed optimizer", "distributed init", "fsdp", "spmd",
                ],
                "runtime_path_terms": [
                    "/distributed/", "distributed_optim", "/parallelisms/",
                    "tensor_parallel", "expert_parallel", "context_parallel",
                    "pipeline_parallel", "/fsdp/", "token_dispatcher",
                ],
            },
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
            "requirement": (
                "a token-aware title or non-test production-path anchor; generic MoE, "
                "overlap, model name, or test path alone is insufficient"
            ),
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "amendment_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "amendment_sha256": payload["amendment_sha256"],
        "failed_initial_case_ids": payload["failed_initial_case_ids"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
