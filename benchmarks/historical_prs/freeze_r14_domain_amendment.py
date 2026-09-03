#!/usr/bin/env python3
"""Freeze a pre-source domain amendment after auditing the first R14 selection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        lowered.startswith(("test/", "tests/"))
        or "/test/" in lowered
        or "/tests/" in lowered
        or lowered.endswith(("_test.py", "_test.cpp", "_test.cu", ".test.ts"))
    )


def _has_anchor(item: dict[str, Any], rule: dict[str, Any]) -> bool:
    title = item["title"].lower()
    paths = " ".join(path for path in item["paths"] if not _is_test_path(path)).lower()
    direct = rule["direct_anchor_terms"]
    if any(term in title or term in paths for term in direct):
        return True
    title_has_topology = any(term in title for term in rule["coupled_topology_title_terms"])
    path_has_runtime = any(term in paths for term in rule["coupled_runtime_path_terms"])
    return title_has_topology and path_has_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--initial-selection", type=Path, required=True)
    parser.add_argument("--superseded-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = _read(args.policy)
    discovery = _read(args.discovery)
    selection = _read(args.initial_selection)
    policy_material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    discovery_material = {
        key: value for key, value in discovery.items() if key != "discovery_sha256"
    }
    if policy["policy_sha256"] != canonical_sha256(policy_material):
        raise SystemExit("R14 policy digest mismatch")
    if discovery["discovery_sha256"] != canonical_sha256(discovery_material):
        raise SystemExit("R14 discovery digest mismatch")
    if selection["selection_lock_sha256"] != canonical_sha256(selection["selection_material"]):
        raise SystemExit("R14 initial selection digest mismatch")

    rule = {
        "rule_id": "explicit-communication-anchor-v0.1",
        "direct_anchor_terms": [
            "all_reduce",
            "all-reduce",
            "allreduce",
            "all_gather",
            "all-gather",
            "allgather",
            "reduce_scatter",
            "reduce-scatter",
            "all_to_all",
            "all-to-all",
            "broadcast",
            "collective",
            "communicator",
            "communication",
            "nccl",
            "xccl",
            "deepep",
            "nixl",
            "p2p",
            "send/recv",
            "send_recv",
            "cuda ipc",
            "cuda_ipc",
            "weight transfer",
            "weight_transfer",
            "weight sync",
            "weight_sync",
            "kv transfer",
            "kv_transfer",
            "ec transfer",
            "ec_transfer",
            "process group",
            "process_group",
            "parallel_state",
            "reshard",
            "copy_services",
            "expert parallel",
            "tensor parallel",
            "context parallel",
            "sequence parallel",
            "pipeline parallel",
            "data parallel",
            "moe_ep",
            "async_ep",
            "ep overlap",
            "ddp",
            "tp deadlock",
        ],
        "coupled_topology_title_terms": [
            "tensor parallel",
            "expert parallel",
            "data parallel",
            "context parallel",
            "sequence parallel",
            "pipeline parallel",
            "tp-shard",
            "tp shard",
            "ep-shard",
            "ep shard",
            "fsdp",
            "distributed optimizer",
            "distributed init",
            "distributed initialization",
        ],
        "coupled_runtime_path_terms": [
            "/distributed/",
            "distributed_optim",
            "/parallelisms/",
            "tensor_parallel",
            "expert_parallel",
            "context_parallel",
            "pipeline_parallel",
            "/fsdp/",
            "token_dispatcher",
        ],
        "requirement": (
            "a direct collective/transport/process-group anchor, or a topology title "
            "coupled to a distributed runtime path"
        ),
        "generic_moe_overlap_or_distributed_path_alone_is_sufficient": False,
        "test_path_anchor_is_sufficient": False,
    }
    failed_initial_cases = [
        item["case_id"]
        for item in selection["selection_material"]["cases"]
        if not _has_anchor(item, rule)
    ]
    superseded_amendment_sha256 = None
    if args.superseded_amendment is not None:
        superseded = _read(args.superseded_amendment)
        superseded_material = {
            key: value for key, value in superseded.items() if key != "amendment_sha256"
        }
        if superseded["amendment_sha256"] != canonical_sha256(superseded_material):
            raise SystemExit("R14 superseded amendment digest mismatch")
        superseded_amendment_sha256 = superseded["amendment_sha256"]

    material = {
        "schema_version": "0.1",
        "protocol_id": "communication-iterative-contract-v0.2.1-r14-domain-amendment",
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        "superseded_selection_lock_sha256": selection["selection_lock_sha256"],
        "superseded_amendment_sha256": superseded_amendment_sha256,
        "stage": "after title/path selection audit and before body/diff/outcome/review access",
        "reason": (
            "the frozen score threshold admitted topology-title plus test-path cases with "
            "no explicit communication anchor"
        ),
        "evidence_used_for_amendment": ["title", "changed paths"],
        "outcome_or_state_used": False,
        "review_or_comment_used": False,
        "ci_or_label_used": False,
        "candidate_body_used": False,
        "diff_content_used": False,
        "identity_specific_exception_used": False,
        "rule": rule,
        "failed_initial_case_ids": failed_initial_cases,
        "failed_initial_case_count": len(failed_initial_cases),
        "canonical_replacement": "selection-lock-amended.json",
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "amendment_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(json.dumps(failed_initial_cases, indent=2))
    print(f"amendment_sha256={payload['amendment_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
