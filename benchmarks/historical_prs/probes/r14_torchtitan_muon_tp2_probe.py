#!/usr/bin/env python3
"""Reduced two-rank exact-head probe for TorchTitan PR #4051."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torchtitan.components.distributed_optimizers.bucketed_redistribution import (
    BucketSpec,
)
from torchtitan.components.distributed_optimizers.muon import Owned
from torchtitan.components.distributed_optimizers.muon_parameter_prep import (
    MuonComputeSharding,
    build_distributed_muon,
)


def _canonical(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    try:
        # Preserve the candidate's named 2-D mesh/placement contract while reducing
        # the FSDP axis to one rank. TP still genuinely shards matrix dimension 1.
        mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("fsdp", "tp"))
        redistribution_mesh = mesh._flatten("optimizer")
        placements = (Shard(0), Shard(1))
        values = [
            torch.arange(35, device=device).reshape(5, 7).float().div_(10),
            torch.arange(35, 65, device=device).reshape(6, 5).float().div_(10),
        ]
        params = [
            torch.nn.Parameter(distribute_tensor(value.clone(), mesh, placements))
            for value in values
        ]
        names = ["layers.0.first", "layers.0.second"]
        optimizer = build_distributed_muon(
            [
                {
                    "params": params,
                    "param_names": names,
                    "compute_sharding": MuonComputeSharding(placement=Owned()),
                }
            ],
            bucket_spec=[
                BucketSpec(
                    patterns=("layers.0.*",),
                    owner_rank_by_fqn={names[0]: 0, names[1]: 1},
                    mesh=redistribution_mesh,
                )
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        grads = [value.flip((0, 1)).contiguous() for value in values]
        for param, grad in zip(params, grads, strict=True):
            param.grad = distribute_tensor(grad, mesh, placements)

        references = [torch.nn.Parameter(value.clone()) for value in values]
        reference_optimizer = torch.optim.Muon(
            references,
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        for reference, grad in zip(references, grads, strict=True):
            reference.grad = grad.clone()

        all_to_all_single = dist.all_to_all_single
        with patch(
            "torchtitan.components.distributed_optimizers.bucketed_redistribution.dist."
            "all_to_all_single",
            wraps=all_to_all_single,
        ) as collective:
            optimizer.step()
        reference_optimizer.step()

        param_errors = []
        momentum_errors = []
        for param, reference in zip(params, references, strict=True):
            expected_param = distribute_tensor(reference.detach(), mesh, placements)
            expected_momentum = distribute_tensor(
                reference_optimizer.state[reference]["momentum_buffer"], mesh, placements
            )
            momentum = optimizer.state[param]["momentum_buffer"]
            param_errors.append(
                float((param.to_local() - expected_param.to_local()).abs().max().item())
            )
            momentum_errors.append(
                float((momentum.to_local() - expected_momentum.to_local()).abs().max().item())
            )
        local = {
            "rank": rank,
            "mesh_shape": [1, 2],
            "placements": ["Shard(0)", "Shard(1)"],
            "all_to_all_call_count": collective.call_count,
            "parameter_max_abs": max(param_errors),
            "momentum_max_abs": max(momentum_errors),
        }
        rows: list[dict[str, Any] | None] = [None, None]
        dist.all_gather_object(rows, local)
        if rank == 0:
            material = {
                "schema_version": "0.1",
                "protocol_id": "r14-torchtitan-4051-reduced-tp2-exact-head-v0.1",
                "case_id": "torchtitan-pr-4051",
                "head_sha": "f0ff6c6925b3c38e6bd007a7d89a4f61b529dcf5",
                "generated_at": datetime.now(UTC).isoformat(),
                "environment": {
                    "hostname": platform.node(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count(),
                },
                "limitation": (
                    "The candidate-owned test requires four GPUs and a 2x2 mesh. This exact-head "
                    "probe preserves TP=2 and both Shard placements but collapses FSDP to size 1."
                ),
                "outcome_review_ci_fields_requested": False,
                "rows": rows,
                "facts": {
                    "all_ranks_match_reference": all(
                        row is not None
                        and row["parameter_max_abs"] == 0
                        and row["momentum_max_abs"] == 0
                        for row in rows
                    ),
                    "all_ranks_used_two_all_to_all_calls": all(
                        row is not None and row["all_to_all_call_count"] == 2 for row in rows
                    ),
                },
            }
            payload = {**material, "evidence_sha256": _canonical(material)}
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
