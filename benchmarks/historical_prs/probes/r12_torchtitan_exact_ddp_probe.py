#!/usr/bin/env python3
"""Execute the exact TorchTitan R12 DDP all-reduce bucketing path on two GPUs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import socket
import subprocess
import sys
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, checkout: str, output_dir: str) -> None:
    sys.path.insert(0, checkout)
    graph_annotations_compatibility_shim = False
    try:
        from torch.cuda._graph_annotations import _is_tools_id_unavailable  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("torch.cuda._graph_annotations")
        shim._is_tools_id_unavailable = lambda *args, **kwargs: True
        sys.modules[shim.__name__] = shim
        graph_annotations_compatibility_shim = True

    from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
    from torch._guards import tracing
    from torch._inductor.fx_passes.overlap_manual_scheduling import (
        ManualOverlapPreservingBucketer,
        ManualOverlapScheduler,
    )
    from torch.utils.checkpoint import checkpoint

    scheduler_compatibility_shim = "bucket_mode" not in inspect.signature(
        ManualOverlapScheduler.__init__
    ).parameters
    if scheduler_compatibility_shim:
        original_scheduler_init = ManualOverlapScheduler.__init__

        def compatible_scheduler_init(
            self,
            gm,
            module_bucket_plans,
            insert_overlap_deps,
            module_stack_fn=None,
            bucket_mode=None,
        ):
            original_scheduler_init(
                self,
                gm,
                module_bucket_plans,
                insert_overlap_deps,
                module_stack_fn=module_stack_fn,
            )
            if bucket_mode is not None:
                self.bucketer.bucket_mode = bucket_mode

        ManualOverlapScheduler.__init__ = compatible_scheduler_init

    manual_bucketer_source = inspect.getsource(ManualOverlapPreservingBucketer)
    bucket_metadata_compatibility_shim = (
        "self.bucketed_node_types" not in manual_bucketer_source
    )
    manual_bucketing_source = inspect.getsource(
        ManualOverlapPreservingBucketer.manual_bucket_collectives
    )
    upstream_all_reduce_bucketing_supported = "all_reduce" in manual_bucketing_source
    if bucket_metadata_compatibility_shim:

        @property
        def bucketed_node_types(self):
            return {
                node: node.meta["manual_bucket_node_type"]
                for node in self.graph.nodes
                if "manual_bucket_node_type" in node.meta
            }

        ManualOverlapPreservingBucketer.bucketed_node_types = bucketed_node_types
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
    from torchtitan.experiments.graph_trainer.fsdp_passes import (
        joint_transformer_block_bucketing_reordering_pass,
    )
    from torchtitan.experiments.graph_trainer.graph_utils import export_joint
    from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
    from torchtitan.models.common.linear import Linear
    from torchtitan.protocols.module import Module, ModuleList

    class ToyModel(Module):
        def __init__(self, dim: int = 16, n_layers: int = 3) -> None:
            super().__init__()

            def make_linear():
                return Linear.Config(in_features=dim, out_features=dim, bias=True).build()

            self.layers = ModuleList([make_linear() for _ in range(n_layers)])

        def forward(self, x):
            for layer in self.layers:
                x = checkpoint(
                    lambda module, value: torch.relu(module(value)),
                    layer,
                    x,
                    use_reentrant=False,
                )
            return x

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        parallel_dims = ParallelDims(
            dp_shard=1,
            dp_replicate=world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=world_size,
        )
        model = ToyModel(n_layers=3).cuda()
        dp_mesh = parallel_dims.get_mesh(["dp_replicate"])
        model = data_parallel(model, device_mesh=dp_mesh, mode="replicate")
        inputs = torch.randn(4, 16, device=f"cuda:{rank}")
        joint_with_descriptors, tracing_context = export_joint(model, (inputs,))
        captured: dict[str, Any] = {}

        def capture_bw_compiler(gm, example_inputs):
            captured["gm"] = gm
            captured["example_inputs"] = example_inputs
            return gm

        with tracing(tracing_context):
            aot_compile_joint_with_descriptors(
                joint_with_descriptors,
                bw_compiler=capture_bw_compiler,
            )
        backward_graph = captured["gm"]
        targets = (
            torch.ops._c10d_functional.all_reduce.default,
            torch.ops._c10d_functional.all_reduce_.default,
        )

        def count_all_reduce() -> int:
            return sum(
                node.op == "call_function" and node.target in targets
                for node in backward_graph.graph.nodes
            )

        parameter_count = sum(1 for _ in model.parameters())
        before = count_all_reduce()
        for node in backward_graph.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops._c10d_functional.all_reduce.default
            ):
                node.meta["custom"] = {_MODULE_FQN: "block"}
                node.meta["autograd_backward"] = True
        joint_transformer_block_bucketing_reordering_pass(
            backward_graph,
            None,
            module_bucket_plans=["block"],
        )
        backward_graph.graph.lint()
        after = count_all_reduce()
        row = {
            "rank": rank,
            "parameter_count": parameter_count,
            "all_reduce_count_before": before,
            "all_reduce_count_after": after,
            "graph_lint_passed": True,
            "scheduler_bucket_mode_compatibility_shim": scheduler_compatibility_shim,
            "bucket_metadata_compatibility_shim": bucket_metadata_compatibility_shim,
            "graph_annotations_compatibility_shim": graph_annotations_compatibility_shim,
            "upstream_all_reduce_bucketing_supported": (
                upstream_all_reduce_bucketing_supported
            ),
            "candidate_contract_observable": upstream_all_reduce_bucketing_supported,
            "candidate_contract_passed": (
                upstream_all_reduce_bucketing_supported
                and before == parameter_count
                and after == 1
            ),
        }
        (Path(output_dir) / f"rank-{rank}.json").write_text(
            json.dumps(row, sort_keys=True), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_sha != args.expected_sha:
        raise SystemExit(f"TorchTitan checkout mismatch: {observed_sha}")
    selection = _read(args.selection)
    plan = _read(args.plan)
    selected = {
        item["case_id"]: item for item in selection["selection_material"]["cases"]
    }
    if selected["torchtitan-pr-3821"]["head_sha"] != args.expected_sha:
        raise SystemExit("TorchTitan expected SHA is not bound to the R12 selection")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R12 plan/selection binding mismatch")
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R12 plan digest mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("exact TorchTitan DDP probe requires two CUDA GPUs")

    started_at = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix="infraswe-r12-torchtitan-") as temporary:
        mp.spawn(
            _worker,
            args=(2, _free_port(), str(args.checkout), temporary),
            nprocs=2,
            join=True,
        )
        rows = [_read(Path(temporary) / f"rank-{rank}.json") for rank in range(2)]
    facts = {
        "exact_head_sha": observed_sha,
        "world_size": 2,
        "rank_rows": rows,
        "all_ranks_candidate_contract_observable": all(
            row["candidate_contract_observable"] for row in rows
        ),
        "all_ranks_candidate_contract_passed": all(
            row["candidate_contract_passed"] for row in rows
        ),
        "execution_status": (
            "pass"
            if all(row["candidate_contract_passed"] for row in rows)
            else "blocked-environment-upstream-bucketer-lacks-all-reduce"
        ),
        "numeric_equivalence_source": "r12 dual-GPU bucket equivalence artifact",
        "test_scope": "exact candidate DDP graph export and bucketing path",
    }
    print(json.dumps(facts, indent=2, sort_keys=True))
    if (
        facts["all_ranks_candidate_contract_observable"]
        and not facts["all_ranks_candidate_contract_passed"]
    ):
        raise SystemExit("TorchTitan exact DDP contract failed")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
    material = {
        "schema_version": "0.1",
        "protocol_id": selection["selection_material"]["protocol_id"],
        "case_id": "torchtitan-pr-3821",
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "test_plan_sha256": plan["test_plan_sha256"],
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "environment": environment,
        "environment_sha256": _canonical(environment),
        "facts": facts,
    }
    payload = {**material, "evidence_sha256": _canonical(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"evidence_sha256={payload['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
