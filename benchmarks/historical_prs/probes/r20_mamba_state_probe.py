#!/usr/bin/env python3
"""Execute the exact candidate Mamba-state Triton functions in isolation."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import tempfile
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    tree = ast.parse(args.source.read_text(encoding="utf-8"))
    wanted = {"_promote_mamba_state_kernel", "_promote_mamba_state_triton"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("exact candidate functions not found")
    extracted = "import torch\nimport triton\nimport triton.language as tl\n\n" + ast.unparse(
        ast.Module(body=nodes, type_ignores=[])
    )

    with tempfile.TemporaryDirectory(prefix="r20-mamba-probe-") as temp_dir:
        module_path = Path(temp_dir) / "candidate_mamba_functions.py"
        module_path.write_text(extracted, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("candidate_mamba_functions", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load extracted candidate functions")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        intermediate = torch.arange(2 * 6 * 4 * 64, device="cuda", dtype=torch.float32).reshape(
            2, 6, 4, 64
        )
        pool = torch.full((2, 8, 2, 64), -7.0, device="cuda")
        destination = pool[:, :, 0, :]
        source_rows = torch.tensor([1, 4, 2], device="cuda")
        accepted_steps = torch.tensor([0, 3, 1], device="cuda")
        destination_blocks = torch.tensor([6, 0, 3], device="cuda")
        module._promote_mamba_state_triton(
            destination,
            intermediate,
            source_rows,
            accepted_steps,
            destination_blocks,
            BLOCK=128,
        )
        torch.cuda.synchronize()

        for generation in range(3):
            assert torch.equal(
                destination[:, int(destination_blocks[generation])],
                intermediate[:, int(source_rows[generation]), int(accepted_steps[generation])],
            )
        assert bool(torch.all(pool[:, :, 1, :] == -7))
        assert bool(torch.all(destination[:, [1, 2, 4, 5, 7], :] == -7))
        print("mamba-strided-copy-pass", tuple(destination.stride()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
