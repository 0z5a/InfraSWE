#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Backend(Enum):
    TORCH_SDPA = "TORCH_SDPA"
    XFORMERS = "XFORMERS"
    FLASH_ATTN = "FLASH_ATTN"
    FLASH_ATTN_VLLM_V1 = "FLASH_ATTN_VLLM_V1"


class FakeBackend:
    def get_name(self) -> str:
        return Backend.TORCH_SDPA.value


def load_mha_class(source: Path) -> type[nn.Module]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MultiHeadAttention"
        ),
        None,
    )
    if class_node is None:
        raise RuntimeError("MultiHeadAttention class not found")
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "F": F,
        "Optional": Optional,
        "_Backend": Backend,
        "backend_name_to_enum": lambda name: Backend(name),
        "get_attn_backend": lambda *args, **kwargs: FakeBackend(),
        "nn": nn,
        "torch": torch,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["MultiHeadAttention"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(20260901)
    source = options.worktree / "vllm" / "attention" / "layer.py"
    mha_class = load_mha_class(source)
    previous_dtype = torch.get_default_dtype()
    runtime_failure: str | None = None
    max_abs_error: float | None = None
    invalid_head_contract = False
    try:
        torch.set_default_dtype(dtype)
        module = mha_class(num_heads=4, head_size=16, scale=0.25, num_kv_heads=2)
        query = torch.randn(2, 5, 64, device=device, dtype=dtype)
        key = torch.randn(2, 7, 32, device=device, dtype=dtype)
        value = torch.randn(2, 7, 32, device=device, dtype=dtype)
        actual = module(query, key, value)

        q_ref = query.view(2, 5, 4, 16).transpose(1, 2)
        k_ref = key.view(2, 7, 2, 16).repeat_interleave(2, dim=2).transpose(1, 2)
        v_ref = value.view(2, 7, 2, 16).repeat_interleave(2, dim=2).transpose(1, 2)
        expected = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=0.25).transpose(1, 2)
        expected = expected.reshape(2, 5, 64)
        max_abs_error = float((actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

        try:
            mha_class(num_heads=3, head_size=16, scale=0.25, num_kv_heads=2)
        except AssertionError:
            invalid_head_contract = True
    except Exception as error:  # pragma: no cover - evidence path
        runtime_failure = f"{type(error).__name__}:{error}"
    finally:
        torch.set_default_dtype(previous_dtype)

    passed = runtime_failure is None and invalid_head_contract
    payload = {
        "schema_version": "0.5",
        "probe_id": "vllm-mha-gqa-contract-v0.5-r1",
        "status": "pass" if passed else "fail",
        "worktree_revision": options.worktree.name,
        "device": str(device),
        "dtype": str(dtype),
        "max_abs_error": max_abs_error,
        "invalid_head_contract": invalid_head_contract,
        "runtime_failure": runtime_failure,
        "failure_codes": [] if passed else ["MHA_GQA_CONTRACT_FAILED"],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
