#!/usr/bin/env python3
"""Blind page-size correctness probe for FlashInfer FA2 MLA paged attention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _set_case_workspace(workspace: Path) -> None:
    """Point every already-imported FlashInfer JIT path at a case-local cache."""
    import flashinfer.jit.attention as attention_jit
    import flashinfer.jit.core as core_jit
    import flashinfer.jit.env as env_jit

    jit_dir = workspace / "cached_ops"
    generated_dir = workspace / "generated"
    for path in (workspace, jit_dir, generated_dir):
        path.mkdir(parents=True, exist_ok=True)
    for module in (env_jit, core_jit, attention_jit):
        if hasattr(module, "FLASHINFER_WORKSPACE_DIR"):
            module.FLASHINFER_WORKSPACE_DIR = workspace
        if hasattr(module, "FLASHINFER_JIT_DIR"):
            module.FLASHINFER_JIT_DIR = jit_dir
        if hasattr(module, "FLASHINFER_GEN_SRC_DIR"):
            module.FLASHINFER_GEN_SRC_DIR = generated_dir


def _attention_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    qo_indptr: list[int],
    kv_indptr: list[int],
    kv_lens: list[int],
    page_size: int,
    causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_parts = []
    lse_parts = []
    for batch_idx, kv_len in enumerate(kv_lens):
        q_start, q_end = qo_indptr[batch_idx : batch_idx + 2]
        page_start, page_end = kv_indptr[batch_idx : batch_idx + 2]
        q = torch.cat([q_nope[q_start:q_end], q_pe[q_start:q_end]], dim=-1)
        key_ckv = ckv[page_start:page_end].reshape(-1, ckv.shape[-1])[:kv_len]
        key_kpe = kpe[page_start:page_end].reshape(-1, kpe.shape[-1])[:kv_len]
        key = torch.cat([key_ckv, key_kpe], dim=-1)
        logits = torch.einsum("mhd,nd->hmn", q.float(), key.float()) * sm_scale
        if causal:
            qo_len = q_end - q_start
            mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
                1
            ) >= torch.arange(kv_len, device=q.device).unsqueeze(0)
            logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        output = torch.einsum("hmn,nd->mhd", probabilities, key_ckv.float())
        lse = torch.logsumexp(logits, dim=-1).transpose(0, 1) * math.log2(math.e)
        output_parts.append(output.to(q_nope.dtype))
        lse_parts.append(lse)
    return torch.cat(output_parts, dim=0), torch.cat(lse_parts, dim=0)


def _run_case(
    name: str,
    kv_lens: list[int],
    qo_lens: list[int],
    page_size: int,
    num_heads: int,
    causal: bool,
) -> dict[str, Any]:
    import flashinfer

    torch.manual_seed(20260901 + sum(kv_lens) + sum(qo_lens))
    device = torch.device("cuda:0")
    dtype = torch.float16
    head_dim_ckv = 512
    head_dim_kpe = 64
    batch_size = len(kv_lens)
    pages_per_sequence = [(length + page_size - 1) // page_size for length in kv_lens]
    qo_indptr_cpu = [0]
    kv_indptr_cpu = [0]
    for qo_len, pages in zip(qo_lens, pages_per_sequence, strict=True):
        qo_indptr_cpu.append(qo_indptr_cpu[-1] + qo_len)
        kv_indptr_cpu.append(kv_indptr_cpu[-1] + pages)

    q_nope = torch.randn(qo_indptr_cpu[-1], num_heads, head_dim_ckv, dtype=dtype, device=device)
    q_pe = torch.randn(qo_indptr_cpu[-1], num_heads, head_dim_kpe, dtype=dtype, device=device)
    ckv = torch.randn(kv_indptr_cpu[-1], page_size, head_dim_ckv, dtype=dtype, device=device)
    kpe = torch.randn(kv_indptr_cpu[-1], page_size, head_dim_kpe, dtype=dtype, device=device)
    qo_indptr = torch.tensor(qo_indptr_cpu, dtype=torch.int32, device=device)
    kv_indptr = torch.tensor(kv_indptr_cpu, dtype=torch.int32, device=device)
    kv_indices = torch.arange(kv_indptr_cpu[-1], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPageAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    output, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
    torch.cuda.synchronize()
    output_ref, lse_ref = _attention_reference(
        q_nope,
        q_pe,
        ckv,
        kpe,
        qo_indptr_cpu,
        kv_indptr_cpu,
        kv_lens,
        page_size,
        causal,
        sm_scale,
    )
    output_error = float((output.float() - output_ref.float()).abs().max().item())
    lse_error = float((lse.float() - lse_ref.float()).abs().max().item())
    torch.testing.assert_close(output, output_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
    return {
        "name": name,
        "batch_size": batch_size,
        "kv_lens": kv_lens,
        "qo_lens": qo_lens,
        "page_size": page_size,
        "num_heads": num_heads,
        "causal": causal,
        "max_abs_output_error": output_error,
        "max_abs_lse_error": lse_error,
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import flashinfer

    _set_case_workspace(args.workspace.resolve())
    flashinfer.mla._batch_mla_modules.clear()
    cases = [
        ("single_partial", [33], [17], 16, 4, False),
        ("multi_partial", [33, 47], [17, 9], 16, 8, False),
        ("causal_partial", [47], [17], 16, 4, True),
        ("divisible_control", [64], [17], 16, 4, False),
    ]
    results = [_run_case(*case) for case in cases]
    payload = {
        "probe": "flashinfer-mla-page-size-v1",
        "flashinfer": flashinfer.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "workspace": str(args.workspace.resolve()),
        "status": "pass",
        "cases": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
