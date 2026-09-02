from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import torch
import triton_kernels
from attention_bench import CASES as ATTENTION_CASES
from attention_bench import load_adapter, make_qkv
from bench_utils import (
    atomic_write_json,
    hardware_manifest,
    module_evidence,
    profiler_evidence,
    utc_now,
)
from classic_bench import CASES as CLASSIC_CASES


def attention_profile(backend: str, case_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = next(item for item in ATTENTION_CASES if item["id"] == case_id)
    adapter = load_adapter(backend)
    q, k, v = make_qkv(case, 97_000 + ATTENTION_CASES.index(case))
    candidate = adapter.prepare(q, k, v, case["causal"])
    for _ in range(5):
        candidate()
    torch.cuda.synchronize()
    return profiler_evidence(candidate), adapter.provenance


def classic_profile(case_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = next(item for item in CLASSIC_CASES if item.case_id == case_id)
    _, candidate = case.prepare(67_000 + CLASSIC_CASES.index(case))
    for _ in range(5):
        candidate()
    torch.cuda.synchronize()
    return profiler_evidence(candidate), [module_evidence(triton_kernels)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("attention", "classic"), required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "schema_version": "0.3",
        "evidence_kind": "per-case-profiler",
        "suite": args.suite,
        "backend": args.backend,
        "case_id": args.case_id,
        "generated_at": utc_now(),
        "hardware": hardware_manifest(),
        "fresh_process": True,
        "status": "running",
    }
    try:
        if args.suite == "attention":
            profile, provenance = attention_profile(args.backend, args.case_id)
        else:
            profile, provenance = classic_profile(args.case_id)
        payload["profiler"] = profile
        payload["implementation_provenance"] = provenance
        payload["status"] = (
            "passed" if profile.get("captured") and profile.get("cuda_events") else "failed"
        )
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    atomic_write_json(args.output, payload)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
