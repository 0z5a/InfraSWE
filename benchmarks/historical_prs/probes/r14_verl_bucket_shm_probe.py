#!/usr/bin/env python3
"""Execute exact candidate SHM bucket helpers on CUDA despite their NPU-only pytest marker."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import torch


def _canonical(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _run(name: str, function: Callable[[], None]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        function()
    except Exception as exception:
        return {
            "name": name,
            "passed": False,
            "duration_seconds": time.monotonic() - started,
            "error": f"{type(exception).__name__}: {exception}",
        }
    return {
        "name": name,
        "passed": True,
        "duration_seconds": time.monotonic() - started,
        "error": None,
    }


def _check_callback_exception_preserved(_candidate_tests: Any) -> None:
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
        BucketedWeightReceiver,
    )

    class _Socket:
        def recv_pyobj(self) -> dict[str, Any]:
            return {
                "bucket_meta": {
                    "layer.weight": {
                        "shape": torch.Size((32, 16)),
                        "dtype": torch.float32,
                        "offset": 0,
                        "handle": None,
                    }
                },
                "is_last": True,
            }

        def send(self, _message: bytes) -> None:
            pass

        def close(self) -> None:
            pass

    shm = shared_memory.SharedMemory(create=True, size=4096)
    receiver = BucketedWeightReceiver(
        zmq_handle="ipc:///tmp/r14-unused.sock",
        device=torch.device("cuda:0"),
        use_shm=True,
        overlap_bucket_processing=True,
    )
    socket = _Socket()

    def _init_socket() -> None:
        receiver.socket = socket

    def _init_buffer() -> None:
        receiver.buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
        receiver.shm = shm

    receiver._init_socket = _init_socket
    receiver._init_buffer = _init_buffer

    def _boom(_weights: object, _is_last: bool) -> None:
        raise RuntimeError("callback boom")

    try:
        receiver.receive_weights(on_bucket_received=_boom)
    except BaseException as exception:
        exception_name = type(exception).__name__
    else:
        exception_name = "no-exception"
    finally:
        receiver.buffer = None
        receiver.shm = None
        gc.collect()
        with contextlib.suppress(BufferError):
            shm.close()
        with contextlib.suppress(FileNotFoundError):
            shm.unlink()
    if exception_name != "RuntimeError":
        raise AssertionError(f"callback RuntimeError was masked by {exception_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=("verl-pr-7591", "verl-pr-7589"), required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from tests.utils import test_bucketed_weight_transfer as candidate_tests

    checks: list[tuple[str, Callable[[], None]]] = [
        (
            "shm_overlap_multiple_buckets",
            lambda: candidate_tests._transfer_and_validate(
                [(f"layer{i}.weight", (128, 128), torch.float32) for i in range(20)],
                bucket_size_mb=1,
                use_shm=True,
                overlap_bucket_processing=True,
            ),
        ),
        (
            "shm_overlap_mixed_dtypes",
            lambda: candidate_tests._transfer_and_validate(
                [
                    ("fp32_param", (64, 64), torch.float32),
                    ("bf16_param", (64, 64), torch.bfloat16),
                    ("fp16_param", (32, 32), torch.float16),
                ],
                bucket_size_mb=1,
                use_shm=True,
                overlap_bucket_processing=True,
            ),
        ),
        (
            "shm_overlap_empty_weights",
            lambda: candidate_tests._transfer_and_validate(
                [],
                bucket_size_mb=1,
                use_shm=True,
                overlap_bucket_processing=True,
            ),
        ),
        (
            "shm_callback_exception_preserved_independent",
            lambda: _check_callback_exception_preserved(candidate_tests),
        ),
    ]
    if hasattr(
        candidate_tests.TestBucketedWeightTransferSHM,
        "test_callback_exception_not_masked_by_cleanup",
    ):
        checks.append(
            (
                "shm_callback_exception_preserved",
                candidate_tests.TestBucketedWeightTransferSHM().test_callback_exception_not_masked_by_cleanup,
            )
        )
    results = [_run(name, function) for name, function in checks]
    material = {
        "schema_version": "0.1",
        "protocol_id": "r14-verl-exact-candidate-shm-on-cuda-v0.1",
        "case_id": args.case_id,
        "head_sha": args.head_sha,
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
        },
        "method": (
            "Calls the exact candidate-owned helpers directly. Their pytest class is skipped on "
            "CUDA only because the project labels SHM as NPU-tested; the implementation chooses "
            "CUDA as the target accelerator and retains all candidate assertions."
        ),
        "outcome_review_ci_fields_requested": False,
        "results": results,
        "all_passed": all(result["passed"] for result in results),
    }
    payload = {**material, "evidence_sha256": _canonical(material)}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": payload["all_passed"], "results": results}))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
