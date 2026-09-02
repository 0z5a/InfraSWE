#!/usr/bin/env python3
"""Execute the exact vLLM MessageQueue.dequeue bodies without importing vLLM."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_dequeue(path: Path) -> Callable[..., Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MessageQueue":
            function = next(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "dequeue"
            )
            break
    if function is None:
        raise ValueError(f"MessageQueue.dequeue not found in {path}")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["dequeue"]


def _harness(path: Path) -> type:
    dequeue = _extract_dequeue(path)

    class MessageQueue:
        @staticmethod
        def recv(socket: Callable[[float | None], Any], timeout: float | None) -> Any:
            return socket(timeout)

    MessageQueue.dequeue = dequeue  # type: ignore[attr-defined]
    dequeue.__globals__["MessageQueue"] = MessageQueue
    return MessageQueue


def _reader(harness: type, socket: Callable[[float | None], Any]) -> Any:
    reader = harness()
    reader._dequeue_lock = threading.Lock()
    reader._is_local_reader = False
    reader._is_remote_reader = True
    reader.remote_socket = socket
    return reader


def _serialization_probe(harness: type) -> dict[str, Any]:
    state_lock = threading.Lock()
    start = threading.Barrier(3)
    active = 0
    max_active = 0
    results: list[int] = []
    errors: list[str] = []

    def socket(_: float | None) -> int:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.025)
        with state_lock:
            active -= 1
        return threading.get_ident()

    reader = _reader(harness, socket)

    def consume() -> None:
        start.wait()
        try:
            results.append(reader.dequeue(timeout=1.0))
        except Exception as error:  # pragma: no cover - evidence capture
            errors.append(f"{type(error).__name__}:{error}")

    threads = [threading.Thread(target=consume) for _ in range(2)]
    for thread in threads:
        thread.start()
    started = time.perf_counter()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "max_concurrent_recv": max_active,
        "result_count": len(results),
        "errors": errors,
        "threads_alive": sum(thread.is_alive() for thread in threads),
    }


def _timeout_probe(harness: type, *, hold_seconds: float, timeout: float) -> dict[str, Any]:
    def socket(requested_timeout: float | None) -> None:
        assert requested_timeout is not None
        time.sleep(requested_timeout)
        raise TimeoutError("simulated socket timeout")

    reader = _reader(harness, socket)
    reader._dequeue_lock.acquire()

    def release() -> None:
        time.sleep(hold_seconds)
        reader._dequeue_lock.release()

    releaser = threading.Thread(target=release)
    releaser.start()
    started = time.perf_counter()
    error_name = None
    try:
        reader.dequeue(timeout=timeout)
    except Exception as error:  # expected
        error_name = type(error).__name__
    elapsed = time.perf_counter() - started
    releaser.join(timeout=1)
    return {
        "hold_seconds": hold_seconds,
        "requested_timeout_seconds": timeout,
        "elapsed_seconds": elapsed,
        "deadline_overshoot_seconds": max(0.0, elapsed - timeout),
        "error": error_name,
    }


def _overhead_probe(harness: type, *, calls: int = 100_000) -> dict[str, Any]:
    reader = _reader(harness, lambda _: 1)
    samples: list[float] = []
    for _ in range(7):
        started = time.perf_counter_ns()
        for _ in range(calls):
            reader.dequeue(timeout=None)
        samples.append((time.perf_counter_ns() - started) / calls)
    return {
        "calls_per_replay": calls,
        "replays": len(samples),
        "median_nanoseconds_per_call": statistics.median(samples),
        "samples_nanoseconds_per_call": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--base-tests", type=Path, required=True)
    parser.add_argument("--head-tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = _harness(args.base)
    head = _harness(args.head)
    base_serial = _serialization_probe(base)
    head_serial = _serialization_probe(head)
    base_timeout = _timeout_probe(base, hold_seconds=0.08, timeout=0.10)
    head_timeout = _timeout_probe(head, hold_seconds=0.08, timeout=0.10)
    base_overhead = _overhead_probe(base)
    head_overhead = _overhead_probe(head)
    added_tests = args.head_tests.read_text(encoding="utf-8").replace(
        args.base_tests.read_text(encoding="utf-8"), ""
    )
    payload = {
        "schema_version": "0.5",
        "probe": "vllm-shm-dequeue-lock-r1",
        "source": {
            "base_sha256": _sha256(args.base),
            "head_sha256": _sha256(args.head),
        },
        "serialization": {"base": base_serial, "head": head_serial},
        "timeout_budget": {"base": base_timeout, "head": head_timeout},
        "uncontended_overhead": {
            "base": base_overhead,
            "head": head_overhead,
            "head_to_base_ratio": (
                head_overhead["median_nanoseconds_per_call"]
                / base_overhead["median_nanoseconds_per_call"]
            ),
        },
        "test_contract": {
            "concurrent_consumer_test_added": "test_dequeue_serializes_concurrent_consumers"
            in args.head_tests.read_text(encoding="utf-8"),
            "lock_wait_deadline_test_added": "deadline_overshoot" in added_tests
            or "hold_lock" in added_tests,
        },
        "findings": {
            "concurrent_recv_is_serialized": head_serial["max_concurrent_recv"] == 1,
            "single_timeout_budget_preserved": head_timeout["elapsed_seconds"] <= 0.13,
            "timeout_is_restarted_after_lock_wait": head_timeout["elapsed_seconds"] >= 0.16,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
