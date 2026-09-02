from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.training.native_pytorch import NativePyTorchAdapter


def _torch():
    import torch

    return torch


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        yield float(value)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def _max_abs_error(left: Any, right: Any) -> float:
    left_values = list(_numbers(left))
    right_values = list(_numbers(right))
    if len(left_values) != len(right_values):
        return math.inf
    return max(
        (
            abs(left_item - right_item)
            for left_item, right_item in zip(left_values, right_values, strict=True)
        ),
        default=0.0,
    )


def _fixture() -> dict[str, Any]:
    batch_size = 4
    sequence_length = 32
    vocab_size = 128
    input_ids = [
        [
            1 + ((row * sequence_length + column) % (vocab_size - 1))
            for column in range(sequence_length)
        ]
        for row in range(batch_size)
    ]
    labels = [[*items[1:], -100] for items in input_ids]
    return {
        "vocab_size": vocab_size,
        "hidden_size": 128,
        "seed": 20260901,
        "device": "cuda:0",
        "dtype": "fp32",
        "learning_rate": 1e-3,
        "input_ids": input_ids,
        "labels": labels,
    }


def _normalized_config() -> dict[str, Any]:
    return {
        "global_batch_tokens": 128,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "sequence_length_policy": "packed-variable",
        "precision": "fp32",
        "loss_reduction": "valid-target-token-mean",
        "gradient_clipping": "global-l2",
        "optimizer": "adamw",
        "learning_rate_schedule": "cosine",
        "activation_checkpointing": False,
        "seed_bundle": {"model": 20260901, "data": 17, "sampling": 23, "dropout": 29},
    }


def _build_adapter() -> tuple[NativePyTorchAdapter, dict[str, Any]]:
    adapter = NativePyTorchAdapter()
    adapter.normalize_config(_normalized_config())
    fixture = _fixture()
    adapter.build_model(fixture)
    batch = adapter.build_data(fixture)
    adapter.build_optimizer(fixture)
    return adapter, batch


def run_child(*, replay_index: int, trace_path: Path | None) -> dict[str, Any]:
    torch = _torch()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    reference, reference_batch = _build_adapter()
    candidate, candidate_batch = _build_adapter()
    reference_result = reference.run_reference_step(reference_batch)
    candidate_result = candidate.run_candidate_step(candidate_batch)
    semantic_fields = ("loss", "logits", "gradients", "parameters")
    semantic_errors = {
        field: _max_abs_error(reference_result[field], candidate_result[field])
        for field in semantic_fields
    }
    reference.shutdown()
    candidate.shutdown()

    measured, batch = _build_adapter()
    for _ in range(2):
        measured.run_candidate_step(batch)
    torch.cuda.synchronize()
    step_times_ms: list[float] = []
    losses: list[float] = []
    final_result: dict[str, Any] | None = None
    for _ in range(10):
        started = time.perf_counter_ns()
        final_result = dict(measured.run_candidate_step(batch))
        torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        losses.append(float(final_result["loss"]))

    trace_exported = False
    profiler_wall_time_seconds = 0.0
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        profiler_started = time.perf_counter()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            measured.run_candidate_step(batch)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(trace_path))
        profiler_wall_time_seconds = time.perf_counter() - profiler_started
        trace_exported = trace_path.is_file() and trace_path.stat().st_size > 0

    assert final_result is not None
    memory = measured.memory_stats()
    callgraph = measured.collect_callgraph()
    shutdown = measured.shutdown()
    passed = (
        all(error <= 1e-7 for error in semantic_errors.values())
        and all(math.isfinite(loss) for loss in losses)
        and len(step_times_ms) == 10
        and callgraph["fallback_calls"] == 0
        and shutdown["status"] == "complete"
    )
    return {
        "schema_version": "0.1",
        "status": "pass" if passed else "fail",
        "replay_index": replay_index,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "semantic_max_abs_error": semantic_errors,
        "fallback_calls": callgraph["fallback_calls"],
        "step_times_ms": step_times_ms,
        "losses": losses,
        "result_digest": _digest(
            {
                "loss": final_result["loss"],
                "gradients": final_result["gradients"],
                "parameters": final_result["parameters"],
            }
        ),
        "memory": memory,
        "profiler_trace": str(trace_path) if trace_exported and trace_path else None,
        "profiler_wall_time_seconds": profiler_wall_time_seconds,
    }


def run_ddp_child(output_dir: Path) -> int:
    torch = _torch()
    import torch.distributed as distributed

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")
    value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    distributed.all_reduce(value)
    distributed.barrier()
    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "all_reduce_value": float(value.cpu()),
        "device_name": torch.cuda.get_device_name(local_rank),
        "status": "pass" if world_size == 2 and float(value.cpu()) == 3.0 else "fail",
    }
    atomic_write_json(output_dir / f"ddp-rank-{rank}.json", payload)
    distributed.destroy_process_group()
    return 0 if payload["status"] == "pass" else 1


def _run_ddp_probe(output_dir: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "--ddp-child",
        "--output-dir",
        str(output_dir),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=120,
        env=environment,
        check=False,
    )
    wall_time_seconds = time.perf_counter() - started
    ranks = []
    for rank in range(2):
        path = output_dir / f"ddp-rank-{rank}.json"
        if path.is_file():
            ranks.append(json.loads(path.read_text(encoding="utf-8")))
    passed = (
        completed.returncode == 0
        and len(ranks) == 2
        and all(item["status"] == "pass" for item in ranks)
    )
    return {
        "status": "pass" if passed else "fail",
        "returncode": completed.returncode,
        "wall_time_seconds": wall_time_seconds,
        "ranks": ranks,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def run_parent(*, output_dir: Path, replay_count: int) -> dict[str, Any]:
    torch = _torch()
    parent_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_count = torch.cuda.device_count()
    if device_count < 2:
        return {
            "schema_version": "0.1",
            "status": "unresolved",
            "failure_codes": ["TWO_GPU_CUDA_CELL_UNAVAILABLE"],
            "device_count": device_count,
            "official_score_published": False,
        }

    replays: list[dict[str, Any]] = []
    for replay_index in range(replay_count):
        result_path = output_dir / f"replay-{replay_index}.json"
        trace_path = output_dir / "torch-profiler-replay-0.json" if replay_index == 0 else None
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--output-dir",
            str(output_dir),
            "--replay-index",
            str(replay_index),
            "--result",
            str(result_path),
        ]
        if trace_path is not None:
            command.extend(["--trace", str(trace_path)])
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(replay_index % device_count)
        child_started = time.perf_counter()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=180,
            env=environment,
            check=False,
        )
        child_wall_time_seconds = time.perf_counter() - child_started
        if result_path.is_file():
            replay = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            replay = {
                "status": "fail",
                "replay_index": replay_index,
                "failure": "child did not emit a result",
            }
        replay["returncode"] = completed.returncode
        replay["wall_time_seconds"] = child_wall_time_seconds
        replay["stdout_tail"] = completed.stdout[-1000:]
        replay["stderr_tail"] = completed.stderr[-2000:]
        replays.append(replay)

    ddp = _run_ddp_probe(output_dir)
    all_step_times = [value for replay in replays for value in replay.get("step_times_ms", [])]
    distinct_pids = {replay.get("pid") for replay in replays if replay.get("pid")}
    passed = (
        replay_count >= 5
        and len(distinct_pids) == replay_count
        and all(replay.get("status") == "pass" for replay in replays)
        and ddp["status"] == "pass"
    )
    digests = {replay.get("result_digest") for replay in replays}
    profiler_seconds = sum(float(replay.get("profiler_wall_time_seconds", 0)) for replay in replays)
    accelerator_seconds = sum(
        float(replay.get("wall_time_seconds", 0)) for replay in replays
    ) + 2 * float(ddp.get("wall_time_seconds", 0))
    wall_time_seconds = time.perf_counter() - parent_started
    return {
        "schema_version": "0.1",
        "status": "pass" if passed else "fail",
        "hardware_cell": "2xl40s-sm89-cuda128-torch211",
        "fresh_process_replays": replay_count,
        "fresh_processes_verified": len(distinct_pids) == replay_count,
        "framework_profiler_grade": "G2",
        "system_trace_grade": "unresolved",
        "kernel_counter_grade": "separate-diagnostic-required",
        "official_score_published": False,
        "official_score_blockers": [
            "DRAFT_NOT_HUMAN_REVIEWED_OR_SEALED",
            "G3_SYSTEM_TRACE_UNAVAILABLE",
        ],
        "benchmark_cost": {
            "status": "partial",
            "wall_time_seconds": wall_time_seconds,
            "accelerator_seconds": accelerator_seconds,
            "compile_seconds": 0.0,
            "profiler_seconds": profiler_seconds,
            "executed_cases": replay_count + 1,
            "skipped_cases": 0,
            "cache_hit_ratio": None,
            "time_to_first_diagnostic_seconds": None,
            "time_to_actionable_decision_seconds": wall_time_seconds,
            "fast_stage_resolution_rate": None,
            "serialization_config_compatibility": "unresolved",
            "failure_codes": [
                "COMPILE_CACHE_NOT_APPLICABLE_EAGER",
                "SERIALIZATION_COMPATIBILITY_NOT_PROBED",
                "DRAFT_STAGE_TIMING_NOT_INSTRUMENTED",
            ],
        },
        "gpu_inventory": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(device_count)
        ],
        "step_time_ms": {
            "count": len(all_step_times),
            "median": statistics.median(all_step_times) if all_step_times else None,
            "p95": _percentile(all_step_times, 0.95) if all_step_times else None,
            "minimum": min(all_step_times) if all_step_times else None,
            "maximum": max(all_step_times) if all_step_times else None,
        },
        "cross_replay_result_digest_count": len(digests),
        "replays": replays,
        "ddp_nccl_probe": ddp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fresh-process native PyTorch training replays on two CUDA GPUs"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=7)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--ddp-child", action="store_true")
    parser.add_argument("--replay-index", type=int, default=0)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.ddp_child:
        return run_ddp_child(args.output_dir)
    if args.child:
        if args.result is None:
            parser.error("--child requires --result")
        payload = run_child(replay_index=args.replay_index, trace_path=args.trace)
        atomic_write_json(args.result, payload)
        return 0 if payload["status"] == "pass" else 1
    payload = run_parent(output_dir=args.output_dir, replay_count=args.replays)
    atomic_write_json(args.output_dir / "remote-training-replays.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
