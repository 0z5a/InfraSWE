from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

repo = Path(os.environ["INFRASWE_REPO"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
workload_dir = Path(os.environ["INFRASWE_WORKLOAD_DIR"])
faults_dir = Path(os.environ["INFRASWE_FAULTS_DIR"])
evidence.mkdir(parents=True, exist_ok=True)


def write_json(name: str, value: object) -> None:
    (evidence / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def load_policy():
    spec = importlib.util.spec_from_file_location(
        "candidate_batch_policy", repo / "batch_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load batch_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, requests, config) -> tuple[Any, str | None, bool]:
    requests_copy = copy.deepcopy(requests)
    config_copy = copy.deepcopy(config)
    try:
        schedule = module.schedule_batches(requests_copy, config_copy)
        error = None
    except Exception as caught:
        schedule = None
        error = f"{type(caught).__name__}: {caught}"
    return schedule, error, requests_copy == requests and config_copy == config


def validate_schedule(
    schedule: Any, requests: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[bool, dict[str, bool]]:
    if not isinstance(schedule, list) or any(not isinstance(batch, list) for batch in schedule):
        return False, {}
    by_id = {request["id"]: request for request in requests}
    flattened = [identifier for batch in schedule for identifier in batch]
    all_once = (
        all(isinstance(identifier, str) and identifier in by_id for identifier in flattened)
        and len(flattened) == len(set(flattened))
        and set(flattened) == set(by_id)
    )
    if not all_once:
        return False, {"all_requests_once": False}
    size_ok = all(0 < len(batch) <= config["max_batch_size"] for batch in schedule)
    model_ok = all(len({by_id[item]["model"] for item in batch}) == 1 for batch in schedule)
    tokens_ok = all(
        sum(by_id[item]["tokens"] for item in batch) <= config["max_batch_tokens"]
        for batch in schedule
    )
    wait_ok = all(
        max(by_id[item]["arrival_ms"] for item in batch)
        - min(by_id[item]["arrival_ms"] for item in batch)
        <= config["max_wait_ms"]
        for batch in schedule
    )
    checks = {
        "all_requests_once": all_once,
        "batch_size_bounded": size_ok,
        "model_isolated": model_ok,
        "token_budget_bounded": tokens_ok,
        "wait_span_bounded": wait_ok,
    }
    return all(checks.values()), checks


def semantic_outcome(schedule: list[list[str]], requests: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {request["id"]: request for request in requests}
    latencies: list[float] = []
    good = 0
    for batch in schedule:
        total_tokens = sum(by_id[item]["tokens"] for item in batch)
        service_ms = 2.0 + total_tokens / 512 + 0.2 * len(batch) ** 2
        finish_ms = max(by_id[item]["arrival_ms"] for item in batch) + service_ms
        for identifier in batch:
            request = by_id[identifier]
            latency = finish_ms - request["arrival_ms"]
            latencies.append(latency)
            good += int(finish_ms <= request["deadline_ms"])
    ordered = sorted(latencies)
    p99 = ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]
    return {
        "goodput_ratio": good / len(requests),
        "latencies_ms": latencies,
        "p99_ms": p99,
    }


def gpu_batches(schedule: list[list[str]]) -> tuple[list[dict[str, Any]], bool, bool]:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    before = int(torch.cuda.memory_allocated(0))
    probes: list[dict[str, Any]] = []
    correct = True
    steady = before
    for index, batch in enumerate(schedule):
        elements = 65536 + len(batch) * 1024
        vector = torch.arange(elements, device="cuda:0", dtype=torch.float32)
        warmup = vector.clone()
        del warmup
        torch.cuda.synchronize(0)
        steady = max(steady, int(torch.cuda.memory_allocated(0)))
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = vector.clone()
        stop.record()
        stop.synchronize()
        elapsed_ms = float(start.elapsed_time(stop))
        matches = bool(torch.equal(output, vector))
        correct = correct and matches and elapsed_ms > 0
        probes.append(
            {
                "batch_index": index,
                "batch_size": len(batch),
                "correct": matches,
                "elapsed_ms": elapsed_ms,
            }
        )
        del output, vector, start, stop
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    after = int(torch.cuda.memory_allocated(0))
    return probes, correct, after <= steady + 1_048_576


try:
    traffic = load_json(workload_dir / "traffic.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "serving_config.json")
    module = load_policy()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}")
    requests = [
        {
            "arrival_ms": index % 4,
            "deadline_ms": 15 + index % 4,
            "id": f"request-{index:02d}",
            "model": "model-a" if index % 2 == 0 else "model-b",
            "tokens": 128 if index % 3 else 192,
        }
        for index in range(16)
    ]
    schedule, schedule_error, immutable = call(module, requests, config)
    schedule_valid, constraint_checks = validate_schedule(schedule, requests, config)
    reversed_schedule, reversed_error, reversed_immutable = call(
        module, list(reversed(requests)), config
    )
    order_independent = (
        schedule_error is None and reversed_error is None and schedule == reversed_schedule
    )
    oversized = [
        {
            "arrival_ms": 0,
            "deadline_ms": 20,
            "id": "oversized",
            "model": "model-a",
            "tokens": int(config.get("max_batch_tokens", 0)) + 1,
        }
    ]
    oversized_rejected = False
    try:
        module.schedule_batches(oversized, config)
    except Exception:
        oversized_rejected = True
    malformed_rejected = False
    try:
        module.schedule_batches({}, config)
    except Exception:
        malformed_rejected = True
    outcome = (
        semantic_outcome(schedule, requests)
        if schedule_valid
        else {
            "goodput_ratio": 0.0,
            "latencies_ms": [999999.0],
            "p99_ms": 999999.0,
        }
    )
    probes, gpu_correct, resources_cleaned = (
        gpu_batches(schedule)
        if schedule_valid
        else (
            [],
            False,
            True,
        )
    )
    regression = {
        **constraint_checks,
        "config_is_deadline_aware": config.get("deadline_aware") is True
        and config.get("group_by_model") is True,
        "input_immutable": immutable and reversed_immutable,
        "malformed_input_rejected": malformed_rejected,
        "order_independent": order_independent,
        "oversized_request_rejected": oversized_rejected,
    }
    write_json(
        "batch-evidence.json",
        {
            "constraints": constraint_checks,
            "gpu_batches": probes,
            "outcome": outcome,
            "schedule": schedule,
            "schedule_error": schedule_error,
        },
    )
    minimum_goodput = float(traffic["minimum_goodput_ratio"])
    maximum_p99 = float(traffic["maximum_p99_ms"])
    assertions = {
        "functional": {
            "all_requests_scheduled_once": constraint_checks.get("all_requests_once", False),
            "gpu_batches_correct": gpu_correct,
        },
        "mechanism": {
            "deadline_aware_microbatching": schedule_valid,
            "models_isolated": constraint_checks.get("model_isolated", False),
        },
        "regression": regression,
        "safety": {
            "no_request_dropped": constraint_checks.get("all_requests_once", False),
            "oversized_request_explicit": oversized_rejected,
            "resources_cleaned": resources_cleaned,
        },
        "slo": {
            "deadline_goodput": outcome["goodput_ratio"] >= minimum_goodput,
            "tail_latency": outcome["p99_ms"] <= maximum_p99,
        },
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "mixed-model-burst": constraint_checks.get("model_isolated", False),
        "oversized-request": oversized_rejected,
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    policy_result = {
        "data_corruption": not gpu_correct,
        "hard_failures": [],
        "passed": gpu_correct and resources_cleaned,
        "resource_leak": not resources_cleaned,
        "silent_fallback": not schedule_valid or outcome["goodput_ratio"] < minimum_goodput,
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "observability_ratio": float((evidence / "batch-evidence.json").is_file()),
        "p99_latency_ms": outcome["p99_ms"],
        "resource_efficiency_ratio": min(1.0, config.get("max_batch_size", 999) / 4),
        "slo_goodput_ratio": min(1.0, outcome["goodput_ratio"] / minimum_goodput),
        "topology_robustness_ratio": regression_ratio,
    }
except Exception as error:
    assertions = {"functional": {"verifier_completed": False}}
    faults_result = {"error": str(error), "passed": False, "score": 0.0}
    policy_result = {
        "data_corruption": False,
        "hard_failures": [],
        "passed": False,
        "resource_leak": False,
        "silent_fallback": False,
        "verifier_error": str(error),
    }
    metrics = {
        "observability_ratio": 0.25,
        "resource_efficiency_ratio": 0.0,
        "slo_goodput_ratio": 0.0,
        "topology_robustness_ratio": 0.0,
    }

write_json("assertions.json", assertions)
write_json("faults.json", faults_result)
write_json("policy.json", policy_result)
write_json("metrics.json", metrics)
passed = all(
    value
    for group in assertions.values()
    for value in (group.values() if isinstance(group, dict) else [group])
)
raise SystemExit(0 if passed and faults_result["passed"] and policy_result["passed"] else 1)
