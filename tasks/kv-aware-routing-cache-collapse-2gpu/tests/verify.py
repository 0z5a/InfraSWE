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
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_router():
    spec = importlib.util.spec_from_file_location("candidate_router", repo / "router.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load router.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def gpu_probe(expected_devices: int) -> tuple[dict[str, Any], bool, bool]:
    actual_devices = torch.cuda.device_count()
    if actual_devices != expected_devices:
        raise RuntimeError(f"expected {expected_devices} visible GPUs, got {actual_devices}")
    devices: list[dict[str, Any]] = []
    correct = True
    cleaned = True
    for index in range(expected_devices):
        torch.cuda.set_device(index)
        torch.cuda.empty_cache()
        before = int(torch.cuda.memory_allocated(index))
        generator = torch.Generator(device=f"cuda:{index}").manual_seed(4100 + index)
        left = torch.randn((512, 512), device=f"cuda:{index}", generator=generator)
        right = torch.randn((512, 512), device=f"cuda:{index}", generator=generator)
        warmup = None
        for _ in range(2):
            warmup = torch.mm(left, right)
        del warmup
        torch.cuda.synchronize(index)
        steady_state = int(torch.cuda.memory_allocated(index))
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        product = torch.mm(left, right)
        stop.record()
        stop.synchronize()
        checksum = float(product.float().sum().item())
        finite = math.isfinite(checksum)
        elapsed_ms = float(start.elapsed_time(stop))
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "checksum": checksum,
                "compute_capability": [properties.major, properties.minor],
                "device": index,
                "elapsed_ms": elapsed_ms,
                "finite": finite,
                "memory_allocated_steady_state_bytes": steady_state,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
        correct = correct and finite and elapsed_ms > 0
        del product, right, left, start, stop
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(index)
        after = int(torch.cuda.memory_allocated(index))
        devices[-1]["memory_allocated_after_bytes"] = after
        devices[-1]["memory_allocated_before_bytes"] = before
        cleaned = cleaned and after <= steady_state + 1_048_576

    peer_access = [
        [
            True if source == target else bool(torch.cuda.can_device_access_peer(source, target))
            for target in range(expected_devices)
        ]
        for source in range(expected_devices)
    ]
    return (
        {
            "cuda_version": torch.version.cuda,
            "device_count": actual_devices,
            "devices": devices,
            "peer_access": peer_access,
            "torch_version": torch.__version__,
        },
        correct,
        cleaned,
    )


def call_router(module, request, workers, available, config) -> tuple[int | None, bool]:
    request_copy = copy.deepcopy(request)
    workers_copy = copy.deepcopy(workers)
    available_copy = copy.deepcopy(available)
    config_copy = copy.deepcopy(config)
    try:
        selected = module.choose_worker(request_copy, workers_copy, available_copy, config_copy)
    except Exception:
        selected = None
    immutable = (
        request_copy == request
        and workers_copy == workers
        and available_copy == available
        and config_copy == config
    )
    if isinstance(selected, bool) or not isinstance(selected, int):
        return None, immutable
    return selected, immutable


def regression_checks(module, config: dict[str, Any]) -> dict[str, bool]:
    prefixes = ["regression-prefix-a", "regression-prefix-b", "regression-prefix-c"]
    workers = [
        {"cache": {}, "id": "gpu-0", "load": 0},
        {"cache": {}, "id": "gpu-1", "load": 0},
    ]
    stable = True
    availability_aware = True
    immutable = True
    for prefix in prefixes:
        request = {"prefix_id": prefix, "request_id": 7, "step": 20}
        selections: list[int | None] = []
        for _ in range(5):
            selected, unchanged = call_router(module, request, workers, [True, True], config)
            selections.append(selected)
            immutable = immutable and unchanged
        stable = stable and len(set(selections)) == 1 and selections[0] in {0, 1}
        if selections[0] in {0, 1}:
            owner = int(selections[0])
            cached_workers = copy.deepcopy(workers)
            cached_workers[owner]["cache"][prefix] = 19
            cached, unchanged = call_router(module, request, cached_workers, [True, True], config)
            fallback, unchanged_fallback = call_router(
                module,
                request,
                cached_workers,
                [index != owner for index in range(2)],
                config,
            )
            stable = stable and cached == owner
            availability_aware = availability_aware and fallback == 1 - owner
            immutable = immutable and unchanged and unchanged_fallback
        else:
            availability_aware = False

    malformed_rejected = False
    try:
        module.choose_worker({}, [], [], config)
    except Exception:
        malformed_rejected = True
    return {
        "availability_aware": availability_aware,
        "capacity_ceiling_preserved": config.get("cache_capacity_entries") == 4,
        "input_immutable": immutable,
        "malformed_input_rejected": malformed_rejected,
        "prefix_route_deterministic": stable,
        "ttl_bounded": isinstance(config.get("cache_ttl_steps"), int)
        and not isinstance(config.get("cache_ttl_steps"), bool)
        and 8 <= config["cache_ttl_steps"] <= 128,
    }


def run_workload(
    module,
    config: dict[str, Any],
    workload: dict[str, Any],
    fault: dict[str, Any],
) -> dict[str, Any]:
    prefixes = workload["prefixes"]
    rounds = int(workload["request_rounds"])
    capacity = int(workload["cache_capacity_per_worker"])
    ttl_steps = config.get("cache_ttl_steps", 0)
    if isinstance(ttl_steps, bool) or not isinstance(ttl_steps, int):
        ttl_steps = 0
    workers: list[dict[str, Any]] = [
        {"cache": {}, "id": "gpu-0", "load": 0},
        {"cache": {}, "id": "gpu-1", "load": 0},
    ]
    delivered = [0, 0]
    events: list[dict[str, Any]] = []
    latencies: list[float] = []
    hits = 0
    errors = 0
    unavailable_routes = 0
    input_immutable = True
    recovered_after_requests: int | None = None
    start_request = int(fault["start_request"])
    end_request = int(fault["end_request"])
    failed_worker = int(fault["worker_index"])

    total_requests = rounds * len(prefixes)
    for request_id in range(total_requests):
        fault_active = start_request <= request_id < end_request
        if request_id == start_request and fault.get("clear_cache_on_failure", False):
            workers[failed_worker]["cache"].clear()
        available = [True, True]
        if fault_active:
            available[failed_worker] = False
        request = {
            "prefix_id": prefixes[request_id % len(prefixes)],
            "request_id": request_id,
            "step": request_id,
        }
        selected, unchanged = call_router(module, request, workers, available, config)
        input_immutable = input_immutable and unchanged
        valid = selected in {0, 1}
        accepted = bool(valid and available[int(selected)])
        hit = False
        if not accepted:
            errors += 1
            latency_ms = float(workload["error_latency_ms"])
            if valid and not available[int(selected)]:
                unavailable_routes += 1
        else:
            worker_index = int(selected)
            cache = workers[worker_index]["cache"]
            cached_at = cache.get(request["prefix_id"])
            hit = (
                isinstance(cached_at, int)
                and cached_at <= request_id
                and request_id - cached_at <= ttl_steps
            )
            if hit:
                hits += 1
                latency_ms = float(workload["hit_latency_ms"])
            else:
                latency_ms = float(workload["miss_latency_ms"])
            cache[request["prefix_id"]] = request_id
            if len(cache) > capacity:
                victim = min(cache, key=lambda prefix: (cache[prefix], prefix))
                del cache[victim]
            workers[worker_index]["load"] += 1
            delivered[worker_index] += 1
            if request_id >= end_request and worker_index == failed_worker:
                recovered_after_requests = recovered_after_requests or request_id - end_request + 1

        latencies.append(latency_ms)
        events.append(
            {
                "accepted": accepted,
                "cache_hit": hit,
                "fault_active": fault_active,
                "latency_ms": latency_ms,
                "prefix_id": request["prefix_id"],
                "request_id": request_id,
                "selected_worker": selected,
            }
        )

    successful = total_requests - errors
    good = sum(
        event["accepted"] and event["latency_ms"] <= float(workload["slo_latency_ms"])
        for event in events
    )
    maximum_delivered = max(delivered)
    fairness = min(delivered) / maximum_delivered if maximum_delivered else 0.0
    return {
        "average_latency_ms": sum(latencies) / len(latencies),
        "cache_hit_ratio": hits / total_requests,
        "delivered_by_worker": delivered,
        "error_rate": errors / total_requests,
        "events": events,
        "fairness_ratio": fairness,
        "input_immutable": input_immutable,
        "p99_latency_ms": percentile(latencies, 0.99),
        "recovered_after_requests": recovered_after_requests,
        "slo_goodput_ratio": good / total_requests,
        "successful_requests": successful,
        "total_requests": total_requests,
        "unavailable_routes": unavailable_routes,
    }


try:
    config = load_json(repo / "routing_config.json")
    workload = load_json(workload_dir / "workload.json")
    faults = load_json(faults_dir / "scenarios.json")
    fault = faults["scenarios"][0]
    module = load_router()
    probe, gpu_correct, resources_cleaned = gpu_probe(2)
    write_json("gpu-probe.json", probe)
    regression = regression_checks(module, config)
    result = run_workload(module, config, workload, fault)
    events = result.pop("events")
    (evidence / "routing-events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    write_json("routing-summary.json", result)

    deadline = int(fault["recovery_deadline_requests"])
    recovered = result["recovered_after_requests"]
    cache_hit_passed = result["cache_hit_ratio"] >= workload["minimum_cache_hit_ratio"]
    assertions = {
        "functional": {
            "all_requests_routed": result["successful_requests"] == result["total_requests"],
            "two_gpu_execution": gpu_correct,
            "valid_worker_for_every_request": result["error_rate"] == 0,
        },
        "mechanism": {
            "kv_affinity_declared": config.get("strategy") == "kv_affinity",
            "prefix_stability_observed": regression["prefix_route_deterministic"],
        },
        "regression": regression,
        "safety": {
            "no_data_corruption": gpu_correct,
            "no_unavailable_worker_route": result["unavailable_routes"] == 0,
            "resources_cleaned": resources_cleaned,
            "worker_recovered_within_deadline": recovered is not None and recovered <= deadline,
        },
        "slo": {
            "cache_hit_ratio": cache_hit_passed,
            "error_rate": result["error_rate"] <= workload["maximum_error_rate"],
            "fairness": result["fairness_ratio"] >= workload["minimum_fairness_ratio"],
            "goodput": result["slo_goodput_ratio"] >= workload["minimum_slo_goodput_ratio"],
            "tail_latency": result["p99_latency_ms"] <= workload["maximum_p99_ms"],
        },
    }
    faults_result = {
        "passed": result["unavailable_routes"] == 0
        and recovered is not None
        and recovered <= deadline,
        "recovered_after_requests": recovered,
        "scenario": fault["id"],
        "score": 1.0
        if result["unavailable_routes"] == 0 and recovered is not None and recovered <= deadline
        else 0.0,
    }
    claimed_affinity = config.get("strategy") == "kv_affinity"
    policy = {
        "data_corruption": not gpu_correct,
        "hard_failures": [],
        "passed": gpu_correct and resources_cleaned,
        "resource_leak": not resources_cleaned,
        "silent_fallback": claimed_affinity and not cache_hit_passed,
    }
    observable = (
        sum(
            path.is_file()
            for path in (
                evidence / "gpu-probe.json",
                evidence / "routing-events.jsonl",
                evidence / "routing-summary.json",
            )
        )
        / 3
    )
    topology_robustness = sum(regression.values()) / len(regression)
    metrics = {
        "cache_hit_ratio": result["cache_hit_ratio"],
        "error_rate": result["error_rate"],
        "fairness_ratio": result["fairness_ratio"],
        "observability_ratio": observable,
        "observed_slo_goodput_ratio": result["slo_goodput_ratio"],
        "p99_latency_ms": result["p99_latency_ms"],
        "resource_efficiency_ratio": min(1.0, 4.0 / max(result["average_latency_ms"], 4.0)),
        "slo_goodput_ratio": min(
            1.0,
            result["slo_goodput_ratio"] / workload["minimum_slo_goodput_ratio"],
        ),
        "topology_robustness_ratio": topology_robustness,
    }
except Exception as error:
    assertions = {"functional": {"verifier_completed": False}}
    faults_result = {"error": str(error), "passed": False, "score": 0.0}
    policy = {
        "data_corruption": False,
        "hard_failures": [],
        "passed": False,
        "resource_leak": False,
        "silent_fallback": False,
        "verifier_error": str(error),
    }
    metrics = {
        "cache_hit_ratio": 0.0,
        "error_rate": 1.0,
        "fairness_ratio": 0.0,
        "observability_ratio": 0.25,
        "p99_latency_ms": 999999.0,
        "resource_efficiency_ratio": 0.0,
        "slo_goodput_ratio": 0.0,
        "topology_robustness_ratio": 0.0,
    }

write_json("assertions.json", assertions)
write_json("faults.json", faults_result)
write_json("policy.json", policy)
write_json("metrics.json", metrics)
all_assertions = all(
    value
    for group in assertions.values()
    for value in (group.values() if isinstance(group, dict) else [group])
)
raise SystemExit(0 if all_assertions and faults_result["passed"] and policy["passed"] else 1)
