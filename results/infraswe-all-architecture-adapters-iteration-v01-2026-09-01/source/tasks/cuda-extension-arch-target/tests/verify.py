from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import os
import re
import time
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
    spec = importlib.util.spec_from_file_location("candidate_arch_policy", repo / "arch_policy.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arch_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict) or set(plan) != {
        "fallback_reported",
        "ptx_target",
        "reason",
        "sass_targets",
        "schema_version",
        "status",
    }:
        return False
    if (
        plan.get("schema_version") != "1"
        or not isinstance(plan.get("reason"), str)
        or not plan["reason"]
        or not isinstance(plan.get("sass_targets"), list)
        or any(isinstance(sm, bool) or not isinstance(sm, int) for sm in plan["sass_targets"])
        or plan["sass_targets"] != sorted(set(plan["sass_targets"]))
        or not isinstance(plan.get("fallback_reported"), bool)
    ):
        return False
    if plan.get("status") == "ready":
        return bool(
            plan["sass_targets"]
            and plan.get("ptx_target") is None
            and not plan["fallback_reported"]
        )
    return bool(
        plan.get("status") == "blocked"
        and not plan["sass_targets"]
        and plan.get("ptx_target") is None
        and plan["fallback_reported"]
    )


def call(module, sms, toolkit, config) -> tuple[Any, str | None, bool, float]:
    sms_copy = copy.deepcopy(sms)
    toolkit_copy = copy.deepcopy(toolkit)
    config_copy = copy.deepcopy(config)
    started = time.perf_counter()
    try:
        plan = module.select_targets(sms_copy, toolkit_copy, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    immutable = sms_copy == sms and toolkit_copy == toolkit and config_copy == config
    return plan, error, immutable, elapsed_ms


def gpu_probe() -> tuple[dict[str, Any], bool, bool]:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    before = int(torch.cuda.memory_allocated(0))
    source = torch.arange(262144, device="cuda:0", dtype=torch.float32)
    warmup = source * 2 + 1
    del warmup
    torch.cuda.synchronize(0)
    steady = int(torch.cuda.memory_allocated(0))
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    product = source * 2 + 1
    stop.record()
    stop.synchronize()
    elapsed_ms = float(start.elapsed_time(stop))
    expected = torch.tensor([1, 3, 5, 7], device="cuda:0", dtype=torch.float32)
    correct = bool(torch.equal(product[:4], expected))
    del expected, product, source, start, stop
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    after = int(torch.cuda.memory_allocated(0))
    return (
        {
            "correct": correct,
            "elapsed_ms": elapsed_ms,
            "memory_after_bytes": after,
            "memory_before_bytes": before,
            "memory_steady_bytes": steady,
        },
        correct and math.isfinite(elapsed_ms) and elapsed_ms > 0,
        after <= steady + 1_048_576,
    )


try:
    workload = load_json(workload_dir / "cases.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "build_config.json")
    module = load_policy()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    actual_sm = properties.major * 10 + properties.minor
    arch_list = torch.cuda.get_arch_list()
    supported_sms = sorted(
        {
            int(match.group(1))
            for value in arch_list
            if (match := re.fullmatch(r"sm_(\d+)", value)) is not None
        }
    )
    actual_plan, actual_error, actual_immutable, _ = call(
        module, [actual_sm], {"supported_sms": supported_sms}, config
    )
    actual_correct = bool(
        valid_plan(actual_plan)
        and actual_plan["status"] == "ready"
        and actual_plan["sass_targets"] == [actual_sm]
    )
    mixed_plan, _, mixed_immutable, _ = call(
        module, [86, 80, 86], {"supported_sms": [75, 80, 86]}, config
    )
    reversed_plan, reversed_error, reversed_immutable, _ = call(
        module, [80, 86], {"supported_sms": [86, 80, 75]}, config
    )
    unsupported_plan, _, unsupported_immutable, _ = call(
        module, [90], {"supported_sms": [75, 80, 86]}, config
    )
    malformed_rejected = False
    try:
        module.select_targets([], {"supported_sms": [80]}, config)
    except Exception:
        malformed_rejected = True
    regression = {
        "config_uses_visible_native_targets": config.get("target_mode") == "visible_devices"
        and config.get("require_native_sass") is True
        and config.get("emit_ptx") is False
        and config.get("allow_unsupported_fallback") is False,
        "input_immutable": actual_immutable
        and mixed_immutable
        and reversed_immutable
        and unsupported_immutable,
        "malformed_fleet_rejected": malformed_rejected,
        "mixed_fleet_targets_each_sm": valid_plan(mixed_plan)
        and mixed_plan["sass_targets"] == [80, 86],
        "order_independent": reversed_error is None and reversed_plan == mixed_plan,
        "unsupported_sm_blocked": valid_plan(unsupported_plan)
        and unsupported_plan["status"] == "blocked",
    }
    latencies = [
        call(module, [actual_sm], {"supported_sms": supported_sms}, config)[3] for _ in range(200)
    ]
    selection_p95_ms = sorted(latencies)[189]
    probe, gpu_correct, resources_cleaned = gpu_probe()
    write_json(
        "arch-evidence.json",
        {
            "actual": {
                "device_name": properties.name,
                "device_sm": actual_sm,
                "error": actual_error,
                "plan": actual_plan,
                "torch_arch_list": arch_list,
            },
            "gpu_probe": probe,
            "synthetic": {"mixed": mixed_plan, "unsupported": unsupported_plan},
        },
    )
    maximum_p95_ms = float(workload["maximum_selection_p95_ms"])
    assertions = {
        "functional": {
            "actual_gpu_native_target": actual_correct,
            "gpu_kernel_correct": gpu_correct,
        },
        "mechanism": {
            "native_sass_only": actual_correct and actual_plan["ptx_target"] is None,
            "toolkit_capability_enforced": regression["unsupported_sm_blocked"],
        },
        "regression": regression,
        "safety": {
            "no_cross_generation_fallback": regression["unsupported_sm_blocked"],
            "no_ptx_fallback": actual_correct and actual_plan["ptx_target"] is None,
            "resources_cleaned": resources_cleaned,
        },
        "slo": {"selection_latency": selection_p95_ms <= maximum_p95_ms},
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "mixed-sm-fleet": regression["mixed_fleet_targets_each_sm"],
        "unsupported-compiler-sm": regression["unsupported_sm_blocked"],
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
        "silent_fallback": not actual_correct or not regression["unsupported_sm_blocked"],
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "observability_ratio": float((evidence / "arch-evidence.json").is_file()),
        "resource_efficiency_ratio": min(1.0, 2.0 / max(selection_p95_ms, 2.0)),
        "selection_p95_ms": selection_p95_ms,
        "slo_goodput_ratio": min(1.0, maximum_p95_ms / max(selection_p95_ms, maximum_p95_ms)),
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
