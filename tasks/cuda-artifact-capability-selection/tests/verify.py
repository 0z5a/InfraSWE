from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import os
import re
import subprocess
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
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_selector():
    spec = importlib.util.spec_from_file_location("candidate_selector", repo / "selector.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load selector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    required = {
        "artifact_id",
        "fallback_reported",
        "mechanism",
        "reason",
        "schema_version",
        "status",
    }
    if set(plan) != required or plan.get("schema_version") != "1":
        return False
    if not isinstance(plan.get("reason"), str) or not plan["reason"]:
        return False
    if not isinstance(plan.get("fallback_reported"), bool):
        return False
    mechanism = plan.get("mechanism")
    if plan.get("status") == "ready":
        if not isinstance(plan.get("artifact_id"), str) or not plan["artifact_id"]:
            return False
        return (mechanism == "native_sass" and not plan["fallback_reported"]) or (
            mechanism == "ptx_jit" and plan["fallback_reported"]
        )
    return bool(
        plan.get("status") == "blocked"
        and plan.get("artifact_id") is None
        and mechanism == "none"
        and plan["fallback_reported"]
    )


def call_selector(
    module,
    request: dict[str, Any],
    artifacts: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[Any, str | None, bool, float]:
    request_copy = copy.deepcopy(request)
    artifacts_copy = copy.deepcopy(artifacts)
    policy_copy = copy.deepcopy(policy)
    started = time.perf_counter()
    try:
        plan = module.select_artifact(request_copy, artifacts_copy, policy_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    immutable = request_copy == request and artifacts_copy == artifacts and policy_copy == policy
    return plan, error, immutable, elapsed_ms


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def run_synthetic_cases(
    module, workload: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    plans: dict[str, Any] = {}
    immutable = True
    deterministic = True
    for case in workload["cases"]:
        plan, error, unchanged, _ = call_selector(
            module, case["request"], case["artifacts"], policy
        )
        reversed_plan, reversed_error, reversed_unchanged, _ = call_selector(
            module, case["request"], list(reversed(case["artifacts"])), policy
        )
        matches = bool(
            valid_plan(plan)
            and plan["status"] == case["expected_status"]
            and plan["mechanism"] == case["expected_mechanism"]
            and plan["artifact_id"] == case["expected_artifact_id"]
        )
        checks[case["id"].replace("-", "_")] = matches
        immutable = immutable and unchanged and reversed_unchanged
        deterministic = (
            deterministic and error is None and reversed_error is None and plan == reversed_plan
        )
        plans[case["id"]] = plan
        details[case["id"]] = {
            "error": error,
            "expected_artifact_id": case["expected_artifact_id"],
            "expected_mechanism": case["expected_mechanism"],
            "expected_status": case["expected_status"],
            "plan": plan,
            "reversed_error": reversed_error,
            "reversed_plan": reversed_plan,
        }

    malformed_rejected = False
    try:
        module.select_artifact({}, [], policy)
    except Exception:
        malformed_rejected = True
    checks.update(
        {
            "artifact_order_independent": deterministic,
            "input_immutable": immutable,
            "malformed_request_rejected": malformed_rejected,
            "policy_disables_cpu_fallback": policy.get("allow_cpu_fallback") is False,
            "policy_requires_explicit_fallback": policy.get("require_explicit_fallback") is True,
            "policy_selection_order": policy.get("selection_order") == ["native_sass", "ptx_jit"],
        }
    )
    return checks, details, plans


def encoded_cuda_version(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.\d+)?\s*", value)
    if not match:
        raise ValueError(f"invalid CUDA version: {value!r}")
    return int(match.group(1)) * 10 + int(match.group(2))


def actual_hardware_request() -> tuple[dict[str, int], dict[str, Any]]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    runtime_text = str(torch.version.cuda)
    nvidia_smi = subprocess.run(
        ["nvidia-smi"],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if nvidia_smi.returncode:
        raise RuntimeError(nvidia_smi.stderr.strip() or "nvidia-smi failed")
    driver_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", nvidia_smi.stdout)
    if not driver_match:
        raise RuntimeError("nvidia-smi did not report driver CUDA capability")
    cxx11_abi = int(bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    request = {
        "cxx11_abi": cxx11_abi,
        "device_sm": properties.major * 10 + properties.minor,
        "driver_cuda": encoded_cuda_version(driver_match.group(1)),
        "runtime_cuda": encoded_cuda_version(runtime_text),
    }
    hardware = {
        "cxx11_abi": cxx11_abi,
        "device_name": properties.name,
        "device_sm": request["device_sm"],
        "driver_cuda": request["driver_cuda"],
        "nvidia_smi": nvidia_smi.stdout,
        "runtime_cuda": request["runtime_cuda"],
        "torch_version": torch.__version__,
    }
    return request, hardware


def gpu_probe() -> tuple[dict[str, Any], bool, bool]:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    before = int(torch.cuda.memory_allocated(0))
    generator = torch.Generator(device="cuda:0").manual_seed(8310)
    left = torch.randn((512, 512), device="cuda:0", generator=generator)
    right = torch.randn((512, 512), device="cuda:0", generator=generator)
    warmup = torch.mm(left, right)
    del warmup
    torch.cuda.synchronize(0)
    steady_state = int(torch.cuda.memory_allocated(0))
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    product = torch.mm(left, right)
    stop.record()
    stop.synchronize()
    elapsed_ms = float(start.elapsed_time(stop))
    checksum = float(product.float().sum().item())
    correct = math.isfinite(checksum) and elapsed_ms > 0
    del product, right, left, start, stop
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    after = int(torch.cuda.memory_allocated(0))
    cleaned = after <= steady_state + 1_048_576
    return (
        {
            "checksum": checksum,
            "elapsed_ms": elapsed_ms,
            "finite": math.isfinite(checksum),
            "memory_allocated_after_bytes": after,
            "memory_allocated_before_bytes": before,
            "memory_allocated_steady_state_bytes": steady_state,
        },
        correct,
        cleaned,
    )


try:
    workload = load_json(workload_dir / "cases.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    policy_config = load_json(repo / "build_policy.json")
    module = load_selector()
    regression, case_details, synthetic_plans = run_synthetic_cases(module, workload, policy_config)

    request, hardware = actual_hardware_request()
    actual_artifact_id = f"sm{request['device_sm']}-native"
    actual_artifacts = [
        {
            "built_cuda": request["runtime_cuda"],
            "cxx11_abi": request["cxx11_abi"],
            "id": f"sm{request['device_sm'] + 10}-incompatible-first",
            "kind": "sass",
            "sms": [request["device_sm"] + 10],
        },
        {
            "compute": request["device_sm"],
            "cxx11_abi": request["cxx11_abi"],
            "id": f"compute{request['device_sm']}-ptx",
            "kind": "ptx",
            "requires_driver_cuda": request["runtime_cuda"],
        },
        {
            "built_cuda": request["runtime_cuda"],
            "cxx11_abi": request["cxx11_abi"],
            "id": actual_artifact_id,
            "kind": "sass",
            "sms": [request["device_sm"]],
        },
    ]
    actual_plan, actual_error, actual_immutable, _ = call_selector(
        module, request, actual_artifacts, policy_config
    )
    latencies_ms = [
        call_selector(module, request, actual_artifacts, policy_config)[3] for _ in range(200)
    ]
    selection_p95_ms = percentile(latencies_ms, 0.95)
    actual_matches = bool(
        valid_plan(actual_plan)
        and actual_plan["status"] == "ready"
        and actual_plan["mechanism"] == "native_sass"
        and actual_plan["artifact_id"] == actual_artifact_id
    )
    probe, gpu_correct, resources_cleaned = gpu_probe()
    hardware["gpu_probe"] = probe
    hardware["request"] = request
    write_json("hardware-evidence.json", hardware)
    write_json(
        "selection-plans.json",
        {
            "actual": {
                "artifacts": actual_artifacts,
                "error": actual_error,
                "plan": actual_plan,
            },
            "synthetic": case_details,
        },
    )

    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "cxx11-abi-mismatch": regression["abi_mismatch_blocked"],
        "driver-runtime-mismatch": regression["driver_runtime_mismatch_blocked"],
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    all_plans = [actual_plan, *synthetic_plans.values()]
    no_cpu_fallback = all(
        not isinstance(plan, dict) or plan.get("mechanism") != "cpu" for plan in all_plans
    )
    explicit_degraded_plans = all(
        not isinstance(plan, dict)
        or plan.get("mechanism") == "native_sass"
        or plan.get("fallback_reported") is True
        for plan in all_plans
    )
    maximum_p95_ms = float(workload["maximum_selection_p95_ms"])
    assertions = {
        "functional": {
            "actual_artifact_compatible": actual_matches,
            "actual_gpu_kernel_correct": gpu_correct,
            "single_gpu_visible": torch.cuda.device_count() == 1,
        },
        "mechanism": {
            "abi_mismatch_blocked": regression["abi_mismatch_blocked"],
            "driver_runtime_mismatch_blocked": regression["driver_runtime_mismatch_blocked"],
            "ptx_fallback_explicit": regression["explicit_ptx_fallback"],
        },
        "regression": regression,
        "safety": {
            "explicit_fail_closed": explicit_degraded_plans,
            "no_cpu_fallback": no_cpu_fallback,
            "no_data_corruption": gpu_correct,
            "resources_cleaned": resources_cleaned,
        },
        "slo": {"selection_latency": selection_p95_ms <= maximum_p95_ms},
    }
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    silent_fallback = bool(not actual_matches or not explicit_degraded_plans or not no_cpu_fallback)
    policy = {
        "data_corruption": not gpu_correct,
        "hard_failures": [],
        "passed": gpu_correct and resources_cleaned,
        "resource_leak": not resources_cleaned,
        "silent_fallback": silent_fallback,
    }
    observability = (
        sum(
            path.is_file()
            for path in (
                evidence / "hardware-evidence.json",
                evidence / "selection-plans.json",
            )
        )
        / 2
    )
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "observability_ratio": observability,
        "resource_efficiency_ratio": min(1.0, 2.0 / max(selection_p95_ms, 2.0)),
        "selection_p95_ms": selection_p95_ms,
        "slo_goodput_ratio": min(1.0, maximum_p95_ms / max(selection_p95_ms, maximum_p95_ms)),
        "topology_robustness_ratio": regression_ratio,
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
        "observability_ratio": 0.25,
        "resource_efficiency_ratio": 0.0,
        "selection_p95_ms": 999999.0,
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
