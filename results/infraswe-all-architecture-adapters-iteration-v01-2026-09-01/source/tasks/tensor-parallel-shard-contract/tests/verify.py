from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
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
        "candidate_shard_policy", repo / "shard_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load shard_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(module, parameters, world_size, config) -> tuple[Any, str | None, bool]:
    parameter_copy = copy.deepcopy(parameters)
    config_copy = copy.deepcopy(config)
    try:
        plan = module.build_shard_plan(parameter_copy, world_size, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    return plan, error, parameter_copy == parameters and config_copy == config


def validate_plan(plan: Any, parameters: list[dict[str, Any]], world_size: int) -> dict[str, bool]:
    expected = {parameter["name"]: parameter for parameter in parameters}
    if not isinstance(plan, dict) or not isinstance(plan.get("parameters"), list):
        return {}
    entries = plan["parameters"]
    base = {
        "fallback_not_silent": plan.get("fallback_reported") is False,
        "parameter_coverage": len(entries) == len(expected)
        and {entry.get("name") for entry in entries if isinstance(entry, dict)} == set(expected),
        "schema_valid": plan.get("schema_version") == "1"
        and plan.get("status") == "ready"
        and plan.get("world_size") == world_size,
        "stable_order": [entry.get("name") for entry in entries if isinstance(entry, dict)]
        == sorted(expected),
    }
    ranges_valid = True
    axes_valid = True
    replication_valid = True
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("name") not in expected:
            ranges_valid = axes_valid = replication_valid = False
            continue
        name = entry["name"]
        shape = expected[name]["shape"]
        expected_axis = (
            0 if name in {"attention.qkv.weight", "mlp.gate_up.weight", "norm.weight"} else 1
        )
        expected_replicated = name == "norm.weight"
        axes_valid = (
            axes_valid and entry.get("axis") == expected_axis and entry.get("shape") == shape
        )
        replication_valid = replication_valid and entry.get("replicated") is expected_replicated
        shards = entry.get("shards")
        if not isinstance(shards, list) or len(shards) != world_size:
            ranges_valid = False
            continue
        dimension = shape[expected_axis]
        if expected_replicated:
            wanted = [{"end": dimension, "rank": rank, "start": 0} for rank in range(world_size)]
        else:
            size = dimension // world_size
            wanted = [
                {"end": (rank + 1) * size, "rank": rank, "start": rank * size}
                for rank in range(world_size)
            ]
        ranges_valid = ranges_valid and shards == wanted
    return {
        **base,
        "axes_match_contract": axes_valid,
        "replication_preserved": replication_valid,
        "shard_ranges_complete": ranges_valid,
    }


try:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()}")
    workload = load_json(workload_dir / "parameters.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "tp_config.json")
    module = load_policy()
    parameters = workload["parameters"]
    world_size = int(workload["world_size"])
    plan, plan_error, immutable = call(module, parameters, world_size, config)
    checks = validate_plan(plan, parameters, world_size)
    reversed_plan, reversed_error, reversed_immutable = call(
        module, list(reversed(parameters)), world_size, config
    )
    order_independent = plan_error is None and reversed_error is None and plan == reversed_plan
    odd = copy.deepcopy(parameters)
    odd[0]["shape"][0] += 1
    _, odd_error, odd_immutable = call(module, odd, world_size, config)
    unclassified = [*parameters, {"name": "unclassified.weight", "shape": [8, 8]}]
    _, unclassified_error, unclassified_immutable = call(module, unclassified, world_size, config)
    regression = {
        **checks,
        "indivisible_dimension_rejected": odd_error is not None,
        "input_immutable": immutable
        and reversed_immutable
        and odd_immutable
        and unclassified_immutable,
        "order_independent": order_independent,
        "unclassified_parameter_rejected": unclassified_error is not None,
    }
    plan_path = evidence / "shard-plan.json"
    write_json(
        "shard-plan.json",
        plan if isinstance(plan, dict) else {"error": plan_error, "plan": plan},
    )
    workload_env = os.environ.copy()
    workload_env.update(
        {
            "INFRASWE_EVIDENCE_DIR": str(evidence),
            "INFRASWE_TP_PLAN": str(plan_path),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "WARN",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        "--master-port=29711",
        str(Path(__file__).with_name("tp_workload.py")),
    ]
    if all(regression.values()):
        completed = subprocess.run(
            command,
            env=workload_env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        (evidence / "torchrun.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (evidence / "torchrun.stderr.log").write_text(completed.stderr, encoding="utf-8")
        returncode = completed.returncode
    else:
        returncode = 125
    rank_results = [
        load_json(evidence / f"rank-{rank}.json")
        for rank in range(world_size)
        if (evidence / f"rank-{rank}.json").is_file()
    ]
    reconstruction_correct = len(rank_results) == world_size and all(
        result.get("correct") is True for result in rank_results
    )
    resources_cleaned = returncode != 124 and all(
        result["memory_after_bytes"] <= result["memory_before_bytes"] + 1_048_576
        for result in rank_results
    )
    collective_ms = max(
        (float(result["collective_ms"]) for result in rank_results), default=999999.0
    )
    maximum_collective = float(workload["maximum_collective_ms"])
    assertions = {
        "functional": {
            "all_parameters_reconstructed": reconstruction_correct,
            "two_ranks_completed": len(rank_results) == world_size and returncode == 0,
        },
        "mechanism": {
            "column_and_row_axes_enforced": checks.get("axes_match_contract", False),
            "replicated_parameter_preserved": checks.get("replication_preserved", False),
        },
        "regression": regression,
        "safety": {
            "no_tensor_data_dropped": checks.get("shard_ranges_complete", False)
            and reconstruction_correct,
            "resources_cleaned": resources_cleaned,
            "unclassified_fails_closed": unclassified_error is not None,
        },
        "slo": {"collective_latency": collective_ms <= maximum_collective},
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "indivisible-shard-dimension": odd_error is not None,
        "unclassified-parameter": unclassified_error is not None,
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    policy_result = {
        "data_corruption": not reconstruction_correct,
        "hard_failures": [],
        "passed": reconstruction_correct and resources_cleaned,
        "resource_leak": not resources_cleaned,
        "silent_fallback": not checks.get("fallback_not_silent", False),
    }
    metrics = {
        "collective_ms": collective_ms,
        "observability_ratio": float(plan_path.is_file() and len(rank_results) == world_size),
        "reconstruction_goodput_ratio": float(reconstruction_correct),
        "resource_efficiency_ratio": min(
            1.0, maximum_collective / max(collective_ms, maximum_collective)
        ),
        "topology_robustness_ratio": sum(regression.values()) / len(regression),
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
        "reconstruction_goodput_ratio": 0.0,
        "resource_efficiency_ratio": 0.0,
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
