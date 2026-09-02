from __future__ import annotations

import copy
import gc
import importlib.util
import json
import os
import re
import subprocess
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
        "candidate_affinity_policy", repo / "affinity_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load affinity_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict) or set(plan) != {
        "cpu_ids",
        "fallback_reported",
        "numa_node",
        "reason",
        "schema_version",
        "status",
    }:
        return False
    if (
        plan.get("schema_version") != "1"
        or not isinstance(plan.get("reason"), str)
        or not plan["reason"]
        or not isinstance(plan.get("cpu_ids"), list)
        or any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in plan["cpu_ids"]
        )
        or plan["cpu_ids"] != sorted(set(plan["cpu_ids"]))
        or not isinstance(plan.get("fallback_reported"), bool)
    ):
        return False
    if plan.get("status") == "ready":
        return bool(
            isinstance(plan.get("numa_node"), int)
            and plan["cpu_ids"]
            and not plan["fallback_reported"]
        )
    return bool(
        plan.get("status") == "blocked"
        and plan.get("numa_node") is None
        and not plan["cpu_ids"]
        and plan["fallback_reported"]
    )


def call(module, gpu, topology, config) -> tuple[Any, str | None, bool]:
    gpu_copy = copy.deepcopy(gpu)
    topology_copy = copy.deepcopy(topology)
    config_copy = copy.deepcopy(config)
    try:
        plan = module.select_affinity(gpu_copy, topology_copy, config_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    immutable = gpu_copy == gpu and topology_copy == topology and config_copy == config
    return plan, error, immutable


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(item))
    return sorted(set(cpus))


def actual_topology() -> tuple[dict[str, Any], dict[str, Any]]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if query.returncode:
        raise RuntimeError(query.stderr.strip() or "nvidia-smi PCI query failed")
    bus = query.stdout.splitlines()[0].strip().lower()
    match = re.search(r"([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2}\.[0-9a-f])$", bus)
    if not match:
        match = re.search(r"[0-9a-f]{4}([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2}\.[0-9a-f])$", bus)
    if not match:
        raise RuntimeError(f"could not normalize PCI bus ID: {bus}")
    pci_id = f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
    numa_path = Path("/sys/bus/pci/devices") / pci_id / "numa_node"
    if not numa_path.is_file():
        raise RuntimeError(f"GPU NUMA sysfs path is missing: {numa_path}")
    numa_node = int(numa_path.read_text(encoding="utf-8").strip())
    numa_source = "pci-sysfs"
    if numa_node < 0:
        topo_query = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        gpu_row = next(
            (
                line.split()
                for line in topo_query.stdout.splitlines()
                if line.lstrip().startswith("GPU0")
            ),
            [],
        )
        try:
            numa_node = int(gpu_row[-2])
        except (IndexError, ValueError) as error:
            raise RuntimeError(
                "GPU NUMA affinity is absent from both sysfs and nvidia-smi topology"
            ) from error
        numa_source = "nvidia-smi-topology"
    allowed = sorted(os.sched_getaffinity(0))
    nodes: dict[str, list[int]] = {}
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        cpulist_path = node_path / "cpulist"
        if cpulist_path.is_file():
            node_id = node_path.name.removeprefix("node")
            nodes[node_id] = parse_cpu_list(cpulist_path.read_text(encoding="utf-8"))
    local = sorted(set(nodes.get(str(numa_node), [])) & set(allowed))
    if len(local) < 5:
        raise RuntimeError(f"not enough allowed CPUs on GPU NUMA node {numa_node}: {local}")
    topology = {
        "allowed_cpus": allowed,
        "nodes": nodes,
        "reserved_cpus": [local[0]],
    }
    evidence_value = {
        "allowed_cpus": allowed,
        "gpu_numa_node": numa_node,
        "local_allowed_cpus": local,
        "numa_source": numa_source,
        "pci_bus_id": bus,
        "pci_sysfs_id": pci_id,
    }
    return {"id": 0, "numa_node": numa_node}, {**topology, "evidence": evidence_value}


def pinned_gpu_probe(cpu_ids: list[int]) -> tuple[dict[str, Any], bool, bool]:
    original_affinity = set(os.sched_getaffinity(0))
    restored = False
    try:
        os.sched_setaffinity(0, set(cpu_ids))
        applied = sorted(os.sched_getaffinity(0))
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        before = int(torch.cuda.memory_allocated(0))
        tensor = torch.arange(1 << 20, device="cuda:0", dtype=torch.float32)
        warmup = tensor.square()
        del warmup
        torch.cuda.synchronize(0)
        steady = int(torch.cuda.memory_allocated(0))
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = tensor.square()
        stop.record()
        stop.synchronize()
        elapsed_ms = float(start.elapsed_time(stop))
        correct = bool(torch.equal(output[:4], torch.tensor([0, 1, 4, 9], device="cuda:0")))
        del output, tensor, start, stop
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        after = int(torch.cuda.memory_allocated(0))
        resources_cleaned = after <= steady + 1_048_576
    finally:
        os.sched_setaffinity(0, original_affinity)
        restored = set(os.sched_getaffinity(0)) == original_affinity
    return (
        {
            "affinity_applied": applied,
            "affinity_restored": restored,
            "correct": correct,
            "elapsed_ms": elapsed_ms,
            "memory_after_bytes": after,
            "memory_before_bytes": before,
        },
        correct and applied == sorted(cpu_ids) and elapsed_ms > 0,
        resources_cleaned and restored,
    )


try:
    workload = load_json(workload_dir / "topology.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    config = load_json(repo / "affinity_config.json")
    module = load_policy()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU, got {torch.cuda.device_count()}")
    gpu, topology_with_evidence = actual_topology()
    topology = {key: value for key, value in topology_with_evidence.items() if key != "evidence"}
    actual_plan, actual_error, actual_immutable = call(module, gpu, topology, config)
    local_candidates = sorted(
        set(topology["nodes"][str(gpu["numa_node"])])
        & set(topology["allowed_cpus"]) - set(topology["reserved_cpus"])
    )
    actual_correct = bool(
        valid_plan(actual_plan)
        and actual_plan["status"] == "ready"
        and actual_plan["numa_node"] == gpu["numa_node"]
        and actual_plan["cpu_ids"] == local_candidates[: config["worker_threads"]]
    )
    synthetic_gpu = {"id": 1, "numa_node": 1}
    synthetic_topology = {
        "allowed_cpus": list(range(16)),
        "nodes": {"0": list(range(8)), "1": list(range(8, 16))},
        "reserved_cpus": [8, 9],
    }
    synthetic_plan, _, synthetic_immutable = call(module, synthetic_gpu, synthetic_topology, config)
    exhausted = copy.deepcopy(synthetic_topology)
    exhausted["reserved_cpus"] = list(range(8, 16))
    exhausted_plan, _, exhausted_immutable = call(module, synthetic_gpu, exhausted, config)
    missing_node_rejected = False
    try:
        module.select_affinity({"id": 2, "numa_node": 2}, synthetic_topology, config)
    except Exception:
        missing_node_rejected = True
    regression = {
        "config_forbids_cross_numa": config.get("forbid_cross_numa") is True
        and config.get("require_gpu_local") is True,
        "input_immutable": actual_immutable and synthetic_immutable and exhausted_immutable,
        "local_cores_exhausted_blocked": valid_plan(exhausted_plan)
        and exhausted_plan["status"] == "blocked",
        "missing_numa_node_rejected": missing_node_rejected,
        "reserved_cpus_excluded": valid_plan(synthetic_plan)
        and synthetic_plan["cpu_ids"] == [10, 11, 12, 13],
        "synthetic_gpu_local": valid_plan(synthetic_plan) and synthetic_plan["numa_node"] == 1,
    }
    probe, gpu_correct, resources_cleaned = (
        pinned_gpu_probe(actual_plan["cpu_ids"])
        if actual_correct
        else (
            {},
            False,
            True,
        )
    )
    write_json(
        "numa-evidence.json",
        {
            "actual": {
                "error": actual_error,
                "gpu": gpu,
                "plan": actual_plan,
                "topology": topology_with_evidence["evidence"],
            },
            "gpu_probe": probe,
            "synthetic": {"exhausted": exhausted_plan, "local": synthetic_plan},
        },
    )
    maximum_probe_ms = float(workload["maximum_gpu_probe_ms"])
    probe_ms = float(probe.get("elapsed_ms", 999999.0))
    assertions = {
        "functional": {
            "actual_affinity_applied": actual_correct and gpu_correct,
            "gpu_kernel_correct": gpu_correct,
        },
        "mechanism": {
            "gpu_numa_local": actual_correct,
            "reserved_cores_excluded": regression["reserved_cpus_excluded"],
        },
        "regression": regression,
        "safety": {
            "affinity_restored": bool(probe.get("affinity_restored", False)),
            "cross_numa_blocked": regression["local_cores_exhausted_blocked"],
            "resources_cleaned": resources_cleaned,
        },
        "slo": {"pinned_gpu_probe": probe_ms <= maximum_probe_ms},
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "gpu-numa-node-missing": missing_node_rejected,
        "local-cores-reserved": regression["local_cores_exhausted_blocked"],
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
        "silent_fallback": not actual_correct or not regression["local_cores_exhausted_blocked"],
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "gpu_probe_ms": probe_ms,
        "observability_ratio": float((evidence / "numa-evidence.json").is_file()),
        "resource_efficiency_ratio": min(1.0, 4 / max(len(actual_plan.get("cpu_ids", [])), 4))
        if isinstance(actual_plan, dict)
        else 0.0,
        "slo_goodput_ratio": min(1.0, maximum_probe_ms / max(probe_ms, maximum_probe_ms)),
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
