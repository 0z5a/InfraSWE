from __future__ import annotations

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
evidence.mkdir(parents=True, exist_ok=True)


def write_json(name: str, value: object) -> None:
    (evidence / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_policy_module():
    spec = importlib.util.spec_from_file_location(
        "candidate_launch_policy", repo / "launch_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load launch_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    required = {"schema_version", "transport", "reason", "fallback_reported", "nccl_env"}
    if set(plan) != required or plan["schema_version"] != "1":
        return False
    if plan["transport"] not in {"p2p", "shm"}:
        return False
    if not isinstance(plan["reason"], str) or not plan["reason"]:
        return False
    if not isinstance(plan["fallback_reported"], bool) or not isinstance(plan["nccl_env"], dict):
        return False
    return all(
        plan["nccl_env"].get(name) in {"0", "1"}
        for name in ("NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE")
    )


def p2p_plan_is_correct(plan: dict[str, Any]) -> bool:
    return (
        valid_plan(plan)
        and plan["transport"] == "p2p"
        and not plan["fallback_reported"]
        and plan["nccl_env"]["NCCL_P2P_DISABLE"] == "0"
        and plan["nccl_env"]["NCCL_SHM_DISABLE"] == "0"
    )


def fallback_plan_is_correct(plan: dict[str, Any]) -> bool:
    return (
        valid_plan(plan)
        and plan["transport"] == "shm"
        and plan["fallback_reported"]
        and plan["reason"] != "configured_preference"
        and plan["nccl_env"]["NCCL_P2P_DISABLE"] == "1"
        and plan["nccl_env"]["NCCL_SHM_DISABLE"] == "0"
    )


try:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()}")
    module = load_policy_module()
    complete = [[True, True], [True, True]]
    unavailable = [[True, False], [False, True]]
    asymmetric = [[True, True], [False, True]]
    complete_plan = module.build_plan(complete)
    unavailable_plan = module.build_plan(unavailable)
    asymmetric_plan = module.build_plan(asymmetric)
    synthetic_checks = {
        "complete_peer_matrix": p2p_plan_is_correct(complete_plan),
        "unavailable_peer_matrix": fallback_plan_is_correct(unavailable_plan),
        "asymmetric_peer_matrix": fallback_plan_is_correct(asymmetric_plan),
    }

    actual_matrix = [
        [
            True if source == target else bool(torch.cuda.can_device_access_peer(source, target))
            for target in range(2)
        ]
        for source in range(2)
    ]
    actual_has_complete_p2p = all(
        actual_matrix[source][target]
        for source in range(2)
        for target in range(2)
        if source != target
    )
    actual_plan = module.build_plan(actual_matrix)
    actual_plan_valid = valid_plan(actual_plan)
    capability_match = actual_plan_valid and (
        (actual_has_complete_p2p and p2p_plan_is_correct(actual_plan))
        or (not actual_has_complete_p2p and fallback_plan_is_correct(actual_plan))
    )
    write_json(
        "transport-plan.json",
        {
            "peer_access": actual_matrix,
            "plan": actual_plan,
            "synthetic": {
                "complete": complete_plan,
                "unavailable": unavailable_plan,
                "asymmetric": asymmetric_plan,
            },
        },
    )

    workload_env = os.environ.copy()
    workload_env.update(
        {str(key): str(value) for key, value in actual_plan.get("nccl_env", {}).items()}
    )
    workload_env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29671",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,COLL",
            "NCCL_DEBUG_FILE": str(evidence / "nccl-%h-%p.log"),
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "INFRASWE_EVIDENCE_DIR": str(evidence),
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        "--master-port=29671",
        str(Path(__file__).with_name("nccl_workload.py")),
    ]
    workload = subprocess.run(
        command,
        env=workload_env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    (evidence / "torchrun.stdout.log").write_text(workload.stdout, encoding="utf-8")
    (evidence / "torchrun.stderr.log").write_text(workload.stderr, encoding="utf-8")
    rank_results = [
        json.loads((evidence / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
        if (evidence / f"rank-{rank}.json").is_file()
    ]
    nccl_logs = sorted(evidence.glob("nccl-*.log"))
    nccl_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in nccl_logs)
    shm_evidence = "SHM" in nccl_text.upper()
    p2p_evidence = "P2P" in nccl_text.upper() and actual_has_complete_p2p
    mechanism_evidence = shm_evidence if actual_plan.get("transport") == "shm" else p2p_evidence
    correct = len(rank_results) == 2 and all(item["correct"] for item in rank_results)
    p95_ms = max((float(item["p95_ms"]) for item in rank_results), default=float("inf"))
    silent_fallback = bool(
        (not capability_match)
        or (
            actual_plan.get("transport") == "p2p"
            and not actual_has_complete_p2p
            and not actual_plan.get("fallback_reported", False)
        )
    )
    assertions = {
        "functional": {
            "all_reduce_correct": correct,
            "two_ranks_completed": len(rank_results) == 2 and workload.returncode == 0,
        },
        "mechanism": {
            "capability_matches_plan": capability_match,
            "nccl_transport_evidence": mechanism_evidence,
        },
        "regression": synthetic_checks,
        "safety": {
            "explicit_fallback_when_required": actual_has_complete_p2p
            or bool(actual_plan.get("fallback_reported", False)),
            "zero_data_corruption": correct,
            "resources_cleaned": workload.returncode != 124,
        },
        "slo": {"collective_latency": p95_ms <= 50.0},
    }
    faults = {
        "passed": (actual_has_complete_p2p or fallback_plan_is_correct(actual_plan)) and correct,
        "score": 1.0
        if (actual_has_complete_p2p or fallback_plan_is_correct(actual_plan)) and correct
        else 0.0,
        "scenario": "p2p-unavailable",
    }
    policy = {
        "passed": correct and workload.returncode == 0,
        "hard_failures": [],
        "silent_fallback": silent_fallback,
        "data_corruption": not correct,
        "resource_leak": workload.returncode == 124,
    }
    synthetic_ratio = sum(synthetic_checks.values()) / len(synthetic_checks)
    evidence_ratio = (
        sum([bool(nccl_logs), len(rank_results) == 2, (evidence / "transport-plan.json").is_file()])
        / 3
    )
    metrics = {
        "slo_goodput_ratio": min(1.0, 50.0 / max(p95_ms, 50.0)),
        "resource_efficiency_ratio": min(1.0, 20.0 / max(p95_ms, 20.0)),
        "topology_robustness_ratio": synthetic_ratio,
        "observability_ratio": evidence_ratio,
        "collective_p95_ms": p95_ms if p95_ms != float("inf") else 999999.0,
    }
except Exception as error:
    assertions = {"functional": {"verifier_completed": False}}
    faults = {"passed": False, "score": 0.0, "error": str(error)}
    policy = {
        "passed": False,
        "hard_failures": [],
        "silent_fallback": False,
        "data_corruption": False,
        "resource_leak": False,
        "verifier_error": str(error),
    }
    metrics = {
        "slo_goodput_ratio": 0.0,
        "resource_efficiency_ratio": 0.0,
        "topology_robustness_ratio": 0.0,
        "observability_ratio": 0.25,
        "collective_p95_ms": 999999.0,
    }

write_json("assertions.json", assertions)
write_json("faults.json", faults)
write_json("policy.json", policy)
write_json("metrics.json", metrics)
all_assertions = all(
    value
    for group in assertions.values()
    for value in (group.values() if isinstance(group, dict) else [group])
)
raise SystemExit(0 if all_assertions and faults["passed"] and policy["passed"] else 1)
