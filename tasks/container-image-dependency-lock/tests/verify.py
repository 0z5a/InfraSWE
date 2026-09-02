from __future__ import annotations

import copy
import importlib.util
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

repo = Path(os.environ["INFRASWE_REPO"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
workload_dir = Path(os.environ["INFRASWE_WORKLOAD_DIR"])
faults_dir = Path(os.environ["INFRASWE_FAULTS_DIR"])
evidence.mkdir(parents=True, exist_ok=True)
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def write_json(name: str, value: object) -> None:
    (evidence / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def load_resolver():
    spec = importlib.util.spec_from_file_location("candidate_resolver", repo / "resolver.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load resolver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    if set(plan) != {
        "candidate_id",
        "fallback_reported",
        "lock",
        "reason",
        "schema_version",
        "status",
    }:
        return False
    if plan.get("schema_version") != "1" or not isinstance(plan.get("reason"), str):
        return False
    if not isinstance(plan.get("fallback_reported"), bool) or not isinstance(
        plan.get("lock"), list
    ):
        return False
    if plan.get("status") == "blocked":
        return bool(
            plan.get("candidate_id") is None and plan["fallback_reported"] and not plan["lock"]
        )
    if plan.get("status") != "ready" or plan["fallback_reported"]:
        return False
    if not isinstance(plan.get("candidate_id"), str) or not plan["candidate_id"]:
        return False
    names: set[str] = set()
    for entry in plan["lock"]:
        if not isinstance(entry, dict) or set(entry) != {
            "architecture",
            "digest",
            "name",
            "version",
        }:
            return False
        if (
            not isinstance(entry["name"], str)
            or not entry["name"]
            or entry["name"] in names
            or not isinstance(entry["version"], str)
            or not isinstance(entry["architecture"], str)
            or not isinstance(entry["digest"], str)
            or not DIGEST.fullmatch(entry["digest"])
        ):
            return False
        names.add(entry["name"])
    return bool(
        plan["lock"]
        and plan["lock"]
        == sorted(plan["lock"], key=lambda item: (item["name"], item["version"], item["digest"]))
    )


def call(module, request, candidates, policy) -> tuple[Any, str | None, bool, float]:
    request_copy = copy.deepcopy(request)
    candidates_copy = copy.deepcopy(candidates)
    policy_copy = copy.deepcopy(policy)
    started = time.perf_counter()
    try:
        plan = module.resolve_image(request_copy, candidates_copy, policy_copy)
        error = None
    except Exception as caught:
        plan = None
        error = f"{type(caught).__name__}: {caught}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    immutable = request_copy == request and candidates_copy == candidates and policy_copy == policy
    return plan, error, immutable, elapsed_ms


def candidate(
    identifier: str,
    *,
    architecture: str = "amd64",
    digest_character: str = "a",
    dependencies: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "dependencies": dependencies
        if dependencies is not None
        else [
            {
                "architecture": architecture,
                "digest": "sha256:" + "b" * 64,
                "name": "cuda-runtime",
                "version": "12.6.0",
            }
        ],
        "digest": "sha256:" + digest_character * 64,
        "id": identifier,
        "name": "inference-server",
        "version": "1.0.0",
    }


try:
    workload = load_json(workload_dir / "cases.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    policy = load_json(repo / "lock_policy.json")
    module = load_resolver()
    request = {
        "architecture": "amd64",
        "name": "inference-server",
        "version": "1.0.0",
    }
    mutable = candidate("mutable-first")
    mutable["digest"] = "latest"
    locked_z = candidate("locked-z", digest_character="c")
    locked_a = candidate("locked-a", digest_character="d")
    ready_candidates = [mutable, locked_z, locked_a]
    ready_plan, ready_error, immutable, _ = call(module, request, ready_candidates, policy)
    reversed_plan, reversed_error, reversed_immutable, _ = call(
        module, request, list(reversed(ready_candidates)), policy
    )

    arch_plan, _, arch_immutable, _ = call(
        module, request, [candidate("arm-only", architecture="arm64")], policy
    )
    conflicting = candidate(
        "conflicting",
        dependencies=[
            {
                "architecture": "amd64",
                "digest": "sha256:" + "e" * 64,
                "name": "torch",
                "version": "2.7.0",
            },
            {
                "architecture": "amd64",
                "digest": "sha256:" + "f" * 64,
                "name": "torch",
                "version": "2.8.0",
            },
        ],
    )
    conflict_plan, _, conflict_immutable, _ = call(module, request, [conflicting], policy)
    unpinned = candidate("unpinned-dependency")
    unpinned["dependencies"][0]["digest"] = "latest"
    unpinned_plan, _, unpinned_immutable, _ = call(module, request, [unpinned], policy)
    malformed_rejected = False
    try:
        module.resolve_image({}, [], policy)
    except Exception:
        malformed_rejected = True

    regression = {
        "architecture_mismatch_blocked": valid_plan(arch_plan) and arch_plan["status"] == "blocked",
        "artifact_order_independent": ready_error is None
        and reversed_error is None
        and ready_plan == reversed_plan,
        "dependency_conflict_blocked": valid_plan(conflict_plan)
        and conflict_plan["status"] == "blocked",
        "input_immutable": immutable
        and reversed_immutable
        and arch_immutable
        and conflict_immutable
        and unpinned_immutable,
        "malformed_request_rejected": malformed_rejected,
        "policy_is_immutable": policy.get("require_digest") is True
        and policy.get("forbid_mutable_tags") is True
        and policy.get("reject_dependency_conflicts") is True,
        "unpinned_dependency_blocked": valid_plan(unpinned_plan)
        and unpinned_plan["status"] == "blocked",
    }
    ready_correct = bool(
        valid_plan(ready_plan)
        and ready_plan["status"] == "ready"
        and ready_plan["candidate_id"] == "locked-a"
    )
    architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    if architecture is None:
        raise RuntimeError(f"unsupported verifier architecture: {platform.machine()}")
    actual_request = {**request, "architecture": architecture}
    actual_candidates = [
        {**mutable, "architecture": architecture},
        candidate("host-locked", architecture=architecture),
    ]
    actual_plan, actual_error, actual_immutable, _ = call(
        module, actual_request, actual_candidates, policy
    )
    actual_correct = bool(
        valid_plan(actual_plan)
        and actual_plan["status"] == "ready"
        and actual_plan["candidate_id"] == "host-locked"
    )
    latencies = [call(module, actual_request, actual_candidates, policy)[3] for _ in range(200)]
    selection_p95_ms = sorted(latencies)[189]
    write_json(
        "resolution-evidence.json",
        {
            "actual": {"error": actual_error, "plan": actual_plan},
            "platform": {
                "machine": platform.machine(),
                "python": platform.python_version(),
                "resolved_architecture": architecture,
            },
            "synthetic": {
                "architecture_mismatch": arch_plan,
                "conflict": conflict_plan,
                "ready": ready_plan,
                "unpinned": unpinned_plan,
            },
        },
    )
    maximum_p95_ms = float(workload["maximum_selection_p95_ms"])
    assertions = {
        "functional": {
            "actual_platform_lock_resolved": actual_correct,
            "immutable_candidate_selected": ready_correct,
        },
        "mechanism": {
            "digest_enforced": regression["unpinned_dependency_blocked"],
            "transitive_conflict_rejected": regression["dependency_conflict_blocked"],
        },
        "regression": regression,
        "safety": {
            "cross_architecture_blocked": regression["architecture_mismatch_blocked"],
            "inputs_unchanged": actual_immutable and regression["input_immutable"],
            "no_mutable_fallback": actual_correct and ready_correct,
            "resources_cleaned": True,
        },
        "slo": {"selection_latency": selection_p95_ms <= maximum_p95_ms},
    }
    expected_faults = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "architecture-mismatch": regression["architecture_mismatch_blocked"],
        "dependency-conflict": regression["dependency_conflict_blocked"],
    }
    faults_passed = expected_faults == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    silent_fallback = (
        not actual_correct
        or not ready_correct
        or any(
            isinstance(plan, dict)
            and plan.get("status") == "blocked"
            and plan.get("fallback_reported") is not True
            for plan in (arch_plan, conflict_plan, unpinned_plan)
        )
    )
    policy_result = {
        "data_corruption": False,
        "hard_failures": [],
        "passed": True,
        "resource_leak": False,
        "silent_fallback": silent_fallback,
    }
    regression_ratio = sum(regression.values()) / len(regression)
    metrics = {
        "observability_ratio": float((evidence / "resolution-evidence.json").is_file()),
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
        "selection_p95_ms": 999999.0,
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
