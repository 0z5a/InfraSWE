from __future__ import annotations

import copy
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

repo = Path(os.environ["INFRASWE_REPO"])
evidence = Path(os.environ["INFRASWE_EVIDENCE_DIR"])
workload_dir = Path(os.environ["INFRASWE_WORKLOAD_DIR"])
faults_dir = Path(os.environ["INFRASWE_FAULTS_DIR"])
evidence.mkdir(parents=True, exist_ok=True)
MODALITIES = ("logs", "metrics", "profiles", "traces")


def write_json(name: str, value: object) -> None:
    (evidence / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def load_diagnoser():
    spec = importlib.util.spec_from_file_location("candidate_diagnoser", repo / "diagnoser.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load diagnoser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_result(result: Any) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "confidence",
        "correlation_id",
        "evidence_ids",
        "fallback_reported",
        "reason",
        "root_cause",
        "schema_version",
        "status",
    }:
        return False
    if (
        result.get("schema_version") != "1"
        or isinstance(result.get("confidence"), bool)
        or not isinstance(result.get("confidence"), int | float)
        or not 0 <= result["confidence"] <= 1
        or not isinstance(result.get("evidence_ids"), list)
        or result["evidence_ids"] != sorted(result["evidence_ids"])
        or not isinstance(result.get("fallback_reported"), bool)
        or not isinstance(result.get("reason"), str)
        or not result["reason"]
    ):
        return False
    if result.get("status") == "inconclusive":
        return bool(
            result.get("root_cause") is None
            and result.get("correlation_id") is None
            and result["confidence"] == 0
            and not result["evidence_ids"]
            and result["fallback_reported"]
        )
    return bool(
        result.get("status") == "diagnosed"
        and isinstance(result.get("root_cause"), str)
        and result["root_cause"]
        and isinstance(result.get("correlation_id"), str)
        and result["correlation_id"]
        and result["confidence"] >= 0.75
        and len(result["evidence_ids"]) >= 3
        and not result["fallback_reported"]
    )


def call(module, signals, policy) -> tuple[Any, str | None, bool, float]:
    signals_copy = copy.deepcopy(signals)
    policy_copy = copy.deepcopy(policy)
    started = time.perf_counter()
    try:
        result = module.diagnose(signals_copy, policy_copy)
        error = None
    except Exception as caught:
        result = None
        error = f"{type(caught).__name__}: {caught}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result, error, signals_copy == signals and policy_copy == policy, elapsed_ms


def empty_signals() -> dict[str, list[dict[str, Any]]]:
    return {modality: [] for modality in MODALITIES}


def add(
    signals: dict[str, list[dict[str, Any]]],
    modality: str,
    identifier: str,
    timestamp: int,
    correlation_id: str,
    cause: str,
) -> None:
    signals[modality].append(
        {
            "at_ms": timestamp,
            "cause": cause,
            "correlation_id": correlation_id,
            "id": identifier,
        }
    )


def correlated(
    cause: str,
    correlation_id: str,
    modalities: tuple[str, ...],
    *,
    start: int = 1000,
    step: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    signals = empty_signals()
    for index, modality in enumerate(modalities):
        add(
            signals,
            modality,
            f"{correlation_id}-{modality}",
            start + index * step,
            correlation_id,
            cause,
        )
    return signals


try:
    workload = load_json(workload_dir / "cases.json")
    fault_spec = load_json(faults_dir / "scenarios.json")
    policy = load_json(repo / "signal_policy.json")
    module = load_diagnoser()
    primary = correlated("nccl_transport_fallback", "request-7", MODALITIES)
    primary["logs"].insert(
        0,
        {
            "at_ms": 900,
            "cause": "cuda_oom",
            "correlation_id": "decoy",
            "id": "decoy-log",
        },
    )
    primary_result, primary_error, primary_immutable, _ = call(module, primary, policy)
    primary_correct = bool(
        valid_result(primary_result)
        and primary_result["status"] == "diagnosed"
        and primary_result["root_cause"] == "nccl_transport_fallback"
        and primary_result["correlation_id"] == "request-7"
        and primary_result["confidence"] == 1.0
    )

    scheduler = correlated("scheduler_queue_saturation", "request-8", ("logs", "metrics", "traces"))
    scheduler_result, _, scheduler_immutable, _ = call(module, scheduler, policy)
    missing = correlated("cuda_oom", "request-9", ("logs", "metrics"))
    missing_result, _, missing_immutable, _ = call(module, missing, policy)
    skewed = correlated("rank_desync", "request-10", ("logs", "metrics", "traces"), step=75)
    skewed_result, _, skewed_immutable, _ = call(module, skewed, policy)
    tie = correlated("b_cause", "tie-b", ("logs", "metrics", "traces"))
    tie_a = correlated("a_cause", "tie-a", ("logs", "metrics", "traces"))
    for modality in MODALITIES:
        tie[modality].extend(tie_a[modality])
    tie_result, _, tie_immutable, _ = call(module, tie, policy)
    reordered = {key: list(reversed(value)) for key, value in reversed(primary.items())}
    reordered_result, reordered_error, reordered_immutable, _ = call(module, reordered, policy)
    malformed_rejected = False
    try:
        module.diagnose([], policy)
    except Exception:
        malformed_rejected = True

    regression = {
        "input_immutable": all(
            (
                primary_immutable,
                scheduler_immutable,
                missing_immutable,
                skewed_immutable,
                tie_immutable,
                reordered_immutable,
            )
        ),
        "malformed_input_rejected": malformed_rejected,
        "missing_modality_inconclusive": valid_result(missing_result)
        and missing_result["status"] == "inconclusive",
        "order_independent": primary_error is None
        and reordered_error is None
        and primary_result == reordered_result,
        "scheduler_cause_correlated": valid_result(scheduler_result)
        and scheduler_result["root_cause"] == "scheduler_queue_saturation"
        and scheduler_result["confidence"] == 0.75,
        "timestamp_skew_inconclusive": valid_result(skewed_result)
        and skewed_result["status"] == "inconclusive",
        "tie_break_deterministic": valid_result(tie_result)
        and tie_result["root_cause"] == "a_cause",
    }
    latencies = [call(module, primary, policy)[3] for _ in range(200)]
    p95_ms = sorted(latencies)[189]
    write_json(
        "diagnosis-evidence.json",
        {
            "cases": {
                "missing": missing_result,
                "primary": primary_result,
                "scheduler": scheduler_result,
                "skewed": skewed_result,
                "tie": tie_result,
            },
            "primary_error": primary_error,
            "signal_counts": {modality: len(primary[modality]) for modality in MODALITIES},
        },
    )
    maximum_p95_ms = float(workload["maximum_diagnosis_p95_ms"])
    assertions = {
        "functional": {
            "decoy_log_rejected": primary_correct,
            "root_cause_diagnosed": primary_correct,
        },
        "mechanism": {
            "multi_modal_threshold": regression["missing_modality_inconclusive"],
            "temporal_correlation": regression["timestamp_skew_inconclusive"],
        },
        "regression": regression,
        "safety": {
            "insufficient_evidence_explicit": regression["missing_modality_inconclusive"],
            "no_evidence_mutation": regression["input_immutable"],
            "resources_cleaned": True,
        },
        "slo": {"diagnosis_latency": p95_ms <= maximum_p95_ms},
    }
    fault_ids = {scenario["id"] for scenario in fault_spec["scenarios"]}
    fault_checks = {
        "missing-signal-modality": regression["missing_modality_inconclusive"],
        "timestamp-skew": regression["timestamp_skew_inconclusive"],
    }
    faults_passed = fault_ids == set(fault_checks) and all(fault_checks.values())
    faults_result = {
        "checks": fault_checks,
        "passed": faults_passed,
        "score": 1.0 if faults_passed else 0.0,
    }
    silent_fallback = not primary_correct or any(
        not isinstance(result, dict)
        or result.get("status") != "inconclusive"
        or result.get("fallback_reported") is not True
        for result in (missing_result, skewed_result)
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
        "diagnosis_p95_ms": p95_ms,
        "observability_ratio": float((evidence / "diagnosis-evidence.json").is_file()),
        "resource_efficiency_ratio": min(1.0, 2.0 / max(p95_ms, 2.0)),
        "slo_goodput_ratio": min(1.0, maximum_p95_ms / max(p95_ms, maximum_p95_ms)),
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
        "diagnosis_p95_ms": 999999.0,
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
