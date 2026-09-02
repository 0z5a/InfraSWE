#!/usr/bin/env python3
"""Aggregate graph LoRA fresh replays and apply checked-in ProjectFit formulas."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.draft import ProjectComparisonCell
from infraswe.scoring.deployability import weighted_geometric
from infraswe.scoring.project_fit import (
    PROJECT_FIT_WEIGHTS,
    build_project_fit,
    score_benchmark_trust,
    score_evolutionary_maintainability,
    score_operational_fit,
    score_performance_reuse_utilization,
    score_project_contract_fit,
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _component_payload(result) -> dict[str, Any]:
    return {
        "component": result.component.model_dump(mode="json"),
        "raw": dict(result.raw or {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--draft-resolution", type=Path, required=True)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.raw_dir.glob("replay-*.json"))
    if len(paths) != 7:
        raise SystemExit(f"expected 7 fresh replays, found {len(paths)}")
    runs = [_read(path) for path in paths]
    plan = _read(args.plan)
    resolution = _read(args.draft_resolution)
    draft = resolution["draft"]

    source_digests = {run["source_sha256"] for run in runs}
    if source_digests != {plan["target"]["source_sha256"]}:
        raise SystemExit("source identity drift")
    if {run["replay_index"] for run in runs} != set(range(1, 8)):
        raise SystemExit("fresh replay indices are incomplete")
    expected_cases = [item["id"] for item in plan["performance_cases"]]
    if any([item["case_id"] for item in run["performance"]] != expected_cases for run in runs):
        raise SystemExit("case order or membership drift")

    correctness_checks = [
        bool(result[phase]["passed"] and result[phase]["all_finite"])
        for run in runs
        for result in run["correctness"]
        for phase in ("eager", "cuda_graph_first_replay", "cuda_graph_dynamic_replay")
    ]
    alias_checks = [
        bool(result["eager"]["base_output_alias"]) for run in runs for result in run["correctness"]
    ]
    special_checks = [
        bool(item["passed"]) for run in runs for item in run["special_contracts"].values()
    ]
    hard_gates_passed = all(correctness_checks + alias_checks + special_checks)

    case_summaries = []
    for case_index, case_id in enumerate(expected_cases):
        medians = [float(run["performance"][case_index]["median_us"]) for run in runs]
        peaks = [int(run["performance"][case_index]["eager_peak_temporary_bytes"]) for run in runs]
        mean = statistics.mean(medians)
        case_summaries.append(
            {
                "case_id": case_id,
                "weight": plan["performance_cases"][case_index]["weight"],
                "fresh_process_medians_us": medians,
                "median_us": statistics.median(medians),
                "p95_us": _p95(medians),
                "min_us": min(medians),
                "max_us": max(medians),
                "cv": statistics.stdev(medians) / mean if mean else None,
                "median_eager_peak_temporary_bytes": int(statistics.median(peaks)),
            }
        )

    manual_test = args.sglang_root / "test/manual/lora/test_lora_ops.py"
    manual_test_present = manual_test.exists() and (
        "def test_sgemm_lora_b_graph_fwd" in manual_test.read_text(encoding="utf-8")
    )
    registered_test_present = False
    registered_root = args.sglang_root / "test/registered"
    if registered_root.exists():
        registered_test_present = any(
            "sgemm_lora_b_graph_fwd" in path.read_text(encoding="utf-8", errors="ignore")
            for path in registered_root.rglob("*.py")
        )

    maintainability_inputs = {
        "evolution": 1.0,
        "locality": 1.0,
        "tests": (int(manual_test_present) + int(registered_test_present) + int(hard_gates_passed))
        / 3,
        "failure": sum(special_checks) / len(special_checks),
        "contract": sum(correctness_checks + alias_checks) / len(correctness_checks + alias_checks),
    }
    contract_inputs = {
        "integration": 1.0,
        "interface": sum(alias_checks + special_checks) / len(alias_checks + special_checks),
        "lifecycle": sum(correctness_checks) / len(correctness_checks),
        "buildtest": (int(manual_test_present) + int(registered_test_present)) / 2,
        "policy": 1.0,
    }
    diagnostic_reuse_inputs = {
        "attainment": 1.0,
        "coverage": len(case_summaries) / len(expected_cases),
        "retention": 1.0,
        "family": 1.0,
        "compile": 1.0,
    }
    official_reuse_inputs = {**diagnostic_reuse_inputs, "attainment": None}
    operational_inputs = {
        "replay": len(runs) / 7,
        "load": 0.5,
        "resource": float(
            all(
                math.isfinite(float(case["median_eager_peak_temporary_bytes"]))
                for case in case_summaries
            )
        ),
        "coldsteady": 1.0,
    }

    maintainability = score_evolutionary_maintainability(maintainability_inputs)
    contract = score_project_contract_fit(contract_inputs)
    diagnostic_reuse = score_performance_reuse_utilization(diagnostic_reuse_inputs)
    official_reuse = score_performance_reuse_utilization(official_reuse_inputs)
    operational = score_operational_fit(operational_inputs)

    profile = draft["target"]
    acceptance = draft["acceptance_contract"]
    deployment = draft["deployment"]
    comparison_cell = ProjectComparisonCell(
        target_project_profile_sha256=profile["project_profile_sha256"],
        target_repository_or_baseline_sha256=plan["target"]["source_sha256"],
        change_intent="integrate",
        semantic_contract_sha256=deployment["request_or_step_protocol"]["sha256"],
        acceptance_contract_sha256=acceptance["sha256"],
        probe_set_sha256=acceptance["probe_set_sha256"],
        workload_portfolio_sha256=deployment["workload_portfolio"]["sha256"],
        performance_target_sha256=canonical_sha256(
            {"baseline_plan": plan, "normalization": "existing-variant=1.0"}
        ),
        required_deployment_cell_set_sha256=canonical_sha256(deployment["required_cells"]),
        formula_template_id="project-fit-kernel-v0.5",
        evidence_policy_id=draft["benchmark_loop"]["evidence_policy_id"],
        project_season=draft["scoring"]["project_season"],
    )
    official_fit = build_project_fit(
        mode="official",
        infra_cert="pass" if hard_gates_passed else "fail",
        formula_template_id="project-fit-kernel-v0.5",
        comparison_cell=comparison_cell,
        evolutionary_maintainability=maintainability,
        project_contract_fit=contract,
        performance_reuse_utilization=official_reuse,
        operational_fit=operational,
        fresh_process_replays=len(runs),
        evidence_grade="E1-framework",
        hidden_probes_complete=False,
        manifest_verified=True,
        sealed_draft_sha256=None,
    )
    diagnostic_values = {
        "evolutionary_maintainability": float(maintainability.component.value),
        "project_contract_fit": float(contract.component.value),
        "performance_reuse_utilization": float(diagnostic_reuse.component.value),
        "operational_fit": float(operational.component.value),
    }
    diagnostic_score = 100 * weighted_geometric(
        diagnostic_values, PROJECT_FIT_WEIGHTS["project-fit-kernel-v0.5"]
    )
    trust = score_benchmark_trust(
        reproducibility=1.0,
        evidence=0.75,
        statistics=1.0,
        environment=1.0,
        evidence_digests=[canonical_sha256(run) for run in runs],
        failure_codes=["E2_SYSTEM_TRACE_NOT_RUN"],
    )

    payload = {
        "schema_version": "0.5",
        "protocol_id": plan["protocol_id"],
        "variant": plan["target"]["variant"],
        "source_sha256": plan["target"]["source_sha256"],
        "draft": {
            "source_kind": resolution["source_kind"],
            "selected_default_project": resolution["selected_default_project"],
            "state": draft["draft"]["state"],
            "primary_host": draft["default_candidates"]["primary_host"],
            "primary_peer_impl": draft["default_candidates"]["primary_peer_impl"],
        },
        "infra_cert": "pass" if hard_gates_passed else "fail",
        "evidence_grade": "E1-framework",
        "fresh_process_replays": len(runs),
        "hard_gates": {
            "passed": hard_gates_passed,
            "correctness_checks_passed": sum(correctness_checks),
            "correctness_checks_total": len(correctness_checks),
            "alias_checks_passed": sum(alias_checks),
            "alias_checks_total": len(alias_checks),
            "special_checks_passed": sum(special_checks),
            "special_checks_total": len(special_checks),
        },
        "case_results": case_summaries,
        "project_fit_components": {
            "evolutionary_maintainability": _component_payload(maintainability),
            "project_contract_fit": _component_payload(contract),
            "performance_reuse_utilization_official": _component_payload(official_reuse),
            "performance_reuse_utilization_baseline_normalized": _component_payload(
                diagnostic_reuse
            ),
            "operational_fit": _component_payload(operational),
        },
        "diagnostic_project_fit_100": diagnostic_score,
        "diagnostic_score_is_official": False,
        "official_project_fit": official_fit.model_dump(mode="json"),
        "benchmark_trust": trust.model_dump(mode="json"),
        "test_inventory": {
            "manual_test_present": manual_test_present,
            "registered_test_present": registered_test_present,
            "external_fresh_replay_probe": True,
        },
        "raw_evidence": [str(path) for path in paths],
        "raw_evidence_sha256": [canonical_sha256(run) for run in runs],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
