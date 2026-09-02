#!/usr/bin/env python3
"""Compare frozen SGLang graph-LoRA baseline and candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
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


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _component_payload(result) -> dict[str, Any]:
    return {
        "component": result.component.model_dump(mode="json"),
        "raw": dict(result.raw or {}),
    }


def _load_fresh_runs(
    raw_dir: Path, expected_benchmark: str
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(raw_dir.glob("replay-*.json"))
    if len(paths) != 7:
        raise SystemExit(f"expected 7 fresh replays in {raw_dir}, found {len(paths)}")
    runs = [_read(path) for path in paths]
    if {int(run["replay_index"]) for run in runs} != set(range(1, 8)):
        raise SystemExit(f"fresh replay indices are incomplete in {raw_dir}")
    if any(run.get("benchmark") != expected_benchmark for run in runs):
        raise SystemExit(f"benchmark identity drift in {raw_dir}")
    if any(run.get("fresh_process") is not True for run in runs):
        raise SystemExit(f"non-fresh replay found in {raw_dir}")
    return paths, runs


def _correctness_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    correctness = [
        bool(result[phase]["passed"] and result[phase]["all_finite"])
        for run in runs
        for result in run["correctness"]
        for phase in ("eager", "cuda_graph_first_replay", "cuda_graph_dynamic_replay")
    ]
    aliases = [
        bool(result["eager"]["base_output_alias"]) for run in runs for result in run["correctness"]
    ]
    special = [bool(item["passed"]) for run in runs for item in run["special_contracts"].values()]
    return {
        "passed": all(correctness + aliases + special),
        "correctness_checks_passed": sum(correctness),
        "correctness_checks_total": len(correctness),
        "alias_checks_passed": sum(aliases),
        "alias_checks_total": len(aliases),
        "special_checks_passed": sum(special),
        "special_checks_total": len(special),
    }


def _absolute_summaries(
    runs: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_ids = [case["id"] for case in cases]
    for run in runs:
        if [item["case_id"] for item in run["performance"]] != expected_ids:
            raise SystemExit("absolute case order or membership drift")
    summaries = []
    for index, case in enumerate(cases):
        medians = [float(run["performance"][index]["median_us"]) for run in runs]
        peaks = [int(run["performance"][index]["eager_peak_temporary_bytes"]) for run in runs]
        summaries.append(
            {
                "case_id": case["id"],
                "fresh_process_medians_us": medians,
                "median_us": statistics.median(medians),
                "p95_us": _p95(medians),
                "min_us": min(medians),
                "max_us": max(medians),
                "median_eager_peak_temporary_bytes": int(statistics.median(peaks)),
            }
        )
    return summaries


def _normalized_geometric(values: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
        values = [(value, weight / total_weight) for value, weight in values]
    if any(value == 0 for value, _ in values):
        return 0.0
    return math.exp(sum(weight * math.log(value) for value, weight in values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-raw-dir", type=Path, required=True)
    parser.add_argument("--candidate-raw-dir", type=Path, required=True)
    parser.add_argument("--paired-raw-dir", type=Path, required=True)
    parser.add_argument("--identity-aa-raw-dir", type=Path, required=True)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--comparison-plan", type=Path, required=True)
    parser.add_argument("--paired-plan", type=Path, required=True)
    parser.add_argument("--identity-aa-plan", type=Path, required=True)
    parser.add_argument("--provenance-plan", type=Path, required=True)
    parser.add_argument("--draft-resolution", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--paired-script", type=Path, required=True)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_paths, baseline_runs = _load_fresh_runs(
        args.baseline_raw_dir, "sglang-graph-lora-b-cuda-graph-v1"
    )
    candidate_paths, candidate_runs = _load_fresh_runs(
        args.candidate_raw_dir, "sglang-graph-lora-b-cuda-graph-v1"
    )
    paired_paths, paired_runs = _load_fresh_runs(
        args.paired_raw_dir, "sglang-graph-lora-b-paired-cuda-graph-v1"
    )
    identity_paths, identity_runs = _load_fresh_runs(
        args.identity_aa_raw_dir, "sglang-graph-lora-b-paired-cuda-graph-v1"
    )
    baseline_plan = _read(args.baseline_plan)
    comparison_plan = _read(args.comparison_plan)
    paired_plan = _read(args.paired_plan)
    identity_plan = _read(args.identity_aa_plan)
    provenance_plan = _read(args.provenance_plan)
    resolution = _read(args.draft_resolution)
    draft = resolution["draft"]

    baseline_sha = comparison_plan["baseline_source_sha256"]
    candidate_sha = comparison_plan["candidate_source_sha256"]
    if {run["source_sha256"] for run in baseline_runs} != {baseline_sha}:
        raise SystemExit("baseline source identity drift")
    if {run["source_sha256"] for run in candidate_runs} != {candidate_sha}:
        raise SystemExit("candidate source identity drift")
    if _file_sha256(args.candidate_source) != candidate_sha:
        raise SystemExit("attached candidate file no longer matches the frozen identity")
    if any(run.get("baseline_source_sha256") != baseline_sha for run in paired_runs):
        raise SystemExit("paired baseline source identity drift")
    if any(run.get("candidate_source_sha256") != candidate_sha for run in paired_runs):
        raise SystemExit("paired candidate source identity drift")
    if any(
        run.get("baseline_source_sha256") != baseline_sha
        or run.get("candidate_source_sha256") != baseline_sha
        for run in identity_runs
    ):
        raise SystemExit("identity A/A source identity drift")
    if provenance_plan["candidate_comparison"] != {
        "baseline_source_sha256": baseline_sha,
        "candidate_source_sha256": candidate_sha,
    }:
        raise SystemExit("provenance plan source identity drift")
    if _file_sha256(args.paired_script) != provenance_plan["paired_script_sha256"]:
        raise SystemExit("paired script identity drift")
    if identity_plan["authority"].split(";")[0] != "diagnostic only":
        raise SystemExit("identity A/A authority drift")

    cases = baseline_plan["performance_cases"]
    expected_ids = [case["id"] for case in cases]
    if any([item["case_id"] for item in run["results"]] != expected_ids for run in paired_runs):
        raise SystemExit("paired case order or membership drift")
    if any([item["case_id"] for item in run["results"]] != expected_ids for run in identity_runs):
        raise SystemExit("identity A/A case order or membership drift")
    if any(int(run["samples_per_case"]) != 5 for run in paired_runs):
        raise SystemExit("paired sample-count drift")
    if any(int(run["iterations_per_sample"]) != 300 for run in paired_runs):
        raise SystemExit("paired iteration-count drift")
    if any(int(run["samples_per_case"]) != 5 for run in identity_runs):
        raise SystemExit("identity A/A sample-count drift")
    if any(int(run["iterations_per_sample"]) != 300 for run in identity_runs):
        raise SystemExit("identity A/A iteration-count drift")

    environment_identities = {
        (
            run["environment"]["gpu"],
            tuple(run["environment"]["compute_capability"]),
            run["environment"]["torch"],
            run["environment"]["cuda"],
        )
        for run in baseline_runs + candidate_runs + paired_runs + identity_runs
    }
    if len(environment_identities) != 1:
        raise SystemExit("environment identity drift across evidence sets")

    baseline_correctness = _correctness_summary(baseline_runs)
    candidate_correctness = _correctness_summary(candidate_runs)
    correctness_gate_passed = bool(
        baseline_correctness["passed"] and candidate_correctness["passed"]
    )
    baseline_absolute = _absolute_summaries(baseline_runs, cases)
    candidate_absolute = _absolute_summaries(candidate_runs, cases)

    threshold = float(paired_plan["gates"]["maximum_per_case_candidate_to_baseline_ratio"])
    case_results = []
    for case_index, case in enumerate(cases):
        process_ratios = []
        pooled_ratios = []
        for run in paired_runs:
            result = run["results"][case_index]
            ratios = [float(pair["candidate_to_baseline"]) for pair in result["pairs"]]
            if len(ratios) != 5:
                raise SystemExit("paired sample membership drift")
            recomputed = statistics.median(ratios)
            if not math.isclose(recomputed, float(result["paired_ratio_median"]), rel_tol=1e-12):
                raise SystemExit("paired median does not match its raw samples")
            process_ratios.append(recomputed)
            pooled_ratios.extend(ratios)
        ratio = statistics.median(process_ratios)
        identity_process_ratios = []
        identity_pooled_ratios = []
        for run in identity_runs:
            result = run["results"][case_index]
            ratios = [float(pair["candidate_to_baseline"]) for pair in result["pairs"]]
            if len(ratios) != 5:
                raise SystemExit("identity A/A sample membership drift")
            recomputed = statistics.median(ratios)
            if not math.isclose(recomputed, float(result["paired_ratio_median"]), rel_tol=1e-12):
                raise SystemExit("identity A/A median does not match its raw samples")
            identity_process_ratios.append(recomputed)
            identity_pooled_ratios.extend(ratios)
        identity_ratio = statistics.median(identity_process_ratios)
        speedup = 1.0 / ratio
        baseline_case = baseline_absolute[case_index]
        candidate_case = candidate_absolute[case_index]
        case_results.append(
            {
                "case_id": case["id"],
                "shape": {
                    key: case[key] for key in ("tokens", "slots", "rank", "slice_dims", "dtype")
                },
                "weight": float(case["weight"]),
                "optimized_branch": int(case["slots"]) >= 4,
                "relative_authority": "paired-interleaved",
                "paired_process_median_ratios": process_ratios,
                "candidate_to_baseline_ratio": ratio,
                "candidate_to_baseline_ratio_p95": _p95(process_ratios),
                "pooled_sample_ratio_median": statistics.median(pooled_ratios),
                "pooled_sample_ratio_p95": _p95(pooled_ratios),
                "speedup_x": speedup,
                "latency_reduction_percent": 100 * (1.0 - ratio),
                "three_percent_gate_passed": ratio <= threshold,
                "identity_aa_diagnostic": {
                    "authority": "diagnostic-only-no-gate-override",
                    "process_median_ratios": identity_process_ratios,
                    "baseline_b_to_a_ratio": identity_ratio,
                    "baseline_b_to_a_ratio_p95": _p95(identity_process_ratios),
                    "pooled_sample_ratio_median": statistics.median(identity_pooled_ratios),
                    "allocation_order_bias_percent": 100 * (identity_ratio - 1.0),
                    "identity_ratio_exceeds_three_percent": identity_ratio > threshold,
                    "bias_normalized_candidate_ratio": ratio / identity_ratio,
                    "bias_normalized_candidate_delta_percent": 100 * (ratio / identity_ratio - 1.0),
                },
                "absolute_latency": {
                    "baseline_median_us": baseline_case["median_us"],
                    "candidate_median_us": candidate_case["median_us"],
                    "sequential_speedup_x": (
                        baseline_case["median_us"] / candidate_case["median_us"]
                    ),
                    "baseline_p95_us": baseline_case["p95_us"],
                    "candidate_p95_us": candidate_case["p95_us"],
                },
                "eager_peak_temporary_bytes": {
                    "baseline_median": baseline_case["median_eager_peak_temporary_bytes"],
                    "candidate_median": candidate_case["median_eager_peak_temporary_bytes"],
                },
            }
        )

    performance_gate_passed = all(result["three_percent_gate_passed"] for result in case_results)
    overall_hard_gate_passed = correctness_gate_passed and performance_gate_passed
    failing_cases = [
        result["case_id"] for result in case_results if not result["three_percent_gate_passed"]
    ]
    identity_bias_gate_collisions = [
        result["case_id"]
        for result in case_results
        if result["identity_aa_diagnostic"]["identity_ratio_exceeds_three_percent"]
    ]

    optimized = [result for result in case_results if result["optimized_branch"]]
    attainment = _normalized_geometric(
        [
            (
                min(1.0, float(result["speedup_x"]) / 1.05),
                float(result["weight"]),
            )
            for result in optimized
        ]
    )
    retention = _normalized_geometric(
        [(min(1.0, float(result["speedup_x"])), float(result["weight"])) for result in case_results]
    )
    optimized_dtypes = {result["shape"]["dtype"] for result in optimized}
    coverage = float(optimized_dtypes == {"float16", "bfloat16"})
    family = float(candidate_correctness["passed"] and coverage == 1.0)
    resource = float(
        all(
            result["eager_peak_temporary_bytes"]["candidate_median"]
            <= result["eager_peak_temporary_bytes"]["baseline_median"]
            for result in case_results
        )
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
        "evolution": 0.5,
        "locality": 1.0,
        "tests": 0.5,
        "failure": float(candidate_correctness["special_checks_passed"])
        / float(candidate_correctness["special_checks_total"]),
        "contract": (
            float(candidate_correctness["correctness_checks_passed"])
            + float(candidate_correctness["alias_checks_passed"])
        )
        / (
            float(candidate_correctness["correctness_checks_total"])
            + float(candidate_correctness["alias_checks_total"])
        ),
    }
    contract_inputs = {
        "integration": 1.0,
        "interface": 1.0,
        "lifecycle": 1.0,
        "buildtest": 0.5,
        "policy": 1.0,
    }
    reuse_inputs = {
        "attainment": attainment,
        "coverage": coverage,
        "retention": retention,
        "family": family,
        "compile": 1.0,
    }
    operational_inputs = {
        "replay": 1.0,
        "load": 0.5,
        "resource": resource,
        "coldsteady": 1.0,
    }
    maintainability = score_evolutionary_maintainability(maintainability_inputs)
    contract = score_project_contract_fit(contract_inputs)
    reuse = score_performance_reuse_utilization(reuse_inputs)
    operational = score_operational_fit(operational_inputs)
    diagnostic_values = {
        "evolutionary_maintainability": float(maintainability.component.value),
        "project_contract_fit": float(contract.component.value),
        "performance_reuse_utilization": float(reuse.component.value),
        "operational_fit": float(operational.component.value),
    }
    diagnostic_score = 100 * weighted_geometric(
        diagnostic_values, PROJECT_FIT_WEIGHTS["project-fit-kernel-v0.5"]
    )

    profile = draft["target"]
    acceptance = draft["acceptance_contract"]
    deployment = draft["deployment"]
    comparison_cell = ProjectComparisonCell(
        target_project_profile_sha256=profile["project_profile_sha256"],
        target_repository_or_baseline_sha256=baseline_sha,
        change_intent="integrate",
        semantic_contract_sha256=deployment["request_or_step_protocol"]["sha256"],
        acceptance_contract_sha256=acceptance["sha256"],
        probe_set_sha256=acceptance["probe_set_sha256"],
        workload_portfolio_sha256=deployment["workload_portfolio"]["sha256"],
        performance_target_sha256=canonical_sha256(
            {
                "comparison_plan": comparison_plan,
                "paired_confirmation_plan": paired_plan,
            }
        ),
        required_deployment_cell_set_sha256=canonical_sha256(deployment["required_cells"]),
        formula_template_id="project-fit-kernel-v0.5",
        evidence_policy_id=draft["benchmark_loop"]["evidence_policy_id"],
        project_season=draft["scoring"]["project_season"],
    )
    infra_cert = "pass" if overall_hard_gate_passed else "fail"
    official_fit = build_project_fit(
        mode="official",
        infra_cert=infra_cert,
        formula_template_id="project-fit-kernel-v0.5",
        comparison_cell=comparison_cell,
        evolutionary_maintainability=maintainability,
        project_contract_fit=contract,
        performance_reuse_utilization=reuse,
        operational_fit=operational,
        fresh_process_replays=7,
        evidence_grade="E1-framework",
        hidden_probes_complete=False,
        manifest_verified=True,
        sealed_draft_sha256=None,
    )

    evidence_payloads = baseline_runs + candidate_runs + paired_runs + identity_runs
    trust = score_benchmark_trust(
        reproducibility=1.0,
        evidence=0.75,
        statistics=1.0,
        environment=1.0,
        evidence_digests=[canonical_sha256(payload) for payload in evidence_payloads],
        failure_codes=["E2_SYSTEM_TRACE_NOT_RUN"],
    )

    decision_codes = []
    required_actions = []
    if failing_cases:
        decision_codes.append("PAIRED_CONTROL_REGRESSION_GT_3_PERCENT")
        required_actions.append(
            "Resolve the source-identical slots=1 control result below the frozen 3% gate."
        )
    if identity_bias_gate_collisions:
        decision_codes.append("PAIRED_HARNESS_IDENTITY_BIAS_GE_3_PERCENT")
        required_actions.append(
            "Counterbalance allocation/capture order or share buffers in the paired harness."
        )
    if not registered_test_present:
        decision_codes.append("PROJECT_OWNED_BRANCH_TEST_MISSING")
        required_actions.append(
            "Add a registered SGLang regression test covering slots=3 and slots=4."
        )
    decision_codes.extend(
        [
            "DRAFT_SEAL_MISSING",
            "E2_SYSTEM_TRACE_NOT_RUN",
            "HIDDEN_PROBES_INCOMPLETE",
            "REQUIRED_DEPLOYMENT_CELLS_INCOMPLETE",
        ]
    )
    required_actions.extend(
        [
            "Run an integrated E2 SGLang server trace.",
            "Cover the frozen required SM80, SM89, and SM90 deployment cells.",
            "Seal the SGLang Draft and complete hidden probes before official scoring.",
        ]
    )

    payload = {
        "schema_version": "0.5",
        "protocol_id": comparison_plan["protocol_id"],
        "primary_project": "sglang",
        "candidate_source_sha256": candidate_sha,
        "baseline_source_sha256": baseline_sha,
        "environment": {
            "gpu": next(iter(environment_identities))[0],
            "compute_capability": list(next(iter(environment_identities))[1]),
            "torch": next(iter(environment_identities))[2],
            "cuda": next(iter(environment_identities))[3],
            "evidence_grade": "E1-framework",
        },
        "fresh_process_replays": {
            "baseline_absolute": 7,
            "candidate_absolute": 7,
            "paired_interleaved_source_identified_v2": 7,
            "identity_aa_diagnostic_source_identified_v2": 7,
        },
        "hard_gates": {
            "passed": overall_hard_gate_passed,
            "infra_cert": infra_cert,
            "correctness_and_contract_passed": correctness_gate_passed,
            "performance_retention_passed": performance_gate_passed,
            "performance_gate_interpretation": (
                "failed-as-frozen-but-confounded-by-identity-aa-allocation-bias"
                if identity_bias_gate_collisions
                else "failed-as-frozen"
                if not performance_gate_passed
                else "passed"
            ),
            "maximum_candidate_to_baseline_ratio": threshold,
            "failing_cases": failing_cases,
            "identity_aa_bias_gate_collisions": identity_bias_gate_collisions,
            "baseline": baseline_correctness,
            "candidate": candidate_correctness,
        },
        "paired_source_identity_verified": True,
        "case_results": case_results,
        "performance_reuse_inputs": reuse_inputs,
        "project_fit_components": {
            "evolutionary_maintainability": _component_payload(maintainability),
            "project_contract_fit": _component_payload(contract),
            "performance_reuse_utilization": _component_payload(reuse),
            "operational_fit": _component_payload(operational),
        },
        "diagnostic_project_fit_100": diagnostic_score,
        "diagnostic_score_is_official": False,
        "official_project_fit": official_fit.model_dump(mode="json"),
        "benchmark_trust": trust.model_dump(mode="json"),
        "test_inventory": {
            "existing_manual_test_present": manual_test_present,
            "existing_manual_test_exercises_new_slots_gte_4_branch": False,
            "registered_test_present": registered_test_present,
            "external_fresh_replay_branch_probe": True,
        },
        "maintainability_rationale": {
            "evolution": (
                "threshold rationale is commented, but num_loras >= 4 is a "
                "hard-coded, non-configurable heuristic"
            ),
            "locality": "the candidate is localized to one SGLang operator file",
            "tests": (
                "external branch coverage passed; project-owned registered branch "
                "coverage is absent"
            ),
        },
        "decision": {
            "verdict": "check",
            "rationale_codes": decision_codes,
            "supported_scope": [
                "single NVIDIA H100 PCIe SM90 operator-level CUDA Graph evidence",
                "preregistered FP16 and BF16 slots>=4 shapes",
            ],
            "excluded_scope": [
                "merge-ready or official ProjectFit claim",
                "integrated SGLang server behavior",
                "SM80, SM89, other GPUs, and other shapes",
                "cross-project ranking",
            ],
            "required_actions": required_actions,
        },
        "official_score_blockers": [
            "candidate failed one frozen 3% per-case performance gate",
            "SGLang Draft is D3-contract-proposed and unsealed",
            "operator evidence is E1-framework rather than E2-system-trace",
            "hidden project probes are incomplete",
            "required SM80 and SM89 deployment cells were not run",
        ],
        "raw_evidence": {
            "baseline": [str(path) for path in baseline_paths],
            "candidate": [str(path) for path in candidate_paths],
            "paired_source_identified_v2": [str(path) for path in paired_paths],
            "identity_aa_source_identified_v2": [str(path) for path in identity_paths],
        },
        "raw_evidence_sha256": {
            "baseline": [canonical_sha256(run) for run in baseline_runs],
            "candidate": [canonical_sha256(run) for run in candidate_runs],
            "paired_source_identified_v2": [canonical_sha256(run) for run in paired_runs],
            "identity_aa_source_identified_v2": [canonical_sha256(run) for run in identity_runs],
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
