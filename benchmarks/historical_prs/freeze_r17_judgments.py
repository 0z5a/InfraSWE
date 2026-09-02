#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the 30-case mixed R17 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.history.triage import CaseContractTriageEvidence, classify_case_contract
from infraswe.io import atomic_write_json

EXPECTED_FILE_SHA256 = {
    "selection": "3d08e7717d77d6114a179a1d926f81890b95bbaf49d4804731173055c74b96d2",
    "plan": "ed6ffeaaa8dc91b62bf2e0034fbfec0ff4605d3c3b62da8f2177ca7dbdc96779",
    "manifest": "198855d31489d52a47c9046003f18273b80d73880b7e63fe39943ae46fb7a075",
    "static": "6666f7142b8de487354be51e13adcd8eac811429ab2a30241c2fb91bd8b9a03c",
    "initial": "fcc19511c3631429ef4b249aeefd33b222bd6e0456f195290524d075b03875e8",
    "followup": "d9e49a77cc87a982a93ef59747cfa92da53c0a8fe6af599a5b54b3434b600a98",
    "rerun": "3356d1aecc9a7f798e9c4a73a2091840261958eb753b1ac4d1f8913f5c0b2f8a",
    "rerun_v2": "8d6a3fc3f428b1bf8067d08de12270b399a6d84eaeb2dc512e99e9adfe73fb46",
    "rerun_v3": "6d7086baacf69925c312c8c1f5780353b04fa565a4c1ded4e6ff1782c2564d42",
    "rerun_v4": "6e9c58eac5fbfe3d61f420ee6aa86a65e2c9761f662c3b9c43ed4d353740364d",
    "sglang_final": "c5a944ff6b50bf44314ee64a652d98857b9d4f847c9a997ee3efd45636e0b732",
}
EXPECTED_SELECTION_SHA256 = "sha256:ede934e0bf3af3c15bbba5bc5f55696077605fc48cbff23e4abce485a6f953fb"
EXPECTED_TEST_PLAN_SHA256 = "sha256:1d9cd3d63d09f97af0c86164a4b537f6a2054d11f4a0acbe7a9e5eb548335a58"
EXPECTED_SOURCE_BUNDLE_SHA256 = "sha256:4458a431ef57c673fa188b8453ba3d1379074652003eed4d720532795ee47c71"
POLICY_ID = "mixed-contract-disposition-split-v0.1-r17"


@dataclass(frozen=True, slots=True)
class Assessment:
    triage: CaseContractTriageEvidence
    technical_contract: str
    findings: tuple[str, ...]
    residual: str | None = None


ACCEPT = CaseContractTriageEvidence(True, True, True, closure_test="frozen-probe")
CHECK = CaseContractTriageEvidence(
    False,
    True,
    True,
    remediation_scope="single-site",
    closure_test="frozen-probe",
    residual_failure_families=1,
)
REJECT_UNPROVEN = CaseContractTriageEvidence(
    False,
    True,
    False,
    remediation_scope="unknown",
    closure_test="missing",
    residual_failure_families=1,
)
REJECT_BROAD = CaseContractTriageEvidence(
    False,
    True,
    True,
    remediation_scope="cross-cutting",
    closure_test="missing",
    design_change_required=True,
    residual_failure_families=2,
)
REJECT_FAILURE = CaseContractTriageEvidence(
    False,
    True,
    False,
    remediation_scope="single-site",
    closure_test="frozen-probe",
    baseline_regression=True,
    safety_or_integrity_failure=True,
    residual_failure_families=1,
)


def assessment(
    triage: CaseContractTriageEvidence,
    technical_contract: str,
    *findings: str,
    residual: str | None = None,
) -> Assessment:
    return Assessment(triage, technical_contract, findings, residual)


ASSESSMENTS: dict[str, Assessment] = {
    "liger-pr-1435": assessment(
        ACCEPT,
        "bounded-gap",
        "The one-file change serializes the two SwiGLU dI dot accumulations and the body supplies a base-distinguishing B200 error matrix plus corrected target results.",
        "SM100/B200 is unavailable locally, but the target-functional evidence and exact local source invariant are stronger than an age-only check.",
    ),
    "liger-pr-1157": assessment(
        ACCEPT,
        "pass",
        "All 32 forward-only fused-linear cross-entropy cases pass on the exact head.",
        "The one-line guard now follows grad_bias ownership, so the None.detach reachability is closed without changing kernel math.",
    ),
    "megatron-pr-7013": assessment(
        ACCEPT,
        "pass",
        "The newly added FP32-shard optimizer test passes and constructs the CPU-offloaded optimizer with leaf shard tensors.",
        "The two-file repair adds detach at the exact alias boundary and preserves shared storage.",
    ),
    "megatron-pr-5047": assessment(
        ACCEPT,
        "bounded-gap",
        "The candidate owns an explicit TP/EP/CP invariance matrix and the exact head compiles; the local one-rank attempt stops only at its declared eight-rank topology.",
        "The scaling factor cancels the TP/CP SUM cardinality algebraically and no independent counterexample was reached.",
    ),
    "slime-pr-1930": assessment(
        REJECT_BROAD,
        "pass",
        "All seven changed data-parallel scheduling tests pass.",
        "The mature seven-file PR is explicitly the first member of an N-part series and has an empty body, so standalone disposition closure and supersession risk remain unresolved.",
        residual="Freeze the final stacked-series member and demonstrate one complete variable-global-batch training step with the integrated rollout owner.",
    ),
    "slime-pr-1959": assessment(
        REJECT_UNPROVEN,
        "bounded-gap",
        "The evaluator path probe passes literal, template, fallback, and evaluation filename resolution.",
        "The mature replay feature has no candidate-owned test and does not execute Sample reconstruction or the live SGLang/offload training path named in its claim.",
        residual="Add exact dump round-trip tests and run one live colocated replay through rollout, offload/onload, and training ownership.",
    ),
    "torchtitan-pr-4398": assessment(
        ACCEPT,
        "pass",
        "All four candidate-owned valid-token collation and trainer tests pass with repository-pinned dependencies.",
        "The eleven-path propagation pops the scalar before model forwarding and enumerates core trainer, validator, forge, and torchft consumers.",
    ),
    "torchtitan-pr-3521": assessment(
        ACCEPT,
        "pass",
        "All four added CPU-offload view replay and keepalive tests pass on the exact head.",
        "The candidate also supplies H100 memory and throughput sweeps while default-off configuration preserves the control path.",
    ),
    "verl-pr-7685": assessment(
        REJECT_FAILURE,
        "fail",
        "After restoring the exact Megatron import path, six tests pass but the newly added non-DDP snapshot test fails by mutating a leaf parameter requiring grad in place.",
        "The exact candidate-owned failure prevents the ten-file offload ownership change from reaching closure.",
        residual="Fix the candidate test and rerun owner/non-owner reload, broadcast, expert replica, and repeated offload state transitions.",
    ),
    "verl-pr-6526": assessment(
        ACCEPT,
        "pass",
        "All eleven candidate optimizer precision and configuration tests pass against exact Megatron source.",
        "The branch matrix covers BF16, FP16, FP32, defaults, overrides, and distributed-optimizer projection while documenting the intended numeric change.",
    ),
    "flashinfer-pr-4861": assessment(
        REJECT_FAILURE,
        "fail",
        "After restoring the package-data JIT boundary, the exact candidate test still reports that each row does not sample from its requested seed stream.",
        "The failure is in the title-scoped per-request seed contract rather than an import or build precondition.",
        residual="Correct seed/offset indexing and pass scalar, length-one, per-row, reproducibility, and chain-speculative matrices.",
    ),
    "flashinfer-pr-3467": assessment(
        ACCEPT,
        "pass",
        "All twelve custom-mask length and shape validation cases pass after the package-data repair.",
        "The two-file fix rejects malformed single-prefill masks at the exact Python/kernel boundary.",
    ),
    "flashinfer-pr-3474": assessment(
        ACCEPT,
        "bounded-gap",
        "The long-running decode suite completes successfully after the one-time JIT window, with target-only parameterizations skipped on A100.",
        "All four decode launch paths zero the fixed counter region per call and the candidate reports its focused target test passing.",
    ),
    "flashinfer-pr-3449": assessment(
        ACCEPT,
        "pass",
        "Six structured tactic tests pass and the only skip is an SM80 capability boundary.",
        "The external cuDNN-frontend API is feature-gated, so unsupported environments retain the prior plan-name path.",
    ),
    "flashinfer-pr-3497": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "All 21 selected NVFP4 cases skip on A100 because the required architecture is unavailable.",
        "The mature body leaves both test-added and all-tests-passing checklist items unchecked, so no independent target closure overrides the not-ready signal.",
        residual="Run the dtype/device/global-scale matrix on supported SM100 hardware and publish passing exact results.",
    ),
    "sglang-pr-37612": assessment(
        ACCEPT,
        "pass",
        "All six newly added hybrid-SSM scheduler latch and retry tests pass.",
        "The three-file change clears the admission latch at the same bounded retry boundary as existing hybrid modes.",
    ),
    "sglang-pr-27312": assessment(
        ACCEPT,
        "pass",
        "Five tests and five subtests pass for server-argument, registry, and environment projection.",
        "The evaluator-owned run closes the body statement that the full local dependency environment had been unavailable.",
    ),
    "sglang-pr-27274": assessment(
        ACCEPT,
        "pass",
        "All four candidate LoRA virtual-expert tests pass after an optional-kernel import alias that fails closed if executed.",
        "The gate/up shrink selection is exercised without relying on the unavailable unrelated FP8 symbol.",
    ),
    "sglang-pr-27203": assessment(
        REJECT_FAILURE,
        "fail",
        "After supplying OpenCV, trimesh, and the optional-kernel import boundary, the exact candidate test constructs the conditioner without initializing its required TP group and fails.",
        "The failure is candidate-owned and title-scoped to camera-modulation caching setup.",
        residual="Initialize or inject the TP group in the test and pass direct/cached value parity plus cache-key and memory-guard cases.",
    ),
    "sglang-pr-27291": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The only changed test launches a four-GPU model server and is not executable on the two-GPU evaluator.",
        "The mature body explicitly leaves Accuracy Tests as TBD, so source plausibility cannot close the cache-hit claim.",
        residual="Add a reduced exact radix-tree prefetch/hit test and run the declared SWA plus HiCache L3 server sequence.",
    ),
    "tensorrt_llm-pr-18596": assessment(
        CHECK,
        "bounded-gap",
        "The recent two-file PR owns an exact cache-key test and reports focused target tests plus repeated GB300 performance results.",
        "The clean worktree reaches the generated bindings boundary; the body QA projection identifies one bounded test-database coverage follow-up.",
        residual="Run the frozen cache-key test in a generated-bindings TensorRT-LLM environment and register the matching QA coverage entry.",
    ),
    "tensorrt_llm-pr-14945": assessment(
        REJECT_BROAD,
        "unresolved",
        "The mature eight-file beam-search change has extensive candidate tests but cannot be imported without TensorRT bindings in the evaluator.",
        "The body explicitly states that the post-commit GPU rerun was not completed, leaving cache ownership and generation integration unclosed.",
        residual="Run all beam ownership tests and the PyTorch LLM integration case with generated bindings on target hardware.",
    ),
    "tensorrt_llm-pr-14911": assessment(
        ACCEPT,
        "bounded-gap",
        "The twelve-path change supplies nine exact pool, cyclic-window, address-stability, transform, and forward-equivalence tests plus a real GB200 run.",
        "Local collection reaches the generated TensorRT binding boundary without a candidate assertion; the complete artifact/state inventory supports bounded target transfer.",
    ),
    "tensorrt_llm-pr-14922": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The six-file timing change compiles statically but local tests cannot cross the generated bindings boundary.",
        "The mature body explicitly says TRT-LLM pytest was not runnable and provides no candidate-owned exact result for the new event lifetime.",
        residual="Run the three new timing/statistics tests and an iteration-to-event lifetime trace with generated bindings.",
    ),
    "tensorrt_llm-pr-14830": assessment(
        REJECT_BROAD,
        "unresolved",
        "The nine-file feature adds a model, weight mapper, server protocol, endpoint, and one heavyweight integration test.",
        "No runnable unit-level mapper, encoding, normalization, dimension, or error matrix closes the mature production serving surface.",
        residual="Add focused unit contracts and run the Qwen3 embedding endpoint against a reference model for float/base64 and dimension boundaries.",
    ),
    "vllm-pr-54990": assessment(
        ACCEPT,
        "pass",
        "All three focused cache-metric tests pass, including the new empty-versus-zero-hit distinction.",
        "The two-file logging-only repair leaves Prometheus counters and scheduler behavior unchanged.",
    ),
    "vllm-pr-44558": assessment(
        ACCEPT,
        "bounded-gap",
        "Five of six focused cadence and saturation-guard tests pass; the remaining multimodal control reaches only the evaluator's fail-closed image-transform shim.",
        "The seven-file scheduler change has candidate exact coverage and measured GB200 latency/throughput results without a reachable algorithmic counterexample.",
    ),
    "vllm-pr-44544": assessment(
        REJECT_FAILURE,
        "fail",
        "The exact availability assertion fails independently of ordering on A100.",
        "The mature body explicitly says Waiting on AITER PR, so the external dependency and candidate failure both veto disposition readiness.",
        residual="Land and package the required AITER API, then pass the full selector/backend matrix on gfx950 and the fallback matrix elsewhere.",
    ),
    "vllm-pr-44572": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The changed CUTLASS test file adds no candidate test and skips entirely because the compiled SM90 backend is unavailable.",
        "The mature kernel performance change removes the Python padding path without an evaluator numeric or compiled target closure.",
        residual="Run odd-M SM90 numeric and performance sweeps against the padded reference through the changed kernel dispatch.",
    ),
    "vllm-pr-44513": assessment(
        REJECT_UNPROVEN,
        "unresolved",
        "The selected online-quantization tests all deselect on A100 because the change is XPU-only.",
        "The mature body leaves test-plan evidence incomplete and depends on a separate fused-MoE kernel PR for part of the declared matrix.",
        residual="Run dense and MoE per-tensor, per-block, and MXFP8 online quantization on XPU with output parity and invalid-config checks.",
    ),
}


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_digest(payload: dict[str, Any], field: str, label: str) -> None:
    material = {key: value for key, value in payload.items() if key != field}
    require(payload.get(field) == canonical_sha256(material), f"{label} digest mismatch")


def binding(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": path.name,
        "evidence_sha256": payload.get("evidence_sha256") or payload.get("evidence_manifest_sha256"),
        "artifact_sha256": canonical_sha256(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(file_sha256(args.selection_lock) == EXPECTED_FILE_SHA256["selection"], "R17 selection file digest mismatch")
    require(file_sha256(args.test_plan) == EXPECTED_FILE_SHA256["plan"], "R17 plan file digest mismatch")
    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    require(selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]), "R17 embedded selection digest mismatch")
    require(selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R17 selection identity changed")
    require(plan["test_plan_sha256"] == canonical_sha256({key: value for key, value in plan.items() if key != "test_plan_sha256"}), "R17 embedded plan digest mismatch")
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R17 plan identity changed")
    require(plan["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R17 plan binding mismatch")
    blind_flags = (
        selection["selection_material"]["review_or_comment_visible"],
        selection["selection_material"]["merge_outcomes_visible"],
        selection["selection_material"]["ci_or_label_visible"],
        selection["selection_material"]["candidate_body_visible"],
        selection["selection_material"]["diff_content_visible"],
        plan["review_or_comment_requested"],
        plan["merge_outcome_or_state_requested"],
        plan["ci_or_label_requested"],
    )
    require(all(value is False for value in blind_flags), "R17 blind boundary is not intact")

    filenames = {
        "manifest": "source-evidence-manifest.json",
        "static": "static-evidence.json",
        "initial": "upstream-test-matrix.json",
        "followup": "upstream-followup-tests.json",
        "rerun": "upstream-followup-tests-rerun.json",
        "rerun_v2": "upstream-followup-tests-rerun-v2.json",
        "rerun_v3": "upstream-followup-tests-rerun-v3.json",
        "rerun_v4": "upstream-followup-tests-rerun-v4.json",
        "sglang_final": "upstream-followup-tests-sglang-final.json",
    }
    paths = {name: args.result_root / filename for name, filename in filenames.items()}
    evidence = {name: read(path) for name, path in paths.items()}
    for name, path in paths.items():
        require(file_sha256(path) == EXPECTED_FILE_SHA256[name], f"{name} file digest mismatch")
    validate_digest(evidence["manifest"], "evidence_manifest_sha256", "manifest")
    for name in filenames.keys() - {"manifest"}:
        validate_digest(evidence[name], "evidence_sha256", name)
    require(evidence["manifest"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256, "R17 source bundle changed")
    require(evidence["static"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256, "R17 static/source binding mismatch")

    initial = {record["case_id"]: record for record in evidence["initial"]["records"]}
    followup = {record["case_id"]: record for record in evidence["followup"]["records"]}
    rerun_v2 = {record["case_id"]: record for record in evidence["rerun_v2"]["records"]}
    sglang_final = {record["case_id"]: record for record in evidence["sglang_final"]["records"]}
    require(initial["megatron-pr-7013"]["returncode"] == 0, "R17 Megatron FP32-shard result changed")
    require("each row must sample" in followup["flashinfer-pr-4861"]["output_tail"], "R17 FlashInfer seed failure changed")
    require("4 passed" in rerun_v2["torchtitan-pr-4398"]["output_tail"], "R17 TorchTitan result changed")
    require("leaf Variable" in followup["verl-pr-7685"]["output_tail"], "R17 verl failure changed")
    require("assert not True" in followup["vllm-pr-44544"]["output_tail"], "R17 vLLM AITER failure changed")
    require("tensor model parallel group is not initialized" in sglang_final["sglang-pr-27203"]["output_tail"], "R17 SGLang failure changed")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 30, "R17 cohort is not 30 cases")
    require(selected.keys() == planned.keys() == ASSESSMENTS.keys(), "R17 case sets differ")
    common_bindings = {name: binding(paths[name], evidence[name]) for name in filenames}
    frozen_at = datetime.now(UTC).isoformat()
    locks: list[dict[str, Any]] = []
    for case_id, selected_case in selected.items():
        planned_case = planned[case_id]
        assessed = ASSESSMENTS[case_id]
        result = classify_case_contract(assessed.triage)
        if result.decision == "check":
            require(selected_case["temporal_band"] == "recent", f"{case_id}: mature case cannot be check")
            require(len(selected_case["paths"]) <= 8, f"{case_id}: check exceeds eight files")
            require(bool(initial[case_id]["test_paths"]), f"{case_id}: check lacks candidate test")
        legacy = "accept_with_scope" if assessed.triage.contract_satisfied else "check"
        records = []
        for name in filenames.keys() - {"manifest", "static"}:
            for index, record in enumerate(evidence[name].get("records", [])):
                if record.get("case_id") == case_id:
                    records.append({
                        "artifact": common_bindings[name],
                        "record_index": index,
                        "returncode": record.get("returncode"),
                        "status": record.get("status"),
                        "output_sha256": record.get("output_sha256"),
                    })
        require(records, f"{case_id}: no execution record")
        lock_material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256({"selection": selected_case, "test_plan": planned_case}),
            "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
            "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
            "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
            "common_evidence_binding_sha256": canonical_sha256(common_bindings),
            "supplemental_evidence_binding_sha256": canonical_sha256(records),
            "technical_contract": assessed.technical_contract,
            "triage_input": asdict(assessed.triage),
            "decision": result.decision,
            "rationale_codes": list(result.rationale_codes),
            "technical_findings": list(assessed.findings),
            "residual_contract": assessed.residual,
            "hot_window_check_eligible": selected_case["temporal_band"] == "recent" and len(selected_case["paths"]) <= 8 and bool(initial[case_id]["test_paths"]),
            "legacy_r10_style_decision": legacy,
            "frozen_at": frozen_at,
        }
        locks.append({"material": lock_material, "lock_sha256": canonical_sha256(lock_material)})

    decision_counts = {
        decision: sum(lock["material"]["decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check", "reject", "unresolved")
    }
    output_material = {
        "schema_version": "0.1",
        "protocol_id": plan["protocol_id"],
        "policy_id": POLICY_ID,
        "review_text_visible_during_machine_judgment": False,
        "merge_outcomes_visible_during_machine_judgment": False,
        "ci_fields_visible_during_machine_judgment": False,
        "learned_model_used": False,
        "trained_weights_used": False,
        "weighted_score_used": False,
        "forced_polarization_used": False,
        "terminology": "check",
        "selection_lock_file_sha256": "sha256:" + EXPECTED_FILE_SHA256["selection"],
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "test_plan_file_sha256": "sha256:" + EXPECTED_FILE_SHA256["plan"],
        "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
        "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "candidate_body_integrity_note": "Bodies were acquired only after the plan lock; four outcome-bearing body blocks were redacted before storage.",
        "common_evidence_bindings": common_bindings,
        "frozen_at": frozen_at,
        "decision_counts": decision_counts,
        "legacy_r10_style_decision_counts": {
            "accept_with_scope": sum(lock["material"]["legacy_r10_style_decision"] == "accept_with_scope" for lock in locks),
            "check": sum(lock["material"]["legacy_r10_style_decision"] == "check" for lock in locks),
        },
        "locks": locks,
    }
    output = {**output_material, "lock_set_sha256": canonical_sha256(output_material)}
    atomic_write_json(args.output, output)
    print(json.dumps({
        "lock_set_sha256": output["lock_set_sha256"],
        "decision_counts": decision_counts,
        "decisions": {lock["material"]["case_id"]: lock["material"]["decision"] for lock in locks},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
