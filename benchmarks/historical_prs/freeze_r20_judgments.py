#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the final 20-case inference R20 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json

EXPECTED_FILE_SHA256 = {
    "selection": "ca31b292db8381a84a2a0e339a3945db7fa0750570d7b138ff181ce85f418c1e",
    "plan": "11fe424f3805104700ce96cb3af02b80166156320b3e58ad9bc0ebf774dea342",
    "manifest": "9cf6db573d1e91918399d916d66181fc1347cfc1557c3c36314b390046000a40",
    "static": "82e61d30ee39ba808971fcaafefad1e8306182faf10129e823f80f869de47b2e",
    "initial": "00c35efc6215846edf957f4ba2f714b0252082b25e20dee99daa1032b5a5ee5d",
    "focused": "963418a5dce690b5fc2a3d6505be7a423b2d0ec147370e048fa397921ecc30c7",
    "narrow": "59813346ce362d94e0bcf1e8d6834f6635a8102bdd7b716860e10a4929f2d2c9",
}
EXPECTED_SELECTION_SHA256 = (
    "sha256:a611e87197fd30a4a3b4441849fbd9eed538086cb3e2169b3f995c93c18a0865"
)
EXPECTED_TEST_PLAN_SHA256 = (
    "sha256:60462a85ca3ccc7c374febdb09f321cd2696f2dcf1e286b5648cb70865bc6fa9"
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "sha256:bc4dad9a7b9c4388fc83449ce11f67532d17209ed83e70d7da3541259931f203"
)
POLICY_ID = "inference-contract-disposition-split-v0.1-r20"


@dataclass(frozen=True, slots=True)
class Assessment:
    decision: str
    technical_contract: str
    findings: tuple[str, ...]
    residual: str | None = None
    rationale_code: str = "TITLE_SCOPED_CONTRACT_CLOSED"


def a(
    decision: str,
    technical: str,
    *findings: str,
    residual: str | None = None,
    code: str | None = None,
) -> Assessment:
    return Assessment(
        decision,
        technical,
        findings,
        residual,
        code
        or (
            "TITLE_SCOPED_CONTRACT_CLOSED"
            if decision == "accept_with_scope"
            else "DISPOSITION_EVIDENCE_INCOMPLETE"
        ),
    )


ASSESSMENTS: dict[str, Assessment] = {
    "flashinfer-pr-4879": a(
        "reject",
        "bounded-gap",
        "Both candidate QK-BF16/PV-FP8 context-accuracy cases collect cleanly but skip at their declared SM100/SM103 boundary on A100.",
        "The six-file mixed-dtype route is internally scoped and the checklist reports success, but this one-day-old PR contains neither an executable target receipt nor an explicit external-review handoff.",
        residual="Run both dense-accuracy cases on SM100/SM103 and obtain an external review or QA handoff.",
        code="RECENT_NO_EXTERNAL_HANDOFF",
    ),
    "flashinfer-pr-3465": a(
        "accept_with_scope",
        "pass",
        "The focused A100 probe proves that the CuTe view advances exactly 8 MiB, preserves the expected remaining extent, and aliases the original workspace.",
        "The frozen source reserves the counter slab only when TRTLLM-Gen shares the buffer, while the body supplies a B200 reference check and before/after benchmark.",
    ),
    "flashinfer-pr-3461": a(
        "accept_with_scope",
        "pass",
        "All 18 existing logits/probability alignment cases pass; the six large-vocabulary scalar-k combinations exercise the new gated fast path.",
        "The path retains a fallback outside k/vocabulary thresholds and the body reports distribution checks plus two-hardware performance sweeps.",
    ),
    "flashinfer-pr-3506": a(
        "accept_with_scope",
        "pass",
        "Two consecutive CUDA-generator probes return Python integers and advance the rounded Philox offsets from 0 to 8 and then 12.",
        "The two-line conversion is limited to the PyTorch 2.11 scalar-state compatibility boundary and leaves the public API unchanged.",
    ),
    "flashinfer-pr-3430": a(
        "accept_with_scope",
        "pass",
        "All four edited modules compile and the exact frozen source contains neither deprecated `cute.core.ThrMma` nor the replaced `cute.make_fragment(` spelling.",
        "The patch is a mechanically complete API migration and does not alter layouts, synchronization, or numerical operations.",
    ),
    "sglang-pr-37620": a(
        "reject",
        "pass",
        "Seven focused LoRA inference-mode cases pass across the exact candidate test selection, and the body reports 28 tests plus a 30-case paired benchmark.",
        "Despite technical closure, the same-day PR has no explicit reviewer- or QA-owned handoff; R20 policy forbids direct hot-window accept from author receipts alone.",
        residual="Obtain an external review/QA disposition while retaining the chunked addmm regression matrix.",
        code="RECENT_NO_EXTERNAL_HANDOFF",
    ),
    "sglang-pr-27257": a(
        "reject",
        "fail",
        "The fail-closed focused run reaches candidate code and four selected tests pass, but the candidate-owned parallel-expand case fails exactly at `req[2]` with `IndexError: list index out of range`.",
        "The new modulo session-parameter selection cannot repair the independently unexpanded `rid` list, so the PR's own advertised n=2 contract is not executable on its frozen head.",
        residual="Make all per-request arrays follow the parallel expansion contract or correct the invalid test setup, then rerun the five selected cases.",
        code="EXACT_CANDIDATE_FAILURE",
    ),
    "sglang-pr-27201": a(
        "accept_with_scope",
        "bounded-gap",
        "The changed MI35x accuracy route cannot launch its two large models on the offline 2xA100 evaluator; the failure occurs before the candidate AITER path executes.",
        "The source closes both separated-layout and three SWA index-dtype boundaries, while the body records MI300 TP2 recovery from 0.041 to 0.904 and an unaffected TP1 control.",
    ),
    "sglang-pr-27300": a(
        "accept_with_scope",
        "pass",
        "All nine candidate enum-interface, conformance-guard, collision, and reserved-name cases pass on the frozen head.",
        "The registration-time reflection uses the enum class dictionary and directly covers the two missing predicates and future interface drift.",
    ),
    "sglang-pr-27290": a(
        "accept_with_scope",
        "pass",
        "The exact candidate EAGLE-v2 custom-logit-processor regression passes behind a fail-closed shim for unrelated optional kernels.",
        "The one-call source change mirrors the v1 ordering and supplies the draft-token batch width required by the processor's shape contract.",
    ),
    "tensorrt_llm-pr-18600": a(
        "check",
        "bounded-gap",
        "Three candidate estimation contracts and a GPT-OSS integration unwaive are present, but local collection stops at unavailable generated TensorRT-LLM bindings.",
        "The recent body contains an explicit QA review handoff with one bounded residual: the changed unit functions are not represented by the provided test-db/QA routing entries.",
        residual="Add or confirm the focused test-db/QA routing for the changed estimation cases and close the explicit QA follow-up.",
        code="EXPLICIT_QA_HANDOFF_ONE_RESIDUAL",
    ),
    "tensorrt_llm-pr-14869": a(
        "accept_with_scope",
        "pass",
        "A source-extracted execution of the exact two Triton functions matches the tensor reference for three gathered accepted steps over a genuinely strided destination.",
        "The probe also proves all adjacent interleaved state bytes and unselected blocks remain untouched; the body supplies the larger B300 throughput sweep.",
    ),
    "tensorrt_llm-pr-14891": a(
        "accept_with_scope",
        "bounded-gap",
        "The frozen patch gates both FP4 and FP8 atom splitting on the exact cached factor-times-atom token invariant.",
        "Two formerly waived multi-GPU FP4 MTP2/MTP3 integration routes are re-enabled, which counts as first-class target evidence even though A100 cannot execute the DSA DSL configuration.",
    ),
    "tensorrt_llm-pr-14970": a(
        "accept_with_scope",
        "bounded-gap",
        "Twelve candidate unit contracts span idle/active queue safety, local and collective dispatch, unsupported backends, encode-only mode, HTTP success, and error mapping.",
        "Generated bindings block local collection, but the final head represents the complete executor-to-LLM-to-server route and updates the API stability reference.",
    ),
    "tensorrt_llm-pr-14844": a(
        "accept_with_scope",
        "bounded-gap",
        "Two candidate tests cover propagation of context-worker chat prompt IDs into the generation request and the context-only response.",
        "Generated TensorRT modules block local collection, but the mature three-file head contains the title-scoped producer and consumer controls; the secondary empty-text token-delta guard remains a bounded unexecuted path.",
    ),
    "vllm-pr-55015": a(
        "reject",
        "bounded-gap",
        "Both candidate routing-selection contracts pass and the SM100 end-to-end MoE case skips only at its declared capability gate.",
        "The recent seven-file backend migration has only a generic PASS receipt and no explicit external-review or QA handoff, so hot-window technical success cannot directly predict acceptance.",
        residual="Run the fused MXFP4/MXFP8 MoE path on SM100 and obtain an external review/QA disposition.",
        code="RECENT_NO_EXTERNAL_HANDOFF",
    ),
    "vllm-pr-44526": a(
        "accept_with_scope",
        "pass",
        "A minimal exact-boundary execution rebuilds an over-computed streaming session and clamps computed and prompt lengths to 6/6.",
        "The broad legacy streaming fixture fails during unrelated multimodal-budget initialization, while the body supplies a completed spec-decode hardware scenario and baseline control.",
    ),
    "vllm-pr-44475": a(
        "accept_with_scope",
        "pass",
        "The centralized helper passes disabled-flag, cold-cache, warm-cache, rounding, and zero-prompt boundaries, including the reported 1632/1645 = 0.9921 case.",
        "The four-file mature refactor routes both chat and completion streaming/non-streaming usage through that helper and the body supplies eight executable server receipts.",
    ),
    "vllm-pr-44527": a(
        "accept_with_scope",
        "bounded-gap",
        "The exact two-line final patch removes the redundant floating-point logits fill and active-row top-k prefill while retaining allocation-time initialization and downstream overwrite contracts.",
        "The ROCm-only route cannot execute on A100, but the body records zero failed requests, accuracy within published error, and profiler elimination of both kernel families.",
    ),
    "vllm-pr-44571": a(
        "accept_with_scope",
        "bounded-gap",
        "The final head supplies the fully qualified vision-embedder prefix to `ColumnParallelLinear`, allowing the compressed-tensors ignore rule to select the unquantized parameter layout at construction.",
        "No candidate test file is added, but the one-file fix has a concrete W4A16 failure signature, head-success text/image receipt, and an unquantized control.",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    filenames = {
        "manifest": "source-evidence-manifest.json",
        "static": "static-evidence.json",
        "initial": "upstream-test-matrix.json",
        "focused": "upstream-focused-followups.json",
        "narrow": "upstream-narrow-rechecks.json",
    }
    require(
        file_sha256(args.selection_lock) == EXPECTED_FILE_SHA256["selection"],
        "R20 selection file changed",
    )
    require(file_sha256(args.test_plan) == EXPECTED_FILE_SHA256["plan"], "R20 plan file changed")
    paths = {name: args.result_root / filename for name, filename in filenames.items()}
    for name, path in paths.items():
        require(file_sha256(path) == EXPECTED_FILE_SHA256[name], f"R20 {name} file changed")

    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    evidence = {name: read(path) for name, path in paths.items()}
    require(
        selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
        "R20 selection identity changed",
    )
    require(
        selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]),
        "R20 selection digest mismatch",
    )
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R20 plan identity changed")
    require(
        plan["test_plan_sha256"]
        == canonical_sha256(
            {key: value for key, value in plan.items() if key != "test_plan_sha256"}
        ),
        "R20 plan digest mismatch",
    )
    require(plan["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R20 plan binding mismatch")
    require(
        evidence["manifest"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256,
        "R20 source bundle changed",
    )
    for name, payload in evidence.items():
        field = "evidence_manifest_sha256" if name == "manifest" else "evidence_sha256"
        require(
            payload[field]
            == canonical_sha256({key: value for key, value in payload.items() if key != field}),
            f"R20 {name} embedded digest mismatch",
        )
        require(
            payload["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
            f"R20 {name} binding mismatch",
        )
    hidden = (
        selection["selection_material"]["review_or_comment_visible"],
        selection["selection_material"]["merge_outcomes_visible"],
        selection["selection_material"]["ci_or_label_visible"],
        selection["selection_material"]["candidate_body_visible"],
        selection["selection_material"]["diff_content_visible"],
        plan["review_or_comment_requested"],
        plan["merge_outcome_or_state_requested"],
        plan["ci_or_label_requested"],
    )
    require(all(value is False for value in hidden), "R20 blind boundary is not intact")

    initial = {record["case_id"]: record for record in evidence["initial"]["records"]}
    focused = {record["case_id"]: record for record in evidence["focused"]["records"]}
    narrow = {record["case_id"]: record for record in evidence["narrow"]["records"]}
    for case_id in (
        "flashinfer-pr-3461",
        "flashinfer-pr-3465",
        "flashinfer-pr-3506",
        "flashinfer-pr-3430",
        "sglang-pr-27290",
        "vllm-pr-44475",
    ):
        require(focused[case_id]["returncode"] == 0, f"R20 focused pass changed: {case_id}")
    require(focused["sglang-pr-27257"]["returncode"] == 1, "R20 SGLang exact failure changed")
    require(
        "IndexError: list index out of range" in focused["sglang-pr-27257"]["output_tail"],
        "R20 SGLang failure signature changed",
    )
    require(narrow["vllm-pr-44526"]["returncode"] == 0, "R20 vLLM narrow result changed")
    require(
        narrow["tensorrt_llm-pr-14869"]["returncode"] == 0, "R20 TensorRT-LLM narrow result changed"
    )
    require(
        initial["flashinfer-pr-4879"]["returncode"] == 0, "R20 recent FlashInfer result changed"
    )
    require(initial["sglang-pr-37620"]["returncode"] == 0, "R20 recent SGLang result changed")
    require(initial["vllm-pr-55015"]["returncode"] == 0, "R20 recent vLLM result changed")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 20, "R20 cohort is not 20 cases")
    require(selected.keys() == planned.keys() == ASSESSMENTS.keys(), "R20 case sets differ")
    bindings = {
        name: {
            "path": path.name,
            "artifact_sha256": canonical_sha256(evidence[name]),
            "evidence_sha256": evidence[name].get("evidence_sha256")
            or evidence[name].get("evidence_manifest_sha256"),
        }
        for name, path in paths.items()
    }
    frozen_at = datetime.now(UTC).isoformat()
    locks: list[dict[str, Any]] = []
    for case_id, selected_case in selected.items():
        assessed = ASSESSMENTS[case_id]
        records = []
        for name, payload in evidence.items():
            for index, record in enumerate(payload.get("records", [])):
                if record.get("case_id") == case_id:
                    records.append(
                        {
                            "artifact": bindings[name],
                            "record_index": index,
                            "returncode": record.get("returncode"),
                            "status": record.get("status"),
                            "output_sha256": record.get("output_sha256"),
                        }
                    )
        require(records, f"{case_id}: no execution record")
        material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256(
                {"selection": selected_case, "test_plan": planned[case_id]}
            ),
            "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
            "test_plan_sha256": EXPECTED_TEST_PLAN_SHA256,
            "source_bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
            "common_evidence_binding_sha256": canonical_sha256(bindings),
            "supplemental_evidence_binding_sha256": canonical_sha256(records),
            "technical_contract": assessed.technical_contract,
            "decision": assessed.decision,
            "rationale_codes": [assessed.rationale_code],
            "technical_findings": list(assessed.findings),
            "residual_contract": assessed.residual,
            "hot_window_check_eligible": case_id == "tensorrt_llm-pr-18600",
            "legacy_r10_style_decision": "accept_with_scope"
            if assessed.decision == "accept_with_scope"
            else "check",
            "frozen_at": frozen_at,
        }
        locks.append({"material": material, "lock_sha256": canonical_sha256(material)})

    counts = {
        decision: sum(lock["material"]["decision"] == decision for lock in locks)
        for decision in ("accept_with_scope", "check", "reject", "unresolved")
    }
    require(
        counts == {"accept_with_scope": 15, "check": 1, "reject": 4, "unresolved": 0},
        "R20 decision distribution changed",
    )
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
        "candidate_body_integrity_note": "Bodies were acquired only after the plan lock; five outcome-bearing blocks were redacted before storage.",
        "common_evidence_bindings": bindings,
        "frozen_at": frozen_at,
        "decision_counts": counts,
        "legacy_r10_style_decision_counts": {
            "accept_with_scope": sum(
                lock["material"]["legacy_r10_style_decision"] == "accept_with_scope"
                for lock in locks
            ),
            "check": sum(
                lock["material"]["legacy_r10_style_decision"] == "check" for lock in locks
            ),
        },
        "locks": locks,
    }
    output = {**output_material, "lock_set_sha256": canonical_sha256(output_material)}
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "lock_set_sha256": output["lock_set_sha256"],
                "decision_counts": counts,
                "decisions": {
                    lock["material"]["case_id"]: lock["material"]["decision"] for lock in locks
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
