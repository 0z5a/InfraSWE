#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the 30-case inference-only R19 cohort."""

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
    "selection": "0c8de38deeaf86de50df188a459a2d99b34b8f8ea6dc9aca6192877667c28a20",
    "plan": "f038fe3c9af40da9da09c456a04f32608c99c01366ff7a3807a3ecab808701f5",
    "manifest": "93c394cdf4a25edab6c594bf9b774a1f3410645ffa3b4c8ff2dd8d5ecf1ab2d2",
    "static": "c5aad02c7062d0bbe8e8088bfd10247fcff3eae5f431cc54aa100fd87051e8c4",
    "initial": "89a9ba7c2fee107dbcbcea1953bfba87718bcf7cc7583ce9d66eee433053b6a5",
    "focused": "d4a76d126e7f2bdebd2777cb7ea0c628c77c0297ab4a5c1860e913a2ea5b8d9a",
}
EXPECTED_SELECTION_SHA256 = "sha256:be22a5d6d6b18a443f32fe7afbc504fcc64a0ef6f558e70cd2608c71707761ed"
EXPECTED_TEST_PLAN_SHA256 = "sha256:6ae2c51e8395f57319f3c5200c00ae779aed2f87639aadf45b61c8d85a4e6ab9"
EXPECTED_SOURCE_BUNDLE_SHA256 = "sha256:262a51da2a0ed0d170dcdf5bd67d9de0e997c0f00991a487ba6f29237f6d5b7c"
POLICY_ID = "inference-contract-disposition-split-v0.1-r19"


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
        code or ("TITLE_SCOPED_CONTRACT_CLOSED" if decision == "accept_with_scope" else "DISPOSITION_EVIDENCE_INCOMPLETE"),
    )


ASSESSMENTS: dict[str, Assessment] = {
    "flashinfer-pr-4900": a(
        "reject", "fail",
        "The exact added block-sparse paged-route matrix fails at its first case with a 19-versus-20 JIT plan argument mismatch.",
        "A second run in a unique FlashInfer workspace reproduces the same candidate failure, excluding cross-head JIT-cache contamination.",
        residual="Align the generated FA2 plan signature and wrapper call, then rerun all five added contract families.",
        code="EXACT_CANDIDATE_FAILURE",
    ),
    "flashinfer-pr-3370": a(
        "accept_with_scope", "bounded-gap",
        "The standalone SM80 preparation contract passes and nine FMHA-v2 cases skip only at their declared SM90/SM12x boundary.",
        "The eight-file change includes candidate prepare and plan/run tests plus an RTX 3080 standalone receipt; the unchanged API path remains explicit.",
    ),
    "flashinfer-pr-3387": a(
        "accept_with_scope", "bounded-gap",
        "All eight selected MXFP8 edge cases reach only the declared SM100 capability skip.",
        "Candidate reference tests cover bitwise MXFP8 scale conversion and NVFP4 fast-math control, with no reachable counterexample on A100.",
    ),
    "flashinfer-pr-3407": a(
        "reject", "bounded-gap",
        "All 17 candidate hybrid B12x/CUTLASS cases skip for CUDA 13 and the otherwise empty body supplies no exact-head target receipt.",
        "The unified dispatch API and two kernel families therefore remain unexecuted at the claimed production boundary.",
        residual="Run the candidate dispatch, numeric-parity, threshold, and invalid-configuration matrix on its CUDA 13 target.",
    ),
    "flashinfer-pr-3412": a(
        "reject", "bounded-gap",
        "The 23,568-case changed MoE matrix skips entirely at the SM100/SM103 gate.",
        "The six-file FP4 bias restoration has no body-level target receipt and every checklist item remains blank.",
        residual="Execute the FP4 bias matrix on SM100 or SM103 and record output parity for bias on and off.",
    ),
    "flashinfer-pr-3434": a(
        "accept_with_scope", "bounded-gap",
        "The four-file ragged-MIS wiring follows the already validated paged dispatch structure and guards the complete pointer tuple.",
        "Although no candidate test file is changed, the body supplies three H200 paged-versus-ragged functional matrices with bounded FP16 deltas.",
    ),
    "flashinfer-pr-3469": a(
        "accept_with_scope", "pass",
        "The one-line explicit false assignment matches both zero initialization and the Context-kernel selector's forced Disabled branch.",
        "The title-scoped no-op invariant is exhaustive and the body records the ragged attention control suite.",
    ),
    "flashinfer-pr-3458": a(
        "accept_with_scope", "pass",
        "The two-file edit replaces four deprecated implicit scalar extractions with the equivalent explicit pointer accessor.",
        "The transformation is mechanically complete in the two monolithic MLA decode variants and changes neither allocation nor synchronization semantics.",
    ),
    "sglang-pr-37638": a(
        "accept_with_scope", "pass",
        "All 270 exact TileLang ragged-tail cases pass on A100 after source-pinning TileLang, Z3, and TVM-FFI dependencies.",
        "The recent two-file fix also provides a deterministic production crash, 44/44 H20 recovery receipt, raw-kernel stray-write checks, and no remaining title-scoped residual.",
    ),
    "sglang-pr-27285": a(
        "accept_with_scope", "pass",
        "The isolated unified-radix suite completes 1,020 passes and 44 subtests; 588 skips are explicit fixture-family exclusions.",
        "The ten-file PP+HiCache L2 change supplies a crash reproduction and PP2/PP4 workload receipts while explicitly excluding L3.",
    ),
    "sglang-pr-27297": a(
        "accept_with_scope", "pass",
        "All 27 candidate LingBot cache and realtime transport unit tests pass behind fail-closed unrelated kernel imports.",
        "The body supplies bitwise 4xH200 consistency and measured steady-state latency plus a single-GPU server receipt.",
    ),
    "sglang-pr-27313": a(
        "accept_with_scope", "pass",
        "All five added strategy, facade, and server-argument contracts pass and the default runtime dispatch remains disabled.",
        "The title is limited to context-parallel abstractions; stack position is treated only as a diversity marker.",
    ),
    "sglang-pr-27228": a(
        "accept_with_scope", "bounded-gap",
        "The source removes a stale top-k veto, replaces tensor-bearing equality with identity, and enables the existing EAGLE invariant matrix.",
        "The nine changed test routes require large external models, but the memory-accounting invariant and the latent equality crash are explicit and narrow.",
    ),
    "sglang-pr-27174": a(
        "reject", "bounded-gap",
        "Three changed radix-force-miss tests pass, but they do not exercise the seven-file JSON/Prometheus load-metric projection or all queue states.",
        "The body supplies no executed response-level test or metric receipt.",
        residual="Add exact load-snapshot tests for cached, uncached, chunked-prefill, disaggregated decode, and both response encodings.",
    ),
    "sglang-pr-27298": a(
        "accept_with_scope", "pass",
        "All 52 serving-chat tests and nine subtests pass behind fail-closed unrelated kernel imports.",
        "The two-file string-level fix additionally reports a 13-tokenizer by four-mode comparison with no regressions.",
    ),
    "tensorrt_llm-pr-18528": a(
        "reject", "bounded-gap",
        "Five CPU-only scheduler tests are present, but generated bindings prevent local collection and the GB300 performance policy cannot execute on A100.",
        "The recent opt-in algorithm documents a p99 regression and follow-up mitigations but contains no explicit external-review or QA handoff proxy.",
        residual="Run the frozen unit tests in a built tree and close the documented deep-request fairness trade-off through review or a bounded policy guard.",
    ),
    "tensorrt_llm-pr-14780": a(
        "accept_with_scope", "bounded-gap",
        "The three-file repair restores an SM90 guard and removes Mamba intercept cost from the one-model draft branch.",
        "The body records same-A100 reproduction and regression verification, and removing the exact waiver re-enables the integration oracle.",
    ),
    "tensorrt_llm-pr-14961": a(
        "reject", "unresolved",
        "Five synthetic GQA/cross-attention tests are added, but collection cannot cross unavailable generated TensorRT modules.",
        "The six-file multi-GPU attention change has an empty description and no test result or target-functional receipt.",
        residual="Execute the 2-GPU GQA, cross-attention, Ulysses, ring-error, and Cosmos parity matrix in a built TensorRT-LLM tree.",
    ),
    "tensorrt_llm-pr-14764": a(
        "accept_with_scope", "pass",
        "All 39 CPU-only MoE-LoRA helper contracts pass with only unavailable backend construction replaced by an inert class marker.",
        "The title-scoped preparatory helper change covers sharing detection, canonical flags, backend/quant guards, shapes, and reference deltas.",
    ),
    "tensorrt_llm-pr-14862": a(
        "reject", "unresolved",
        "Six NVFP4 production files change while the only test artifact is one QA-list entry.",
        "The body describes rearrangements and a follow-up without an exact output, graph-safety, or Hopper execution receipt.",
        residual="Run the newly enabled Hopper QA case and add exact dequantization, graph-safety, and unsupported-SM checks.",
    ),
    "tensorrt_llm-pr-14816": a(
        "accept_with_scope", "bounded-gap",
        "Four candidate scheduler/recompute tests and five changed test routes cover victim selection, serialization, stats, and unbounded replay semantics.",
        "Generated bindings block local collection, but the exact frozen head and body enumerate the focused commands and preserve V1 defaults.",
    ),
    "tensorrt_llm-pr-14806": a(
        "reject", "unresolved",
        "The eight-file C++/nanobind/Python connector change includes substantial tests but cannot compile or run without generated bindings.",
        "The body explicitly leaves full CI pending, so the new hash-commit API and SWA guard remain technically unclosed.",
        residual="Compile the nanobind surface and pass the C++ hash-chain, Python connector, and persistent-cache integration controls.",
    ),
    "tensorrt_llm-pr-14979": a(
        "reject", "unresolved",
        "The four-file shared-pointer lifetime port changes async ownership but adds no failure-directed candidate test.",
        "Existing multi-GPU test edits only adapt construction types and do not reproduce broken-promise cleanup or cancellation races.",
        residual="Add a deterministic peer-drop/worker-cleanup lifetime test that proves every promise reaches a terminal state.",
    ),
    "vllm-pr-54993": a(
        "reject", "bounded-gap",
        "All 25 focused Mamba scheduling contracts pass, but the recent body explicitly calls the work a draft and names unresolved abstraction, coverage, and Hopper questions.",
        "Its own benchmark records a 12.9% regression at the 1024-token budget and contains no external-review or QA handoff proxy.",
        residual="Resolve the scheduler abstraction and performance regression, then add the named boundary, decode, larger-model, and Hopper coverage.",
    ),
    "vllm-pr-44577": a(
        "reject", "bounded-gap",
        "Five structural view/offset tests pass, but the seven-file packed-KV change alters NIXL and offloading registration without connector execution.",
        "No body-level test or end-to-end RDMA receipt distinguishes the claimed 92-to-1 region reduction from structural plausibility.",
        residual="Run send/receive and offload connector tests over the packed block and verify byte reconstruction and region registration.",
    ),
    "vllm-pr-44492": a(
        "accept_with_scope", "pass",
        "All three exact draft metadata bound, zero-padding, and clamp tests pass.",
        "The four-file fix also supplies two ROCm EAGLE/EAGLE3 failure-before and pass-after receipts at the production spec-decode boundary.",
    ),
    "vllm-pr-44514": a(
        "reject", "bounded-gap",
        "The changed FP8 integration test reaches an unavailable local compiled-UVA operator rather than a candidate assertion.",
        "However, the three-file removal of 133 lines has no recorded test result and no narrow base/head proof for old-to-new online MoE quantization parity.",
        residual="Run FP8 online linear and MoE model loading across the replacement frontend and prove configuration and output parity.",
    ),
    "vllm-pr-44535": a(
        "accept_with_scope", "pass",
        "All 16 selected multi-window profiler state, validation, reset, stop, and Torch-profiler tests pass.",
        "The five-file feature contains direct state-machine coverage for disjoint windows and preserves the single-window form.",
    ),
    "vllm-pr-44568": a(
        "accept_with_scope", "bounded-gap",
        "The one-file change delegates multimodal embedding to the language model interface expected by Model Runner V2.",
        "The body supplies the exact Cohere ASR initialization failure and a passing 72-second frozen-head control.",
    ),
    "vllm-pr-44566": a(
        "accept_with_scope", "bounded-gap",
        "The two-file config projection preserves explicit user precedence and carries the proposal token count before SpeculativeConfig validation.",
        "A complete base-failure/head-success EngineArgs reproduction records the derived value and production config path.",
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
    }
    require(file_sha256(args.selection_lock) == EXPECTED_FILE_SHA256["selection"], "R19 selection file changed")
    require(file_sha256(args.test_plan) == EXPECTED_FILE_SHA256["plan"], "R19 plan file changed")
    paths = {name: args.result_root / filename for name, filename in filenames.items()}
    for name, path in paths.items():
        require(file_sha256(path) == EXPECTED_FILE_SHA256[name], f"R19 {name} file changed")

    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    evidence = {name: read(path) for name, path in paths.items()}
    require(selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R19 selection identity changed")
    require(selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]), "R19 selection digest mismatch")
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R19 plan identity changed")
    require(plan["test_plan_sha256"] == canonical_sha256({key: value for key, value in plan.items() if key != "test_plan_sha256"}), "R19 plan digest mismatch")
    require(plan["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R19 plan binding mismatch")
    require(evidence["manifest"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256, "R19 source bundle changed")
    for name, payload in evidence.items():
        field = "evidence_manifest_sha256" if name == "manifest" else "evidence_sha256"
        require(payload[field] == canonical_sha256({key: value for key, value in payload.items() if key != field}), f"R19 {name} embedded digest mismatch")
        require(payload["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, f"R19 {name} binding mismatch")
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
    require(all(value is False for value in hidden), "R19 blind boundary is not intact")

    initial = {record["case_id"]: record for record in evidence["initial"]["records"]}
    focused = {record["case_id"]: record for record in evidence["focused"]["records"]}
    require(focused["flashinfer-pr-4900"]["returncode"] == 1, "R19 exact FlashInfer failure changed")
    require("19 but got 20" in focused["flashinfer-pr-4900"]["output_tail"], "R19 FlashInfer failure signature changed")
    for case_id in ("sglang-pr-37638", "sglang-pr-27285", "sglang-pr-27297", "sglang-pr-27174", "sglang-pr-27298", "tensorrt_llm-pr-14764"):
        require(focused[case_id]["returncode"] == 0, f"R19 focused pass changed: {case_id}")
    require(initial["vllm-pr-54993"]["returncode"] == 0, "R19 Mamba scheduler result changed")
    require(initial["vllm-pr-44492"]["returncode"] == 0, "R19 spec-decode result changed")
    require(initial["vllm-pr-44535"]["returncode"] == 0, "R19 profiler result changed")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 30, "R19 cohort is not 30 cases")
    require(selected.keys() == planned.keys() == ASSESSMENTS.keys(), "R19 case sets differ")
    bindings = {
        name: {
            "path": path.name,
            "artifact_sha256": canonical_sha256(evidence[name]),
            "evidence_sha256": evidence[name].get("evidence_sha256") or evidence[name].get("evidence_manifest_sha256"),
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
                    records.append({
                        "artifact": bindings[name],
                        "record_index": index,
                        "returncode": record.get("returncode"),
                        "status": record.get("status"),
                        "output_sha256": record.get("output_sha256"),
                    })
        require(records, f"{case_id}: no execution record")
        material = {
            "schema_version": "0.1",
            "policy_id": POLICY_ID,
            "case_id": case_id,
            "candidate_sha256": canonical_sha256({"selection": selected_case, "test_plan": planned[case_id]}),
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
            "hot_window_check_eligible": False,
            "legacy_r10_style_decision": "accept_with_scope" if assessed.decision == "accept_with_scope" else "check",
            "frozen_at": frozen_at,
        }
        locks.append({"material": material, "lock_sha256": canonical_sha256(material)})

    counts = {decision: sum(lock["material"]["decision"] == decision for lock in locks) for decision in ("accept_with_scope", "check", "reject", "unresolved")}
    require(counts == {"accept_with_scope": 18, "check": 0, "reject": 12, "unresolved": 0}, "R19 decision distribution changed")
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
            "accept_with_scope": sum(lock["material"]["legacy_r10_style_decision"] == "accept_with_scope" for lock in locks),
            "check": sum(lock["material"]["legacy_r10_style_decision"] == "check" for lock in locks),
        },
        "locks": locks,
    }
    output = {**output_material, "lock_set_sha256": canonical_sha256(output_material)}
    atomic_write_json(args.output, output)
    print(json.dumps({
        "lock_set_sha256": output["lock_set_sha256"],
        "decision_counts": counts,
        "decisions": {lock["material"]["case_id"]: lock["material"]["decision"] for lock in locks},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
