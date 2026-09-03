#!/usr/bin/env python3
# ruff: noqa: E501
"""Freeze outcome-blind judgments for the 30-case inference-only R18 cohort."""

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
    "selection": "cad2609af0a34df1b07fc1417ab8e783f6753de3d199f7139619c58419416862",
    "plan": "83069338afeac27d39451be0d3ee8452fb0b49c476220d8e1485443d55dffa72",
    "manifest": "7ae52c74f75333c619d5066f5df100ddd0fde6aea7cf22152ef7618895f8d1a2",
    "static": "73db85a8a43a4c76d783241375c2a8f60d76ca246084bd5f949fff616dcaef4f",
    "initial": "aaaf6ab5b301c7643dc3d82c599dbdc938bd830d469c8b4c8f995bdbf154dd0d",
    "followup": "bf454e8216f1178d940d89ca7e6a3ea8516cf4347a37a3936d80c5e75fe34585",
    "vllm": "7aa165cf12c32c747418637e9b360156a19c567a1b16340597a8321bda4d6968",
    "vllm_conftest": "56a321d699192193afee5c2bcfc65319dc3654b336d9b54b29e6cb6a56def448",
    "vllm_conftest_v2": "74c19ae6a52cf336dbcc0c57c661d2e26899ec1e16ee7c57b4f42363e72bd41f",
    "focused": "8d386fbcb2407befdc02b0b462201655875823685561f43a679f5a7a32d5ccdc",
}
EXPECTED_SELECTION_SHA256 = (
    "sha256:8589c445a9337530e50a3b166f0838bf454299c453c7cee5e393c36069f5c622"
)
EXPECTED_TEST_PLAN_SHA256 = (
    "sha256:60299d98042deaa88a1b39ba11e6c62c9d3eb04d9b61b067dcca33b076206dc2"
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "sha256:518c3bff124896ad5043ac7b407c960a0cbb65f573a1b37b275a267c9eecc9f1"
)
POLICY_ID = "inference-contract-disposition-split-v0.1-r18"


@dataclass(frozen=True, slots=True)
class Assessment:
    decision: str
    technical_contract: str
    findings: tuple[str, ...]
    residual: str | None = None


def a(decision: str, technical: str, *findings: str, residual: str | None = None) -> Assessment:
    return Assessment(decision, technical, findings, residual)


ASSESSMENTS: dict[str, Assessment] = {
    "flashinfer-pr-4850": a(
        "check",
        "bounded-gap",
        "The two-file SM120 dispatch extension has exact target geometry, collaborator-owned TP1/TP2 image-serving receipts, and a candidate numerical matrix that skips only on the evaluator's SM80 GPU.",
        "A broader open refactor can supersede this narrow route and FlashInfer GPU CI is the single remaining closure step.",
        residual="Run the frozen numerical matrix in FlashInfer SM120 CI and resolve the explicitly documented supersession order.",
    ),
    "flashinfer-pr-3485": a(
        "accept_with_scope",
        "pass",
        "The warmed exact head completes 192 FP8 paged/ragged prefill cases on A100.",
        "The body supplies SM120/SM121 before-and-after sweeps and the staging-aware heuristic preserves non-FP8 code paths.",
    ),
    "flashinfer-pr-3357": a(
        "accept_with_scope",
        "bounded-gap",
        "All 156 selected new CuTe-DSL swizzle cases skip at their declared SM100 boundary rather than fail.",
        "The candidate owns CUDA/CuTe bitwise-parity tests and B200 backend comparisons; default public behavior is unchanged.",
    ),
    "flashinfer-pr-3355": a(
        "accept_with_scope",
        "bounded-gap",
        "The candidate autotuner smoke test reaches its explicit SM100 skip on A100.",
        "A complete B200 batch/sequence performance grid and default-off autotune gate close the scoped backend selection change.",
    ),
    "flashinfer-pr-3503": a(
        "accept_with_scope",
        "pass",
        "All three added raw and packed custom-mask length contracts pass on the exact head.",
        "The wrapper-only check is narrow, leaves the CUDA coordinate guard intact, and reports two existing runtime controls passing.",
    ),
    "flashinfer-pr-3457": a(
        "accept_with_scope",
        "bounded-gap",
        "The 55 selected quantization cases skip only because MXFP8/NVFP4 requires SM100.",
        "Candidate tests distinguish rank-preserving 2D/3D layouts, per-batch padding, default compatibility, and the unsupported CuTe branch.",
    ),
    "flashinfer-pr-3393": a(
        "accept_with_scope",
        "bounded-gap",
        "The full 97,172-case decode collection is cleanly target-skipped on SM80 with no reachable assertion failure.",
        "The body reports a 576/576 B200 matrix and the three small source changes close selector, reduction offset, and workspace-layout invariants.",
    ),
    "sglang-pr-37643": a(
        "check",
        "pass",
        "All three candidate DP-attention broadcast tests pass and the new fake collective reproduces the None-source failure on base.",
        "The recent three-file fix has one bounded residual: execute the two-hop TP-by-CP collective on a real multi-GPU topology.",
        residual="Run one real attn_tp>1 and attn_cp>1 request broadcast and confirm every rank receives the leader lists.",
    ),
    "sglang-pr-27159": a(
        "reject",
        "unresolved",
        "The four-file NPU attention path has no candidate-owned test, leaves accuracy/performance and every checklist item blank, and explicitly lists unsupported CP/scoring families.",
        "A tens-of-GiB memory claim plus a 22% causal regression cannot close from source plausibility on A100.",
        residual="Add NPU exact output tests for encoder and causal variable-length batches plus measured memory and supported-family guards.",
    ),
    "sglang-pr-27180": a(
        "reject",
        "bounded-gap",
        "Three IPv6 parser tests pass, but the six-file PR combines IPv6 ZMQ transport, benchmark sampling projection, and logging behavior.",
        "All three body test-plan items remain unchecked, leaving the production multinode socket and two unrelated routes unexecuted.",
        residual="Split or execute exact IPv6 snapshot transport, benchmark parameter forwarding, and warning-level controls.",
    ),
    "sglang-pr-27181": a(
        "accept_with_scope",
        "bounded-gap",
        "All 15 isolated LoRA overlap-loader state, capacity, eviction, and synchronization tests pass; the broad run stops only on gated model access.",
        "The body reports single/two-adapter and TP2 integration plus a controlled TTFT comparison, while the default capacity semantics are explicit.",
    ),
    "sglang-pr-27239": a(
        "accept_with_scope",
        "pass",
        "All ten selected local-path and Hub-identifier cases pass on the exact head.",
        "The two-file guard is confined to unambiguously local spellings and preserves remote repository IDs.",
    ),
    "sglang-pr-27192": a(
        "accept_with_scope",
        "bounded-gap",
        "All three newly added registry ownership and gathered-DP tests pass.",
        "The structural migration inventories legacy containers and consumers; stack membership is treated only as a diversity signal, with GPU replay retained as a bounded gap.",
    ),
    "sglang-pr-27188": a(
        "accept_with_scope",
        "bounded-gap",
        "The AMD MI35x integration cannot launch on A100, but the exact target suite and workflow registration are present and static integrity holds.",
        "The body supplies repeatable crash-before/pass-after TP2 and TP4 accuracy controls for the one-value metadata gate.",
    ),
    "sglang-pr-27183": a(
        "reject",
        "pass",
        "All nine YOCO eligibility and truncate/restore unit contracts pass behind fail-closed unrelated kernel imports.",
        "The body nevertheless leaves both MMLU arms TODO and explicitly says they must be filled before review, so performance alone does not close correctness readiness.",
        residual="Run the declared E2B/E4B flag-off/on MMLU comparison and record tolerance-bounded parity before review.",
    ),
    "tensorrt_llm-pr-18538": a(
        "reject",
        "unresolved",
        "The recent four-file C++ KV-cache cleanup has an empty description and test section and no executable candidate Python test.",
        "A generic checked checklist cannot establish behavior preservation for the 358-line manager change.",
        residual="Name and run the changed C++ KV-cache manager cases, including capacity, allocation, free, reuse, and executor API parity.",
    ),
    "tensorrt_llm-pr-14849": a(
        "accept_with_scope",
        "bounded-gap",
        "The exact test reaches only the unavailable generated TensorRT binding boundary, not a candidate assertion.",
        "The three-file rollback removal has an explicit history/capacity invariant and reports all 11 focused GB200 tests passing.",
    ),
    "tensorrt_llm-pr-14887": a(
        "accept_with_scope",
        "bounded-gap",
        "The two-file repair clamps the pool key with the same max-sequence normalization already applied to attention windows.",
        "The existing model-registry accuracy route is re-enabled and the body records same-GPU and regression verification.",
    ),
    "tensorrt_llm-pr-14765": a(
        "accept_with_scope",
        "pass",
        "All 18 CPU-only sidecar schema and flag-projection tests pass without loading unavailable generated bindings.",
        "The five-file title scope is metadata convention and propagation, not the explicitly deferred cache-size reduction; unset sidecars preserve false flags.",
    ),
    "tensorrt_llm-pr-14910": a(
        "accept_with_scope",
        "bounded-gap",
        "The four-path change rewrites generation usage from the concurrent context response and enables block/partial reuse in the exact integration configurations.",
        "The body records the original same-GPU failure and regression verification; local execution is blocked only by the packaged multi-node runtime.",
    ),
    "tensorrt_llm-pr-14853": a(
        "accept_with_scope",
        "bounded-gap",
        "Two exact attention-DP dummy-slot tests are present and collection reaches only the generated TensorRT binding boundary.",
        "The five-file change reserves the title-scoped slot for the documented max-batch-one disaggregated Mamba case.",
    ),
    "tensorrt_llm-pr-14845": a(
        "accept_with_scope",
        "bounded-gap",
        "Candidate scheduler and disk-cache validation tests are present; local collection stops at generated bindings.",
        "The five-file API-to-manager projection is narrow and retains validation at the public configuration boundary.",
    ),
    "vllm-pr-54979": a(
        "check",
        "pass",
        "All eight focused terminal-registration, async in-flight, and Mamba copy-on-write cases pass on the exact head.",
        "The recent five-file PR includes red-green evidence and an A40 token/cache oracle, while adjacent cache-lifecycle proposals leave one bounded integration review surface.",
        residual="Review the shared Mamba publication interaction with the named adjacent proposals and run the external-connector boundary if it enters scope.",
    ),
    "vllm-pr-44564": a(
        "reject",
        "bounded-gap",
        "The eleven-file new SQuat backend adds calibration, storage kernels, registry/configuration, and serving behavior without a candidate-owned test path.",
        "Published benchmark tables do not independently close rotation loading, invalid configuration, numeric kernel parity, or fallback reachability.",
        residual="Add exact store/dequant parity, rotation validation, fallback, page-layout, and server integration tests across the declared models.",
    ),
    "vllm-pr-44456": a(
        "reject",
        "unresolved",
        "The mature ten-file third stack member runs only two empty-parameter skips in the changed offloading test.",
        "Its body explicitly defers GPU and multi-node Mamba bind validation until before merge, so the ownership migration is not ready.",
        residual="Execute both Mamba bind paths and NIXL/Mooncake registration on the stated GPU multi-node P/D topology.",
    ),
    "vllm-pr-44450": a(
        "accept_with_scope",
        "pass",
        "Both candidate Model Runner V2 multimodal-LoRA state and mapping tests pass with a minimal fail-closed import boundary.",
        "The three-file fix also reports the original real Qwen-VL failure and its passing end-to-end control.",
    ),
    "vllm-pr-44528": a(
        "accept_with_scope",
        "pass",
        "All 17 selected Mooncake PP ownership, transfer-region, bootstrap, and request-finish tests pass.",
        "The body supplies 162-unit, four-node GLM, GSM8K, and 32K TTFT receipts while explicitly excluding decode-side PP.",
    ),
    "vllm-pr-44431": a(
        "accept_with_scope",
        "pass",
        "All three candidate retry, disabled-cache, and once-scheduled prefix-cache statistics tests pass.",
        "The body supplies red-green tuples and broader 64/36-test controls; duplicate handling does not change the frozen member's technical closure.",
    ),
    "vllm-pr-44584": a(
        "accept_with_scope",
        "pass",
        "Both candidate bit-exact sliding-window tile-base tests pass on A100.",
        "The three-file change preserves tensor-descriptor and non-window paths while reducing the analytically enumerated 2D-pointer tile count.",
    ),
    "vllm-pr-44518": a(
        "accept_with_scope",
        "pass",
        "All five native packed-audio attention, weight-remap, and forward metadata tests pass.",
        "The two-file replacement also reports real L40S offline and HTTP serving validation and removes the standalone dependency from the production route.",
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
        "followup": "upstream-followup-tests.json",
        "vllm": "upstream-followup-tests-vllm.json",
        "vllm_conftest": "upstream-followup-tests-vllm-conftest.json",
        "vllm_conftest_v2": "upstream-followup-tests-vllm-conftest-v2.json",
        "focused": "upstream-focused-followups.json",
    }
    require(
        file_sha256(args.selection_lock) == EXPECTED_FILE_SHA256["selection"],
        "R18 selection file changed",
    )
    require(file_sha256(args.test_plan) == EXPECTED_FILE_SHA256["plan"], "R18 plan file changed")
    paths = {name: args.result_root / filename for name, filename in filenames.items()}
    for name, path in paths.items():
        require(file_sha256(path) == EXPECTED_FILE_SHA256[name], f"R18 {name} file changed")

    selection = read(args.selection_lock)
    plan = read(args.test_plan)
    evidence = {name: read(path) for name, path in paths.items()}
    require(
        selection["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
        "R18 selection identity changed",
    )
    require(
        selection["selection_lock_sha256"] == canonical_sha256(selection["selection_material"]),
        "R18 selection digest mismatch",
    )
    require(plan["test_plan_sha256"] == EXPECTED_TEST_PLAN_SHA256, "R18 plan identity changed")
    require(
        plan["test_plan_sha256"]
        == canonical_sha256(
            {key: value for key, value in plan.items() if key != "test_plan_sha256"}
        ),
        "R18 plan digest mismatch",
    )
    require(plan["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256, "R18 plan binding mismatch")
    require(
        evidence["manifest"]["source_bundle_sha256"] == EXPECTED_SOURCE_BUNDLE_SHA256,
        "R18 source bundle changed",
    )
    for name, payload in evidence.items():
        field = "evidence_manifest_sha256" if name == "manifest" else "evidence_sha256"
        require(
            payload[field]
            == canonical_sha256({key: value for key, value in payload.items() if key != field}),
            f"R18 {name} embedded digest mismatch",
        )
        if name != "manifest":
            require(
                payload["selection_lock_sha256"] == EXPECTED_SELECTION_SHA256,
                f"R18 {name} binding mismatch",
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
    require(all(value is False for value in hidden), "R18 blind boundary is not intact")

    initial = {record["case_id"]: record for record in evidence["initial"]["records"]}
    focused = {record["case_id"]: record for record in evidence["focused"]["records"]}
    vllm = {record["case_id"]: record for record in evidence["vllm"]["records"]}
    require(initial["sglang-pr-37643"]["returncode"] == 0, "R18 recent SGLang result changed")
    require(
        "192 passed" in evidence["followup"]["records"][0]["output_tail"],
        "R18 FlashInfer warm result changed",
    )
    require(focused["sglang-pr-27181"]["returncode"] == 0, "R18 SGLang LoRA result changed")
    require(focused["sglang-pr-27183"]["returncode"] == 0, "R18 SGLang YOCO result changed")
    require(
        focused["tensorrt_llm-pr-14765"]["returncode"] == 0, "R18 TensorRT sidecar result changed"
    )
    require(focused["vllm-pr-44450"]["returncode"] == 0, "R18 vLLM LoRA result changed")
    require(focused["vllm-pr-44518"]["returncode"] == 0, "R18 vLLM audio result changed")
    require(vllm["vllm-pr-54979"]["returncode"] == 0, "R18 recent vLLM result changed")

    selected = {case["case_id"]: case for case in selection["selection_material"]["cases"]}
    planned = {case["case_id"]: case for case in plan["cases"]}
    require(len(selected) == 30, "R18 cohort is not 30 cases")
    require(selected.keys() == planned.keys() == ASSESSMENTS.keys(), "R18 case sets differ")
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
        if assessed.decision == "check":
            require(
                selected_case["temporal_band"] == "recent", f"{case_id}: mature check forbidden"
            )
            require(
                int(selected_case["changed_files"]) <= 8, f"{case_id}: check exceeds eight files"
            )
            require(bool(initial[case_id]["test_paths"]), f"{case_id}: check lacks candidate test")
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
            "technical_findings": list(assessed.findings),
            "residual_contract": assessed.residual,
            "hot_window_check_eligible": selected_case["temporal_band"] == "recent"
            and int(selected_case["changed_files"]) <= 8
            and bool(initial[case_id]["test_paths"]),
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
        counts == {"accept_with_scope": 21, "check": 3, "reject": 6, "unresolved": 0},
        "R18 decision distribution changed",
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
        "candidate_body_integrity_note": "Bodies were acquired only after the plan lock; eight outcome-bearing blocks were redacted before storage.",
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
