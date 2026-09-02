#!/usr/bin/env python3
"""Freeze twenty case-specific R11 contracts before source diff inspection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import HistoricalPRCandidate

CASE_PLANS: dict[str, dict[str, Any]] = {
    "cutlass-pr-3352": {
        "claim": "CuTe DSL c_pointer lifecycle does not retain dead Python owners.",
        "execution_tier": "exact base/head ownership and repeated-GC isolation probe",
        "questions": [
            "Does releasing the final public c_pointer wrapper release its owner after GC?",
            "Does repeated construction avoid monotonically retaining owner or wrapper objects?",
            "Are live-pointer type, value, and representation semantics unchanged?",
            "Does a direct test exercise both collection and live-object controls?",
        ],
        "decision_rule": (
            "Accept only if head removes a base retention signal while preserving live-pointer "
            "semantics. A bounded remaining ownership edge with a closure probe is revise; an "
            "unbounded cycle, semantic no-op, or new lifetime hazard is reject."
        ),
    },
    "cutlass-pr-3380": {
        "claim": (
            "i8-to-bf16 conversion and s2t copy execute only on architectures/CTAs that "
            "support them."
        ),
        "execution_tier": "exact predicate truth table plus CUDA source/compile contract",
        "questions": [
            "Are Ampere and Ada excluded while Hopper and newer remain enabled?",
            "Is the s2t operation guarded to the leader CTA at its exact side-effect site?",
            "Are unrelated conversion paths and documentation claims coherent with the gate?",
            "Do boundary controls cover sm80, sm89, sm90, sm100 and leader/nonleader CTAs?",
        ],
        "decision_rule": (
            "Accept only if both gates satisfy the frozen truth tables without collateral "
            "restriction. One demonstrably local missing gate is revise; architecture-wide "
            "misclassification or unsafe duplicate copy is reject."
        ),
    },
    "deepgemm-pr-327": {
        "claim": (
            "the clean-logits kernel compiles under CUDA 12.8 NVRTC without changing "
            "negative-infinity behavior."
        ),
        "execution_tier": "exact base/head CUDA 12.8 compilation and constant-semantics probe",
        "questions": [
            "Does the exact affected source fail in base and compile in head under CUDA 12.8?",
            "Does the replacement constant retain IEEE negative-infinity semantics?",
            "Are finite and NaN neighboring controls unchanged?",
            "Is the compatibility fix covered by a direct compile or regression test?",
        ],
        "decision_rule": (
            "Accept if head fixes the exact compile failure and preserves numeric semantics. A "
            "working local fix missing only direct coverage is revise; a no-op or finite-value "
            "substitution is reject."
        ),
    },
    "deepgemm-pr-337": {
        "claim": "UE8M0 packing extracts FP32 exponent bits without mantissa contamination.",
        "execution_tier": "exact expression plus exhaustive bit-pattern boundary probe",
        "questions": [
            "Do equal exponents with different mantissas pack to the same UE8M0 value?",
            "Are exponent boundaries, zero, subnormal, infinity, and NaN handled explicitly?",
            "Does head match an independent bit-mask oracle for every frozen pattern?",
            "Does direct coverage include nonzero mantissas that distinguish base from head?",
        ],
        "decision_rule": (
            "Accept only if head removes mantissa contamination across the full matrix and has "
            "direct distinguishing coverage. A correct bounded implementation needing one edge "
            "policy is revise; a semantic no-op or broad mismatch is reject."
        ),
    },
    "flashattention-pr-2662": {
        "claim": (
            "the SM100 LPT scheduler computes long-context tile indices without int32 overflow."
        ),
        "execution_tier": "exact scheduler arithmetic isolation against an integer oracle",
        "questions": [
            "Do values around 2^31 avoid wraparound and match an unbounded-integer reference?",
            "Are ordinary short-context schedules unchanged?",
            "Are resulting tile indices monotonic, in range, and collision-free where required?",
            "Does direct coverage include a value that overflows the base implementation?",
        ],
        "decision_rule": (
            "Accept if long-context arithmetic matches the oracle and short cases are preserved. "
            "A single remaining cast with a direct closure case is revise; systemic overflow or "
            "new index corruption is reject."
        ),
    },
    "flashattention-pr-2678": {
        "claim": "is_fake_mode remains correct and traceable under torch.compile.",
        "execution_tier": "exact eager, FakeTensor, and torch.compile execution matrix",
        "questions": [
            "Can torch.compile trace the helper without an active_fake_mode graph break?",
            "Does eager execution still distinguish normal and fake tensors correctly?",
            "Do fullgraph and repeated compiled calls remain stable?",
            "Does direct coverage include compiled and eager neighboring controls?",
        ],
        "decision_rule": (
            "Accept if head fixes the compiled path and preserves fake/eager semantics. A local "
            "traceability gap with a closure test is revise; silently returning a wrong mode or "
            "requiring a cross-cutting compiler workaround is reject."
        ),
    },
    "flashinfer-pr-3930": {
        "claim": (
            "find_loaded_library never selects look-alike CUDA libraries such as libcudart_stub."
        ),
        "execution_tier": "exact library-name matcher over path-order and version variants",
        "questions": [
            "Are canonical unversioned and versioned library names accepted?",
            "Are prefix/suffix look-alikes rejected regardless of enumeration order?",
            "Is behavior consistent for Linux shared objects and supported platform forms?",
            "Does direct coverage include a stub appearing before the real library?",
        ],
        "decision_rule": (
            "Accept if exact head matching chooses only the intended library across all orderings. "
            "A bounded missing supported suffix is revise; continued look-alike selection or "
            "nondeterministic binding is reject."
        ),
    },
    "flashinfer-pr-3990": {
        "claim": (
            "B300 MNNVL all-up detection reports true only for complete usable peer connectivity."
        ),
        "execution_tier": "exact topology predicate with mocked complete and degraded matrices",
        "questions": [
            "Do complete supported B300 topologies report all-up?",
            "Does any missing, asymmetric, or unusable peer edge force false?",
            "Are diagonal/self entries and device-count boundaries handled correctly?",
            "Do non-B300 or topology-query failures retain a safe behavior?",
        ],
        "decision_rule": (
            "Accept only if complete/degraded truth tables are exact and failures are safe. One "
            "bounded topology shape omission is revise; a false-positive all-up result is reject "
            "because it can select an unsafe transport path."
        ),
    },
    "liger-pr-1251": {
        "claim": "liger_cross_entropy with inplace=False preserves logits and upstream gradients.",
        "execution_tier": "exact branched-autograd comparison against PyTorch reference",
        "questions": [
            "Are input logits byte-identical after forward when inplace=False?",
            "Does a branch consuming the same logits receive the correct upstream gradient?",
            "Are loss and primary gradients equal to the reference across reductions?",
            "Are explicit inplace=True and default compatibility semantics unchanged?",
        ],
        "decision_rule": (
            "Accept if head fixes the corruption and matches the full reference matrix. A local "
            "dtype/reduction omission with a direct closure test is revise; continued silent "
            "mutation or gradient corruption is reject."
        ),
    },
    "liger-pr-1283": {
        "claim": "FLCE grad_weight accumulation aligns input-chunk dtype under AMP.",
        "execution_tier": "exact mixed-dtype accumulation matrix against FP32 reference",
        "questions": [
            "Does accumulation avoid addmm dtype errors for frozen AMP dtype pairs?",
            "Is the cast applied to the input chunk rather than silently narrowing grad_weight?",
            "Are accumulated values within dtype-appropriate tolerance?",
            "Do same-dtype and repeated-accumulation controls remain unchanged?",
        ],
        "decision_rule": (
            "Accept if head fixes all mixed-dtype cases without precision-policy regression. A "
            "single missing AMP pairing is revise; silent accumulator downcast or broad numeric "
            "divergence is reject."
        ),
    },
    "megatron-pr-5726": {
        "claim": (
            "multimodal add_document derives default modes from the number of lengths entries."
        ),
        "execution_tier": "exact builder call matrix with mocked document metadata",
        "questions": [
            "When modes are omitted, is one default mode produced per lengths entry?",
            "Are one, multiple, and empty/malformed length collections handled deterministically?",
            "Are explicit valid modes preserved and mismatches rejected or documented?",
            "Does direct coverage distinguish document count from lengths count?",
        ],
        "decision_rule": (
            "Accept if default cardinality follows lengths and explicit behavior is preserved. A "
            "bounded malformed-input policy gap is revise; continued metadata misalignment or "
            "silent truncation is reject."
        ),
    },
    "megatron-pr-5759": {
        "claim": (
            "remove_sharded_tensors remains a working public API across checkpoint strategies."
        ),
        "execution_tier": "exact import/signature and nested checkpoint filtering probe",
        "questions": [
            "Can callers import and invoke the documented public function?",
            "Are selected sharded tensors removed while unrelated nested values are retained?",
            "Do affected torch checkpoint strategy callers use the restored contract?",
            "Is the re-enabled direct unit test behaviorally distinguishing?",
        ],
        "decision_rule": (
            "Accept if API, filtering semantics, and affected callers all work. A local "
            "unsupported container with a closure test is revise; a broken public import or "
            "destructive filtering of unrelated state is reject."
        ),
    },
    "sglang-pr-31339": {
        "claim": "ReqTimeStats survives scheduler IPC serialization without field loss.",
        "execution_tier": "exact encode/decode round-trip over populated and default records",
        "questions": [
            "Are all populated timing fields preserved exactly across IPC round-trip?",
            "Are None/default and backward-compatible payloads handled deterministically?",
            "Do repeated and collection round-trips retain record identity semantics?",
            "Does direct coverage fail on the base field omission?",
        ],
        "decision_rule": (
            "Accept if head preserves every field and compatibility control. A bounded optional "
            "field default gap is revise; continued silent metric loss or payload incompatibility "
            "is reject."
        ),
    },
    "sglang-pr-31351": {
        "claim": "DeepSeek streaming detectors never leak partial bot-token bytes.",
        "execution_tier": "exact byte/chunk boundary matrix for both changed detectors",
        "questions": [
            "Do splits at every byte of the bot token emit no partial marker text?",
            "Does ordinary UTF-8 content reconstruct exactly once around marker boundaries?",
            "Are complete markers detected with unchanged state transitions?",
            "Do direct tests cover both detector variants, Unicode, and incomplete streams?",
        ],
        "decision_rule": (
            "Accept only if both detectors satisfy exhaustive frozen chunk splits. One bounded "
            "detector omission with the same local remedy is revise; unretractable leakage across "
            "multiple boundary families or corrupted user text is reject."
        ),
    },
    "torchtitan-pr-3861": {
        "claim": "RL checkpoint loading preserves buffer dtype instead of silently downcasting it.",
        "execution_tier": "exact state-loading probe over parameter and buffer dtype pairs",
        "questions": [
            "Do FP32/FP64/integer buffers retain checkpoint dtype through the affected path?",
            "Are parameter casting semantics unchanged?",
            "Do nested modules and persistent/nonpersistent buffer controls behave correctly?",
            "Does direct coverage distinguish a buffer from a parameter?",
        ],
        "decision_rule": (
            "Accept if every frozen buffer dtype is preserved without changing parameter policy. "
            "A bounded buffer class omission is revise; continued silent downcast or checkpoint "
            "value corruption is reject."
        ),
    },
    "torchtitan-pr-3869": {
        "claim": "AsyncTP is enabled exactly when its runtime prerequisites hold.",
        "execution_tier": "exact configuration predicate truth table with mocked capabilities",
        "questions": [
            "Does the intended valid tensor-parallel configuration enable AsyncTP?",
            "Do degree-one, unsupported backend, or missing compile prerequisites remain disabled?",
            "Are explicit user disable/enable controls respected with actionable errors?",
            (
                "Does direct coverage include the previously misclassified valid case and invalid "
                "neighbors?"
            ),
        ],
        "decision_rule": (
            "Accept if the enablement truth table is exact and fails safely. A single local "
            "predicate omission is revise; enabling on an unsupported runtime or disabling broad "
            "valid classes is reject."
        ),
    },
    "verl-pr-7010": {
        "claim": "waiting for rollout capacity does not hold the fully-async state lock.",
        "execution_tier": "exact bounded concurrency schedule with progress and state invariants",
        "questions": [
            "Can an independent state operation acquire the lock while capacity wait is blocked?",
            "Does the waiter resume exactly once after capacity becomes available?",
            "Are counters, wakeups, and cancellation paths free of lost updates?",
            "Does direct CPU coverage deterministically distinguish base from head?",
        ],
        "decision_rule": (
            "Accept if head restores concurrent progress and preserves state invariants. A local "
            "cancellation/wakeup omission with a deterministic closure test is revise; deadlock, "
            "lost capacity, or a cross-cutting lock redesign is reject."
        ),
    },
    "verl-pr-7046": {
        "claim": "BaseTool initialization resolves fallbacks without circular recursion.",
        "execution_tier": (
            "exact constructor matrix over base, subclass, missing, and explicit values"
        ),
        "questions": [
            "Do missing optional values terminate without recursive fallback?",
            "Are subclass defaults and explicit caller values resolved with stable precedence?",
            "Do invalid required values fail once with an actionable exception?",
            "Does direct coverage include the cycle-triggering base case and valid controls?",
        ],
        "decision_rule": (
            "Accept if all constructors terminate and precedence is preserved. A bounded missing "
            "default case is revise; recursion, silent invalid configuration, or broken subclass "
            "initialization is reject."
        ),
    },
    "vllm-pr-48754": {
        "claim": (
            "local speculators containing dots in their names are not misclassified as custom "
            "classes."
        ),
        "execution_tier": "exact speculative-config classifier truth table",
        "questions": [
            "Are existing local paths and registered local names with dots classified as local?",
            "Are genuine module.Class references still classified as custom classes?",
            "Are built-in and nonexistent-name error paths unchanged?",
            "Does direct coverage include ambiguous dotted controls rather than one literal?",
        ],
        "decision_rule": (
            "Accept if the full classifier truth table is exact. A bounded path-form omission is "
            "revise; misrouting genuine classes or broad local names is reject."
        ),
    },
    "vllm-pr-48755": {
        "claim": (
            "InternLM2 streaming deltas derive from raw input and reconstruct tool arguments "
            "exactly once."
        ),
        "execution_tier": "exact parser execution across exhaustive frozen JSON chunk boundaries",
        "questions": [
            "Do emitted argument deltas concatenate to the final raw argument JSON exactly once?",
            "Are splits inside keys, escapes, Unicode, values, and delimiters correct?",
            "Are multiple calls, incomplete input, and ordinary text handled deterministically?",
            "Do direct tests distinguish raw input offsets from re-serialized JSON offsets?",
        ],
        "decision_rule": (
            "Accept only if all frozen chunk families reconstruct exactly. One bounded parser "
            "state with a local remedy is revise; unretractable corruption across multiple "
            "boundary families or a required parser redesign is reject."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    material = selection["selection_material"]
    if selection["selection_lock_sha256"] != canonical_sha256(material):
        raise SystemExit("R11 selection lock digest mismatch")
    if material["review_text_visible_to_machine_judge"] is not False:
        raise SystemExit("R11 selection exposes review text")
    if material["merge_outcomes_visible_to_machine_judge"] is not False:
        raise SystemExit("R11 selection exposes outcomes")
    cases = [HistoricalPRCandidate.model_validate(item) for item in material["cases"]]
    if {item.case_id for item in cases} != set(CASE_PLANS):
        raise SystemExit("R11 selection and plan case sets differ")

    plan_material = {
        "schema_version": "0.1",
        "protocol_id": material["protocol_id"],
        "selection_lock_sha256": selection["selection_lock_sha256"],
        "machine_policy_id": material["machine_policy_id"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "review_text_requested": False,
        "frozen_before_source_diff_content_inspection": True,
        "scoring_policy": {
            "kind": "ordered exact-contract judgment with repairability triage",
            "weighted_score_used": False,
            "forced_polarization_used": False,
            "decisions": ["accept_with_scope", "revise", "reject", "unresolved"],
            "revise_is_bounded_repairability_claim": True,
            "missing_environment_evidence": "unresolved, never candidate fail",
        },
        "cases": [
            {
                "case_id": item.case_id,
                "project": item.project,
                "repository": item.repository,
                "pull_number": item.pull_number,
                "base_sha": item.base_sha,
                "head_sha": item.head_sha,
                "changed_paths": item.paths,
                **CASE_PLANS[item.case_id],
            }
            for item in cases
        ],
    }
    payload = {**plan_material, "test_plan_sha256": canonical_sha256(plan_material)}
    atomic_write_json(args.output, payload)
    print(f"case_count={len(cases)}")
    print(f"test_plan_sha256={payload['test_plan_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
