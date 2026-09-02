from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from infraswe.models.training import (
    CheckpointEvidence,
    DAPOSemanticEvidence,
    GRPOSemanticEvidence,
    MuonSemanticEvidence,
    SFTSemanticEvidence,
    TensorComparisonEvidence,
    TrainingCertification,
    TrainingEvidenceBundle,
    TrainingGate,
)

COMMON_GATE_NAMES = (
    "FORWARD_SEMANTICS",
    "BACKWARD_SEMANTICS",
    "LOSS_FINITE",
    "OPTIMIZER_UPDATE",
    "CHECKPOINT_RESUME",
    "RNG_CONTINUITY",
    "NO_SILENT_FALLBACK",
    "NO_DEADLOCK",
    "NO_RESOURCE_LEAK",
    "EVIDENCE_INTEGRITY",
    "ALGORITHM_FIDELITY",
)


def _gate(
    status: str,
    *,
    refs: Iterable[str],
    codes: Iterable[str] = (),
    reason: str | None = None,
) -> TrainingGate:
    return TrainingGate(
        status=status,
        failure_codes=sorted(set(codes)),
        evidence_refs=sorted(set(refs)),
        reason=reason,
    )


def _merge_gates(*gates: TrainingGate) -> TrainingGate:
    statuses = {gate.status for gate in gates}
    status = (
        "fail" if "fail" in statuses else ("unresolved" if "unresolved" in statuses else "pass")
    )
    reasons = sorted({gate.reason for gate in gates if gate.reason})
    return _gate(
        status,
        refs=(ref for gate in gates for ref in gate.evidence_refs),
        codes=(code for gate in gates for code in gate.failure_codes),
        reason="; ".join(reasons) if reasons else None,
    )


def _comparison_gate(
    evidence: TensorComparisonEvidence | None,
    *,
    failure_code: str,
    evidence_ref: str,
) -> TrainingGate:
    if evidence is None:
        return _gate(
            "unresolved",
            refs=(evidence_ref,),
            reason=f"missing {failure_code.lower()} comparison evidence",
        )
    if len(evidence.reference) != len(evidence.candidate):
        return _gate("fail", refs=(evidence_ref,), codes=(failure_code,))
    finite = all(math.isfinite(value) for value in (*evidence.reference, *evidence.candidate))
    close = finite and all(
        math.isclose(candidate, reference, rel_tol=evidence.rtol, abs_tol=evidence.atol)
        for reference, candidate in zip(evidence.reference, evidence.candidate, strict=True)
    )
    return _gate(
        "pass" if close else "fail",
        refs=(evidence_ref,),
        codes=() if close else (failure_code,),
    )


def _verify_sft(evidence: SFTSemanticEvidence | None, ref: str) -> TrainingGate:
    if evidence is None:
        return _gate("unresolved", refs=(ref,), reason="SFT semantic evidence is missing")
    failures: list[str] = []
    valid_count = sum(evidence.target_mask)
    finite_losses = all(math.isfinite(loss) for loss in evidence.token_losses)
    if valid_count == 0 or not finite_losses:
        failures.append("LOSS_MASK_MISMATCH")
    else:
        expected_loss = (
            sum(
                loss
                for loss, selected in zip(evidence.token_losses, evidence.target_mask, strict=True)
                if selected
            )
            / valid_count
        )
        if evidence.observed_denominator != valid_count or not math.isclose(
            evidence.observed_loss, expected_loss, rel_tol=1e-9, abs_tol=1e-12
        ):
            failures.append("LOSS_MASK_MISMATCH")
    token_count = len(evidence.packed_sample_ids)
    for source, target in evidence.observed_attention_edges:
        if not (0 <= source < token_count and 0 <= target < token_count):
            failures.append("SFT_ATTENTION_EDGE_INVALID")
        elif evidence.packed_sample_ids[source] != evidence.packed_sample_ids[target]:
            failures.append("PACK_BOUNDARY_LEAK")
    return _gate(
        "fail" if failures else "pass",
        refs=(ref,),
        codes=failures,
    )


def _verify_grpo(evidence: GRPOSemanticEvidence | None, ref: str) -> TrainingGate:
    if evidence is None:
        return _gate("unresolved", refs=(ref,), reason="GRPO rollout evidence is missing")
    failures: list[str] = []
    samples_by_group: dict[tuple[str, str], list] = defaultdict(list)
    prompts_by_group: dict[str, set[str]] = defaultdict(set)
    sample_ids: set[str] = set()
    for sample in evidence.samples:
        samples_by_group[(sample.prompt_id, sample.group_id)].append(sample)
        prompts_by_group[sample.group_id].add(sample.prompt_id)
        if sample.sample_id in sample_ids:
            failures.append("GRPO_GROUP_MISMATCH")
        sample_ids.add(sample.sample_id)
        staleness = sample.train_policy_version - sample.policy_version
        if staleness < 0 or staleness > evidence.max_policy_staleness:
            failures.append("POLICY_VERSION_STALE")
        if not all(math.isfinite(value) for value in (*sample.old_log_probs, sample.reward)):
            failures.append("GRPO_GROUP_MISMATCH")
    if any(len(prompts) != 1 for prompts in prompts_by_group.values()):
        failures.append("GRPO_GROUP_MISMATCH")
    for samples in samples_by_group.values():
        if len(samples) != evidence.expected_group_size:
            failures.append("GRPO_GROUP_MISMATCH")
            continue
        rewards = [sample.reward for sample in samples]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        stddev = math.sqrt(variance)
        expected = (
            [0.0] * len(rewards)
            if stddev <= evidence.advantage_epsilon
            else [(reward - mean) / stddev for reward in rewards]
        )
        for sample, expected_advantage in zip(samples, expected, strict=True):
            if not math.isclose(
                sample.observed_advantage,
                expected_advantage,
                rel_tol=evidence.advantage_tolerance,
                abs_tol=evidence.advantage_tolerance,
            ):
                failures.append("GRPO_GROUP_MISMATCH")
    return _gate(
        "fail" if failures else "pass",
        refs=(ref,),
        codes=failures,
    )


def _verify_dapo(
    evidence: DAPOSemanticEvidence | None,
    *,
    certification_scope: str,
    ref: str,
) -> TrainingGate:
    if evidence is None:
        return _gate("unresolved", refs=(ref,), reason="DAPO component evidence is missing")
    required = {
        "token_level_policy_gradient": evidence.token_level_policy_gradient,
        "asymmetric_clip_higher": evidence.asymmetric_clip_higher,
        "overlong_policy_exact": evidence.overlong_policy_exact,
        "soft_overlong_punishment_exact": evidence.soft_overlong_punishment_exact,
        "reward_aggregation_exact": evidence.reward_aggregation_exact,
    }
    if certification_scope in {"dapo-recipe-contract", "dapo-online"}:
        required["dynamic_sampling"] = evidence.dynamic_sampling
    missing = sorted(name for name, present in required.items() if not present)
    failures = ["DAPO_COMPONENT_MISSING"] if missing else []
    if "dynamic_sampling" in missing:
        failures.append("DYNAMIC_SAMPLING_MISMATCH")
    if any(name.startswith("overlong") or name.startswith("soft_overlong") for name in missing):
        failures.append("OVERLONG_POLICY_MISMATCH")
    return _gate(
        "fail" if failures else "pass",
        refs=(ref,),
        codes=failures,
        reason="missing DAPO components: " + ", ".join(missing) if missing else None,
    )


def _verify_muon(evidence: MuonSemanticEvidence | None, ref: str) -> TrainingGate:
    if evidence is None:
        return _gate("unresolved", refs=(ref,), reason="Muon optimizer evidence is missing")
    failures: list[str] = []
    trainable = set(evidence.trainable_parameters)
    grouped_names = [record.name for record in evidence.parameter_groups]
    if len(grouped_names) != len(set(grouped_names)) or set(grouped_names) != trainable:
        failures.append("OPTIMIZER_GROUP_MISMATCH")
    for record in evidence.parameter_groups:
        should_use_muon = record.semantic_role == "hidden-matrix" and len(record.shape) == 2
        expected_optimizer = "muon" if should_use_muon else "adamw"
        if record.optimizer != expected_optimizer or record.update_count != 1:
            failures.append("OPTIMIZER_GROUP_MISMATCH")
    update = _comparison_gate(
        evidence.update_comparison,
        failure_code="MUON_ORTHOGONALIZATION_MISMATCH",
        evidence_ref=ref,
    )
    grouping = _gate(
        "fail" if failures else "pass",
        refs=(ref,),
        codes=failures,
    )
    return _merge_gates(grouping, update)


def _verify_checkpoint(
    evidence: CheckpointEvidence | None,
    *,
    online_algorithm: bool,
    ref: str,
) -> tuple[TrainingGate, TrainingGate]:
    if evidence is None:
        unresolved = _gate(
            "unresolved", refs=(ref,), reason="checkpoint/resume evidence is missing"
        )
        return unresolved, unresolved
    required = {"weights", "optimizer", "scheduler", "data_cursor"}
    required_rng = {"data_rng", "dropout_rng"}
    if online_algorithm:
        required.update({"rollout_cursor", "policy_version"})
        required_rng.add("sampling_rng")
    saved = set(evidence.saved_components)
    restored = set(evidence.restored_components)
    missing_state = sorted(required - saved | required - restored)
    missing_rng = sorted(required_rng - set(evidence.rng_streams_restored))
    comparison = _comparison_gate(
        evidence.next_step_comparison,
        failure_code="RESUME_DIVERGENCE",
        evidence_ref=ref,
    )
    checkpoint_failures = []
    if missing_state or not evidence.fresh_process:
        checkpoint_failures.append("CHECKPOINT_INCOMPLETE")
    checkpoint = _merge_gates(
        _gate(
            "fail" if checkpoint_failures else "pass",
            refs=(ref,),
            codes=checkpoint_failures,
            reason=("missing checkpoint state: " + ", ".join(missing_state))
            if missing_state
            else None,
        ),
        comparison,
    )
    rng = _gate(
        "fail" if missing_rng else "pass",
        refs=(ref,),
        codes=("RESUME_DIVERGENCE",) if missing_rng else (),
        reason="missing RNG streams: " + ", ".join(missing_rng) if missing_rng else None,
    )
    return checkpoint, rng


def verify_training_evidence(bundle: TrainingEvidenceBundle) -> TrainingCertification:
    """Evaluate training hard gates without issuing any performance score.

    Missing artifacts become ``unresolved``. Numeric zero is reserved for measured values;
    it is never used as a substitute for absent evidence.
    """

    ref = bundle.evidence_manifest_sha256
    gates: dict[str, TrainingGate] = {
        "FORWARD_SEMANTICS": _comparison_gate(
            bundle.forward,
            failure_code="TRAIN_FORWARD_MISMATCH",
            evidence_ref=ref,
        ),
        "BACKWARD_SEMANTICS": _comparison_gate(
            bundle.backward,
            failure_code="TRAIN_BACKWARD_MISMATCH",
            evidence_ref=ref,
        ),
        "OPTIMIZER_UPDATE": _comparison_gate(
            bundle.optimizer_update,
            failure_code="OPTIMIZER_STATE_MISMATCH",
            evidence_ref=ref,
        ),
    }
    if bundle.runtime is None:
        for name in ("LOSS_FINITE", "NO_SILENT_FALLBACK", "NO_DEADLOCK", "NO_RESOURCE_LEAK"):
            gates[name] = _gate(
                "unresolved", refs=(ref,), reason="runtime safety evidence is missing"
            )
    else:
        losses_finite = all(math.isfinite(loss) for loss in bundle.runtime.loss_values)
        gates["LOSS_FINITE"] = _gate(
            "pass" if losses_finite else "fail",
            refs=(ref,),
            codes=() if losses_finite else ("LOSS_NONFINITE",),
        )
        no_silent_fallback = bundle.runtime.silent_fallback_count == 0
        gates["NO_SILENT_FALLBACK"] = _gate(
            "pass" if no_silent_fallback else "fail",
            refs=(ref,),
            codes=() if no_silent_fallback else ("SILENT_FRAMEWORK_FALLBACK",),
        )
        live = (
            not bundle.runtime.deadlock
            and bundle.runtime.watchdog_passed
            and bundle.runtime.half_batch_updates == 0
        )
        liveness_codes = []
        if bundle.runtime.deadlock or not bundle.runtime.watchdog_passed:
            liveness_codes.append("COLLECTIVE_DEADLOCK")
        if bundle.runtime.half_batch_updates:
            liveness_codes.append("TRAIN_HALF_BATCH_UPDATE")
        gates["NO_DEADLOCK"] = _gate(
            "pass" if live else "fail",
            refs=(ref,),
            codes=liveness_codes,
        )
        no_leak = not bundle.runtime.resource_leaks
        gates["NO_RESOURCE_LEAK"] = _gate(
            "pass" if no_leak else "fail",
            refs=(ref,),
            codes=() if no_leak else ("ROLLOUT_WORKER_LEAK",),
            reason=("resource leaks: " + ", ".join(bundle.runtime.resource_leaks))
            if bundle.runtime.resource_leaks
            else None,
        )

    checkpoint, rng = _verify_checkpoint(
        bundle.checkpoint,
        online_algorithm=bundle.certification_scope in {"grpo-online", "dapo-online"},
        ref=ref,
    )
    gates["CHECKPOINT_RESUME"] = checkpoint
    gates["RNG_CONTINUITY"] = rng

    if bundle.integrity is None:
        gates["EVIDENCE_INTEGRITY"] = _gate(
            "unresolved", refs=(ref,), reason="evidence integrity record is missing"
        )
    else:
        integrity_ok = (
            bundle.integrity.manifest_sha256 == ref
            and bundle.integrity.timeline_consistent
            and bundle.integrity.versions_exact
        )
        gates["EVIDENCE_INTEGRITY"] = _gate(
            "pass" if integrity_ok else "fail",
            refs=(ref, *bundle.integrity.raw_evidence_digests),
            codes=() if integrity_ok else ("EVIDENCE_INTEGRITY_MISMATCH",),
        )

    if bundle.algorithm == "sft":
        algorithm = _verify_sft(bundle.sft, ref)
    elif bundle.algorithm == "grpo":
        algorithm = _verify_grpo(bundle.grpo, ref)
    elif bundle.algorithm == "dapo":
        algorithm = _merge_gates(
            _verify_grpo(bundle.grpo, ref),
            _verify_dapo(bundle.dapo, certification_scope=bundle.certification_scope, ref=ref),
        )
    else:
        algorithm = _gate(
            "unresolved",
            refs=(ref,),
            reason=f"no core semantic verifier is registered for {bundle.algorithm}",
        )
    if bundle.optimizer == "muon-plus-adamw":
        gates["OPTIMIZER_UPDATE"] = _merge_gates(
            gates["OPTIMIZER_UPDATE"], _verify_muon(bundle.muon, ref)
        )
    gates["ALGORITHM_FIDELITY"] = algorithm

    if set(gates) != set(COMMON_GATE_NAMES):
        raise RuntimeError("internal error: training gate set drifted from the frozen contract")
    statuses = {gate.status for gate in gates.values()}
    status = (
        "fail" if "fail" in statuses else ("unresolved" if "unresolved" in statuses else "pass")
    )
    failures = sorted({code for gate in gates.values() for code in gate.failure_codes})
    return TrainingCertification(
        status=status,
        algorithm=bundle.algorithm,
        certification_scope=bundle.certification_scope,
        gates=gates,
        failure_codes=failures,
        evidence_manifest_sha256=ref,
    )
