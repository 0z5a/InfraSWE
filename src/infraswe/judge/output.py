from __future__ import annotations

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.judge.pack import resolve_evidence_ref
from infraswe.judge.profile import audit_judge_cell
from infraswe.models.judge import (
    JudgeCell,
    JudgeInputPackManifest,
    JudgeOutput,
    JudgeProfile,
    JudgeRubric,
    JudgeRunRecord,
)


def validate_judge_output(
    output: JudgeOutput,
    *,
    profile: JudgeProfile,
    cell: JudgeCell,
    rubric: JudgeRubric,
    input_pack: JudgeInputPackManifest,
    member_id: str,
    repetition: int,
    decoding_seed: int,
    candidate_agent_family: str | None = None,
) -> JudgeRunRecord:
    """Ground a strict model output; failures invalidate the vote, not the candidate."""

    failures = audit_judge_cell(cell)
    if cell.profile_id != profile.profile_id:
        failures.append("JUDGE_OUTPUT_PROFILE_CELL_MISMATCH")
    if cell.profile_sha256 != canonical_sha256(profile):
        failures.append("JUDGE_OUTPUT_PROFILE_DIGEST_MISMATCH")
    if cell.rubric_sha256 != canonical_sha256(rubric):
        failures.append("JUDGE_OUTPUT_RUBRIC_CELL_MISMATCH")
    if input_pack.rubric_sha256 != cell.rubric_sha256:
        failures.append("JUDGE_OUTPUT_PACK_RUBRIC_MISMATCH")
    if output.judge_cell_sha256 != cell.judge_cell_sha256:
        failures.append("JUDGE_OUTPUT_CELL_DIGEST_MISMATCH")
    if output.input_pack_sha256 != input_pack.pack_sha256:
        failures.append("JUDGE_OUTPUT_PACK_DIGEST_MISMATCH")
    if output.rubric_sha256 != cell.rubric_sha256:
        failures.append("JUDGE_OUTPUT_RUBRIC_DIGEST_MISMATCH")
    if output.mode != profile.mode:
        failures.append("JUDGE_OUTPUT_MODE_MISMATCH")

    member = next((item for item in profile.panel if item.member_id == member_id), None)
    if member is None:
        raise ValueError(f"Judge panel member is not sealed in profile: {member_id}")
    if repetition > member.repetitions:
        failures.append("JUDGE_OUTPUT_REPETITION_OUT_OF_RANGE")

    expected = {
        item.criterion_id: item
        for item in rubric.criteria
        if item.owner_type == "semantic-judge" and item.required
    }
    optional = {
        item.criterion_id: item
        for item in rubric.criteria
        if item.owner_type == "semantic-judge" and not item.required
    }
    observed = {item.criterion_id: item for item in output.criteria}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected) - set(optional))
    failures.extend(f"JUDGE_OUTPUT_REQUIRED_CRITERION_MISSING:{item}" for item in missing)
    failures.extend(f"JUDGE_OUTPUT_CRITERION_NOT_JUDGE_OWNED:{item}" for item in extra)

    for criterion_id, result in observed.items():
        criterion = expected.get(criterion_id) or optional.get(criterion_id)
        if criterion is None:
            continue
        if result.ordinal_grade is not None:
            expected_verdict = criterion.scale.anchors[result.ordinal_grade]
            if result.verdict != expected_verdict:
                failures.append(f"JUDGE_OUTPUT_ANCHOR_MISMATCH:{criterion_id}")
        refs = [*result.evidence_refs, *result.counterevidence_refs]
        resolved = [resolve_evidence_ref(input_pack, ref) for ref in refs]
        if any(item is None for item in resolved):
            failures.append(f"JUDGE_OUTPUT_EVIDENCE_REF_UNRESOLVED:{criterion_id}")
        if result.normalized_value is not None:
            if not result.evidence_refs:
                failures.append(f"JUDGE_OUTPUT_EVIDENCE_REF_MISSING:{criterion_id}")
            grounding = [
                item
                for item in (resolve_evidence_ref(input_pack, ref) for ref in result.evidence_refs)
                if item is not None
                and item.authority in {"target-authority", "deterministic-evidence"}
            ]
            if not grounding:
                failures.append(f"JUDGE_OUTPUT_EVIDENCE_GROUNDING_FAILED:{criterion_id}")
            available_types = {artifact.evidence_type for artifact in input_pack.artifacts}
            if not set(criterion.required_evidence_types).issubset(available_types):
                failures.append(f"JUDGE_OUTPUT_REQUIRED_EVIDENCE_MISSING:{criterion_id}")
        if output.global_status.prompt_injection_suspected and result.normalized_value is not None:
            failures.append(f"JUDGE_OUTPUT_INJECTION_REQUIRES_ABSTENTION:{criterion_id}")
        if output.global_status.out_of_scope and result.normalized_value is not None:
            failures.append(f"JUDGE_OUTPUT_OUT_OF_SCOPE_REQUIRES_ABSTENTION:{criterion_id}")
        if result.security_flags and result.normalized_value is not None:
            failures.append(f"JUDGE_OUTPUT_SECURITY_FLAG_REQUIRES_ABSTENTION:{criterion_id}")

    excluded = bool(
        candidate_agent_family is not None
        and member.model.family == candidate_agent_family
        and profile.same_family_as_candidate_policy == "leave-family-out-or-zero-weight"
    )
    criterion_weights = {
        criterion_id: 0.0 if excluded else member.calibration_weights.get(criterion_id, 1.0)
        for criterion_id in expected | optional
    }
    return JudgeRunRecord(
        member_id=member.member_id,
        model_family=member.model.family,
        repetition=repetition,
        decoding_seed=decoding_seed,
        calibration_weight=0.0 if excluded else 1.0,
        criterion_calibration_weights=criterion_weights,
        candidate_family_excluded=excluded,
        validation_status="invalid" if failures else "valid",
        failure_codes=sorted(set(failures)),
        output=output,
    )
