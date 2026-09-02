from __future__ import annotations

from collections.abc import Sequence

from infraswe.models.capability import (
    CandidateCapabilityUseObservation,
    CandidateCapabilityUseVerdict,
)


def evaluate_candidate_capability_use(
    observations: Sequence[CandidateCapabilityUseObservation],
) -> CandidateCapabilityUseVerdict:
    failures: list[str] = []
    unresolved = False
    for observation in observations:
        if observation.forbidden:
            failures.append("FORBIDDEN_CAPABILITY_USED:" + observation.capability_id)
        if not observation.declared:
            failures.append("CAPABILITY_CONTRACT_VIOLATION:" + observation.capability_id)
        if observation.native_required and observation.silent_fallback:
            failures.append("MECHANISM_PROOF_FAILED:" + observation.capability_id)
        elif observation.native_required and not observation.native_proved:
            unresolved = True
    status = "VALID_FAIL" if failures else "UNRESOLVED" if unresolved else "PASS"
    return CandidateCapabilityUseVerdict(
        status=status,
        failure_codes=sorted(set(failures)),
    )
