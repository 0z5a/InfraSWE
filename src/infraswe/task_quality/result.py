from __future__ import annotations

from collections.abc import Collection, Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.task_quality import (
    INFRA_CERT_BUCKETS,
    ObligationObservation,
    TaskAcceptanceContract,
    TaskSeal,
    VerifierResult,
    VerifierResultDiagnostics,
    bucket_pass_fraction,
)
from infraswe.task_quality.seal import audit_task_seal

_ZERO_DIGEST = "sha256:" + "0" * 64


def build_verifier_result(
    *,
    seal: TaskSeal,
    contract: TaskAcceptanceContract,
    candidate_sha256: str,
    observations: Sequence[ObligationObservation],
    sealed_evidence_refs: Collection[str] | None = None,
) -> VerifierResult:
    """Validate exact obligation coverage and derive Candidate result ownership."""

    if audit_task_seal(seal):
        raise ValueError("TaskSeal digest mismatch")
    if seal.acceptance_contract_sha256 != contract.contract_sha256:
        raise ValueError("AcceptanceContract does not match TaskSeal")
    contract_material = contract.model_dump(mode="json", exclude={"contract_sha256"})
    if contract.contract_sha256 != canonical_sha256(contract_material):
        raise ValueError("AcceptanceContract digest mismatch")

    observed_list = list(observations)
    identifiers = [item.obligation_id for item in observed_list]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate obligation in VerifierResult")
    obligation_by_id = {item.obligation_id: item for item in contract.obligations}
    missing = sorted(set(obligation_by_id) - set(identifiers))
    unknown = sorted(set(identifiers) - set(obligation_by_id))
    if missing:
        raise ValueError("missing mandatory obligation(s): " + ", ".join(missing))
    if unknown:
        raise ValueError("unknown obligation(s): " + ", ".join(unknown))

    sealed_refs = set(sealed_evidence_refs) if sealed_evidence_refs is not None else None
    metadata_failures: list[str] = []
    for observation in observed_list:
        obligation = obligation_by_id[observation.obligation_id]
        if observation.bucket != obligation.bucket or observation.severity != obligation.severity:
            metadata_failures.append(observation.obligation_id)
        if sealed_refs is not None:
            missing_refs = sorted(set(observation.evidence_refs) - sealed_refs)
            if missing_refs:
                raise ValueError(
                    "evidence ref not present in sealed EvidencePack: " + ", ".join(missing_refs)
                )
    if metadata_failures:
        raise ValueError(
            "obligation metadata differs from AcceptanceContract: "
            + ", ".join(sorted(metadata_failures))
        )

    environment_observations = [
        item for item in observed_list if item.bucket == "environment-sentinel"
    ]
    if any(item.status == "BENCHMARK_DEFECT" for item in environment_observations):
        environment_status = "BENCHMARK_DEFECT"
        result_status = "BENCHMARK_DEFECT"
        infra_cert = None
    elif any(item.status in {"FAIL", "INFRA_INVALID"} for item in environment_observations):
        environment_status = "INFRA_INVALID"
        result_status = "INFRA_INVALID"
        infra_cert = None
    elif any(item.status != "PASS" for item in environment_observations):
        environment_status = "PASS"
        result_status = "UNRESOLVED"
        infra_cert = None
    else:
        environment_status = "PASS"
        hard_ids = {
            item.obligation_id
            for item in contract.obligations
            if item.severity == "hard"
            and (
                item.bucket in INFRA_CERT_BUCKETS
                or (item.bucket == "maintenance-probe" and item.release_gate)
            )
        }
        hard_observations = [item for item in observed_list if item.obligation_id in hard_ids]
        if any(item.status == "FAIL" for item in hard_observations):
            result_status = "VALID_FAIL"
            infra_cert = 0
        elif any(item.status not in {"PASS", "FAIL"} for item in hard_observations):
            result_status = "UNRESOLVED"
            infra_cert = None
        else:
            result_status = "VALID_PASS"
            infra_cert = 1

    ordered = sorted(
        observed_list,
        key=lambda item: next(
            index
            for index, obligation in enumerate(contract.obligations)
            if obligation.obligation_id == item.obligation_id
        ),
    )
    all_failures = [
        item.obligation_id
        for item in ordered
        if item.status in {"FAIL", "INFRA_INVALID", "BENCHMARK_DEFECT"}
    ]
    buckets = {item.bucket for item in contract.obligations}
    diagnostics = VerifierResultDiagnostics(
        bucket_pass_fractions={
            bucket: bucket_pass_fraction([item for item in ordered if item.bucket == bucket])
            for bucket in sorted(buckets)
        },
        first_failure=all_failures[0] if all_failures else None,
        all_failures=all_failures,
    )
    preliminary = VerifierResult(
        task_id=seal.task_id,
        task_seal_sha256=seal.task_seal_sha256,
        candidate_sha256=candidate_sha256,
        environment_status=environment_status,
        obligations=ordered,
        infra_cert=infra_cert,
        result_status=result_status,
        diagnostics=diagnostics,
        verifier_result_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"verifier_result_sha256"})
    return preliminary.model_copy(update={"verifier_result_sha256": canonical_sha256(material)})
