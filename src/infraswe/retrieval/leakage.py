from __future__ import annotations

from collections.abc import Iterable, Sequence

from infraswe.models.retrieval import (
    LeakageAudit,
    LeakageExclusion,
    PrecedentRecord,
    QueryPlan,
)


def audit_leakage(
    records: Sequence[PrecedentRecord],
    plan: QueryPlan,
    *,
    known_solution_fingerprints: Iterable[str] = (),
    suspected_near_duplicate_ids: Iterable[str] = (),
) -> LeakageAudit:
    forbidden = set(plan.forbidden_source_ids)
    solution_fingerprints = set(known_solution_fingerprints)
    suspected = set(suspected_near_duplicate_ids)
    exclusions: list[LeakageExclusion] = []
    allowed: list[str] = []
    known_solution_leaked = False
    near_duplicate = False
    for record in records:
        reason = None
        if record.source_event_id in forbidden or record.precedent_id in forbidden:
            reason = "forbidden-source-id"
            known_solution_leaked = True
        elif record.observed_at > plan.corpus_cutoff:
            reason = "after-corpus-cutoff"
        elif (
            record.change_fingerprint is not None
            and record.change_fingerprint in solution_fingerprints
        ):
            reason = "known-solution-fingerprint"
            known_solution_leaked = True
        elif record.precedent_id in suspected:
            reason = "suspected-near-duplicate"
            near_duplicate = True
        if reason is None:
            allowed.append(record.precedent_id)
        else:
            exclusions.append(LeakageExclusion(precedent_id=record.precedent_id, reason=reason))
    status = "fail" if known_solution_leaked else "unresolved" if near_duplicate else "pass"
    failure_codes = []
    if known_solution_leaked:
        failure_codes.append("KNOWN_SOLUTION_LEAKED")
    if near_duplicate:
        failure_codes.append("SUSPECTED_NEAR_DUPLICATE_REQUIRES_HUMAN_AUDIT")
    return LeakageAudit(
        status=status,
        allowed_precedent_ids=allowed,
        exclusions=exclusions,
        known_solution_leaked=known_solution_leaked,
        suspected_near_duplicate=near_duplicate,
        failure_codes=failure_codes,
    )
