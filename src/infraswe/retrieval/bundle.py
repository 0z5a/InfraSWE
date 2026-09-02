from __future__ import annotations

from collections.abc import Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.retrieval import (
    CandidateFootprint,
    ChannelHit,
    ConflictSet,
    FusedHit,
    HumanRuleDecision,
    LeakageAudit,
    PrecedentRecord,
    PrecedentSet,
    QueryPlan,
    RepositorySnapshot,
    RetrievalBundle,
    RetrievalCoverage,
    RetrievalTrustCard,
    RuleCandidate,
)

_POSITIVE_KINDS = {"accepted-pattern", "migration-precedent", "explicit-contract"}
_NEGATIVE_KINDS = {
    "rejected-pattern",
    "regression-precedent",
    "superseded-precedent",
    "conflicting-precedent",
}


def detect_conflicts(records: Sequence[PrecedentRecord]) -> list[ConflictSet]:
    """Detect same-scope positive/negative history without interpreting untrusted prose."""

    groups: dict[str, list[PrecedentRecord]] = {}
    for record in records:
        scope_id = canonical_sha256(record.scope)
        groups.setdefault(scope_id, []).append(record)
    conflicts: list[ConflictSet] = []
    for scope_id, scoped in sorted(groups.items()):
        positive = [item for item in scoped if item.kind in _POSITIVE_KINDS]
        negative = [item for item in scoped if item.kind in _NEGATIVE_KINDS]
        if not positive or not negative:
            continue
        current_contract = [item for item in positive if item.kind == "explicit-contract"]
        disposition = (
            "superseded-by-current-contract" if current_contract else "human-review-required"
        )
        reason = (
            "current explicit contract and older negative history share the same typed scope"
            if current_contract
            else "positive and negative precedents share the same typed scope"
        )
        conflicts.append(
            ConflictSet(
                conflict_id=f"scope-{scope_id.removeprefix('sha256:')[:16]}",
                precedent_ids=sorted(item.precedent_id for item in [*positive, *negative]),
                disposition=disposition,
                reason=reason,
            )
        )
    return conflicts


def _footprint_anchors(footprint: CandidateFootprint) -> set[tuple[str, str]]:
    fields = {
        "file": footprint.files,
        "symbol": [*footprint.symbols, *footprint.callers, *footprint.dispatcher_points],
        "build-target": footprint.build_targets,
        "test": footprint.tests,
        "config": footprint.config_keys,
        "failure": footprint.failure_signatures,
        "lifecycle": footprint.resource_lifecycles,
    }
    return {(kind, value) for kind, values in fields.items() for value in values if value}


def _record_anchors(records: Sequence[PrecedentRecord]) -> set[tuple[str, str]]:
    anchors: set[tuple[str, str]] = set()
    for record in records:
        fields = {
            "file": record.scope.files,
            "symbol": record.scope.symbols,
            "build-target": record.scope.build_targets,
            "test": record.scope.tests,
            "config": record.scope.configs,
            "failure": record.scope.failure_signatures,
            "lifecycle": record.scope.lifecycle_tags,
        }
        anchors.update(
            (kind, value) for kind, values in fields.items() for value in values if value
        )
    return anchors


def build_retrieval_assessment(
    *,
    snapshot: RepositorySnapshot,
    footprint: CandidateFootprint,
    plan: QueryPlan,
    records: Sequence[PrecedentRecord],
    conflicts: Sequence[ConflictSet],
    leakage_audit: LeakageAudit,
) -> tuple[RetrievalCoverage, RetrievalTrustCard]:
    required_passes = sorted(item.id for item in plan.passes if item.required)
    missing_sources = sorted(snapshot.unparsed_files)
    unresolved = sorted(footprint.unresolved_surfaces)
    if leakage_audit.status == "fail":
        status = "blocked"
    elif conflicts:
        status = "conflicting"
    elif leakage_audit.status == "unresolved" or snapshot.partial or unresolved:
        status = "partial"
    else:
        status = "complete"
    coverage = RetrievalCoverage(
        status=status,
        required_passes_completed=required_passes,
        missing_sources=missing_sources,
        unresolved_surfaces=unresolved,
        known_blind_spots=(
            ["semantic-sidecar-not-required"]
            if any(item.id == "semantic" for item in plan.passes)
            else []
        ),
    )

    footprint_anchors = _footprint_anchors(footprint)
    matched_anchors = footprint_anchors & _record_anchors(records)
    anchor_coverage = len(matched_anchors) / len(footprint_anchors) if footprint_anchors else 1.0
    complete_provenance = sum(
        bool(
            record.source_locator
            and record.source_event_id
            and record.validity.repository
            and record.validity.first_revision
        )
        for record in records
    )
    provenance = complete_provenance / len(records) if records else 0.0
    trust = RetrievalTrustCard(
        snapshot_integrity="unresolved" if snapshot.partial else "pass",
        deterministic_replay="pass",
        parser_coverage=0.0 if snapshot.partial else 1.0,
        anchor_coverage=anchor_coverage,
        provenance_completeness=provenance,
        conflict_detection_status="partial" if snapshot.partial else "complete",
        leakage_audit=leakage_audit.status,
        unresolved_sources=missing_sources,
    )
    return coverage, trust


def build_retrieval_bundle(
    *,
    snapshot: RepositorySnapshot,
    footprint: CandidateFootprint,
    query_plan: QueryPlan,
    channel_hits: Sequence[ChannelHit],
    fused_ranking: Sequence[FusedHit],
    leakage_audit: LeakageAudit,
    coverage: RetrievalCoverage,
    rules: Sequence[RuleCandidate],
    trust: RetrievalTrustCard,
    precedent_set: PrecedentSet,
    human_decisions: Sequence[HumanRuleDecision] = (),
) -> RetrievalBundle:
    if query_plan.target_snapshot_sha256 != canonical_sha256(snapshot):
        raise ValueError("query plan is not bound to the retrieval snapshot")
    if query_plan.candidate_footprint_sha256 != canonical_sha256(footprint):
        raise ValueError("query plan is not bound to the retrieval footprint")
    if precedent_set.target_snapshot_sha256 != query_plan.target_snapshot_sha256:
        raise ValueError("PrecedentSet is not bound to the retrieval snapshot")
    if precedent_set.digest != canonical_sha256(
        precedent_set.model_dump(mode="json", exclude={"digest"})
    ):
        raise ValueError("PrecedentSet digest is invalid")
    preliminary = RetrievalBundle(
        snapshot=snapshot,
        footprint=footprint,
        query_plan=query_plan,
        channel_hits=list(channel_hits),
        fused_ranking=list(fused_ranking),
        leakage_audit=leakage_audit,
        coverage=coverage,
        rules=list(rules),
        human_decisions=list(human_decisions),
        trust=trust,
        precedent_set=precedent_set,
        bundle_sha256="sha256:" + "0" * 64,
    )
    material = preliminary.model_dump(mode="json", exclude={"bundle_sha256"})
    return preliminary.model_copy(update={"bundle_sha256": canonical_sha256(material)})


def audit_retrieval_bundle_digest(bundle: RetrievalBundle) -> bool:
    material = bundle.model_dump(mode="json", exclude={"bundle_sha256"})
    return bundle.bundle_sha256 == canonical_sha256(material)
