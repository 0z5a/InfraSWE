from __future__ import annotations

from collections.abc import Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.retrieval import (
    ConflictSet,
    LeakageAudit,
    PrecedentGraphEdge,
    PrecedentRecord,
    PrecedentSet,
    QueryPlan,
    RuleCandidate,
)


def compile_rule_candidates(records: Sequence[PrecedentRecord]) -> list[RuleCandidate]:
    rules: list[RuleCandidate] = []
    for record in sorted(records, key=lambda item: item.precedent_id):
        for index, template in enumerate(record.proposed_rule_templates, start=1):
            rules.append(
                RuleCandidate(
                    rule_id=f"{record.precedent_id}-rule-{index}",
                    modality="MUST" if record.kind == "explicit-contract" else "SHOULD",
                    template=template,
                    arguments={
                        "domain_tags": record.scope.domain_tags,
                        "lifecycle_tags": record.scope.lifecycle_tags,
                    },
                    source_precedents=[record.precedent_id],
                    authority=record.authority,
                    confidence=record.confidence,
                )
            )
    return rules


def build_precedent_set(
    *,
    draft_id: str,
    draft_revision: int,
    target_snapshot_sha256: str,
    query_plan: QueryPlan,
    records: Sequence[PrecedentRecord],
    graph_edges: Sequence[PrecedentGraphEdge],
    conflicts: Sequence[ConflictSet],
    leakage_audit: LeakageAudit,
    omitted_records_path: str,
) -> PrecedentSet:
    if leakage_audit.status != "pass":
        raise ValueError("only a passed leakage audit can produce a sealable PrecedentSet")
    allowed = set(leakage_audit.allowed_precedent_ids)
    selected = sorted(
        (record for record in records if record.precedent_id in allowed),
        key=lambda item: item.precedent_id,
    )
    selected_ids = {record.precedent_id for record in selected}
    selected_edges = sorted(
        (
            edge
            for edge in graph_edges
            if edge.source_id in selected_ids and edge.target_id in selected_ids
        ),
        key=lambda item: (item.source_id, item.target_id, item.kind),
    )
    preliminary = PrecedentSet(
        draft_id=draft_id,
        draft_revision=draft_revision,
        target_snapshot_sha256=target_snapshot_sha256,
        corpus_cutoff=query_plan.corpus_cutoff,
        query_policy_id=query_plan.policy_id,
        leakage_policy_id=query_plan.leakage_policy_id,
        records=selected,
        graph_edges=selected_edges,
        conflict_sets=sorted(conflicts, key=lambda item: item.conflict_id),
        omitted_records_path=omitted_records_path,
        digest="sha256:" + "0" * 64,
    )
    material = preliminary.model_dump(mode="json")
    material.pop("digest")
    return PrecedentSet.model_validate({**material, "digest": canonical_sha256(material)})


def audit_precedent_set_digest(precedent_set: PrecedentSet) -> bool:
    material = precedent_set.model_dump(mode="json")
    digest = material.pop("digest")
    return digest == canonical_sha256(material)
