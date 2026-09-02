from __future__ import annotations

from collections.abc import Mapping, Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.retrieval import (
    CandidateFootprint,
    ChannelHit,
    FusedHit,
    GraphExpansionBudget,
    PrecedentRecord,
    QueryPass,
    QueryPlan,
    RetrievalChannel,
    RRFPolicy,
)
from infraswe.retrieval.store import PrecedentStore


def build_default_query_plan(
    *,
    target_snapshot_sha256: str,
    footprint: CandidateFootprint,
    corpus_cutoff,
    forbidden_source_ids: Sequence[str] = (),
) -> QueryPlan:
    passes = [
        QueryPass(id="exact", required=True, budget=100, features=["identity"]),
        QueryPass(id="graph", required=True, budget=100, features=["bounded-neighborhood"]),
        QueryPass(id="failure", required=True, budget=100, features=["failure-signature"]),
        QueryPass(id="lifecycle", required=True, budget=100, features=["resource-lifecycle"]),
        QueryPass(id="lexical", required=False, budget=50, features=["fts5"]),
        QueryPass(id="semantic", required=False, budget=50, features=["optional-sidecar"]),
        QueryPass(id="negative", required=True, budget=100, features=["revert-regression"]),
    ]
    return QueryPlan(
        policy_id="precedent-retrieval-v0.5.1",
        target_snapshot_sha256=target_snapshot_sha256,
        candidate_footprint_sha256=canonical_sha256(footprint),
        corpus_cutoff=corpus_cutoff,
        leakage_policy_id="historical-blind-v1",
        forbidden_source_ids=sorted(set(forbidden_source_ids)),
        passes=passes,
        rrf=RRFPolicy(
            k=60,
            channel_weights={
                "exact": 1.00,
                "graph": 0.95,
                "failure": 1.00,
                "lifecycle": 0.85,
                "lexical": 0.50,
                "semantic": 0.25,
                "negative": 1.10,
            },
        ),
        graph=GraphExpansionBudget(
            max_hops=2,
            per_node_fanout=20,
            maximum_records=200,
            edge_allowlist=[
                "TESTED_BY",
                "FAILED_BY",
                "FIXES",
                "REVERTS",
                "SUPERSEDES",
                "REGRESSES",
                "TOUCHES_LIFECYCLE",
            ],
        ),
    )


def reciprocal_rank_fusion(
    channel_results: Mapping[RetrievalChannel, Sequence[str]],
    policy: RRFPolicy,
) -> list[FusedHit]:
    scores: dict[str, float] = {}
    ranks: dict[str, dict[RetrievalChannel, int]] = {}
    for channel in sorted(channel_results):
        weight = policy.channel_weights[channel]
        for rank, precedent_id in enumerate(dict.fromkeys(channel_results[channel]), start=1):
            scores[precedent_id] = scores.get(precedent_id, 0.0) + weight / (policy.k + rank)
            ranks.setdefault(precedent_id, {})[channel] = rank
    return [
        FusedHit(precedent_id=precedent_id, score=score, channel_ranks=ranks[precedent_id])
        for precedent_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def execute_retrieval(
    store: PrecedentStore,
    footprint: CandidateFootprint,
    plan: QueryPlan,
) -> tuple[list[ChannelHit], list[FusedHit], list[PrecedentRecord]]:
    if plan.candidate_footprint_sha256 != canonical_sha256(footprint):
        raise ValueError("query plan is not bound to this candidate footprint")
    budgets = {item.id: item.budget for item in plan.passes}
    exact = store.query_exact(footprint, budget=budgets["exact"])
    channel_results: dict[RetrievalChannel, list[str]] = {
        "exact": exact,
        "graph": store.expand_graph(
            exact,
            edge_allowlist=plan.graph.edge_allowlist,
            max_hops=plan.graph.max_hops,
            per_node_fanout=plan.graph.per_node_fanout,
            maximum_records=min(plan.graph.maximum_records, budgets["graph"]),
        ),
        "failure": store.query_failures(footprint, budget=budgets["failure"]),
        "lifecycle": store.query_lifecycle(footprint, budget=budgets["lifecycle"]),
        "lexical": store.query_lexical(
            [
                *footprint.symbols,
                *footprint.config_keys,
                *footprint.failure_signatures,
                *footprint.resource_lifecycles,
            ],
            budget=budgets["lexical"],
        ),
        "semantic": [],
        "negative": store.query_negative(budget=budgets["negative"]),
    }
    hits = [
        ChannelHit(precedent_id=precedent_id, channel=channel, rank=rank)
        for channel in sorted(channel_results)
        for rank, precedent_id in enumerate(channel_results[channel], start=1)
    ]
    fused = reciprocal_rank_fusion(channel_results, plan.rrf)
    records = store.get_records(item.precedent_id for item in fused)
    return hits, fused, records
