from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    TopologyContract,
    TopologyGraph,
    TopologyMatchResult,
    TopologyRelationRequirement,
    TopologyVertex,
)


def audit_topology_graph(graph: TopologyGraph) -> list[str]:
    material = graph.model_dump(mode="json", exclude={"graph_sha256"})
    return (
        []
        if graph.graph_sha256 == canonical_sha256(material)
        else ["TOPOLOGY_GRAPH_DIGEST_MISMATCH"]
    )


def _vertices_for_role(vertices: list[TopologyVertex], role: str) -> list[TopologyVertex]:
    return [item for item in vertices if item.role == role]


def _edge_pairs(graph: TopologyGraph, kinds: set[str]) -> set[frozenset[str]]:
    return {
        frozenset((item.source, item.target))
        for item in graph.edges
        if not kinds or item.kind in kinds
    }


def _same_attribute(vertices: list[TopologyVertex], attribute: str) -> bool:
    values = [item.attributes.get(attribute) for item in vertices]
    return bool(values) and None not in values and len(set(values)) == 1


def _match_relation(
    relation: TopologyRelationRequirement,
    graph: TopologyGraph,
) -> bool | None:
    sources = _vertices_for_role(graph.vertices, relation.source_role)
    targets = (
        _vertices_for_role(graph.vertices, relation.target_role)
        if relation.target_role
        else sources
    )
    if not sources or not targets:
        return False
    if relation.pattern == "same-node":
        return _same_attribute([*sources, *targets], "host_id")
    if relation.pattern == "same-socket":
        return _same_attribute([*sources, *targets], "socket_id")
    if relation.pattern == "same-numa":
        return _same_attribute([*sources, *targets], "numa_node")
    if relation.pattern == "same-root-complex":
        return _same_attribute([*sources, *targets], "root_complex")
    if relation.pattern == "cross-node":
        values = {item.attributes.get("host_id") for item in [*sources, *targets]}
        return None not in values and len(values) > 1
    if relation.pattern == "anti-affinity":
        values = [item.attributes.get("host_id") for item in sources]
        return None not in values and len(values) == len(set(values))

    pairs = _edge_pairs(graph, set(relation.via_edge_kinds))
    if relation.pattern in {"all-pairs", "mesh"}:
        vertices = sources if targets is sources else [*sources, *targets]
        return all(
            frozenset((left.vertex_id, right.vertex_id)) in pairs
            for left, right in combinations(vertices, 2)
        )
    if relation.pattern == "ring":
        if len(sources) < 2:
            return False
        degree: defaultdict[str, int] = defaultdict(int)
        source_ids = {item.vertex_id for item in sources}
        for pair in pairs:
            if len(pair) == 2 and pair <= source_ids:
                for vertex_id in pair:
                    degree[vertex_id] += 1
        required_degree = 1 if len(sources) == 2 else 2
        return all(degree[item.vertex_id] >= required_degree for item in sources)
    if relation.pattern == "one-nic-per-k-gpu":
        if relation.k is None:
            return None
        source_ids = {item.vertex_id for item in sources}
        target_ids = {item.vertex_id for item in targets}
        connected_sources = {
            next(iter(pair & source_ids))
            for pair in pairs
            if pair & source_ids and pair & target_ids
        }
        minimum_nics = (len(sources) + relation.k - 1) // relation.k
        connected_targets = {
            next(iter(pair & target_ids))
            for pair in pairs
            if pair & source_ids and pair & target_ids
        }
        return len(connected_sources) == len(sources) and len(connected_targets) >= minimum_nics
    return None


def match_topology(
    contract: TopologyContract,
    graph: TopologyGraph,
) -> TopologyMatchResult:
    if audit_topology_graph(graph):
        return TopologyMatchResult(
            status="probe-defect",
            graph_sha256=graph.graph_sha256,
            failure_codes=["TOPOLOGY_GRAPH_DIGEST_MISMATCH"],
        )
    material = contract.model_dump(mode="json", exclude={"contract_sha256"})
    if contract.contract_sha256 != canonical_sha256(material):
        return TopologyMatchResult(
            status="probe-defect",
            graph_sha256=graph.graph_sha256,
            failure_codes=["TOPOLOGY_CONTRACT_DIGEST_MISMATCH"],
        )
    failures: list[str] = []
    unresolved: list[str] = []
    for requirement in contract.vertices:
        matches = [
            item
            for item in graph.vertices
            if item.role == requirement.role
            and item.kind == requirement.kind
            and all(
                item.attributes.get(name) == value
                for name, value in requirement.attribute_equals.items()
            )
        ]
        if len(matches) != requirement.count:
            failures.append("TOPOLOGY_VERTEX_COUNT_MISMATCH:" + requirement.role)
    matched: list[str] = []
    for relation in contract.relations:
        result = _match_relation(relation, graph)
        if result is True:
            matched.append(relation.relation_id)
        elif result is False:
            failures.append("TOPOLOGY_RELATION_UNSATISFIED:" + relation.relation_id)
        else:
            unresolved.append("TOPOLOGY_RELATION_UNRESOLVED:" + relation.relation_id)
    status = "unsatisfied" if failures else "unresolved" if unresolved else "satisfied"
    return TopologyMatchResult(
        status=status,
        graph_sha256=graph.graph_sha256,
        matched_relations=matched,
        failure_codes=[*failures, *unresolved],
    )
