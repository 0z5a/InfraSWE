from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.retrieval import (
    CandidateFootprint,
    PrecedentGraphEdge,
    PrecedentRecord,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _record_anchors(record: PrecedentRecord) -> list[tuple[str, str]]:
    fields = {
        "file": record.scope.files,
        "symbol": record.scope.symbols,
        "build-target": record.scope.build_targets,
        "test": record.scope.tests,
        "config": record.scope.configs,
        "failure": record.scope.failure_signatures,
        "lifecycle": record.scope.lifecycle_tags,
        "domain": record.scope.domain_tags,
    }
    return sorted((kind, value) for kind, values in fields.items() for value in values if value)


def _footprint_anchors(footprint: CandidateFootprint) -> list[tuple[str, str]]:
    fields = {
        "file": footprint.files,
        "symbol": [*footprint.symbols, *footprint.callers, *footprint.dispatcher_points],
        "build-target": footprint.build_targets,
        "test": footprint.tests,
        "config": footprint.config_keys,
        "failure": footprint.failure_signatures,
        "lifecycle": footprint.resource_lifecycles,
    }
    anchors = {(kind, value) for kind, values in fields.items() for value in values if value}
    return sorted(anchors)


class PrecedentStore:
    """Versioned local SQLite/FTS precedent store with deterministic queries."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                precedent_id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL,
                kind TEXT NOT NULL,
                authority TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS anchors (
                precedent_id TEXT NOT NULL REFERENCES records(precedent_id) ON DELETE CASCADE,
                anchor_kind TEXT NOT NULL,
                anchor_value TEXT NOT NULL,
                PRIMARY KEY (precedent_id, anchor_kind, anchor_value)
            );
            CREATE INDEX IF NOT EXISTS anchors_lookup
                ON anchors(anchor_kind, anchor_value, precedent_id);
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL REFERENCES records(precedent_id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES records(precedent_id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, kind)
            );
            CREATE INDEX IF NOT EXISTS edges_reverse ON edges(target_id, kind, source_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
                precedent_id UNINDEXED,
                text,
                tokenize = 'unicode61'
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> PrecedentStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def set_metadata(self, key: str, value: object) -> None:
        encoded = _canonical_json(value)
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, encoded),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> object | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row is not None else None

    def upsert_record(self, record: PrecedentRecord) -> str:
        payload = record.model_dump(mode="json")
        digest = canonical_sha256(payload)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO records(
                    precedent_id, source_kind, kind, authority, observed_at,
                    record_sha256, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(precedent_id) DO UPDATE SET
                    source_kind=excluded.source_kind,
                    kind=excluded.kind,
                    authority=excluded.authority,
                    observed_at=excluded.observed_at,
                    record_sha256=excluded.record_sha256,
                    payload=excluded.payload
                """,
                (
                    record.precedent_id,
                    record.source_kind,
                    record.kind,
                    record.authority,
                    record.observed_at.isoformat(),
                    digest,
                    _canonical_json(payload),
                ),
            )
            self.connection.execute(
                "DELETE FROM anchors WHERE precedent_id = ?", (record.precedent_id,)
            )
            self.connection.executemany(
                "INSERT INTO anchors(precedent_id, anchor_kind, anchor_value) VALUES (?, ?, ?)",
                [(record.precedent_id, kind, value) for kind, value in _record_anchors(record)],
            )
            self.connection.execute(
                "DELETE FROM records_fts WHERE precedent_id = ?", (record.precedent_id,)
            )
            searchable = " ".join(
                [
                    record.text,
                    record.source_locator,
                    record.source_event_id,
                    *[value for _, value in _record_anchors(record)],
                ]
            )
            self.connection.execute(
                "INSERT INTO records_fts(precedent_id, text) VALUES (?, ?)",
                (record.precedent_id, searchable),
            )
        return digest

    def add_edge(self, edge: PrecedentGraphEdge) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO edges(source_id, target_id, kind) VALUES (?, ?, ?)",
                (edge.source_id, edge.target_id, edge.kind),
            )

    def get_record(self, precedent_id: str) -> PrecedentRecord | None:
        row = self.connection.execute(
            "SELECT payload FROM records WHERE precedent_id = ?", (precedent_id,)
        ).fetchone()
        return PrecedentRecord.model_validate_json(row["payload"]) if row else None

    def get_records(self, precedent_ids: Iterable[str]) -> list[PrecedentRecord]:
        records = [self.get_record(identifier) for identifier in precedent_ids]
        return [record for record in records if record is not None]

    def query_anchor_pairs(self, anchors: Sequence[tuple[str, str]], *, budget: int) -> list[str]:
        if not anchors:
            return []
        clauses = " OR ".join("(anchor_kind = ? AND anchor_value = ?)" for _ in anchors)
        parameters = [value for pair in anchors for value in pair]
        rows = self.connection.execute(
            f"""
            SELECT precedent_id, COUNT(*) AS matched
            FROM anchors
            WHERE {clauses}
            GROUP BY precedent_id
            ORDER BY matched DESC, precedent_id ASC
            LIMIT ?
            """,
            [*parameters, budget],
        ).fetchall()
        return [row["precedent_id"] for row in rows]

    def query_exact(self, footprint: CandidateFootprint, *, budget: int) -> list[str]:
        return self.query_anchor_pairs(_footprint_anchors(footprint), budget=budget)

    def query_failures(self, footprint: CandidateFootprint, *, budget: int) -> list[str]:
        anchors = [("failure", value) for value in footprint.failure_signatures]
        return self.query_anchor_pairs(anchors, budget=budget)

    def query_lifecycle(self, footprint: CandidateFootprint, *, budget: int) -> list[str]:
        anchors = [("lifecycle", value) for value in footprint.resource_lifecycles]
        return self.query_anchor_pairs(anchors, budget=budget)

    def query_negative(self, *, budget: int) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT precedent_id FROM records
            WHERE kind IN (
                'rejected-pattern', 'regression-precedent',
                'superseded-precedent', 'conflicting-precedent'
            ) OR source_kind IN ('revert', 'regression', 'ci-failure')
            ORDER BY observed_at DESC, precedent_id ASC
            LIMIT ?
            """,
            (budget,),
        ).fetchall()
        return [row["precedent_id"] for row in rows]

    def query_lexical(self, terms: Sequence[str], *, budget: int) -> list[str]:
        tokens = sorted({term.strip() for term in terms if term.strip()})
        if not tokens:
            return []
        expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        rows = self.connection.execute(
            """
            SELECT precedent_id, bm25(records_fts) AS score
            FROM records_fts
            WHERE records_fts MATCH ?
            ORDER BY score ASC, precedent_id ASC
            LIMIT ?
            """,
            (expression, budget),
        ).fetchall()
        return [row["precedent_id"] for row in rows]

    def expand_graph(
        self,
        seeds: Sequence[str],
        *,
        edge_allowlist: Sequence[str],
        max_hops: int,
        per_node_fanout: int,
        maximum_records: int,
    ) -> list[str]:
        seen = set(seeds)
        ordered = list(dict.fromkeys(seeds))
        frontier = list(ordered)
        for _ in range(max_hops):
            next_frontier: list[str] = []
            for node in frontier:
                placeholders = ",".join("?" for _ in edge_allowlist)
                rows = self.connection.execute(
                    f"""
                    SELECT neighbor FROM (
                        SELECT target_id AS neighbor, kind FROM edges WHERE source_id = ?
                        UNION
                        SELECT source_id AS neighbor, kind FROM edges WHERE target_id = ?
                    )
                    WHERE kind IN ({placeholders})
                    ORDER BY neighbor ASC
                    LIMIT ?
                    """,
                    (node, node, *edge_allowlist, per_node_fanout),
                ).fetchall()
                for row in rows:
                    neighbor = row["neighbor"]
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    ordered.append(neighbor)
                    next_frontier.append(neighbor)
                    if len(ordered) >= maximum_records:
                        return ordered
            frontier = next_frontier
            if not frontier:
                break
        return ordered

    def edges_between(self, precedent_ids: Iterable[str]) -> list[PrecedentGraphEdge]:
        identifiers = sorted(set(precedent_ids))
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        rows = self.connection.execute(
            f"""
            SELECT source_id, target_id, kind FROM edges
            WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
            ORDER BY source_id, target_id, kind
            """,
            (*identifiers, *identifiers),
        ).fetchall()
        return [PrecedentGraphEdge.model_validate(dict(row)) for row in rows]
