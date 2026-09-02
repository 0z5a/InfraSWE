from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from infraswe.cli import app
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.retrieval import (
    CandidateFootprint,
    CommunicationFootprint,
    ConflictSet,
    FootprintExtractionRequest,
    HumanRuleDecision,
    MemoryTieringFootprint,
    PrecedentGraphEdge,
    PrecedentRecord,
    PrecedentScope,
    PrecedentValidity,
    RepositorySnapshot,
)
from infraswe.retrieval import (
    PrecedentStore,
    apply_human_rule_decisions,
    audit_leakage,
    audit_precedent_set_digest,
    build_default_query_plan,
    build_precedent_set,
    compile_rule_candidates,
    contract_executable_rules,
    detect_conflicts,
    execute_retrieval,
    extract_candidate_footprint,
    reciprocal_rank_fusion,
)

NOW = datetime(2026, 9, 2, tzinfo=UTC)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _footprint() -> CandidateFootprint:
    return CandidateFootprint(
        draft_id="communication-change",
        draft_revision=1,
        candidate_sha256=_digest("a"),
        files=["src/collective.py"],
        symbols=["enqueue_collective"],
        config_keys=["COMM_ASYNC_ERROR_HANDLING"],
        failure_signatures=["watchdog-timeout-rank-skew-v2"],
        resource_lifecycles=["communicator-destroy"],
        communication=CommunicationFootprint(
            family="collective-library",
            collectives=["all_reduce"],
            communicator_lifecycle=["destroy"],
        ),
    )


def _record(
    precedent_id: str,
    *,
    kind: str = "accepted-pattern",
    source_kind: str = "pull-request",
    event_id: str | None = None,
    observed_at: datetime = NOW - timedelta(days=30),
    target_authority: bool = True,
    fingerprint: str | None = None,
) -> PrecedentRecord:
    return PrecedentRecord(
        precedent_id=precedent_id,
        source_kind=source_kind,
        source_locator=f"exports/{precedent_id}.json",
        source_digest=_digest("b"),
        source_event_id=event_id or precedent_id,
        observed_at=observed_at,
        validity=PrecedentValidity(
            repository="target/project" if target_authority else "reference/project",
            first_revision="abc123",
        ),
        kind=kind,
        authority=(
            "regression-or-revert-precedent"
            if kind == "regression-precedent"
            else "repeated-accepted-precedent"
            if target_authority
            else "advisory-cross-project"
        ),
        target_authority=target_authority,
        confidence=0.9,
        scope=PrecedentScope(
            files=["src/collective.py"],
            symbols=["enqueue_collective"],
            configs=["COMM_ASYNC_ERROR_HANDLING"],
            failure_signatures=["watchdog-timeout-rank-skew-v2"],
            lifecycle_tags=["communicator-destroy"],
            domain_tags=["communication"],
        ),
        text="collective watchdog timeout communicator destroy ordering",
        change_fingerprint=fingerprint,
        proposed_rule_templates=["bounded-resource"],
    )


def test_snapshot_and_footprint_fail_closed_on_partial_or_mixed_domains() -> None:
    with pytest.raises(ValidationError, match="partial=true"):
        RepositorySnapshot(
            repository="target/project",
            revision="abc123",
            repository_sha256=_digest("1"),
            source_manifest_sha256=_digest("2"),
            permission_snapshot_sha256=_digest("3"),
            captured_at=NOW,
            corpus_cutoff=NOW,
            unparsed_files=["generated.cu"],
        )
    payload = _footprint().model_dump(mode="json")
    payload["memory_tiering"] = MemoryTieringFootprint(
        offload_object_kind="kv-cache",
        mutability="request-scoped",
        source_tier="device",
        destination_tier="host-pinned",
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="cannot mix"):
        CandidateFootprint.model_validate(payload)


def test_query_plan_requires_deterministic_core_and_optional_semantic() -> None:
    footprint = _footprint()
    plan = build_default_query_plan(
        target_snapshot_sha256=_digest("4"),
        footprint=footprint,
        corpus_cutoff=NOW,
    )
    assert {item.id for item in plan.passes if item.required} >= {
        "exact",
        "graph",
        "failure",
        "negative",
    }
    assert next(item for item in plan.passes if item.id == "semantic").required is False
    payload = plan.model_dump(mode="json")
    payload["passes"] = [item for item in payload["passes"] if item["id"] != "negative"]
    payload["rrf"]["channel_weights"].pop("negative")
    with pytest.raises(ValidationError, match="must be required"):
        type(plan).model_validate(payload)


def test_footprint_extractor_emits_typed_communication_and_memory_anchors() -> None:
    communication_request = FootprintExtractionRequest(
        draft_id="communication-change",
        draft_revision=1,
        candidate_sha256=_digest("a"),
        files=["src/collective.py"],
    )
    communication = extract_candidate_footprint(
        communication_request,
        {
            "src/collective.py": """
def destroy_communicator():
    nccl_all_reduce()
    raise RuntimeError("watchdog timeout")
""",
        },
    )
    assert communication.communication is not None
    assert communication.communication.family == "collective-library"
    assert communication.communication.collectives == ["all_reduce"]
    assert "timeout" in communication.failure_signatures
    assert "destroy_communicator" in communication.resource_lifecycles

    memory_request = FootprintExtractionRequest(
        draft_id="kv-offload",
        draft_revision=1,
        candidate_sha256=_digest("b"),
        files=["src/kv_offload.py"],
    )
    memory = extract_candidate_footprint(
        memory_request,
        {
            "src/kv_offload.py": """
def prefetch_kv_cache_version():
    allocate_host_pinned()
    # DEVICE_RESIDENT -> HOST_RESIDENT
""",
        },
    )
    assert memory.memory_tiering is not None
    assert memory.memory_tiering.offload_object_kind == "kv-cache"
    assert memory.memory_tiering.destination_tier == "host-pinned"
    assert memory.unresolved_surfaces == []


def test_footprint_extractor_fails_closed_on_ambiguous_composite_patch() -> None:
    request = FootprintExtractionRequest(
        draft_id="composite-change",
        draft_revision=1,
        candidate_sha256=_digest("a"),
        files=["runtime.py"],
    )
    with pytest.raises(ValueError, match="split the patch or select a domain"):
        extract_candidate_footprint(
            request,
            {"runtime.py": "def prefetch_kv_cache(): nccl_all_reduce()"},
        )


def test_sqlite_store_exact_graph_negative_and_fts_are_replayable(tmp_path) -> None:
    footprint = _footprint()
    plan = build_default_query_plan(
        target_snapshot_sha256=_digest("4"),
        footprint=footprint,
        corpus_cutoff=NOW,
    )
    accepted = _record("pr-accepted")
    regression = _record(
        "revert-regression",
        kind="regression-precedent",
        source_kind="revert",
    )
    with PrecedentStore(tmp_path / "precedents.sqlite") as store:
        first_digest = store.upsert_record(accepted)
        assert first_digest == store.upsert_record(accepted)
        store.upsert_record(regression)
        store.add_edge(
            PrecedentGraphEdge(
                source_id="revert-regression",
                target_id="pr-accepted",
                kind="REVERTS",
            )
        )
        hits, fused, records = execute_retrieval(store, footprint, plan)
        assert {item.channel for item in hits} >= {"exact", "graph", "negative"}
        assert {item.precedent_id for item in fused} == {
            "pr-accepted",
            "revert-regression",
        }
        assert {item.precedent_id for item in records} == {
            "pr-accepted",
            "revert-regression",
        }
        assert store.query_lexical(["watchdog"], budget=10)
        assert store.edges_between(item.precedent_id for item in records)[0].kind == "REVERTS"


def test_rrf_uses_frozen_channel_weights_without_becoming_a_candidate_score() -> None:
    plan = build_default_query_plan(
        target_snapshot_sha256=_digest("4"),
        footprint=_footprint(),
        corpus_cutoff=NOW,
    )
    fused = reciprocal_rank_fusion(
        {
            "exact": ["positive", "negative"],
            "graph": [],
            "failure": ["negative"],
            "lifecycle": [],
            "lexical": [],
            "semantic": [],
            "negative": ["negative"],
        },
        plan.rrf,
    )
    assert fused[0].precedent_id == "negative"
    assert not hasattr(fused[0], "candidate_score")


def test_leakage_audit_blocks_target_solution_and_near_duplicate() -> None:
    footprint = _footprint()
    plan = build_default_query_plan(
        target_snapshot_sha256=_digest("4"),
        footprint=footprint,
        corpus_cutoff=NOW,
        forbidden_source_ids=["target-pr"],
    )
    target = _record("target-record", event_id="target-pr")
    audit = audit_leakage([target], plan)
    assert audit.status == "fail"
    assert audit.known_solution_leaked
    suspected = audit_leakage(
        [_record("similar")],
        plan.model_copy(update={"forbidden_source_ids": []}),
        suspected_near_duplicate_ids=["similar"],
    )
    assert suspected.status == "unresolved"


def test_precedent_set_digest_and_allowlisted_rule_compilation() -> None:
    footprint = _footprint()
    plan = build_default_query_plan(
        target_snapshot_sha256=_digest("4"),
        footprint=footprint,
        corpus_cutoff=NOW,
    )
    records = [_record("accepted"), _record("regression", kind="regression-precedent")]
    audit = audit_leakage(records, plan)
    precedent_set = build_precedent_set(
        draft_id=footprint.draft_id,
        draft_revision=1,
        target_snapshot_sha256=_digest("4"),
        query_plan=plan,
        records=records,
        graph_edges=[
            PrecedentGraphEdge(source_id="regression", target_id="accepted", kind="REVERTS")
        ],
        conflicts=[
            ConflictSet(
                conflict_id="lifecycle-policy",
                precedent_ids=["accepted", "regression"],
                disposition="human-review-required",
                reason="accepted behavior was later regressed",
            )
        ],
        leakage_audit=audit,
        omitted_records_path="retrieval/omitted.jsonl",
    )
    assert audit_precedent_set_digest(precedent_set)
    assert not audit_precedent_set_digest(precedent_set.model_copy(update={"digest": _digest("f")}))
    rules = compile_rule_candidates(records)
    assert {item.template for item in rules} == {"bounded-resource"}
    assert all(item.status == "proposed" for item in rules)


def test_typed_scope_conflicts_do_not_depend_on_review_prose() -> None:
    conflicts = detect_conflicts(
        [_record("accepted"), _record("regression", kind="regression-precedent")]
    )
    assert len(conflicts) == 1
    assert conflicts[0].disposition == "human-review-required"
    assert conflicts[0].precedent_ids == ["accepted", "regression"]


def test_only_digest_bound_human_accepted_rules_can_enter_d3_contract(tmp_path) -> None:
    rule = compile_rule_candidates([_record("accepted")])[0]
    accepted = rule.model_copy(update={"status": "accepted"})
    decision = HumanRuleDecision(
        rule_id=rule.rule_id,
        action="accept",
        before_sha256=canonical_sha256(rule),
        after_sha256=canonical_sha256(accepted),
        reviewer="maintainer@example.org",
        reason="matches current project contract",
        reviewed_at=NOW,
    )
    reviewed = apply_human_rule_decisions([rule], [decision])
    assert reviewed == [accepted]
    assert contract_executable_rules([rule]) == []
    assert contract_executable_rules(reviewed) == [accepted]
    with pytest.raises(ValueError, match="before digest mismatch"):
        apply_human_rule_decisions(
            [rule],
            [decision.model_copy(update={"before_sha256": _digest("f")})],
        )

    rules_path = tmp_path / "rules.json"
    decisions_path = tmp_path / "decisions.jsonl"
    reviewed_path = tmp_path / "reviewed.json"
    contract_path = tmp_path / "contract.json"
    rules_path.write_text(
        json.dumps([rule.model_dump(mode="json")]),
        encoding="utf-8",
    )
    decisions_path.write_text(decision.model_dump_json() + "\n", encoding="utf-8")
    cli_result = CliRunner().invoke(
        app,
        [
            "precedent",
            "review-rules",
            "--rules",
            str(rules_path),
            "--decisions",
            str(decisions_path),
            "--output",
            str(reviewed_path),
            "--contract-output",
            str(contract_path),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert json.loads(contract_path.read_text(encoding="utf-8")) == [
        accepted.model_dump(mode="json")
    ]


def test_cross_project_precedent_cannot_claim_target_authority() -> None:
    payload = _record("advisory", target_authority=False).model_dump(mode="json")
    payload["authority"] = "explicit-profile"
    with pytest.raises(ValidationError, match="must remain advisory"):
        PrecedentRecord.model_validate(payload)


def test_precedent_cli_indexes_plans_retrieves_and_audits(tmp_path) -> None:
    runner = CliRunner()
    snapshot = RepositorySnapshot(
        repository="target/project",
        revision="abc123",
        repository_sha256=_digest("1"),
        source_manifest_sha256=_digest("2"),
        permission_snapshot_sha256=_digest("3"),
        captured_at=NOW,
        corpus_cutoff=NOW,
        parser_versions={"normalized-jsonl": "1"},
    )
    snapshot_path = tmp_path / "snapshot.json"
    footprint_path = tmp_path / "footprint.json"
    records_path = tmp_path / "records.jsonl"
    index_path = tmp_path / "precedents.sqlite"
    plan_path = tmp_path / "query-plan.json"
    retrieval_path = tmp_path / "retrieval"
    snapshot_path.write_text(json.dumps(snapshot.model_dump(mode="json")), encoding="utf-8")
    footprint_path.write_text(json.dumps(_footprint().model_dump(mode="json")), encoding="utf-8")
    records_path.write_text(_record("accepted").model_dump_json() + "\n", encoding="utf-8")

    indexed = runner.invoke(
        app,
        [
            "precedent",
            "index",
            "--snapshot",
            str(snapshot_path),
            "--records",
            str(records_path),
            "--output",
            str(index_path),
        ],
    )
    assert indexed.exit_code == 0, indexed.output
    planned = runner.invoke(
        app,
        [
            "precedent",
            "plan",
            "--snapshot",
            str(snapshot_path),
            "--footprint",
            str(footprint_path),
            "--output",
            str(plan_path),
        ],
    )
    assert planned.exit_code == 0, planned.output
    retrieved = runner.invoke(
        app,
        [
            "precedent",
            "retrieve",
            "--index",
            str(index_path),
            "--footprint",
            str(footprint_path),
            "--plan",
            str(plan_path),
            "--output",
            str(retrieval_path),
        ],
    )
    assert retrieved.exit_code == 0, retrieved.output
    precedent_set_path = retrieval_path / "precedent-set.json"
    audited = runner.invoke(
        app,
        ["precedent", "audit", str(precedent_set_path)],
    )
    assert audited.exit_code == 0, audited.output
    assert "valid" in audited.output
    bundle_path = retrieval_path / "retrieval-bundle.json"
    audited_bundle = runner.invoke(
        app,
        ["precedent", "audit-bundle", str(bundle_path)],
    )
    assert audited_bundle.exit_code == 0, audited_bundle.output
    trust = json.loads((retrieval_path / "trust-card.json").read_text(encoding="utf-8"))
    assert trust["candidate_score_effect"] == "none"
    assert trust["deterministic_replay"] == "pass"


def test_precedent_cli_extracts_a_frozen_footprint(tmp_path) -> None:
    runner = CliRunner()
    source_root = tmp_path / "source"
    source_path = source_root / "src" / "collective.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def destroy_communicator(): nccl_all_reduce()\n",
        encoding="utf-8",
    )
    request = FootprintExtractionRequest(
        draft_id="communication-change",
        draft_revision=1,
        candidate_sha256=_digest("a"),
        files=["src/collective.py"],
    )
    request_path = tmp_path / "footprint-request.json"
    output_path = tmp_path / "candidate-footprint.json"
    request_path.write_text(
        json.dumps(request.model_dump(mode="json")),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "precedent",
            "footprint",
            "--request",
            str(request_path),
            "--source-root",
            str(source_root),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    extracted = CandidateFootprint.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert extracted.communication is not None
    assert extracted.communication.collectives == ["all_reduce"]
