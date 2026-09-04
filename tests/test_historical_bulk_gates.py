from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from infraswe.draft.lifecycle import canonical_sha256


def test_non_improving_campaign_skips_publish_without_skipping_shutdown(
    project_root: Path,
) -> None:
    for name in ("run_training_bulk_campaign.sh", "run_inference_bulk_campaign.sh"):
        script = (project_root / "benchmarks" / "historical_prs" / name).read_text(encoding="utf-8")
        assert 'export PYTHONPATH="src:benchmarks/historical_prs' in script
        assert "git -c user.name=" in script
        assert 'git_commit_email="${INFRASWE_GIT_USER_EMAIL:' in script
        assert "aggregate target metric did not improve; skipping commit and push" in script
        assert "refusing publish and shutdown" not in script
        assert script.index("skipping commit and push") < script.index("vastai stop instance")


def test_last_bulk_campaign_owns_credential_cleanup_and_shutdown(project_root: Path) -> None:
    training = (
        project_root / "benchmarks" / "historical_prs" / "run_training_bulk_campaign.sh"
    ).read_text(encoding="utf-8")
    inference = (
        project_root / "benchmarks" / "historical_prs" / "run_inference_bulk_campaign.sh"
    ).read_text(encoding="utf-8")

    assert "inference_pending=false" in training
    assert '"${inference_pending}" != true' in training
    assert "training_pending=false" in inference
    assert '"${training_pending}" != true' in inference


def test_training_campaign_uses_large_groups_after_safe_boundary(project_root: Path) -> None:
    supervisor = (
        project_root / "benchmarks" / "historical_prs" / "infraswe-training-bulk-supervisor.sh"
    ).read_text(encoding="utf-8")
    campaign = (
        project_root / "benchmarks" / "historical_prs" / "run_training_bulk_campaign.sh"
    ).read_text(encoding="utf-8")

    assert "training-95pct-queue-ready" in supervisor
    assert "queue-lock-95pct-groups3000.json" in supervisor
    assert 'run_training_bulk_campaign.sh 11 "${end_group}"' in supervisor
    assert "group_case_count + github_batch_size - 1" in campaign
    assert "/ github_batch_size + 40" in campaign
    assert "group_case_count * 2" not in campaign

    inference_supervisor = (
        project_root / "benchmarks" / "historical_prs" / "infraswe-inference-bulk-supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "INFRASWE_TRAINING_LANES_PER_PROJECT=16" in supervisor
    assert "INFRASWE_INFERENCE_LANES_PER_PROJECT=16" in inference_supervisor
    training_round = (
        project_root / "benchmarks" / "historical_prs" / "run_training_bulk_round.sh"
    ).read_text(encoding="utf-8")
    inference_round = (
        project_root / "benchmarks" / "historical_prs" / "run_inference_bulk_round.sh"
    ).read_text(encoding="utf-8")
    assert '"$group_case_count" -ge 1000' in training_round
    assert "lanes_per_project=16" in training_round
    assert '"${group_case_count}" -ge 1000' in inference_round
    assert "lanes_per_project=16" in inference_round


def test_95pct_queue_composition_preserves_executed_prefix(project_root: Path) -> None:
    compose = _load(project_root, "compose_training_95pct_queue")
    repositories = {"a": "owner/a", "b": "owner/b"}
    coverage_cases = [
        {
            "case_id": f"{project}-pr-{number}",
            "project": project,
            "repository": f"owner/{project}",
            "pull_number": number,
            "queue_index": index,
            "group_index": index // 3,
            "group_offset": index % 3,
        }
        for index, (project, number) in enumerate(
            [("a", 3), ("a", 1), ("b", 2), ("a", 2), ("b", 1)]
        )
    ]
    coverage_material = {
        "profile": "training",
        "target_fraction": 0.95,
        "seed": "seed",
        "identity_source": "git-refs",
        "available_count_after_prior_exclusions": 6,
        "repositories": repositories,
        "acquisitions": {},
        "project_quotas": {"a": 3, "b": 2},
        "cases": coverage_cases,
    }
    coverage = {
        **coverage_material,
        "queue_lock_sha256": canonical_sha256(coverage_material),
    }
    original_cases = [dict(coverage_cases[1]), dict(coverage_cases[2]), *coverage_cases[2:]]
    for index, case in enumerate(original_cases):
        case["queue_index"] = index
        case["group_index"] = index // 2
        case["group_offset"] = index % 2
    original_material = {"cases": original_cases}
    original = {
        **original_material,
        "queue_lock_sha256": canonical_sha256(original_material),
    }
    override_cases = [dict(original_cases[1])]
    override_cases[0].update({"source_queue_index": 1, "group_index": 7, "group_offset": 0})
    override_material = {"cases": override_cases}
    override = {
        **override_material,
        "queue_lock_sha256": canonical_sha256(override_material),
    }

    payload = compose.compose_queue(
        coverage=coverage,
        original=original,
        prefix_overrides=[override],
        processed_count=2,
        group_index_base=8,
        group_size=2,
        created_at="2026-09-04T00:00:00+00:00",
    )

    assert [case["case_id"] for case in payload["cases"][:2]] == ["a-pr-1", "b-pr-2"]
    assert payload["cases"][1]["group_index"] == 7
    assert [case["group_index"] for case in payload["cases"][2:]] == [8, 8, 9]
    assert payload["target_count"] == 5
    assert payload["last_group_size"] == 1
    material = {key: value for key, value in payload.items() if key != "queue_lock_sha256"}
    assert payload["queue_lock_sha256"] == canonical_sha256(material)


def _load(project_root: Path, name: str) -> ModuleType:
    path = project_root / "benchmarks" / "historical_prs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case() -> dict[str, object]:
    return {
        "case_id": "vllm-pr-45239",
        "project": "vllm",
        "repository": "vllm-project/vllm",
        "pull_number": 45239,
        "created_at": None,
        "selected_ref_sha": "0" * 40,
        "group_index": 0,
        "group_offset": 0,
        "queue_index": 0,
    }


def _pull(number: int) -> dict[str, object]:
    return {
        "number": number,
        "title": f"PR {number}",
        "createdAt": "2026-09-01T00:00:00Z",
        "baseRefName": "main",
        "baseRefOid": "1" * 40,
        "headRefOid": "2" * 40,
        "changedFiles": 0,
        "additions": 0,
        "deletions": 0,
        "author": {"login": "author"},
        "authorAssociation": "CONTRIBUTOR",
        "files": {"totalCount": 0, "nodes": []},
        "commits": {"nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
    }


def test_metadata_batch_projects_multiple_prs_with_one_query(
    project_root: Path, monkeypatch
) -> None:
    acquire = _load(project_root, "acquire_training_bulk_group")
    cases = [_case(), {**_case(), "case_id": "vllm-pr-2", "pull_number": 2}]
    calls: list[list[int]] = []

    def query_batch(_repository: str, numbers: list[int]):
        calls.append(numbers)
        return {
            number: {
                "data": {
                    "repository": {"pullRequest": _pull(number)},
                    "rateLimit": {"cost": 1, "remaining": 4000, "resetAt": None},
                }
            }
            for number in numbers
        }

    monkeypatch.setattr(acquire, "_query_batch", query_batch)
    results = acquire._acquire_batch(cases)

    assert calls == [[45239, 2]]
    assert [item[1][0]["acquisition_status"] for item in results] == [
        "acquired",
        "acquired",
    ]
    assert [item[1][0]["pull_number"] for item in results] == [45239, 2]


def test_reveal_batch_projects_multiple_outcomes_with_one_query(
    project_root: Path, monkeypatch
) -> None:
    reveal = _load(project_root, "reveal_training_bulk_group")
    cases = [
        {**_case(), "acquisition_status": "acquired", "head_sha": "2" * 40},
        {
            **_case(),
            "case_id": "vllm-pr-2",
            "pull_number": 2,
            "acquisition_status": "acquired",
            "head_sha": "2" * 40,
        },
    ]
    calls: list[list[int]] = []

    def query_batch(_repository: str, numbers: list[int]):
        calls.append(numbers)
        return {
            number: {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "number": number,
                            "state": "MERGED" if number == 2 else "CLOSED",
                            "merged": number == 2,
                            "mergedAt": "2026-09-02T00:00:00Z" if number == 2 else None,
                            "closedAt": "2026-09-02T00:00:00Z",
                            "headRefOid": "2" * 40,
                            "url": f"https://github.com/vllm-project/vllm/pull/{number}",
                        }
                    },
                    "rateLimit": {"cost": 1, "remaining": 4000, "resetAt": None},
                }
            }
            for number in numbers
        }

    monkeypatch.setattr(reveal, "_query_batch", query_batch)
    results = reveal._acquire_batch(cases)

    assert calls == [[45239, 2]]
    assert [item[1]["availability"] for item in results] == ["available", "available"]
    assert [item[1]["merged"] for item in results] == [False, True]


def test_reveal_batch_preserves_a_removed_pr_as_an_invalid_attempt(
    project_root: Path, monkeypatch
) -> None:
    reveal = _load(project_root, "reveal_training_bulk_group")
    cases = [
        {**_case(), "acquisition_status": "acquired", "head_sha": "2" * 40},
        {
            **_case(),
            "case_id": "vllm-pr-2",
            "pull_number": 2,
            "acquisition_status": "acquired",
            "head_sha": "2" * 40,
        },
    ]
    available = {
        "number": 45239,
        "state": "OPEN",
        "merged": False,
        "mergedAt": None,
        "closedAt": None,
        "headRefOid": "2" * 40,
        "url": "https://github.com/vllm-project/vllm/pull/45239",
    }

    monkeypatch.setattr(
        reveal,
        "_query_batch",
        lambda *_args: {
            45239: {
                "data": {
                    "repository": {"pullRequest": available},
                    "rateLimit": {"cost": 1, "remaining": 4000, "resetAt": None},
                }
            },
            2: {
                "data": {
                    "repository": {"pullRequest": None},
                    "rateLimit": {"cost": 1, "remaining": 4000, "resetAt": None},
                }
            },
        },
    )

    results = reveal._acquire_batch(cases)

    assert results[0][1]["availability"] == "available"
    assert results[1][1]["availability"] == "invalid"
    assert results[1][1]["failure_code"] == "PULL_REQUEST_NOT_FOUND"


def test_hard_merged_recall_gate_can_override_an_exact_accuracy_loss(
    project_root: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(project_root / "benchmarks" / "historical_prs"))
    derive = _load(project_root, "derive_training_bulk_policy_iteration")

    assert derive.MERGED_ACCEPT_RECALL_MINIMUM == 0.95
    assert derive.MERGED_ACCEPT_RECALL_REPAIR_MARGIN == 0.015

    assert derive._should_promote_candidate(
        candidate_available=True,
        current_merged_gate_satisfied=False,
        candidate_exact_matches=2485,
        current_exact_matches=2508,
    )
    assert not derive._should_promote_candidate(
        candidate_available=True,
        current_merged_gate_satisfied=True,
        candidate_exact_matches=2485,
        current_exact_matches=2508,
    )
    assert not derive._should_promote_candidate(
        candidate_available=False,
        current_merged_gate_satisfied=False,
        candidate_exact_matches=0,
        current_exact_matches=2508,
    )


def test_merged_recall_guard_is_narrow_and_outcome_blind(project_root: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(project_root / "benchmarks" / "historical_prs"))
    freeze = _load(project_root, "freeze_training_bulk_group")
    policy = {
        **freeze.DEFAULT_POLICY,
        "merged_recall_guard_projects": ["sglang"],
        "merged_recall_guard_max_changed_lines": 100,
        "merged_recall_guard_author_associations": ["CONTRIBUTOR"],
        "merged_recall_guard_review_modes": ["reviewed", "unreviewed"],
    }
    case = {
        "acquisition_status": "acquired",
        "title": "Improve bounded runtime path",
        "created_at": "2025-01-01T00:00:00Z",
        "human_non_author_reviews": [],
        "human_non_author_review_state_counts": {
            "APPROVED": 0,
            "CHANGES_REQUESTED": 0,
            "COMMENTED": 0,
            "DISMISSED": 0,
            "PENDING": 0,
        },
        "final_head_human_non_author_review_state_counts": {
            "APPROVED": 0,
            "CHANGES_REQUESTED": 0,
            "COMMENTED": 0,
            "DISMISSED": 0,
            "PENDING": 0,
        },
        "files": [
            {
                "path": "python/sglang/runtime.py",
                "change_type": "modified",
            }
        ],
        "additions": 60,
        "deletions": 20,
        "project": "sglang",
        "pr_author_association": "CONTRIBUTOR",
    }
    frozen_at = datetime(2026, 9, 4, tzinfo=UTC)

    assert freeze._decision(case, "bounded-gap", policy, frozen_at)[0] == "accept_with_scope"
    assert (
        freeze._decision({**case, "project": "vllm"}, "bounded-gap", policy, frozen_at)[0]
        == "reject"
    )
    assert (
        freeze._decision(
            {**case, "additions": 101, "deletions": 0}, "bounded-gap", policy, frozen_at
        )[0]
        == "reject"
    )
    assert (
        freeze._decision(
            {**case, "pr_author_association": "NONE"}, "bounded-gap", policy, frozen_at
        )[0]
        == "reject"
    )


def test_unavailable_metadata_becomes_auditable_invalid_attempt(
    project_root: Path, monkeypatch
) -> None:
    acquire = _load(project_root, "acquire_training_bulk_group")

    def unavailable(repository: str, number: int):
        raise acquire.MetadataUnavailable(
            "PULL_REQUEST_NOT_FOUND", f"{repository}#{number} was removed"
        )

    monkeypatch.setattr(acquire, "_query", unavailable)
    projected, response_digest, rate_limit = acquire._acquire_case(_case())

    assert projected["acquisition_status"] == "invalid"
    assert projected["acquisition_failure_code"] == "PULL_REQUEST_NOT_FOUND"
    assert projected["files"] == []
    assert response_digest.startswith("sha256:")
    assert rate_limit["cost"] == 0


def test_partial_graphql_metadata_is_not_treated_as_a_valid_pr(
    project_root: Path, monkeypatch
) -> None:
    acquire = _load(project_root, "acquire_training_bulk_group")
    monkeypatch.setattr(
        acquire,
        "_query",
        lambda *_args: {
            "data": {
                "repository": {
                    "pullRequest": {
                        "number": 45239,
                        "files": None,
                        "commits": {"nodes": []},
                        "reviews": {"nodes": [], "totalCount": 0},
                    }
                },
                "rateLimit": {"cost": 1, "remaining": 100, "resetAt": None},
            }
        },
    )

    projected, _, _ = acquire._acquire_case(_case())

    assert projected["acquisition_status"] == "invalid"
    assert projected["acquisition_failure_code"] == "GITHUB_METADATA_INCOMPLETE"


def test_invalid_attempt_never_queries_reveal_or_enters_policy_oracle(
    project_root: Path, monkeypatch
) -> None:
    reveal = _load(project_root, "reveal_training_bulk_group")
    freeze = _load(project_root, "freeze_training_bulk_group")
    case = {
        **_case(),
        "acquisition_status": "invalid",
        "acquisition_failure_code": "PULL_REQUEST_NOT_FOUND",
        "head_sha": "0" * 40,
    }

    def forbidden_query(*_args, **_kwargs):
        raise AssertionError("invalid attempts must not query a reveal-time oracle")

    monkeypatch.setattr(reveal, "_query", forbidden_query)
    _, outcome, _ = reveal._acquire(case)
    oracle, reason = reveal._oracle(case, outcome, datetime.now(UTC))
    decision, rationale = freeze._decision(case, "bounded-gap", {}, datetime.now(UTC))

    assert outcome["availability"] == "invalid"
    assert oracle == "unresolved"
    assert reason == "PULL_REQUEST_NOT_FOUND"
    assert decision == "unresolved"
    assert rationale == ["INVALID_INPUT_PULL_REQUEST_NOT_FOUND"]


def test_invalid_attempt_is_preserved_but_excluded_from_accuracy(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    audit = _load(project_root, "audit_training_bulk_group")
    judgment_material = {
        "schema_version": "0.1",
        "group_index": 0,
        "policy": {"domain": "inference"},
    }
    judgment = {
        **judgment_material,
        "lock_set_sha256": canonical_sha256(judgment_material),
    }
    valid_case = {
        "case_id": "vllm-pr-1",
        "machine_decision": "accept_with_scope",
        "legacy_decision": "check",
        "oracle_decision": "accept",
        "machine_rationale_codes": ["CANDIDATE_TEST_OR_COMPILE_CONTRACT_CLOSED"],
        "technical_contract": "test-pass",
        "outcome": {
            "availability": "available",
            "failure_code": None,
            "merged": True,
        },
    }
    invalid_case = {
        "case_id": "vllm-pr-2",
        "machine_decision": "unresolved",
        "legacy_decision": "unresolved",
        "oracle_decision": "unresolved",
        "machine_rationale_codes": ["INVALID_INPUT_PULL_REQUEST_NOT_FOUND"],
        "technical_contract": "bounded-gap",
        "outcome": {
            "availability": "invalid",
            "failure_code": "PULL_REQUEST_NOT_FOUND",
            "merged": False,
        },
    }
    reveal_material = {
        "schema_version": "0.1",
        "judgment_lock_set_sha256": judgment["lock_set_sha256"],
        "cases": [valid_case, invalid_case],
    }
    reveal = {
        **reveal_material,
        "reveal_sha256": canonical_sha256(reveal_material),
    }
    judgment_path = tmp_path / "judgment.json"
    reveal_path = tmp_path / "reveal.json"
    output_path = tmp_path / "audit.json"
    judgment_path.write_text(json.dumps(judgment), encoding="utf-8")
    reveal_path.write_text(json.dumps(reveal), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_training_bulk_group.py",
            "--judgment-locks",
            str(judgment_path),
            "--reveal",
            str(reveal_path),
            "--output",
            str(output_path),
        ],
    )

    assert audit.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["cases"] == 2
    assert payload["summary"]["eligible_cases"] == 1
    assert payload["summary"]["invalid_cases"] == 1
    assert payload["summary"]["exact_accuracy"] == 1.0
    assert payload["summary"]["legacy_exact_accuracy"] == 0.0
    assert payload["invalid_cases"][0]["oracle_failure_code"] == "PULL_REQUEST_NOT_FOUND"
