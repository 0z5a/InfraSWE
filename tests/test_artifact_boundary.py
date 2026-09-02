from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from infraswe.artifact_boundary import (
    audit_candidate_manifest,
    audit_evidence_pack,
    audit_pristine_apply_result,
    audit_pristine_build_result,
    audit_score_binding,
    audit_timing_integrity,
    audit_transport_envelope,
    audit_trial_seal,
    build_evidence_pack,
    build_pristine_apply_result,
    build_pristine_build_result,
    build_timing_integrity_report,
    build_transport_envelope,
    build_trial_seal,
    capture_git_candidate_patch,
    collect_candidate_artifacts,
    sanitize_pristine_environment,
)
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.artifact_boundary import (
    ArtifactAllowRule,
    ArtifactBuildPolicy,
    ArtifactPolicy,
    EvidenceArtifact,
    EvidenceProducerIdentity,
    OfficialTimingSample,
    PristineBase,
    ScoreEvidenceBinding,
    WorkspaceFreezeAttestation,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _seal(model, field: str):
    material = model.model_dump(mode="json", exclude={field})
    return model.model_copy(update={field: canonical_sha256(material)})


def _policy() -> ArtifactPolicy:
    policy = ArtifactPolicy(
        policy_id="source-patch-pristine-rebuild-v1",
        collection_roots=["/workspace/repo"],
        allow=[
            ArtifactAllowRule(
                logical_name="candidate.patch",
                source_glob="candidate.patch",
                kind="source-patch",
                max_bytes=65_536,
            )
        ],
        deny=["verifier/**", "tests/hidden/**", "**/.git/**"],
        build=ArtifactBuildPolicy(cache_policy_sha256=_digest("1")),
        policy_sha256=_digest("0"),
    )
    return _seal(policy, "policy_sha256")


def _freeze() -> WorkspaceFreezeAttestation:
    attestation = WorkspaceFreezeAttestation(
        environment_id="candidate-env-1",
        frozen_at=datetime(2026, 9, 2, tzinfo=UTC),
        filesystem_frozen=True,
        candidate_processes_terminated=True,
        accelerator_work_drained=True,
        mutation_channel_closed=True,
        environment_sha256=_digest("2"),
        attestation_sha256=_digest("0"),
    )
    return _seal(attestation, "attestation_sha256")


def _collect(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "candidate.patch").write_text(
        "diff --git a/src/op.py b/src/op.py\n"
        "--- a/src/op.py\n"
        "+++ b/src/op.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n",
        encoding="utf-8",
    )
    output = tmp_path / "collected"
    manifest = collect_candidate_artifacts(
        root=root,
        requested={"candidate.patch": "candidate.patch"},
        output=output,
        policy=_policy(),
        task_seal_sha256=_digest("3"),
        candidate_id="candidate-001",
        freeze=_freeze(),
    )
    return manifest, output


def test_collection_is_allowlisted_content_addressed_and_read_only(tmp_path) -> None:
    manifest, output = _collect(tmp_path)
    assert audit_candidate_manifest(manifest, root=output) == []
    assert manifest.artifacts[0].origin_trust == "T1_COLLECTOR_CAPTURED"
    assert manifest.artifacts[0].authority == "NONE"
    assert (output / "candidate.patch").stat().st_mode & 0o222 == 0


def test_collection_rejects_escape_symlink_denied_and_secret_material(tmp_path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside.patch"
    outside.write_text("diff --git a/a b/a\n", encoding="utf-8")
    (root / "candidate.patch").symlink_to(outside)
    with pytest.raises(ValueError, match=r"escapes|symlink"):
        collect_candidate_artifacts(
            root=root,
            requested={"candidate.patch": "candidate.patch"},
            output=tmp_path / "symlink-output",
            policy=_policy(),
            task_seal_sha256=_digest("3"),
            candidate_id="candidate-001",
            freeze=_freeze(),
        )

    (root / "candidate.patch").unlink()
    (root / "candidate.patch").write_text(
        "diff --git a/src/op.py b/src/op.py\n+API_KEY=abcdefghijklmnop123456\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="secret-like"):
        collect_candidate_artifacts(
            root=root,
            requested={"candidate.patch": "candidate.patch"},
            output=tmp_path / "secret-output",
            policy=_policy(),
            task_seal_sha256=_digest("3"),
            candidate_id="candidate-001",
            freeze=_freeze(),
        )

    with pytest.raises(ValueError, match="repository-relative"):
        collect_candidate_artifacts(
            root=root,
            requested={"candidate.patch": "../outside.patch"},
            output=tmp_path / "escape-output",
            policy=_policy(),
            task_seal_sha256=_digest("3"),
            candidate_id="candidate-001",
            freeze=_freeze(),
        )


def test_candidate_claim_cannot_impersonate_verifier_or_meter() -> None:
    producer = EvidenceProducerIdentity(
        producer_id="candidate",
        role="pristine-verifier",
        implementation_sha256=_digest("1"),
        image_sha256=_digest("2"),
        configuration_sha256=_digest("3"),
    )
    with pytest.raises(ValidationError, match="below its claimed authority"):
        EvidenceArtifact(
            evidence_id="evidence://verifier/result",
            relative_path="verifier/result.json",
            media_type="application/json",
            sha256=_digest("4"),
            size_bytes=10,
            origin_trust="T1_COLLECTOR_CAPTURED",
            authority="VERIFIER",
            producer=producer,
        )


def test_transport_trial_and_evidence_pack_form_one_digest_chain(tmp_path) -> None:
    candidate_manifest, _ = _collect(tmp_path)
    envelope = build_transport_envelope(
        candidate_manifest,
        base_revision=_digest("4"),
        source_environment_id="candidate-env-1",
        destination_policy_id="pristine-build-v1",
    )
    assert audit_transport_envelope(envelope, candidate_manifest) == []
    trial = build_trial_seal(
        task_seal_sha256=candidate_manifest.task_seal_sha256,
        draft_seal_sha256=_digest("5"),
        artifact_policy_sha256=candidate_manifest.artifact_policy_sha256,
        cache_policy_sha256=_digest("1"),
        capability_resolution_sha256=_digest("6"),
        runner_attestation_sha256=_digest("7"),
        candidate_artifact_manifest_sha256=candidate_manifest.manifest_sha256,
        build_environment_sha256=_digest("8"),
        verifier_environment_sha256=_digest("9"),
        meter_environment_sha256=_digest("a"),
        resource_lease_sha256=_digest("b"),
        benchmark_cell_sha256=_digest("c"),
        environment_sentinel_policy_sha256=_digest("d"),
    )
    assert audit_trial_seal(trial) == []
    producer = EvidenceProducerIdentity(
        producer_id="trusted-meter-v1",
        role="trusted-meter",
        implementation_sha256=_digest("e"),
        image_sha256=_digest("f"),
        configuration_sha256=_digest("1"),
    )
    evidence = EvidenceArtifact(
        evidence_id="evidence://measurement/timing",
        relative_path="measurement/timing.json",
        media_type="application/json",
        sha256=_digest("2"),
        size_bytes=100,
        origin_trust="T3_TRUSTED_METERED",
        authority="METER",
        producer=producer,
    )
    pack = build_evidence_pack(
        trial_seal=trial,
        draft_seal_sha256=trial.draft_seal_sha256,
        candidate_artifact_manifest_sha256=candidate_manifest.manifest_sha256,
        runner_attestation_sha256=trial.runner_attestation_sha256,
        verifier_result_sha256=_digest("3"),
        measurement_set_sha256=_digest("4"),
        score_input_sha256=_digest("5"),
        artifacts=[evidence],
    )
    assert audit_evidence_pack(pack, trial_seal=trial) == []

    binding = ScoreEvidenceBinding(
        component_id="performance",
        value=0.91,
        status="VALID",
        required_authority="METER",
        evidence_refs=[evidence.evidence_id],
        evidence_pack_sha256=pack.evidence_pack_sha256,
    )
    assert audit_score_binding(binding, pack) == []

    tampered = pack.model_copy(update={"measurement_set_sha256": _digest("f")})
    assert "EVIDENCE_PACK_DIGEST_MISMATCH" in audit_evidence_pack(tampered, trial_seal=trial)


def test_score_binding_fails_closed_on_wrong_authority(tmp_path) -> None:
    candidate_manifest, _ = _collect(tmp_path)
    trial = build_trial_seal(
        task_seal_sha256=candidate_manifest.task_seal_sha256,
        draft_seal_sha256=_digest("5"),
        artifact_policy_sha256=candidate_manifest.artifact_policy_sha256,
        cache_policy_sha256=_digest("1"),
        capability_resolution_sha256=_digest("6"),
        runner_attestation_sha256=_digest("7"),
        candidate_artifact_manifest_sha256=candidate_manifest.manifest_sha256,
        build_environment_sha256=_digest("8"),
        verifier_environment_sha256=_digest("9"),
        meter_environment_sha256=_digest("a"),
        resource_lease_sha256=_digest("b"),
        benchmark_cell_sha256=_digest("c"),
        environment_sentinel_policy_sha256=_digest("d"),
    )
    verifier = EvidenceProducerIdentity(
        producer_id="verifier-v1",
        role="pristine-verifier",
        implementation_sha256=_digest("1"),
        image_sha256=_digest("2"),
        configuration_sha256=_digest("3"),
    )
    evidence = EvidenceArtifact(
        evidence_id="evidence://verifier/correctness",
        relative_path="verifier/correctness.json",
        media_type="application/json",
        sha256=_digest("4"),
        size_bytes=10,
        origin_trust="T2_PRISTINE_REEXECUTED",
        authority="VERIFIER",
        producer=verifier,
    )
    pack = build_evidence_pack(
        trial_seal=trial,
        draft_seal_sha256=trial.draft_seal_sha256,
        candidate_artifact_manifest_sha256=candidate_manifest.manifest_sha256,
        runner_attestation_sha256=trial.runner_attestation_sha256,
        verifier_result_sha256=_digest("3"),
        measurement_set_sha256=_digest("4"),
        score_input_sha256=_digest("5"),
        artifacts=[evidence],
    )
    binding = ScoreEvidenceBinding(
        component_id="performance",
        value=0.5,
        status="VALID",
        required_authority="METER",
        evidence_refs=[evidence.evidence_id],
        evidence_pack_sha256=pack.evidence_pack_sha256,
    )
    failures = audit_score_binding(binding, pack)
    assert "SCORE_EVIDENCE_AUTHORITY_MISMATCH:evidence://verifier/correctness" in failures
    assert "VALID_SCORE_COMPONENT_HAS_INVALID_EVIDENCE" in failures


def test_git_capture_includes_committed_staged_unstaged_and_untracked(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            text=True,
            capture_output=True,
            check=True,
        )

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "tracked.py").write_text("base = 1\n", encoding="utf-8")
    (repo / "staged.py").write_text("base = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").stdout.strip()

    (repo / "tracked.py").write_text("committed = 2\n", encoding="utf-8")
    git("add", "tracked.py")
    git("commit", "-qm", "candidate commit")
    (repo / "staged.py").write_text("staged = 2\n", encoding="utf-8")
    git("add", "staged.py")
    (repo / "tracked.py").write_text("unstaged = 3\n", encoding="utf-8")
    (repo / "new.py").write_text("untracked = 4\n", encoding="utf-8")

    patch = capture_git_candidate_patch(repo, base_revision=base).decode()
    assert "committed = 2" not in patch
    assert "unstaged = 3" in patch
    assert "staged = 2" in patch
    assert "untracked = 4" in patch
    assert "diff --git a/new.py b/new.py" in patch


def test_pristine_apply_build_and_environment_do_not_inherit_agent_state(tmp_path) -> None:
    manifest, _ = _collect(tmp_path)
    envelope = build_transport_envelope(
        manifest,
        base_revision=_digest("4"),
        source_environment_id="candidate-env-1",
        destination_policy_id="pristine-build-v1",
    )
    pristine = PristineBase(
        repository_sha256=_digest("1"),
        base_revision=_digest("4"),
        image_sha256=_digest("2"),
        toolchain_sha256=_digest("3"),
        dependency_lock_sha256=_digest("4"),
        build_script_sha256=_digest("5"),
        environment_allowlist_sha256=_digest("6"),
        network_policy_id="no-egress-v1",
        pristine_base_sha256=_digest("0"),
    )
    pristine = _seal(pristine, "pristine_base_sha256")
    environment = sanitize_pristine_environment(
        {
            "LANG": "C.UTF-8",
            "PATH": "/trusted/bin",
            "PYTHONPATH": "/candidate/injection",
            "LD_PRELOAD": "/candidate/shim.so",
            "GITHUB_TOKEN": "secret",
        },
        allowed_names={"LANG", "PATH", "PYTHONPATH", "LD_PRELOAD", "GITHUB_TOKEN"},
    )
    assert environment == {"LANG": "C.UTF-8", "PATH": "/trusted/bin"}

    applied = build_pristine_apply_result(
        pristine_base=pristine,
        envelope=envelope,
        manifest=manifest,
        environment_allowlist_sha256=pristine.environment_allowlist_sha256,
        patch_applied=True,
        resulting_source_tree_sha256=_digest("7"),
    )
    assert applied.status == "APPLIED"
    assert audit_pristine_apply_result(applied) == []
    failed_apply = build_pristine_apply_result(
        pristine_base=pristine,
        envelope=envelope,
        manifest=manifest,
        environment_allowlist_sha256=pristine.environment_allowlist_sha256,
        patch_applied=False,
    )
    assert failed_apply.status == "APPLY_FAILED"
    assert failed_apply.owner == "candidate"

    built = build_pristine_build_result(
        apply_result=applied,
        build_policy=_policy().build,
        build_environment_sha256=_digest("8"),
        toolchain_sha256=pristine.toolchain_sha256,
        dependency_lock_sha256=pristine.dependency_lock_sha256,
        network_observed=False,
        build_succeeded=True,
        output_sha256=_digest("9"),
        log_evidence_ref="evidence://build/log",
    )
    assert built.status == "BUILT"
    assert audit_pristine_build_result(built) == []
    network_violation = build_pristine_build_result(
        apply_result=applied,
        build_policy=_policy().build,
        build_environment_sha256=_digest("8"),
        toolchain_sha256=pristine.toolchain_sha256,
        dependency_lock_sha256=pristine.dependency_lock_sha256,
        network_observed=True,
        build_succeeded=True,
        output_sha256=_digest("9"),
        log_evidence_ref="evidence://build/log",
    )
    assert network_violation.status == "BUILD_FAILED"
    assert network_violation.owner == "candidate"


def _timing_sample(pair: int, role: str, position: str) -> OfficialTimingSample:
    return OfficialTimingSample(
        sample_id=f"sample-{pair}-{role}",
        pair_id=f"pair-{pair}",
        pair_position=position,
        role=role,
        repetition=pair,
        work_units=100,
        host_elapsed_seconds=0.01,
        device_elapsed_seconds=0.009,
        synchronization="event-completion",
        completion_counter_before=0,
        completion_counter_after=100,
        output_consumed_by_trusted_plane=True,
        deferred_error_check_passed=True,
        evidence_refs=[f"evidence://measurement/sample-{pair}-{role}"],
    )


def test_official_timing_requires_complete_counterbalanced_pairs() -> None:
    samples = [_timing_sample(1, role, "baseline-first") for role in ("baseline", "candidate")] + [
        _timing_sample(2, role, "candidate-first") for role in ("baseline", "candidate")
    ]
    report = build_timing_integrity_report(samples, expected_repetitions=2)
    assert report.status == "PASS"
    assert report.official_timing_profiled is False
    assert audit_timing_integrity(report) == []

    incomplete = build_timing_integrity_report(samples[:-1], expected_repetitions=2)
    assert incomplete.status == "BENCHMARK_DEFECT"
    assert "TIMING_PAIR_ROLE_INCOMPLETE:pair-2" in incomplete.failure_codes
