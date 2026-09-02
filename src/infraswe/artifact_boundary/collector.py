from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.artifact_boundary import (
    ArtifactAllowRule,
    ArtifactPolicy,
    CandidateArtifactEntry,
    CandidateArtifactManifest,
    WorkspaceFreezeAttestation,
)

_ZERO_DIGEST = "sha256:" + "0" * 64
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"),
    re.compile(rb"gh[opusr]_[A-Za-z0-9]{30,}"),
)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def audit_artifact_policy(policy: ArtifactPolicy) -> list[str]:
    material = policy.model_dump(mode="json", exclude={"policy_sha256"})
    return (
        []
        if policy.policy_sha256 == canonical_sha256(material)
        else ["ARTIFACT_POLICY_DIGEST_MISMATCH"]
    )


def audit_freeze_attestation(attestation: WorkspaceFreezeAttestation) -> list[str]:
    failures: list[str] = []
    material = attestation.model_dump(mode="json", exclude={"attestation_sha256"})
    if attestation.attestation_sha256 != canonical_sha256(material):
        failures.append("FREEZE_ATTESTATION_DIGEST_MISMATCH")
    checks = {
        "filesystem_frozen": "WORKSPACE_NOT_FROZEN",
        "candidate_processes_terminated": "CANDIDATE_PROCESSES_STILL_RUNNING",
        "accelerator_work_drained": "CANDIDATE_ACCELERATOR_WORK_NOT_DRAINED",
        "mutation_channel_closed": "WORKSPACE_MUTATION_CHANNEL_OPEN",
    }
    for field, code in checks.items():
        if not getattr(attestation, field):
            failures.append(code)
    return failures


def _normalize_relative_path(raw: str, *, max_length: int) -> str:
    if "\x00" in raw or "\\" in raw or len(raw) > max_length:
        raise ValueError("unsafe or overlong artifact path")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must be normalized and repository-relative")
    return path.as_posix()


def _matches(pattern: str, relative: str) -> bool:
    return PurePosixPath(relative).match(pattern)


def _select_rule(
    policy: ArtifactPolicy,
    logical_name: str,
    relative: str,
) -> ArtifactAllowRule:
    matches = [
        item
        for item in policy.allow
        if item.logical_name == logical_name and _matches(item.source_glob, relative)
    ]
    if len(matches) != 1:
        raise ValueError("artifact must match exactly one allow rule: " + logical_name)
    if any(_matches(pattern, relative) for pattern in policy.deny):
        raise ValueError("artifact path is denied by policy: " + relative)
    return matches[0]


def _validate_patch_paths(payload: bytes) -> None:
    import shlex

    text = payload.decode("utf-8", errors="surrogateescape")
    for line in text.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as error:
            raise ValueError("malformed patch header") from error
        if len(tokens) != 4 or tokens[:2] != ["diff", "--git"]:
            raise ValueError("non-canonical patch header is not accepted")
        for token, prefix in zip(tokens[2:], ("a/", "b/"), strict=True):
            if not token.startswith(prefix):
                raise ValueError("patch path prefix is invalid")
            value = token.removeprefix(prefix)
            _normalize_relative_path(value, max_length=4096)


def capture_git_candidate_patch(repo: Path, *, base_revision: str) -> bytes:
    """Capture committed, staged, and unstaged tracked changes plus untracked files."""

    resolved = repo.resolve(strict=True)

    def run(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                "git",
                "-c",
                "core.quotePath=true",
                "-c",
                "diff.external=",
                "-C",
                str(resolved),
                *arguments,
            ],
            capture_output=True,
            check=False,
        )

    checked = run("cat-file", "-e", base_revision + "^{commit}")
    if checked.returncode:
        raise ValueError("base revision is not a valid commit")
    tracked = run(
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--submodule=diff",
        base_revision,
        "--",
        ".",
    )
    if tracked.returncode:
        raise ValueError("failed to capture tracked Candidate changes")
    untracked_result = run("ls-files", "--others", "--exclude-standard", "-z")
    if untracked_result.returncode:
        raise ValueError("failed to enumerate untracked Candidate files")
    untracked = sorted(
        path.decode("utf-8", errors="surrogateescape")
        for path in untracked_result.stdout.split(b"\0")
        if path
    )
    payload = bytearray(tracked.stdout)
    for relative in untracked:
        normalized = _normalize_relative_path(relative, max_length=4096)
        path = resolved.joinpath(*PurePosixPath(normalized).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("untracked Candidate artifact is not a regular file")
        addition = run(
            "diff",
            "--no-index",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--",
            "/dev/null",
            normalized,
        )
        if addition.returncode not in {0, 1}:
            raise ValueError("failed to capture untracked Candidate file: " + normalized)
        payload.extend(addition.stdout)
    result = bytes(payload)
    _validate_patch_paths(result)
    return result


def _read_stable_regular_file(path: Path, *, max_bytes: int) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("symlink candidate artifact rejected")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("special candidate artifact rejected")
    if before.st_size > max_bytes:
        raise ValueError("candidate artifact exceeds per-file size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        payload = b""
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
            if len(payload) > max_bytes:
                raise ValueError("candidate artifact exceeds per-file size limit")
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_opened or identity_opened != identity_after:
        raise ValueError("candidate artifact changed during collection")
    return payload, before


def _contains_secret(payload: bytes) -> bool:
    return any(pattern.search(payload) for pattern in _SECRET_PATTERNS)


def collect_candidate_artifacts(
    *,
    root: Path,
    requested: Mapping[str, str],
    output: Path,
    policy: ArtifactPolicy,
    task_seal_sha256: str,
    candidate_id: str,
    freeze: WorkspaceFreezeAttestation,
    canonicalizer_version: str = "artifact-canonicalizer-v1",
) -> CandidateArtifactManifest:
    """Collect only frozen, allowlisted regular files into a trusted output root."""

    policy_failures = audit_artifact_policy(policy)
    freeze_failures = audit_freeze_attestation(freeze)
    if policy_failures or freeze_failures:
        raise ValueError("; ".join([*policy_failures, *freeze_failures]))
    if len(requested) > policy.filesystem.max_file_count:
        raise ValueError("candidate artifact count exceeds policy")
    missing = sorted(
        item.logical_name
        for item in policy.allow
        if item.required and item.logical_name not in requested
    )
    if missing:
        raise ValueError("required candidate artifact missing: " + ", ".join(missing))

    resolved_root = root.resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise ValueError("trusted artifact output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[CandidateArtifactEntry] = []
    total = 0
    for logical_name, raw_relative in sorted(requested.items()):
        relative = _normalize_relative_path(
            raw_relative, max_length=policy.filesystem.max_path_length
        )
        rule = _select_rule(policy, logical_name, relative)
        source = resolved_root.joinpath(*PurePosixPath(relative).parts)
        resolved_source = source.resolve(strict=True)
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            raise ValueError("candidate artifact escapes collection root")
        payload, metadata = _read_stable_regular_file(source, max_bytes=rule.max_bytes)
        if policy.filesystem.hardlink_policy == "reject" and metadata.st_nlink > 1:
            raise ValueError("hardlinked candidate artifact rejected")
        if rule.kind == "binary" and policy.build.binary_submission == "forbidden":
            raise ValueError("binary candidate artifacts are forbidden")
        if rule.kind == "source-patch" and policy.canonicalization.normalize_patch_headers:
            _validate_patch_paths(payload)
        if _contains_secret(payload) and policy.filesystem.secret_policy == "reject":
            raise ValueError("secret-like material found in candidate artifact")
        total += len(payload)
        if total > policy.filesystem.max_total_bytes:
            raise ValueError("candidate artifact set exceeds total size policy")

        destination = output / logical_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name("." + destination.name + ".tmp")
        temporary.write_bytes(payload)
        os.chmod(temporary, stat.S_IMODE(metadata.st_mode) & 0o755)
        temporary.replace(destination)
        media_type = (
            "text/x-diff"
            if rule.kind == "source-patch"
            else "application/octet-stream"
            if rule.kind == "binary"
            else "text/plain"
        )
        entries.append(
            CandidateArtifactEntry(
                logical_name=logical_name,
                relative_path=logical_name,
                kind=rule.kind,
                media_type=media_type,
                sha256=_sha256_bytes(payload),
                size_bytes=len(payload),
                mode=f"0{stat.S_IMODE(metadata.st_mode) & 0o777:03o}",
                origin_trust="T1_COLLECTOR_CAPTURED",
                authority="NONE",
            )
        )

    preliminary = CandidateArtifactManifest(
        artifact_policy_id=policy.policy_id,
        artifact_policy_sha256=policy.policy_sha256,
        task_seal_sha256=task_seal_sha256,
        candidate_id=candidate_id,
        artifacts=entries,
        collection_environment_sha256=freeze.environment_sha256,
        freeze_attestation_sha256=freeze.attestation_sha256,
        canonicalizer_version=canonicalizer_version,
        manifest_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"manifest_sha256"})
    manifest = preliminary.model_copy(update={"manifest_sha256": canonical_sha256(material)})
    # Durability is delegated to the controller; prevent later candidate writes in local mode.
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    return manifest


def audit_candidate_manifest(
    manifest: CandidateArtifactManifest,
    *,
    root: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    material = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if manifest.manifest_sha256 != canonical_sha256(material):
        failures.append("CANDIDATE_ARTIFACT_MANIFEST_DIGEST_MISMATCH")
    if root is not None:
        resolved = root.resolve(strict=True)
        for entry in manifest.artifacts:
            path = resolved / entry.relative_path
            if not path.is_file() or path.is_symlink():
                failures.append("CANDIDATE_ARTIFACT_MISSING_OR_UNSAFE:" + entry.logical_name)
                continue
            payload = path.read_bytes()
            if len(payload) != entry.size_bytes:
                failures.append("CANDIDATE_ARTIFACT_SIZE_MISMATCH:" + entry.logical_name)
            if _sha256_bytes(payload) != entry.sha256:
                failures.append("CANDIDATE_ARTIFACT_DIGEST_MISMATCH:" + entry.logical_name)
    return failures
