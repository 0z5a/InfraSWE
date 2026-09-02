from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path, PurePosixPath

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.judge import (
    JudgeInputArtifact,
    JudgeInputPackManifest,
    JudgeInputPackSpec,
)

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*"
        r"[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
)
_IDENTITY_LINE = re.compile(
    r"(?im)^(?:author|from|signed-off-by|co-authored-by|maintainer-approved)\s*:.*$"
)
_FORBIDDEN_BLINDNESS_CUES = (
    re.compile(r'(?i)"(?:candidate_)?author"\s*:'),
    re.compile(r'(?i)"(?:agent|model)_name"\s*:'),
    re.compile(r'(?i)"(?:project_fit|deployability|overall_score|leaderboard_rank)"\s*:'),
    re.compile(r'(?i)"human_final_decision"\s*:'),
)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_source_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Judge pack source path escapes source root: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Judge pack source file is missing or escapes root: {relative}")
    return path


def _scan_secrets(text: str, *, ref_id: str) -> None:
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise ValueError(f"JUDGE_INPUT_SECRET_DETECTED:{ref_id}")


def _scan_blindness(text: str, *, ref_id: str, authority: str) -> None:
    if authority in {"rubric", "candidate-controlled"}:
        return
    if any(pattern.search(text) for pattern in _FORBIDDEN_BLINDNESS_CUES):
        raise ValueError(f"JUDGE_INPUT_BLINDNESS_CUE_DETECTED:{ref_id}")


def _wrap_untrusted(text: str, *, source: str, source_sha256: str) -> str:
    redacted = _IDENTITY_LINE.sub("[REDACTED_IDENTITY_CUE]", text)
    escaped = html.escape(redacted, quote=False)
    source_attribute = html.escape(source, quote=True)
    return (
        f'<UNTRUSTED_CANDIDATE_CONTENT source="{source_attribute}" '
        f'digest="{source_sha256}">\n'
        f"{escaped}\n"
        "</UNTRUSTED_CANDIDATE_CONTENT>\n"
    )


def _artifact_filename(index: int, ref_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", ref_id).strip("-")
    return f"artifacts/{index:03d}-{slug}.txt"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_input_pack(
    spec: JudgeInputPackSpec,
    *,
    source_root: Path,
    output: Path,
) -> JudgeInputPackManifest:
    """Build a content-addressed, identity-blinded, read-only Judge input pack."""

    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Judge input source root does not exist: {root}")

    artifacts: list[JudgeInputArtifact] = []
    rendered: list[tuple[str, str]] = []
    for index, item in enumerate(spec.artifacts):
        path = _validate_source_path(root, item.path)
        try:
            source_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Judge input must be UTF-8 text: {item.ref_id}") from error
        _scan_secrets(source_text, ref_id=item.ref_id)
        _scan_blindness(source_text, ref_id=item.ref_id, authority=item.authority)
        source_sha256 = _sha256_bytes(source_text.encode())
        if item.candidate_controlled:
            content = _wrap_untrusted(
                source_text,
                source=item.path,
                source_sha256=source_sha256,
            )
            encoding = "html-escaped-untrusted-v1"
        else:
            content = source_text if source_text.endswith("\n") else source_text + "\n"
            encoding = "none"
        pack_path = _artifact_filename(index, item.ref_id)
        artifacts.append(
            JudgeInputArtifact(
                ref_id=item.ref_id,
                pack_path=pack_path,
                evidence_type=item.evidence_type,
                authority=item.authority,
                source_sha256=source_sha256,
                content_sha256=_sha256_bytes(content.encode()),
                candidate_controlled=item.candidate_controlled,
                boundary_encoding=encoding,
            )
        )
        rendered.append((pack_path, content))

    material = {
        "schema_version": "0.5.3",
        "draft_id": spec.draft_id,
        "draft_revision": spec.draft_revision,
        "candidate_sha256": spec.candidate_sha256,
        "target_revision_sha256": spec.target_revision_sha256,
        "rubric_sha256": spec.rubric_sha256,
        "blindness": spec.blindness.model_dump(mode="json"),
        "data_egress": spec.data_egress,
        "secret_scan_status": "pass",
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
    }
    manifest = JudgeInputPackManifest.model_validate(
        {**material, "pack_sha256": canonical_sha256(material)}
    )
    for relative, content in rendered:
        _atomic_write_text(output / relative, content)
    atomic_write_json(output / "blindness.json", spec.blindness.model_dump(mode="json"))
    atomic_write_json(output / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def audit_input_pack(manifest: JudgeInputPackManifest, *, root: Path) -> list[str]:
    failures: list[str] = []
    material = manifest.model_dump(mode="json", exclude={"pack_sha256"})
    if manifest.pack_sha256 != canonical_sha256(material):
        failures.append("JUDGE_INPUT_PACK_DIGEST_MISMATCH")
    resolved_root = root.resolve()
    for artifact in manifest.artifacts:
        relative = PurePosixPath(artifact.pack_path)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"JUDGE_INPUT_ARTIFACT_PATH_INVALID:{artifact.ref_id}")
            continue
        path = (resolved_root / artifact.pack_path).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            failures.append(f"JUDGE_INPUT_ARTIFACT_MISSING:{artifact.ref_id}")
            continue
        payload = path.read_bytes()
        if _sha256_bytes(payload) != artifact.content_sha256:
            failures.append(f"JUDGE_INPUT_ARTIFACT_DIGEST_MISMATCH:{artifact.ref_id}")
        if artifact.candidate_controlled:
            text = payload.decode("utf-8", errors="replace")
            if not (
                text.startswith("<UNTRUSTED_CANDIDATE_CONTENT ")
                and text.endswith("</UNTRUSTED_CANDIDATE_CONTENT>\n")
            ):
                failures.append(f"JUDGE_INPUT_UNTRUSTED_BOUNDARY_MISSING:{artifact.ref_id}")
    return sorted(set(failures))


def resolve_evidence_ref(
    manifest: JudgeInputPackManifest,
    evidence_ref: str,
) -> JudgeInputArtifact | None:
    ref_id = evidence_ref.split("#", 1)[0]
    return next((artifact for artifact in manifest.artifacts if artifact.ref_id == ref_id), None)
