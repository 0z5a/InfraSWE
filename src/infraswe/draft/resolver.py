from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from infraswe.draft.defaults import build_default_draft, select_default_project
from infraswe.models.draft import (
    DefaultDraftProject,
    DraftCandidate,
    DraftSourceResolution,
    DraftSpec,
    RemoteGitDraftLocation,
)

RemoteReader = Callable[[RemoteGitDraftLocation], str]


def parse_draft_document(text: str, *, source: str) -> DraftSpec:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(f"cannot parse Draft document from {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Draft document from {source} must contain an object")
    return DraftSpec.model_validate(payload)


def read_remote_git_draft(location: RemoteGitDraftLocation) -> str:
    """Fetch exactly one revision and read one repository-relative Draft file."""

    with tempfile.TemporaryDirectory(prefix="infraswe-draft-") as temporary:
        repository = Path(temporary) / "repository"
        environment = dict(os.environ)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        commands = [
            ["git", "init", "--quiet", str(repository)],
            ["git", "-C", str(repository), "remote", "add", "origin", location.repository],
            [
                "git",
                "-C",
                str(repository),
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                location.revision,
            ],
        ]
        for command in commands:
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=90,
                    env=environment,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                detail = getattr(error, "stderr", None) or str(error)
                raise ValueError(f"cannot fetch remote Git Draft: {detail.strip()}") from error
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "show",
                    f"FETCH_HEAD:{location.path}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=environment,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise ValueError(f"cannot read remote Git Draft: {detail.strip()}") from error
        return result.stdout


def resolve_draft(
    *,
    local_draft: Path | None = None,
    remote_git_draft: RemoteGitDraftLocation | None = None,
    candidate: DraftCandidate | None = None,
    default_project: DefaultDraftProject | None = None,
    target_hint: str | None = None,
    created_by: str = "infraswe-default-resolver",
    remote_reader: RemoteReader = read_remote_git_draft,
) -> DraftSourceResolution:
    """Resolve local > remote Git > pinned default catalog without a silent main update."""

    if local_draft is not None:
        path = local_draft.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"local Draft does not exist: {path}")
        flags = ["REMOTE_GIT_DRAFT_SHADOWED_BY_LOCAL"] if remote_git_draft else []
        return DraftSourceResolution(
            source_kind="local",
            source=str(path),
            draft=parse_draft_document(path.read_text(encoding="utf-8"), source=str(path)),
            selection_reason="explicit-local-draft",
            audit_flags=flags,
        )
    if remote_git_draft is not None:
        source = (
            f"{remote_git_draft.repository}@{remote_git_draft.revision}#{remote_git_draft.path}"
        )
        return DraftSourceResolution(
            source_kind="remote-git",
            source=source,
            draft=parse_draft_document(remote_reader(remote_git_draft), source=source),
            selection_reason="explicit-remote-git-draft",
        )
    if candidate is None:
        raise ValueError("default Draft resolution requires a candidate descriptor")
    if default_project is None:
        project, reason, flags = select_default_project(
            target_hint=target_hint, candidate=candidate
        )
    else:
        project = default_project
        reason = "explicit-default-catalog-project"
        flags = ["DEFAULT_TARGET_REPORTED"]
    draft, profile = build_default_draft(
        project=project,
        candidate=candidate,
        created_by=created_by,
    )
    return DraftSourceResolution(
        source_kind="default-catalog",
        source=f"builtin://default-projects-v0.5/{project}",
        draft=draft,
        bundled_profile=profile,
        selected_default_project=project,
        selection_reason=reason,
        audit_flags=flags,
    )
