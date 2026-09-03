#!/usr/bin/env python3
"""Hydrate a partial clone's object store from an exact GitHub source archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tarfile
import zlib
from pathlib import Path


def _git_dir(repository: Path) -> Path:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--absolute-git-dir"],
        text=True,
    ).strip()
    return Path(value)


def _install_blob(objects: Path, data: bytes) -> tuple[str, bool]:
    payload = f"blob {len(data)}\0".encode() + data
    object_sha = hashlib.sha1(payload, usedforsecurity=False).hexdigest()
    destination = objects / object_sha[:2] / object_sha[2:]
    if destination.exists():
        return object_sha, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(zlib.compress(payload))
    try:
        os.link(temporary, destination)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    return object_sha, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    object_format = subprocess.run(
        [
            "git",
            "-C",
            str(args.repository),
            "config",
            "--get",
            "extensions.objectformat",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if object_format not in {"", "sha1"}:
        raise SystemExit(f"unsupported Git object format: {object_format}")

    objects = _git_dir(args.repository) / "objects"
    member_count = 0
    installed_count = 0
    byte_count = 0
    with tarfile.open(args.archive, mode="r|*") as archive:
        for member in archive:
            data: bytes | None = None
            if member.isfile():
                source = archive.extractfile(member)
                if source is None:
                    raise SystemExit(f"could not extract archive member: {member.name}")
                data = source.read()
            elif member.issym():
                data = member.linkname.encode()
            if data is None:
                continue
            _sha, installed = _install_blob(objects, data)
            member_count += 1
            installed_count += int(installed)
            byte_count += len(data)

    print(f"archive_member_blob_count={member_count}")
    print(f"installed_blob_count={installed_count}")
    print(f"archive_payload_bytes={byte_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
