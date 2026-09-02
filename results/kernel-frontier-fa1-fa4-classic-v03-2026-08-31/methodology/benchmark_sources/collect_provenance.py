from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import triton


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def evidence(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def tree_evidence(path: Path, root: Path, pattern: str = "*.py") -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file())
    size_bytes = 0
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size_bytes += len(block)
    return {
        "path": path.relative_to(root).as_posix(),
        "pattern": pattern,
        "file_count": len(files),
        "size_bytes": size_bytes,
        "sha256": "sha256:" + digest.hexdigest(),
    }


def variant_manifest(root: Path, name: str) -> dict[str, Any]:
    source = root / "sources" / name
    target = root / "envs" / name
    binaries = sorted(target.rglob("*.so"))
    binary_evidence = []
    for binary in binaries:
        item = evidence(binary, root)
        try:
            item["cuda_elf_images"] = command("cuobjdump", "--list-elf", str(binary)).splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            item["cuda_elf_error"] = f"{type(error).__name__}: {error}"
        binary_evidence.append(item)
    if source.exists():
        source_commit = command("git", "rev-parse", "HEAD", cwd=source)
        source_status = command(
            "git", "status", "--porcelain=v1", "--untracked-files=no", cwd=source
        )
        submodules = command("git", "submodule", "status", "--recursive", cwd=source).splitlines()
    else:
        source_commit = "pypi:flash-attn-4==4.0.0b28"
        source_status = "not-applicable-pypi-wheel"
        submodules = []
    package_root = target / "flash_attn"
    return {
        "source_commit": source_commit,
        "source_status": source_status,
        "submodules": submodules,
        "binaries": binary_evidence,
        "python_source_tree": tree_evidence(package_root, root) if package_root.exists() else None,
        "distribution_metadata": [
            evidence(path, root) for path in sorted(target.glob("*.dist-info/METADATA"))
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/infraswe"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    benchmark_root = root / "benchmarks" / "kernel_frontier"
    payload = {
        "schema_version": "0.3",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "torch_build_config": torch.__config__.show(),
            "nvcc": command("nvcc", "--version"),
            "nvidia_smi": command("nvidia-smi", "-q"),
        },
        "implementations": {
            name: variant_manifest(root, name) for name in ("fa1", "fa2", "fa3", "fa4")
        },
        "evaluator_sources": [
            evidence(path, root)
            for path in sorted(benchmark_root.iterdir())
            if path.is_file() and path.suffix in {".py", ".sh"}
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
