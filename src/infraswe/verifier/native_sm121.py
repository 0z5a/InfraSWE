from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.gb10 import FEATURE_CONTRACTS, MATCHER_VERSION, target_satisfies, version_tuple

PROVIDED_SASS_SUFFIXES = (".sass", ".sass.txt", ".disasm", ".disasm.txt")
BINARY_SUFFIXES = (".cubin", ".fatbin", ".so")
MAX_TEXT_BYTES = 64 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _strip_ptx_comments(text: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", without_blocks, flags=re.MULTILINE)


def extract_ptx_entries(text: str) -> list[dict[str, str]]:
    clean = _strip_ptx_comments(text)
    target_match = re.search(r"(?m)^\s*\.target\s+(sm_[0-9]+(?:[af])?)\b", clean)
    version_match = re.search(r"(?m)^\s*\.version\s+([0-9]+(?:\.[0-9]+)+)\b", clean)
    target = target_match.group(1) if target_match else ""
    version = version_match.group(1) if version_match else ""
    entries = []
    pattern = re.compile(r"(?:\.visible\s+)?\.entry\s+([^\s(]+)\s*\(")
    for match in pattern.finditer(clean):
        opening = clean.find("{", match.end())
        if opening < 0:
            continue
        depth = 0
        closing = -1
        for position in range(opening, len(clean)):
            character = clean[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    closing = position
                    break
        if closing >= 0:
            entries.append(
                {
                    "name": match.group(1),
                    "body": clean[opening + 1 : closing],
                    "target": target,
                    "ptx_isa": version,
                }
            )
    return entries


def _matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES * 4:
            continue
        items.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def artifact_set_sha256(inventory: list[dict[str, Any]]) -> str:
    stable = [
        {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
        for item in inventory
    ]
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _disassemble(binary_paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    parts = []
    commands = []
    for binary in binary_paths:
        if shutil.which("cuobjdump"):
            argv = ["cuobjdump", "--dump-sass", str(binary)]
        elif shutil.which("nvdisasm") and binary.suffix == ".cubin":
            argv = ["nvdisasm", str(binary)]
        else:
            continue
        completed = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=60)
        commands.append(
            {
                "argv": argv,
                "returncode": completed.returncode,
                "artifact": binary.name,
            }
        )
        if completed.returncode == 0:
            parts.append(completed.stdout + completed.stderr)
    return "\n".join(parts), commands


def _entry_gate(
    entries: list[dict[str, str]], *, feature_id: str, requested_entry: str | None
) -> dict[str, Any]:
    contract = FEATURE_CONTRACTS[feature_id]
    if contract.required_target is None:
        return {
            "passed": False,
            "selected": None,
            "candidate_count": 0,
            "reason": "feature is not a GPU PTX contract",
        }
    reports = []
    for entry in entries:
        if requested_entry is not None and entry["name"] != requested_entry:
            continue
        version_passed = True
        if contract.minimum_ptx_isa:
            try:
                version_passed = version_tuple(entry["ptx_isa"]) >= version_tuple(
                    contract.minimum_ptx_isa
                )
            except ValueError:
                version_passed = False
        matched_all = _matches(entry["body"], contract.ptx_require_all)
        matched_any = _matches(entry["body"], contract.ptx_require_any)
        report = {
            "entry": entry["name"],
            "target": entry["target"] or None,
            "ptx_isa": entry["ptx_isa"] or None,
            "target_passed": target_satisfies(entry["target"], contract.required_target),
            "version_passed": version_passed,
            "matched_all": matched_all,
            "missing_all": [
                pattern for pattern in contract.ptx_require_all if pattern not in matched_all
            ],
            "matched_any": matched_any,
            "any_passed": not contract.ptx_require_any or bool(matched_any),
        }
        report["passed"] = bool(
            report["target_passed"]
            and report["version_passed"]
            and not report["missing_all"]
            and report["any_passed"]
        )
        reports.append(report)
    passing = next((report for report in reports if report["passed"]), None)
    return {
        "passed": passing is not None,
        "selected": passing or (reports[0] if reports else None),
        "candidate_count": len(reports),
    }


def _dynamic_gate(
    payload: dict[str, Any] | None,
    *,
    feature_id: str,
    artifact_sha256: str,
    capability_fingerprint: str | None,
    requested_entry: str | None,
) -> dict[str, Any]:
    if payload is None:
        return {"present": False, "passed": False, "failure_codes": ["DYNAMIC_EVIDENCE_MISSING"]}
    failures = []
    if payload.get("schema_version") != "0.1":
        failures.append("DYNAMIC_SCHEMA_VERSION_MISMATCH")
    if payload.get("feature_id") != feature_id:
        failures.append("DYNAMIC_FEATURE_ID_MISMATCH")
    if payload.get("artifact_set_sha256") != artifact_sha256:
        failures.append("DYNAMIC_ARTIFACT_BINDING_MISMATCH")
    if (
        not capability_fingerprint
        or payload.get("capability_fingerprint") != capability_fingerprint
    ):
        failures.append("DYNAMIC_CAPABILITY_BINDING_MISMATCH")
    if not payload.get("correctness", {}).get("passed"):
        failures.append("DYNAMIC_CORRECTNESS_FAILED")
    liveness = payload.get("liveness", {})
    if not liveness.get("completed") or not liveness.get("watchdog_passed"):
        failures.append("DYNAMIC_LIVENESS_FAILED")
    observed_entries = [str(value) for value in payload.get("observed_entries", [])]
    if requested_entry and requested_entry not in observed_entries:
        failures.append("DYNAMIC_ENTRY_NOT_OBSERVED")
    if payload.get("forbidden_library_calls"):
        failures.append("DYNAMIC_FORBIDDEN_LIBRARY_CALL")
    return {"present": True, "passed": not failures, "failure_codes": failures}


def verify_gpu_feature(
    *,
    artifact_root: Path,
    feature_id: str,
    requested_entry: str | None = None,
    dynamic_evidence: dict[str, Any] | None = None,
    capability_fingerprint: str | None = None,
) -> dict[str, Any]:
    if feature_id not in FEATURE_CONTRACTS:
        raise ValueError(f"unknown SM121 feature: {feature_id}")
    contract = FEATURE_CONTRACTS[feature_id]
    if contract.required_target is None:
        raise ValueError(f"{feature_id} is not verified through PTX/cubin evidence")
    inventory = _inventory(artifact_root)
    artifact_sha256 = artifact_set_sha256(inventory)
    paths = [artifact_root / item["path"] for item in inventory]
    ptx_paths = [path for path in paths if path.suffix == ".ptx"]
    binary_paths = [path for path in paths if path.suffix in BINARY_SUFFIXES]
    provided_sass_paths = [
        path
        for path in paths
        if any(path.name.endswith(suffix) for suffix in PROVIDED_SASS_SUFFIXES)
    ]
    entries = []
    clean_ptx = []
    for path in ptx_paths:
        text = _read_text(path)
        clean_ptx.append(_strip_ptx_comments(text))
        entries.extend(extract_ptx_entries(text))
    entry_gate = _entry_gate(entries, feature_id=feature_id, requested_entry=requested_entry)
    generated_sass, commands = _disassemble(binary_paths)
    provided_sass = "\n".join(_read_text(path) for path in provided_sass_paths)
    sass_text = "\n".join((provided_sass, generated_sass))
    binary_gate = {
        "passed": bool(binary_paths and (provided_sass.strip() or generated_sass.strip())),
        "binary_count": len(binary_paths),
        "successful_disassembly": any(item["returncode"] == 0 for item in commands),
        "provided_sass": bool(provided_sass.strip()),
    }
    forbidden = _matches("\n".join((*clean_ptx, sass_text)), contract.forbidden_patterns)
    fallback_gate = {"passed": not forbidden, "matched_forbidden_patterns": forbidden}
    dynamic_gate = _dynamic_gate(
        dynamic_evidence,
        feature_id=feature_id,
        artifact_sha256=artifact_sha256,
        capability_fingerprint=capability_fingerprint,
        requested_entry=requested_entry,
    )
    failures = []
    if not inventory:
        failures.append("ARTIFACT_SET_EMPTY")
    if not entry_gate["passed"]:
        failures.append("REACHABLE_PTX_GATE_FAILED")
    if not binary_gate["passed"]:
        failures.append("NATIVE_BINARY_OR_DISASSEMBLY_MISSING")
    if not fallback_gate["passed"]:
        failures.append("GB10_FORBIDDEN_FEATURE_OR_FALLBACK")
    failures.extend(dynamic_gate["failure_codes"])
    static_passed = bool(
        inventory and entry_gate["passed"] and binary_gate["passed"] and fallback_gate["passed"]
    )
    certified = static_passed and dynamic_gate["passed"]
    status = "certified" if certified else "static_only" if static_passed else "failed"
    return {
        "schema_version": "0.1",
        "verifier": "infraswe-native-sm121",
        "matcher_version": MATCHER_VERSION,
        "feature_id": feature_id,
        "title": contract.title,
        "namespace": contract.namespace,
        "artifact_set_sha256": artifact_sha256,
        "capability_fingerprint": capability_fingerprint,
        "status": status,
        "certified": certified,
        "gates": {
            "reachable_ptx": entry_gate,
            "native_binary": binary_gate,
            "fallback": fallback_gate,
            "dynamic": dynamic_gate,
        },
        "failure_codes": sorted(set(failures)),
        "artifacts": inventory,
        "tool_commands": commands,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed SM121 PTX/cubin verifier")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--feature-id", choices=sorted(FEATURE_CONTRACTS), required=True)
    parser.add_argument("--entry")
    parser.add_argument("--dynamic-evidence", type=Path)
    parser.add_argument("--capability-fingerprint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-certified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dynamic = None
    if args.dynamic_evidence:
        dynamic = json.loads(args.dynamic_evidence.read_text(encoding="utf-8"))
    result = verify_gpu_feature(
        artifact_root=args.artifact_root,
        feature_id=args.feature_id,
        requested_entry=args.entry,
        dynamic_evidence=dynamic,
        capability_fingerprint=args.capability_fingerprint,
    )
    atomic_write_json(args.output, result)
    if args.require_certified and not result["certified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
