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
from infraswe.kernel.ada_sm89 import (
    FEATURE_CONTRACTS,
    MATCHER_VERSION,
    target_satisfies,
    version_tuple,
)

PROVIDED_SASS_SUFFIXES = (".sass", ".sass.txt", ".disasm", ".disasm.txt")
PROVIDED_SYMBOL_SUFFIXES = (".symbols", ".symbols.txt", ".elf.txt")
BINARY_SUFFIXES = (".cubin", ".fatbin", ".so")
MAX_TEXT_BYTES = 64 * 1024 * 1024
OFFICIAL_FRESH_REPLAYS = 7


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


def _run(argv: list[str]) -> tuple[dict[str, Any], str]:
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=60)
        output = completed.stdout + completed.stderr
        return {
            "argv": argv,
            "returncode": completed.returncode,
        }, output
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": argv,
            "returncode": 127,
            "error": f"{type(error).__name__}: {error}",
        }, ""


def _binary_evidence(binary_paths: list[Path]) -> tuple[str, str, list[dict[str, Any]]]:
    sass_parts: list[str] = []
    symbol_parts: list[str] = []
    commands: list[dict[str, Any]] = []
    for binary in binary_paths:
        if shutil.which("cuobjdump"):
            record, output = _run(["cuobjdump", "--dump-sass", str(binary)])
            record.update({"artifact": binary.name, "kind": "sass"})
            commands.append(record)
            if record["returncode"] == 0:
                sass_parts.append(output)
            record, output = _run(["cuobjdump", "--dump-elf", str(binary)])
            record.update({"artifact": binary.name, "kind": "symbols"})
            commands.append(record)
            if record["returncode"] == 0:
                symbol_parts.append(output)
        elif shutil.which("nvdisasm") and binary.suffix == ".cubin":
            record, output = _run(["nvdisasm", str(binary)])
            record.update({"artifact": binary.name, "kind": "sass"})
            commands.append(record)
            if record["returncode"] == 0:
                sass_parts.append(output)
    return "\n".join(sass_parts), "\n".join(symbol_parts), commands


def _entry_gate(
    entries: list[dict[str, str]], *, feature_id: str, requested_entry: str | None
) -> dict[str, Any]:
    contract = FEATURE_CONTRACTS[feature_id]
    if contract.required_target is None:
        return {
            "passed": False,
            "selected": None,
            "candidate_count": 0,
            "reason": "feature requires runtime/external evidence rather than PTX entry evidence",
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
    artifact_file_hashes: set[str],
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
    if int(payload.get("silent_fallback_count", 0)) != 0:
        failures.append("DYNAMIC_SILENT_FALLBACK")
    if int(payload.get("fresh_process_replays", 0)) < OFFICIAL_FRESH_REPLAYS:
        failures.append("DYNAMIC_FRESH_REPLAYS_BELOW_7")
    loaded_module_hash = str(payload.get("loaded_module_container_sha256", ""))
    if not loaded_module_hash.startswith("sha256:"):
        failures.append("DYNAMIC_LOADED_MODULE_HASH_MISSING")
    elif loaded_module_hash not in artifact_file_hashes:
        failures.append("DYNAMIC_LOADED_MODULE_HASH_MISMATCH")
    if feature_id == "SM89-TARGET-001":
        dispatch_modes = payload.get("dispatch_modes", {})
        if not dispatch_modes.get("native_cubin", {}).get("passed"):
            failures.append("DYNAMIC_NATIVE_CUBIN_DISPATCH_MISSING")
        if not dispatch_modes.get("ptx_jit", {}).get("passed"):
            failures.append("DYNAMIC_PTX_JIT_DISPATCH_MISSING")
    if (
        feature_id == "SM89-FP8-MMA-001"
        and int(payload.get("allocation_audit", {}).get("full_size_fp16_temporaries", -1)) != 0
    ):
        failures.append("DYNAMIC_FP16_MATERIALIZATION_NOT_EXCLUDED")
    return {"present": True, "passed": not failures, "failure_codes": failures}


def verify_gpu_feature(
    *,
    artifact_root: Path,
    feature_id: str,
    requested_entry: str | None = None,
    dynamic_evidence: dict[str, Any] | None = None,
    capability_fingerprint: str | None = None,
    allow_provided_disassembly: bool = False,
) -> dict[str, Any]:
    if feature_id not in FEATURE_CONTRACTS:
        raise ValueError(f"unknown Ada SM89 feature: {feature_id}")
    contract = FEATURE_CONTRACTS[feature_id]
    if contract.required_target is None:
        raise ValueError(f"{feature_id} requires runtime/external evidence")
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
    provided_symbol_paths = [
        path
        for path in paths
        if any(path.name.endswith(suffix) for suffix in PROVIDED_SYMBOL_SUFFIXES)
    ]
    entries = []
    clean_ptx = []
    for path in ptx_paths:
        text = _read_text(path)
        clean_ptx.append(_strip_ptx_comments(text))
        entries.extend(extract_ptx_entries(text))
    entry_gate = _entry_gate(entries, feature_id=feature_id, requested_entry=requested_entry)
    generated_sass, generated_symbols, commands = _binary_evidence(binary_paths)
    provided_sass = "\n".join(_read_text(path) for path in provided_sass_paths)
    provided_symbols = "\n".join(_read_text(path) for path in provided_symbol_paths)
    accepted_sass = provided_sass if allow_provided_disassembly else ""
    accepted_symbols = provided_symbols if allow_provided_disassembly else ""
    sass_text = "\n".join((accepted_sass, generated_sass))
    symbol_text = "\n".join((accepted_symbols, generated_symbols))
    generated_disassembly = bool(
        generated_sass.strip()
        and any(item["kind"] == "sass" and item["returncode"] == 0 for item in commands)
    )
    binary_gate = {
        "passed": bool(binary_paths and sass_text.strip()),
        "binary_count": len(binary_paths),
        "successful_disassembly": generated_disassembly,
        "provided_sass": bool(provided_sass.strip()),
        "provided_sass_accepted": bool(allow_provided_disassembly and provided_sass.strip()),
    }
    sass_matches = _matches(sass_text, contract.sass_require_any)
    sass_gate = {
        "passed": not contract.sass_require_any or bool(sass_matches),
        "required_any": list(contract.sass_require_any),
        "matched": sass_matches,
    }
    audit_text = "\n".join((*clean_ptx, sass_text, symbol_text))
    forbidden = _matches(audit_text, contract.forbidden_patterns)
    fallback_gate = {"passed": not forbidden, "matched_forbidden_patterns": forbidden}
    dynamic_gate = _dynamic_gate(
        dynamic_evidence,
        feature_id=feature_id,
        artifact_sha256=artifact_sha256,
        artifact_file_hashes={str(item["sha256"]) for item in inventory},
        capability_fingerprint=capability_fingerprint,
        requested_entry=requested_entry,
    )
    failures = []
    if not inventory:
        failures.append("ARTIFACT_SET_EMPTY")
    if not entry_gate["passed"]:
        failures.append("SM89_REACHABLE_PTX_GATE_FAILED")
    if not binary_gate["passed"]:
        failures.append("SM89_NATIVE_BINARY_OR_DISASSEMBLY_MISSING")
    if not sass_gate["passed"]:
        failures.append("SM89_REQUIRED_SASS_PATH_MISSING")
    if not fallback_gate["passed"]:
        failures.append("SM89_FORBIDDEN_FEATURE_OR_FALLBACK")
    failures.extend(dynamic_gate["failure_codes"])
    static_passed = bool(
        inventory
        and entry_gate["passed"]
        and binary_gate["passed"]
        and sass_gate["passed"]
        and fallback_gate["passed"]
    )
    certified = static_passed and dynamic_gate["passed"]
    status = "certified" if certified else "static_only" if static_passed else "failed"
    return {
        "schema_version": "0.1",
        "verifier": "infraswe-native-sm89",
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
            "required_sass": sass_gate,
            "fallback": fallback_gate,
            "dynamic": dynamic_gate,
        },
        "failure_codes": sorted(set(failures)),
        "artifacts": inventory,
        "tool_commands": commands,
    }


def parse_args() -> argparse.Namespace:
    native_features = sorted(
        feature_id
        for feature_id, contract in FEATURE_CONTRACTS.items()
        if contract.required_target is not None
    )
    parser = argparse.ArgumentParser(description="Fail-closed Ada SM89 PTX/cubin verifier")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--feature-id", choices=native_features, required=True)
    parser.add_argument("--entry")
    parser.add_argument("--dynamic-evidence", type=Path)
    parser.add_argument("--capability-fingerprint")
    parser.add_argument(
        "--allow-provided-disassembly",
        action="store_true",
        help="test-fixture escape hatch; production verification must disassemble the binary",
    )
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
        allow_provided_disassembly=args.allow_provided_disassembly,
    )
    atomic_write_json(args.output, result)
    if args.require_certified and not result["certified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
