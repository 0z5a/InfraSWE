from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.kernel.blackwell import (
    FEATURE_CONTRACTS,
    NATIVE_EVIDENCE_SCHEMA_VERSION,
    NATIVE_MATCHER_VERSION,
    FeatureContract,
    target_satisfies,
    version_tuple,
)

PROVIDED_SASS_SUFFIXES = (".sass", ".sass.txt", ".disasm", ".disasm.txt")
BINARY_SUFFIXES = (".cubin", ".fatbin", ".so")
MAX_TEXT_BYTES = 64 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _strip_ptx_comments(text: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", without_blocks, flags=re.MULTILINE)


def extract_ptx_entries(text: str) -> list[dict[str, str]]:
    clean = _strip_ptx_comments(text)
    target_match = re.search(r"(?m)^\s*\.target\s+(sm_[0-9]+(?:[af])?)\b", clean)
    version_match = re.search(r"(?m)^\s*\.version\s+([0-9]+(?:\.[0-9]+)+)\b", clean)
    target = target_match.group(1) if target_match else ""
    version = version_match.group(1) if version_match else ""
    entries: list[dict[str, str]] = []
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
        if closing < 0:
            continue
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


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES * 4:
            continue
        inventory.append(
            {
                "path": _relative(path, root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def artifact_set_sha256(inventory: list[dict[str, Any]]) -> str:
    stable = [
        {"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
        for item in inventory
    ]
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _run_and_preserve(
    argv: list[str], *, destination: Path, timeout_seconds: int = 60
) -> tuple[dict[str, Any], str]:
    try:
        completed = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        text = completed.stdout + completed.stderr
        returncode = completed.returncode
        error = None
    except (OSError, subprocess.TimeoutExpired) as exception:
        text = f"{type(exception).__name__}: {exception}\n"
        returncode = 127
        error = text.strip()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return (
        {
            "argv": argv,
            "returncode": returncode,
            "output_path": destination.name,
            "output_sha256": sha256_file(destination),
            "error": error,
        },
        text,
    )


def _collect_binary_evidence(
    binary_paths: list[Path], *, artifact_root: Path, evidence_dir: Path
) -> tuple[str, str, list[dict[str, Any]]]:
    sass_parts: list[str] = []
    symbol_parts: list[str] = []
    commands: list[dict[str, Any]] = []
    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    nm = shutil.which("nm")
    for index, binary in enumerate(binary_paths, start=1):
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", _relative(binary, artifact_root))
        disassembled = False
        if cuobjdump:
            record, output = _run_and_preserve(
                [cuobjdump, "--dump-sass", str(binary)],
                destination=evidence_dir / f"{index:02d}-{label}.cuobjdump.sass.txt",
            )
            record["kind"] = "sass"
            record["artifact"] = _relative(binary, artifact_root)
            commands.append(record)
            if record["returncode"] == 0:
                sass_parts.append(output)
                disassembled = True
        if not disassembled and nvdisasm and binary.suffix == ".cubin":
            record, output = _run_and_preserve(
                [nvdisasm, str(binary)],
                destination=evidence_dir / f"{index:02d}-{label}.nvdisasm.txt",
            )
            record["kind"] = "sass"
            record["artifact"] = _relative(binary, artifact_root)
            commands.append(record)
            if record["returncode"] == 0:
                sass_parts.append(output)
        if cuobjdump:
            record, output = _run_and_preserve(
                [cuobjdump, "--dump-elf", str(binary)],
                destination=evidence_dir / f"{index:02d}-{label}.cuobjdump.elf.txt",
            )
            record["kind"] = "symbols"
            record["artifact"] = _relative(binary, artifact_root)
            commands.append(record)
            if record["returncode"] == 0:
                symbol_parts.append(output)
        if nm and binary.suffix == ".so":
            record, output = _run_and_preserve(
                [nm, "-D", str(binary)],
                destination=evidence_dir / f"{index:02d}-{label}.nm.txt",
            )
            record["kind"] = "symbols"
            record["artifact"] = _relative(binary, artifact_root)
            commands.append(record)
            if record["returncode"] == 0:
                symbol_parts.append(output)
    return "\n".join(sass_parts), "\n".join(symbol_parts), commands


def _entry_report(
    *, entries: list[dict[str, str]], contract: FeatureContract, requested_entry: str | None
) -> dict[str, Any]:
    candidates = [
        entry for entry in entries if requested_entry is None or entry["name"] == requested_entry
    ]
    reports = []
    for entry in candidates:
        matched_all = _matches(entry["body"], contract.ptx_require_all)
        matched_any = _matches(entry["body"], contract.ptx_require_any)
        version_ok = False
        if entry["ptx_isa"]:
            try:
                version_ok = version_tuple(entry["ptx_isa"]) >= version_tuple(
                    contract.minimum_ptx_isa
                )
            except ValueError:
                version_ok = False
        report = {
            "entry": entry["name"],
            "target": entry["target"] or None,
            "ptx_isa": entry["ptx_isa"] or None,
            "target_passed": target_satisfies(entry["target"], contract.required_target),
            "version_passed": version_ok,
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
    selected = passing or (reports[0] if reports else None)
    return {
        "passed": passing is not None,
        "requested_entry": requested_entry,
        "selected": selected,
        "candidate_count": len(reports),
        "note": (
            "required PTX opcodes must coexist in one comment-stripped .entry body; "
            "this is a minimum reachability screen, not a complete CFG proof"
        ),
    }


def _dynamic_event_names(profiler: dict[str, Any]) -> list[str]:
    names = [str(name) for name in profiler.get("kernel_names", [])]
    events = profiler.get("device_events", profiler.get("cuda_events", []))
    names.extend(str(event.get("name", "")) for event in events if isinstance(event, dict))
    return [name for name in names if name]


def validate_dynamic_evidence(
    payload: dict[str, Any] | None,
    *,
    feature_id: str,
    requested_entry: str | None,
    expected_artifact_sha256: str,
    expected_capability_fingerprint: str | None,
) -> dict[str, Any]:
    if payload is None:
        return {"present": False, "passed": False, "failure_codes": ["DYNAMIC_EVIDENCE_MISSING"]}
    failures: list[str] = []
    if payload.get("schema_version") != NATIVE_EVIDENCE_SCHEMA_VERSION:
        failures.append("DYNAMIC_SCHEMA_VERSION_MISMATCH")
    if payload.get("feature_id") != feature_id:
        failures.append("DYNAMIC_FEATURE_ID_MISMATCH")
    if payload.get("status") != "passed":
        failures.append("DYNAMIC_EXECUTION_FAILED")
    if not isinstance(payload.get("replay_index"), int) or int(payload["replay_index"]) < 1:
        failures.append("DYNAMIC_REPLAY_INDEX_INVALID")
    if payload.get("artifact_set_sha256") != expected_artifact_sha256:
        failures.append("DYNAMIC_ARTIFACT_BINDING_MISMATCH")
    if expected_capability_fingerprint is None:
        failures.append("CAPABILITY_BINDING_MISSING")
    elif payload.get("capability_fingerprint") != expected_capability_fingerprint:
        failures.append("DYNAMIC_CAPABILITY_BINDING_MISMATCH")
    evaluator = payload.get("evaluator", {})
    if evaluator.get("owner") != "infraswe" or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(evaluator.get("evaluator_sha256", ""))
    ):
        failures.append("DYNAMIC_EVALUATOR_IDENTITY_MISSING")

    correctness = payload.get("correctness", {})
    if not correctness.get("passed"):
        failures.append("DYNAMIC_CORRECTNESS_FAILED")
    if not correctness.get("dynamic_input_changes_output"):
        failures.append("DYNAMIC_INPUT_PROBE_FAILED")
    liveness = payload.get("liveness", {})
    if not liveness.get("completed") or not liveness.get("watchdog_passed"):
        failures.append("DYNAMIC_LIVENESS_FAILED")
    mutation = payload.get("mutation", {})
    if not mutation.get("performed") or not mutation.get("passed"):
        failures.append("DYNAMIC_MUTATION_GATE_FAILED")
    profiler = payload.get("profiler", {})
    event_names = _dynamic_event_names(profiler)
    if not profiler.get("captured") or not event_names:
        failures.append("DYNAMIC_PROFILER_EVIDENCE_MISSING")
    if requested_entry and not (
        profiler.get("entry_observed")
        or any(requested_entry.lower() in name.lower() for name in event_names)
    ):
        failures.append("DYNAMIC_ENTRY_NOT_OBSERVED")
    return {
        "present": True,
        "passed": not failures,
        "replay_index": payload.get("replay_index"),
        "evaluator_sha256": evaluator.get("evaluator_sha256"),
        "event_names": event_names[:64],
        "failure_codes": failures,
    }


def verify_feature(
    *,
    artifact_root: Path,
    feature_id: str,
    evidence_dir: Path,
    dynamic_evidence: dict[str, Any] | None = None,
    requested_entry: str | None = None,
    expected_capability_fingerprint: str | None = None,
) -> dict[str, Any]:
    if feature_id not in FEATURE_CONTRACTS:
        raise ValueError(f"unknown SM100 feature contract: {feature_id}")
    if evidence_dir.resolve().is_relative_to(artifact_root.resolve()):
        raise ValueError(
            "evidence_dir must be outside artifact_root to keep artifact hashing stable"
        )
    contract = FEATURE_CONTRACTS[feature_id]
    inventory = _artifact_inventory(artifact_root)
    artifact_sha256 = artifact_set_sha256(inventory)
    paths = [artifact_root / item["path"] for item in inventory]
    ptx_paths = [path for path in paths if path.suffix == ".ptx"]
    binary_paths = [path for path in paths if path.suffix in BINARY_SUFFIXES]
    provided_sass_paths = [
        path
        for path in paths
        if any(path.name.endswith(suffix) for suffix in PROVIDED_SASS_SUFFIXES)
    ]
    provided_symbol_paths = [path for path in paths if path.name.endswith(".symbols.txt")]

    ptx_entries: list[dict[str, str]] = []
    clean_ptx_parts = []
    for path in ptx_paths:
        text = _read_text(path)
        clean_ptx_parts.append(_strip_ptx_comments(text))
        for entry in extract_ptx_entries(text):
            entry["artifact"] = _relative(path, artifact_root)
            ptx_entries.append(entry)
    entry_gate = _entry_report(
        entries=ptx_entries,
        contract=contract,
        requested_entry=requested_entry,
    )

    provided_sass = "\n".join(_read_text(path) for path in provided_sass_paths)
    generated_sass, generated_symbols, commands = _collect_binary_evidence(
        binary_paths,
        artifact_root=artifact_root,
        evidence_dir=evidence_dir,
    )
    sass_text = "\n".join((provided_sass, generated_sass))
    symbols = "\n".join((*(_read_text(path) for path in provided_symbol_paths), generated_symbols))
    sass_matched_all = _matches(sass_text, contract.sass_require_all)
    sass_matched_any = _matches(sass_text, contract.sass_require_any)
    sass_gate = {
        "passed": bool(
            binary_paths
            and len(sass_matched_all) == len(contract.sass_require_all)
            and (not contract.sass_require_any or sass_matched_any)
        ),
        "binary_present": bool(binary_paths),
        "provided_sass_present": bool(provided_sass.strip()),
        "successful_disassembly": any(
            command["kind"] == "sass" and command["returncode"] == 0 for command in commands
        ),
        "matched_all": sass_matched_all,
        "missing_all": [
            pattern for pattern in contract.sass_require_all if pattern not in sass_matched_all
        ],
        "matched_any": sass_matched_any,
    }

    fallback_text = "\n".join((*clean_ptx_parts, sass_text, symbols))
    forbidden_matches = _matches(fallback_text, contract.forbidden_patterns)
    fallback_gate = {
        "passed": not forbidden_matches,
        "matched_forbidden_patterns": forbidden_matches,
    }
    dynamic_gate = validate_dynamic_evidence(
        dynamic_evidence,
        feature_id=feature_id,
        requested_entry=requested_entry,
        expected_artifact_sha256=artifact_sha256,
        expected_capability_fingerprint=expected_capability_fingerprint,
    )

    failures: list[str] = []
    if contract.phase == "preview-disabled":
        failures.append("FEATURE_PREVIEW_DISABLED")
    if not inventory:
        failures.append("ARTIFACT_SET_EMPTY")
    if not entry_gate["passed"]:
        failures.append("REACHABLE_PTX_GATE_FAILED")
    if not binary_paths:
        failures.append("NATIVE_BINARY_MISSING")
    if not sass_gate["passed"]:
        failures.append("SASS_OPCODE_GATE_FAILED")
    if not fallback_gate["passed"]:
        failures.append("FALLBACK_SYMBOL_DETECTED")
    failures.extend(dynamic_gate["failure_codes"])

    static_passed = bool(
        contract.phase != "preview-disabled"
        and inventory
        and entry_gate["passed"]
        and sass_gate["passed"]
        and fallback_gate["passed"]
    )
    certified = static_passed and dynamic_gate["passed"]
    if certified:
        status = "certified"
    elif static_passed and not dynamic_gate["present"]:
        status = "static_only"
    else:
        status = "failed"
    return {
        "schema_version": NATIVE_EVIDENCE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "verifier": "infraswe-native-sm100",
        "matcher_version": NATIVE_MATCHER_VERSION,
        "feature_id": feature_id,
        "title": contract.title,
        "namespace": contract.namespace,
        "required_target": contract.required_target,
        "minimum_ptx_isa": contract.minimum_ptx_isa,
        "artifact_root": str(artifact_root.resolve()),
        "artifact_set_sha256": artifact_sha256,
        "capability_fingerprint": expected_capability_fingerprint,
        "status": status,
        "certified": certified,
        "gates": {
            "reachable_ptx": entry_gate,
            "native_sass": sass_gate,
            "fallback": fallback_gate,
            "dynamic": dynamic_gate,
        },
        "failure_codes": sorted(set(failures)),
        "artifacts": inventory,
        "tool_commands": commands,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed SM100 PTX/SASS/dynamic-evidence verifier"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--feature-id", choices=sorted(FEATURE_CONTRACTS), required=True)
    parser.add_argument("--entry")
    parser.add_argument("--dynamic-evidence", type=Path)
    parser.add_argument("--capability-fingerprint")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-certified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dynamic = None
    if args.dynamic_evidence:
        dynamic = json.loads(args.dynamic_evidence.read_text(encoding="utf-8"))
    result = verify_feature(
        artifact_root=args.artifact_root,
        feature_id=args.feature_id,
        evidence_dir=args.evidence_dir,
        dynamic_evidence=dynamic,
        requested_entry=args.entry,
        expected_capability_fingerprint=args.capability_fingerprint,
    )
    atomic_write_json(args.output, result)
    if args.require_certified and not result["certified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
