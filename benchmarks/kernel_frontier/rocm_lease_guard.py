from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
    os.replace(temporary, path)


def _pids_from_json(payload: Any) -> set[int]:
    pids: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if "pid" in str(key).lower():
                if isinstance(value, int) and value > 0:
                    pids.add(value)
                elif isinstance(value, str):
                    pids.update(int(item) for item in re.findall(r"\b[1-9][0-9]*\b", value))
            pids.update(_pids_from_json(value))
    elif isinstance(payload, list):
        for value in payload:
            pids.update(_pids_from_json(value))
    return pids


def _rocm_smi_pids(device_index: int) -> tuple[set[int], dict[str, Any]] | None:
    command = ["rocm-smi", "--showpids", "--json"]
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    raw = completed.stdout.strip() or completed.stderr.strip()
    evidence: dict[str, Any] = {
        "method": "rocm-smi-showpids",
        "command": command,
        "exit_code": completed.returncode,
        "raw": raw,
    }
    if completed.returncode:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        evidence["parse_error"] = "rocm-smi did not return JSON"
        return None
    selected = parsed.get(f"card{device_index}", parsed) if isinstance(parsed, dict) else parsed
    return _pids_from_json(selected), evidence


def _kfd_pids() -> tuple[set[int], dict[str, Any]] | None:
    if shutil.which("fuser") is None or not Path("/dev/kfd").exists():
        return None
    command = ["fuser", "/dev/kfd"]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    raw = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    evidence = {
        "method": "fuser-dev-kfd",
        "command": command,
        "exit_code": completed.returncode,
        "raw": raw,
    }
    if completed.returncode not in (0, 1):
        return None
    return {int(item) for item in re.findall(r"\b[1-9][0-9]*\b", raw)}, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probes = [
        result for result in (_rocm_smi_pids(args.device_index), _kfd_pids()) if result is not None
    ]
    if not probes:
        payload = {
            "schema_version": "0.3",
            "evidence_kind": "exclusive-device-lease",
            "generated_at": utc_now(),
            "accelerator_vendor": "amd",
            "device_index": args.device_index,
            "status": "invalid",
            "failure_code": "LIVENESS_EXCLUSIVE_LEASE_UNVERIFIABLE",
            "active_pids": [],
        }
        exit_code = 2
    else:
        pids = set().union(*(result[0] for result in probes))
        payload = {
            "schema_version": "0.3",
            "evidence_kind": "exclusive-device-lease",
            "generated_at": utc_now(),
            "accelerator_vendor": "amd",
            "device_index": args.device_index,
            "status": "passed" if not pids else "failed",
            "failure_code": None if not pids else "LIVENESS_DEVICE_NOT_EXCLUSIVE",
            "active_pids": sorted(pids),
            "probes": [result[1] for result in probes],
        }
        exit_code = 0 if not pids else 3
    atomic_write_json(args.output, payload)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
