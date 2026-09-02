from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from infraswe.io import atomic_write_json
from infraswe.kernel.gb10 import FEATURE_CONTRACTS


def run(argv: list[str]) -> dict:
    if shutil.which(argv[0]) is None:
        return {"argv": argv, "returncode": 127, "stdout": "", "stderr": "command not found"}
    completed = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=60)
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def verify(binary: Path, dynamic_evidence: dict | None = None) -> dict:
    contract = FEATURE_CONTRACTS["GB10-ARM-ORDER-001"]
    header = run(["readelf", "-h", str(binary)])
    disassembly = run(["objdump", "-d", str(binary)])
    header_text = header["stdout"] + header["stderr"]
    disassembly_text = disassembly["stdout"] + disassembly["stderr"]
    aarch64 = bool(re.search(r"Machine:\s+AArch64", header_text, flags=re.IGNORECASE))
    matched = [
        pattern
        for pattern in contract.host_isa_require_any
        if re.search(pattern, disassembly_text, flags=re.IGNORECASE)
    ]
    dynamic_passed = bool(
        dynamic_evidence
        and dynamic_evidence.get("passed")
        and int(dynamic_evidence.get("iterations", 0)) >= 1_000_000
    )
    failures = []
    if not aarch64:
        failures.append("HOST_ELF_NOT_AARCH64")
    if not matched:
        failures.append("HOST_LSE_INSTRUCTION_NOT_OBSERVED")
    if not dynamic_passed:
        failures.append("ARM_MEMORY_ORDERING_STRESS_MISSING_OR_FAILED")
    return {
        "schema_version": "0.1",
        "feature_id": contract.feature_id,
        "status": "certified" if not failures else "failed",
        "certified": not failures,
        "gates": {
            "aarch64_elf": {"passed": aarch64},
            "lse_instruction": {"passed": bool(matched), "matched": matched},
            "dynamic_stress": {"passed": dynamic_passed},
        },
        "failure_codes": failures,
        "commands": {"readelf": header, "objdump": disassembly},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GB10 AArch64/LSE task evidence")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--dynamic-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-certified", action="store_true")
    args = parser.parse_args()
    dynamic = (
        json.loads(args.dynamic_evidence.read_text(encoding="utf-8"))
        if args.dynamic_evidence
        else None
    )
    result = verify(args.binary, dynamic)
    atomic_write_json(args.output, result)
    if args.require_certified and not result["certified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
