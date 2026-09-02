from __future__ import annotations

import argparse
import json
from pathlib import Path

from infraswe.io import atomic_write_json


def attribute_value(manifest: dict, name: str):
    return (
        manifest.get("runtime_attributes", {})
        .get("parsed", {})
        .get("attributes", {})
        .get(name, {})
        .get("value")
    )


def verify(manifest: dict, *, scaleout_requested: bool) -> dict:
    platform_passed = manifest.get("gates", {}).get("platform", {}).get("status") == "pass"
    compile_passed = manifest.get("gates", {}).get("compile", {}).get("status") == "pass"
    failures = []
    if not platform_passed:
        failures.append("GB10_PLATFORM_GATE_FAILED")
    if not compile_passed:
        failures.append("GB10_COMPILE_GATE_FAILED")
    if scaleout_requested:
        ibv = manifest.get("topology", {}).get("ibv_devices", {})
        if ibv.get("returncode") != 0:
            failures.append("ROCE_DEVICE_NOT_ESTABLISHED")
    gpudirect = attribute_value(manifest, "gpudirect_rdma_supported")
    result = {
        "schema_version": "0.1",
        "status": "pass" if not failures else "fail",
        "single_node_eligible": platform_passed and compile_passed,
        "scaleout_status": "pending" if scaleout_requested and not failures else "not_applicable",
        "gpudirect_rdma_observed": gpudirect,
        "gpudirect_rdma_assumed": False,
        "pinned_host_staging_required": scaleout_requested and gpudirect != 1,
        "failure_codes": failures,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="GB10 single-node/RoCE topology gate")
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--scaleout", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        json.loads(args.capability.read_text(encoding="utf-8")),
        scaleout_requested=args.scaleout,
    )
    atomic_write_json(args.output, result)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
