from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.kernel.blackwell import FEATURE_CONTRACTS, MVP_FEATURE_IDS
from infraswe.verifier.native_sm100 import verify_feature

REPLAY_SCHEMA_VERSION = "0.1"


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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def replay_feature(
    *,
    feature_id: str,
    replay_index: int,
    artifact_root: Path,
    dynamic_root: Path,
    evidence_root: Path,
    capability: dict[str, Any],
    capability_fingerprint: str,
) -> dict[str, Any]:
    contract = FEATURE_CONTRACTS[feature_id]
    capability_record = capability["features"][feature_id]
    if contract.phase == "preview-disabled":
        return {
            "feature_id": feature_id,
            "namespace": contract.namespace,
            "status": "not_applicable",
            "certified": False,
            "reason": "preview lane is disabled in the CUDA 13.3 / PTX 9.3 baseline",
        }
    if capability_record["support_state"] != "supported":
        return {
            "feature_id": feature_id,
            "namespace": contract.namespace,
            "status": "blocked",
            "certified": False,
            "reason": capability_record["reason"],
        }
    if capability_record["runtime_available"] is False:
        return {
            "feature_id": feature_id,
            "namespace": contract.namespace,
            "status": "not_applicable",
            "certified": False,
            "reason": "the leased topology does not satisfy this feature's runtime scope",
        }

    feature_artifacts = artifact_root / feature_id
    dynamic_path = dynamic_root / f"replay-{replay_index}" / f"{feature_id}.json"
    if not feature_artifacts.is_dir():
        return {
            "feature_id": feature_id,
            "namespace": contract.namespace,
            "status": "pending",
            "certified": False,
            "artifact_root": str(feature_artifacts),
            "dynamic_evidence": str(dynamic_path),
            "reason": "candidate artifact directory is absent",
        }
    dynamic = load_json(dynamic_path) if dynamic_path.is_file() else None
    native = verify_feature(
        artifact_root=feature_artifacts,
        feature_id=feature_id,
        evidence_dir=evidence_root / feature_id,
        dynamic_evidence=dynamic,
        expected_capability_fingerprint=capability_fingerprint,
    )
    return {
        "feature_id": feature_id,
        "namespace": contract.namespace,
        "status": native["status"],
        "certified": native["certified"],
        "artifact_root": str(feature_artifacts),
        "dynamic_evidence": str(dynamic_path) if dynamic_path.is_file() else None,
        "native_verification": native,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one fresh-process B200 feature replay")
    parser.add_argument("--replay-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--require-certified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capability = load_json(args.capability)
    capability_manifest_sha256 = sha256_file(args.capability)
    capability_fingerprint = capability["capability_fingerprint"]
    feature_ids = list(MVP_FEATURE_IDS)
    if args.include_optional:
        feature_ids.append("BW-FABRIC-001")
    features = [
        replay_feature(
            feature_id=feature_id,
            replay_index=args.replay_index,
            artifact_root=args.artifact_root,
            dynamic_root=args.dynamic_root,
            evidence_root=args.evidence_root,
            capability=capability,
            capability_fingerprint=capability_fingerprint,
        )
        for feature_id in feature_ids
    ]
    all_certified = all(
        item["certified"] for item in features if item["status"] != "not_applicable"
    )
    has_failure = any(item["status"] in {"failed", "blocked"} for item in features)
    status = "certified" if all_certified else "failed" if has_failure else "pending"
    result = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "suite_id": "b200-sm100-compiler-features-v0.1",
        "replay_index": args.replay_index,
        "capability_path": str(args.capability.resolve()),
        "capability_manifest_sha256": capability_manifest_sha256,
        "capability_fingerprint": capability_fingerprint,
        "hardware": capability["hardware"]["selected_gpu"],
        "toolchain": {
            "status": capability["toolchain"]["status"],
            "detected_versions": capability["toolchain"]["detected_versions"],
        },
        "status": status,
        "all_certified": all_certified,
        "features": features,
    }
    atomic_write_json(args.output, result)
    if args.require_certified and not all_certified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
