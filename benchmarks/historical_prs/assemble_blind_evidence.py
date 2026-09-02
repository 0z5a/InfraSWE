#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import assert_outcome_free, canonical_sha256
from infraswe.io import atomic_write_json
from infraswe.models.history import (
    BlindEvaluationEvidence,
    HistoricalCheckResult,
    HistoricalPRCandidate,
)


def _raw_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _artifact_check(
    root: Path, payload: dict[str, Any]
) -> tuple[HistoricalCheckResult, str, datetime]:
    artifact = root / payload.pop("artifact")
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    digest = _raw_sha256(artifact)
    stream = payload.pop("artifact_stream", "stdout")
    if stream == "stdout":
        payload["stdout_sha256"] = digest
    elif stream == "stderr":
        payload["stderr_sha256"] = digest
    elif stream != "raw":
        raise ValueError(f"unsupported artifact stream: {stream}")
    return HistoricalCheckResult.model_validate(payload), digest, _timestamp(artifact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--static-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    candidate_payload = json.loads(options.candidates.read_text(encoding="utf-8"))
    test_plan = json.loads(options.test_plan.read_text(encoding="utf-8"))
    environment = json.loads(options.environment.read_text(encoding="utf-8"))
    static_payload = json.loads(options.static_results.read_text(encoding="utf-8"))
    manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
    for value in (candidate_payload, test_plan, environment, static_payload, manifest):
        assert_outcome_free(value)

    candidates = {
        item.case_id: item for item in map(HistoricalPRCandidate.model_validate, candidate_payload)
    }
    static_cases = {item["case_id"]: item for item in static_payload["cases"]}
    manifest_cases = manifest["cases"]
    if candidates.keys() != manifest_cases.keys() or candidates.keys() != static_cases.keys():
        raise SystemExit("candidate, static-result, and evidence-manifest case sets differ")

    test_plan_sha256 = canonical_sha256(test_plan)
    environment_sha256 = canonical_sha256(environment)
    common_raw_digests = {
        _raw_sha256(options.static_results),
        _raw_sha256(options.manifest),
        _raw_sha256(options.environment),
        _raw_sha256(options.test_plan),
    }
    common_times = [
        _timestamp(options.static_results),
        _timestamp(options.manifest),
        _timestamp(options.environment),
        _timestamp(options.test_plan),
    ]

    evidence: list[BlindEvaluationEvidence] = []
    for case_id in sorted(candidates):
        candidate = candidates[case_id]
        static = static_cases[case_id]
        case_manifest = manifest_cases[case_id]
        checks = [
            HistoricalCheckResult(
                name="exact-head-checkout",
                category="checkout",
                status="pass",
                command=static["head_commit"]["command"],
                return_code=static["head_commit"]["return_code"],
                duration_seconds=static["head_commit"]["duration_seconds"],
                stdout_sha256=static["head_commit"]["stdout_sha256"],
                stderr_sha256=static["head_commit"]["stderr_sha256"],
                details=f"exact head SHA {candidate.head_sha}",
            ),
            HistoricalCheckResult(
                name="diff-path-and-parse-contract",
                category="static",
                status="pass",
                command=static["diff_check"]["command"],
                return_code=static["diff_check"]["return_code"],
                duration_seconds=static["diff_check"]["duration_seconds"],
                stdout_sha256=static["diff_check"]["stdout_sha256"],
                stderr_sha256=static["diff_check"]["stderr_sha256"],
                details=(
                    f"path_parity={static['path_parity_pass']}; "
                    f"head_parse={static['head_parse']['status']}; "
                    f"base_parse={static['base_parse']['status']}"
                ),
            ),
        ]
        raw_digests = set(common_raw_digests)
        times = list(common_times)
        for check_payload in case_manifest["checks"]:
            check, digest, observed = _artifact_check(options.root, dict(check_payload))
            checks.append(check)
            raw_digests.add(digest)
            times.append(observed)

        item = BlindEvaluationEvidence(
            case_id=case_id,
            candidate_sha256=canonical_sha256(candidate),
            test_plan_sha256=test_plan_sha256,
            environment_sha256=environment_sha256,
            started_at=min(times),
            finished_at=max(times),
            stage=case_manifest["stage"],
            checks=checks,
            candidate_failure_codes=case_manifest.get("candidate_failure_codes", []),
            infrastructure_failure_codes=case_manifest.get("infrastructure_failure_codes", []),
            raw_evidence_digests=sorted(raw_digests),
        )
        evidence.append(item)

    payload = [item.model_dump(mode="json") for item in evidence]
    assert_outcome_free(payload)
    atomic_write_json(options.output, payload)
    print(f"test_plan_sha256={test_plan_sha256}")
    print(f"environment_sha256={environment_sha256}")
    print(f"cases={len(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
