#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from infraswe.history.blind import (
    audit_prediction_lock,
    build_calibration_report,
    canonical_sha256,
    join_revealed_case,
)
from infraswe.io import atomic_write_json
from infraswe.models.history import (
    HistoricalGroundTruth,
    HistoricalPRCandidate,
    HistoricalPredictionLock,
)


def api_sha256(payload: str) -> str:
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--prediction-locks", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    options = parser.parse_args()

    candidates = {
        item.case_id: item
        for item in map(
            HistoricalPRCandidate.model_validate,
            json.loads(options.candidates.read_text(encoding="utf-8")),
        )
    }
    locks = {
        item.material.case_id: item
        for item in map(
            HistoricalPredictionLock.model_validate,
            json.loads(options.prediction_locks.read_text(encoding="utf-8")),
        )
    }
    if candidates.keys() != locks.keys():
        raise SystemExit("refusing reveal: every candidate needs exactly one prediction lock")
    if not all(audit_prediction_lock(lock) for lock in locks.values()):
        raise SystemExit("refusing reveal: prediction lock audit failed")

    truths = []
    cases = []
    for case_id in sorted(candidates):
        candidate = candidates[case_id]
        lock = locks[case_id]
        endpoint = f"repos/{candidate.repository}/pulls/{candidate.pull_number}"
        process = subprocess.run(
            ["gh", "api", endpoint],
            check=True,
            text=True,
            capture_output=True,
        )
        raw = process.stdout
        answer = json.loads(raw)
        observed_at = datetime.now(UTC)
        truth = HistoricalGroundTruth(
            case_id=case_id,
            repository=candidate.repository,
            pull_number=candidate.pull_number,
            state=answer["state"],
            merged=answer["merged"],
            merged_at=answer["merged_at"],
            closed_at=answer["closed_at"],
            merge_commit_sha=answer["merge_commit_sha"] if answer["merged"] else None,
            html_url=answer["html_url"],
            observed_at=observed_at,
            prediction_lock_sha256=lock.lock_sha256,
            api_response_sha256=api_sha256(raw),
        )
        truths.append(truth)
        cases.append(join_revealed_case(lock, truth))

    report = build_calibration_report(cases)
    options.output_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        options.output_directory / "ground-truth.json",
        [truth.model_dump(mode="json") for truth in truths],
    )
    atomic_write_json(
        options.output_directory / "calibration-report.json",
        report.model_dump(mode="json"),
    )
    binding = {
        "prediction_locks_sha256": canonical_sha256(
            [locks[case_id].model_dump(mode="json") for case_id in sorted(locks)]
        ),
        "ground_truth_sha256": canonical_sha256(
            [truth.model_dump(mode="json") for truth in truths]
        ),
        "calibration_report_sha256": canonical_sha256(report),
    }
    atomic_write_json(options.output_directory / "reveal-binding.json", binding)
    print(
        f"covered={report.covered_cases}/{report.total_cases} "
        f"correct={report.correct_cases} accuracy={report.accuracy}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
