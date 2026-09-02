#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from infraswe.history.blind import (
    assert_outcome_free,
    audit_prediction_lock,
    compile_prediction,
    freeze_prediction,
)
from infraswe.io import atomic_write_json
from infraswe.models.history import BlindEvaluationEvidence, HistoricalPRCandidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    candidate_payload = json.loads(options.candidates.read_text(encoding="utf-8"))
    evidence_payload = json.loads(options.evidence.read_text(encoding="utf-8"))
    assert_outcome_free(candidate_payload)
    assert_outcome_free(evidence_payload)
    candidates = {
        item.case_id: item for item in map(HistoricalPRCandidate.model_validate, candidate_payload)
    }
    evidence = {
        item.case_id: item for item in map(BlindEvaluationEvidence.model_validate, evidence_payload)
    }
    if candidates.keys() != evidence.keys():
        missing = sorted(candidates.keys() - evidence.keys())
        extra = sorted(evidence.keys() - candidates.keys())
        raise SystemExit(f"candidate/evidence set mismatch: missing={missing}, extra={extra}")

    locks = [
        freeze_prediction(compile_prediction(candidates[case_id], evidence[case_id]))
        for case_id in sorted(candidates)
    ]
    if not all(audit_prediction_lock(lock) for lock in locks):
        raise SystemExit("prediction lock self-audit failed")
    payload = [lock.model_dump(mode="json") for lock in locks]
    atomic_write_json(options.output, payload)
    for lock in locks:
        print(f"{lock.material.case_id}\t{lock.material.predicted_outcome}\t{lock.lock_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
