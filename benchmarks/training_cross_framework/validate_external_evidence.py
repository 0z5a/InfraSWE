from __future__ import annotations

import argparse
import json
from pathlib import Path

from infraswe.io import atomic_write_json
from infraswe.models.training import (
    TrainingComparability,
    TrainingEvidenceBundle,
    TrainingScoreInput,
)
from infraswe.scoring.training import build_training_result
from infraswe.training.semantics import verify_training_evidence


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an external training evidence bundle")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--score-input", type=Path)
    parser.add_argument("--comparability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = TrainingEvidenceBundle.model_validate(_read_json(args.evidence))
    certification = verify_training_evidence(bundle)
    payload = {"training_cert": certification.model_dump(mode="json")}
    if bool(args.score_input) != bool(args.comparability):
        parser.error("--score-input and --comparability must be provided together")
    if args.score_input and args.comparability:
        result = build_training_result(
            bundle=bundle,
            certification=certification,
            score_input=TrainingScoreInput.model_validate(_read_json(args.score_input)),
            comparability=TrainingComparability.model_validate(_read_json(args.comparability)),
        )
        payload["result"] = result.model_dump(mode="json")
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if certification.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
