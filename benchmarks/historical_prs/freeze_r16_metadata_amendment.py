#!/usr/bin/env python3
"""Freeze R16 metadata-only domain and duplicate corrections before source access."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--initial-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = read(args.policy)
    discovery = read(args.discovery)
    selection = read(args.initial_selection)
    if policy["policy_sha256"] != canonical_sha256(
        {key: value for key, value in policy.items() if key != "policy_sha256"}
    ):
        raise SystemExit("R16 policy digest mismatch")
    if discovery["discovery_sha256"] != canonical_sha256(
        {key: value for key, value in discovery.items() if key != "discovery_sha256"}
    ):
        raise SystemExit("R16 discovery digest mismatch")
    if selection["selection_lock_sha256"] != canonical_sha256(selection["selection_material"]):
        raise SystemExit("R16 initial selection digest mismatch")
    if selection["selection_material"]["candidate_body_visible"] is not False:
        raise SystemExit("R16 initial selection crossed the source boundary")

    material = {
        "schema_version": "0.1",
        "protocol_id": "training-iterative-contract-v0.1.1-r16-metadata-amendment",
        "stage": "after title/path audit and before body/diff/outcome/review access",
        "policy_sha256": policy["policy_sha256"],
        "discovery_sha256": discovery["discovery_sha256"],
        "superseded_selection_lock_sha256": selection["selection_lock_sha256"],
        "canonical_replacement": "selection-lock-amended.json",
        "rules": {
            "exclude_inference_only": {
                "rule": (
                    "Exclude a candidate when every non-test runtime source path is "
                    "under a project's explicit inference subtree and the title has "
                    "no direct train/trainer/optimizer/checkpoint/backward/gradient term."
                ),
                "inference_path_fragments": ["/inference/", "/generation/", "text_generation"],
                "training_override_title_terms": [
                    "train",
                    "trainer",
                    "optimizer",
                    "checkpoint",
                    "backward",
                    "gradient",
                ],
            },
            "deduplicate_metadata_signature": {
                "rule": (
                    "Within one project, retain only the highest-ranked candidate for "
                    "an identical normalized title plus identical ordered non-test "
                    "runtime path tuple."
                ),
                "identity_specific_exception": False,
            },
        },
        "reason": (
            "The initial title/path audit exposed an inference-only false anchor and "
            "near-duplicate production changes; both corrections are general metadata "
            "rules."
        ),
        "outcome_or_state_used": False,
        "review_or_comment_used": False,
        "ci_or_label_used": False,
        "candidate_body_used": False,
        "diff_content_used": False,
        "identity_specific_exception_used": False,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "amendment_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "amendment_sha256": payload["amendment_sha256"],
                "superseded_selection_lock_sha256": material["superseded_selection_lock_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
