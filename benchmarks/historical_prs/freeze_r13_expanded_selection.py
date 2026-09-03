#!/usr/bin/env python3
"""Bind the original fourteen and fifteen-case extension into R13-29."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _validate_selection(payload: dict[str, Any]) -> None:
    if payload["selection_lock_sha256"] != canonical_sha256(payload["selection_material"]):
        raise SystemExit("selection lock digest mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--extension-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = _read(args.base_selection)
    extension = _read(args.extension_selection)
    _validate_selection(base)
    _validate_selection(extension)
    base_cases = base["selection_material"]["cases"]
    extension_cases = extension["selection_material"]["cases"]
    if len(base_cases) != 14 or len(extension_cases) != 15:
        raise SystemExit("R13 expanded components must contain 14 and 15 cases")
    case_ids = [case["case_id"] for case in [*base_cases, *extension_cases]]
    identities = [
        (case["repository"], case["pull_number"]) for case in [*base_cases, *extension_cases]
    ]
    if len(set(case_ids)) != 29 or len(set(identities)) != 29:
        raise SystemExit("R13 expanded selection contains duplicates")

    material = {
        "schema_version": "0.1",
        "protocol_id": "historical-pr-blind-training-v0.1-r13-expanded-29",
        "component_selection_lock_sha256": [
            base["selection_lock_sha256"],
            extension["selection_lock_sha256"],
        ],
        "base_case_count": 14,
        "extension_case_count": 15,
        "case_count": 29,
        "review_text_visible_to_machine_judge": False,
        "merge_outcomes_visible_to_machine_judge": False,
        "ci_fields_visible_to_machine_judge": False,
        "candidate_body_visible_during_extension_selection": False,
        "diff_content_visible_during_extension_selection": False,
        "boundary_provenance": {
            "base_fourteen": "bound to the original R13 pre-diff selection and test-plan locks",
            "extension_fifteen": (
                "selected and bound before extension diff or candidate-body acquisition"
            ),
        },
        "frozen_at": datetime.now(UTC).isoformat(),
        "cases": [*base_cases, *extension_cases],
    }
    payload = {
        "selection_material": material,
        "selection_lock_sha256": canonical_sha256(material),
    }
    atomic_write_json(args.output, payload)
    print(f"case_count={len(case_ids)}")
    print(f"selection_lock_sha256={payload['selection_lock_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
