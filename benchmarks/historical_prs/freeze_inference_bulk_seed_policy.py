#!/usr/bin/env python3
"""Freeze the prospective policy used before a domain bulk group zero."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument(
        "--domain",
        choices=("inference", "communication"),
        default="inference",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_policy.read_text(encoding="utf-8"))
    source_digest = source.get("policy_sha256")
    material = {
        "schema_version": "0.1",
        "policy_id": f"{args.domain}-bulk-disposition-v0.1-g0000",
        "domain": args.domain,
        "derived_from_round": source.get("round"),
        "source_policy_sha256": source_digest,
        "seeded_at": datetime.now(UTC).isoformat(),
        "recent_pr_max_age_days": 30,
        "active_review_max_idle_days": 14,
        "maintainer_associations": ["COLLABORATOR", "MEMBER", "OWNER"],
        "small_change_max_lines": 120,
        "small_compile_accept_enabled": False,
        "explicit_revert_accept_enabled": True,
        "active_final_head_review_priority": False,
        "maintainer_precedes_review_without_approval": False,
        "maintainer_requires_runtime_source": True,
        "mature_review_without_approval_is_reject": True,
        "uncertain_disposition": "reject",
        "check_requires_recent_final_head_human_review": True,
        "normal_test_budget_seconds": 60,
        "tensorrt_llm_test_budget_seconds": 20,
        "timeout_disposition": "abandoned-time-budget-neutral",
        "forced_polarization_used": False,
        "review_text_visible": False,
        "outcomes_visible": False,
    }
    payload = {**material, "policy_sha256": canonical_sha256(material)}
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "policy_id": payload["policy_id"],
                "source_policy_sha256": source_digest,
                "policy_sha256": payload["policy_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
