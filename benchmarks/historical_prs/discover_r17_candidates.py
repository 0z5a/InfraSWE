#!/usr/bin/env python3
"""Acquire a deep outcome-free GitHub metadata pool for the mixed R17 group."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from discover_r15_candidates import acquire_band

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def main(round_label: str = "R17", mature_eligible_multiplier: int = 8) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    material = {key: value for key, value in policy.items() if key != "policy_sha256"}
    if policy["policy_sha256"] != canonical_sha256(material):
        raise SystemExit(f"{round_label} policy digest mismatch")
    if any(
        policy["blindness"][key] is not False
        for key in (
            "candidate_body_visible",
            "diff_content_visible",
            "state_or_merge_visible",
            "review_or_comment_visible",
            "ci_or_label_visible",
        )
    ):
        raise SystemExit(f"{round_label} policy exposes forbidden evidence")

    window = policy["created_at_window"]
    discoveries: dict[str, Any] = {
        domain: {} for domain in policy["domains_in_order"]
    }
    tasks = [
        (domain, project_name, project)
        for domain in policy["domains_in_order"]
        for project_name, project in policy["projects"][domain].items()
    ]

    def acquire_slice(
        task: tuple[str, str, dict[str, Any]],
    ) -> tuple[str, str, dict[str, Any]]:
        domain, project_name, project = task
        repository = project["repository"]
        recent, recent_metadata = acquire_band(
            repository,
            domain,
            project,
            policy,
            band="recent",
            start=window["recent_start"][:10],
            end=window["observation_cutoff"][:10],
            eligible_target=max(8, int(project["count"]) * 2),
            page_limit=8,
        )
        mature, mature_metadata = acquire_band(
            repository,
            domain,
            project,
            policy,
            band="mature",
            start=window["start"][:10],
            end=window["mature_end"][:10],
            eligible_target=max(40, int(project["count"]) * mature_eligible_multiplier),
            page_limit=20,
        )
        by_number = {item["number"]: item for item in mature}
        by_number.update({item["number"]: item for item in recent})
        return domain, project_name, {
            "repository": repository,
            "queries": [recent_metadata, mature_metadata],
            "candidates": list(by_number.values()),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        acquired = list(executor.map(acquire_slice, tasks))
    for domain, project_name, result in acquired:
        discoveries[domain][project_name] = result

    output_material = {
        "schema_version": "0.1",
        "protocol_id": policy["protocol_id"],
        "policy_sha256": policy["policy_sha256"],
        "allowed_graphql_fields": policy["blindness"]["allowed_selection_fields"],
        "outcome_fields_requested": False,
        "review_or_comment_fields_requested": False,
        "ci_or_label_fields_requested": False,
        "candidate_body_requested": False,
        "diff_content_requested": False,
        "excluded_resolution_gray_zone_queried": False,
        "discovery_acquisition_policy": {
            "mature_eligible_multiplier": mature_eligible_multiplier,
            "mature_page_limit": 20,
            "recent_eligible_multiplier": 2,
            "recent_page_limit": 8,
        },
        "discoveries": discoveries,
    }
    payload = {
        **output_material,
        "discovery_sha256": canonical_sha256(output_material),
    }
    atomic_write_json(args.output, payload)
    counts = {
        f"{domain}/{project}": len(item["candidates"])
        for domain, projects in discoveries.items()
        for project, item in projects.items()
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"discovery_sha256={payload['discovery_sha256']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
