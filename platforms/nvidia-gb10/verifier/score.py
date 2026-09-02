from __future__ import annotations

import argparse
import json
from pathlib import Path

from infraswe.io import atomic_write_json
from infraswe.kernel.gb10 import FEATURE_CONTRACTS, MINIMUM_RELEASE_FEATURE_IDS


def summarize(results: list[dict]) -> dict:
    by_feature = {result["feature_id"]: result for result in results}
    missing = [feature for feature in MINIMUM_RELEASE_FEATURE_IDS if feature not in by_feature]
    certified = [
        feature
        for feature in MINIMUM_RELEASE_FEATURE_IDS
        if by_feature.get(feature, {}).get("certified") is True
    ]
    failures = sorted({code for result in results for code in result.get("failure_codes", [])})
    return {
        "schema_version": "0.4",
        "status": "certified" if len(certified) == len(MINIMUM_RELEASE_FEATURE_IDS) else "partial",
        "infra_cert": "pass" if not failures and not missing else "unresolved",
        "deployability_100": None,
        "deployability_reason": (
            "GB10 feature coverage cannot replace v0.4 C/U/M scoring; "
            "5 or 7 replay concurrency/reuse/maintenance evidence is required"
        ),
        "single_node_feature_coverage": {
            "certified": certified,
            "required": list(MINIMUM_RELEASE_FEATURE_IDS),
            "missing": missing,
        },
        "cell_scorecard": {
            namespace: [
                feature_id
                for feature_id, contract in FEATURE_CONTRACTS.items()
                if contract.namespace == namespace
                and by_feature.get(feature_id, {}).get("certified")
            ]
            for namespace in sorted({contract.namespace for contract in FEATURE_CONTRACTS.values()})
        },
        "roce_scaleout_mixed_into_single_node": False,
        "failure_codes": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GB10 evidence under scoring RFC v0.4")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    atomic_write_json(args.output, summarize(payloads))


if __name__ == "__main__":
    main()
