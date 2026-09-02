from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.ada_sm89 import FEATURE_CONTRACTS, MINIMUM_RELEASE_FEATURE_IDS
from infraswe.models.score import ScoreResult


def _result_map(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for result in results:
        feature_id = str(result.get("feature_id", ""))
        if feature_id in mapped:
            raise ValueError(f"duplicate feature result: {feature_id}")
        mapped[feature_id] = result
    return mapped


def summarize(
    results: list[dict[str, Any]], *, v04_score: dict[str, Any] | None = None
) -> dict[str, Any]:
    by_feature = _result_map(results)
    missing = [feature for feature in MINIMUM_RELEASE_FEATURE_IDS if feature not in by_feature]
    certified = [
        feature
        for feature in MINIMUM_RELEASE_FEATURE_IDS
        if by_feature.get(feature, {}).get("certified") is True
    ]
    static_only = [
        feature
        for feature in MINIMUM_RELEASE_FEATURE_IDS
        if by_feature.get(feature, {}).get("status") == "static_only"
    ]
    failed = [
        feature
        for feature in MINIMUM_RELEASE_FEATURE_IDS
        if by_feature.get(feature, {}).get("status") == "failed"
    ]
    unresolved = [
        feature
        for feature in MINIMUM_RELEASE_FEATURE_IDS
        if feature not in certified and feature not in failed and feature not in static_only
    ]
    failures = sorted({code for result in results for code in result.get("failure_codes", [])})

    validated_v04 = None
    if v04_score is not None:
        validated_v04 = ScoreResult.model_validate(v04_score).model_dump(mode="json")
        if validated_v04["schema_version"] != "0.4":
            raise ValueError("Ada release scoring accepts only a v0.4 score envelope")

    if failed:
        infra_cert = "fail"
    elif len(certified) == len(MINIMUM_RELEASE_FEATURE_IDS):
        infra_cert = "pass"
    else:
        infra_cert = "unresolved"
    deployability = validated_v04.get("deployability") if validated_v04 else None
    effective = (
        validated_v04.get("leaderboard_effective_deployability_100")
        if validated_v04
        else None
    )
    return {
        "schema_version": "0.4",
        "track": "nvidia-ada-sm89",
        "status": "certified" if infra_cert == "pass" else "failed" if failed else "partial",
        "infra_cert": infra_cert,
        "score_authority": "infraswe-scoring-v0.4",
        "deployability": deployability,
        "deployability_100": deployability.get("score_100") if deployability else None,
        "leaderboard_effective_deployability_100": effective,
        "deployability_reason": (
            "the Ada architecture feature overlay cannot replace v0.4 C/U/M; official scoring "
            "requires frozen load cells, at least five (recommended seven) fresh-process replays, "
            "reuse evidence, maintainability probes, and E2-or-better evidence"
            if validated_v04 is None
            else "validated from the attached scoring RFC v0.4 ScoreResult envelope"
        ),
        "feature_coverage": {
            "certified": certified,
            "static_only": static_only,
            "unresolved": unresolved,
            "failed": failed,
            "required": list(MINIMUM_RELEASE_FEATURE_IDS),
            "missing": missing,
        },
        "cell_scorecard": {
            namespace: {
                "certified": [
                    feature_id
                    for feature_id, contract in FEATURE_CONTRACTS.items()
                    if contract.namespace == namespace and feature_id in certified
                ],
                "unresolved": [
                    feature_id
                    for feature_id, contract in FEATURE_CONTRACTS.items()
                    if contract.namespace == namespace and feature_id not in certified
                ],
            }
            for namespace in sorted({contract.namespace for contract in FEATURE_CONTRACTS.values()})
        },
        "architecture_overlay_cross_cell_ranking_allowed": False,
        "absolute_l40s_vs_l20_ranking_published": False,
        "pcie_multigpu_mixed_into_single_gpu": False,
        "vgpu_mixed_into_bare_metal": False,
        "missing_evidence_is_zero": False,
        "failure_codes": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ada SM89 evidence under scoring v0.4")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--v04-score", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    flattened = []
    for payload in payloads:
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            flattened.extend(payload["results"])
        else:
            flattened.append(payload)
    v04_score = (
        json.loads(args.v04_score.read_text(encoding="utf-8")) if args.v04_score else None
    )
    atomic_write_json(args.output, summarize(flattened, v04_score=v04_score))


if __name__ == "__main__":
    main()
