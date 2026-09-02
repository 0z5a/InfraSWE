from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infraswe.io import atomic_write_json
from infraswe.kernel.ada_sm89 import CANONICAL_PLATFORM_CELLS
from infraswe.scoring.ada_sm89 import score_cross_sku_reuse
from infraswe.scoring.deployability import score_concurrent_stability

VALIDATOR_ID = "infraswe-ada-sm89-external-v1"


def _unresolved(feature_id: str, code: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "0.4",
        "validator": VALIDATOR_ID,
        "feature_id": feature_id,
        "status": "unresolved",
        "certified": False,
        "reason": reason,
        "failure_codes": [code],
    }


def validate_concurrency(payload: Mapping[str, Any]) -> dict[str, Any]:
    feature_id = "ADA-CONCURRENCY-001"
    load_cells = payload.get("load_cells")
    if not isinstance(load_cells, list):
        return _unresolved(feature_id, "ADA_CONCURRENCY_LOAD_CELLS_MISSING", "load_cells missing")
    replays = int(payload.get("fresh_process_replays", 0))
    dimension = score_concurrent_stability(load_cells, fresh_process_replays=replays)
    failures = list(dimension.failure_codes)
    if replays < 7:
        failures.append("ADA_FRESH_REPLAYS_BELOW_7")
    certified = dimension.component.status == "scored" and not failures
    return {
        "schema_version": "0.4",
        "validator": VALIDATOR_ID,
        "feature_id": feature_id,
        "status": "certified" if certified else "unresolved",
        "certified": certified,
        "replay_count": replays,
        "component": dimension.component.model_dump(mode="json"),
        "raw_metrics": dimension.raw_metrics,
        "failure_codes": sorted(set(failures)),
    }


def validate_cross_sku(payload: Mapping[str, Any]) -> dict[str, Any]:
    feature_id = "ADA-CROSS-SKU-001"
    cells = payload.get("cells")
    weights = payload.get("shape_weights")
    if not isinstance(cells, Mapping) or not isinstance(weights, Mapping):
        return _unresolved(
            feature_id,
            "ADA_CROSS_SKU_EVIDENCE_MISSING",
            "cells and shape_weights are required",
        )
    diagnostic = score_cross_sku_reuse(cells, shape_weights=weights)
    replays = int(payload.get("fresh_process_replays", 0))
    replay_failures = ["ADA_FRESH_REPLAYS_BELOW_7"] if replays < 7 else []
    certified = (
        diagnostic["status"] == "diagnostic"
        and not diagnostic["failure_codes"]
        and not replay_failures
    )
    status = "certified" if certified else diagnostic["status"]
    return {
        "schema_version": "0.4",
        "validator": VALIDATOR_ID,
        "feature_id": feature_id,
        "status": status,
        "certified": certified,
        "replay_count": replays,
        "diagnostic": diagnostic,
        "failure_codes": sorted({*diagnostic["failure_codes"], *replay_failures}),
    }


def validate_torchcompile(payload: Mapping[str, Any]) -> dict[str, Any]:
    feature_id = "ADA-TORCHCOMPILE-001"
    cells = payload.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(CANONICAL_PLATFORM_CELLS):
        return _unresolved(
            feature_id,
            "ADA_TORCHCOMPILE_CELL_MISSING",
            "both L40S and L20 cell records are required",
        )
    codegen_keys = {str(cell.get("codegen_cache_key", "")) for cell in cells.values()}
    tuning_keys = {str(cell.get("board_tuning_cache_key", "")) for cell in cells.values()}
    failures = []
    if "" in codegen_keys or len(codegen_keys) != 1:
        failures.append("ADA_TORCHCOMPILE_CODEGEN_CACHE_NOT_REUSED")
    if "" in tuning_keys or len(tuning_keys) != len(CANONICAL_PLATFORM_CELLS):
        failures.append("ADA_TORCHCOMPILE_TUNING_CACHE_NOT_ISOLATED")
    hard_failures = []
    budget_warnings = []
    observed = {}
    for cell_name, cell in cells.items():
        values = {
            "unique_graphs": int(cell.get("unique_graphs", -1)),
            "unique_kernels": int(cell.get("unique_kernels", -1)),
            "compile_seconds_cold": float(cell.get("compile_seconds_cold", -1)),
            "generated_source_mib": float(cell.get("generated_source_mib", -1)),
            "steady_compile_events": int(cell.get("steady_compile_events", -1)),
            "online_unbounded_autotune": bool(cell.get("online_unbounded_autotune", True)),
        }
        observed[cell_name] = values
        if any(values[name] < 0 for name in values if name != "online_unbounded_autotune"):
            failures.append(f"ADA_TORCHCOMPILE_METRIC_MISSING:{cell_name}")
        limits = {
            "unique_graphs": 8,
            "unique_kernels": 32,
            "compile_seconds_cold": 120,
            "generated_source_mib": 8,
        }
        budget_warnings.extend(
            f"{cell_name}:{name}" for name, limit in limits.items() if values[name] > limit
        )
        if values["online_unbounded_autotune"]:
            hard_failures.append(f"ADA_ONLINE_UNBOUNDED_AUTOTUNE:{cell_name}")
        if values["steady_compile_events"] > 0:
            hard_failures.append(f"ADA_STEADY_COMPILE_EVENT:{cell_name}")
    failures.extend(hard_failures)
    replays = int(payload.get("fresh_process_replays", 0))
    if replays < 7:
        failures.append("ADA_FRESH_REPLAYS_BELOW_7")
    certified = not failures
    return {
        "schema_version": "0.4",
        "validator": VALIDATOR_ID,
        "feature_id": feature_id,
        "status": (
            "certified_with_budget_warning"
            if certified and budget_warnings
            else "certified"
            if certified
            else "failed"
            if hard_failures
            else "unresolved"
        ),
        "certified": certified,
        "replay_count": replays,
        "budget_warnings": sorted(budget_warnings),
        "observed": observed,
        "failure_codes": sorted(set(failures)),
    }


def validate_external(feature_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    validators = {
        "ADA-CONCURRENCY-001": validate_concurrency,
        "ADA-CROSS-SKU-001": validate_cross_sku,
        "ADA-TORCHCOMPILE-001": validate_torchcompile,
    }
    return validators[feature_id](payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Ada SM89 external task evidence")
    parser.add_argument(
        "--feature-id",
        choices=("ADA-CONCURRENCY-001", "ADA-CROSS-SKU-001", "ADA-TORCHCOMPILE-001"),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-certified", action="store_true")
    args = parser.parse_args()
    input_bytes = args.input.read_bytes()
    payload = json.loads(input_bytes)
    result = validate_external(args.feature_id, payload)
    result["input_evidence_sha256"] = "sha256:" + hashlib.sha256(input_bytes).hexdigest()
    atomic_write_json(args.output, result)
    if args.require_certified and not result["certified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
