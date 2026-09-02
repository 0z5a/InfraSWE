from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from infraswe.kernel.ada_sm89 import CANONICAL_PLATFORM_CELLS


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _weighted_geometric(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    if set(values) != set(weights):
        raise ValueError("ratio and weight shape sets must match exactly")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("shape weights must sum to one")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("shape weights cannot be negative")
    return math.exp(sum(weights[name] * math.log(values[name]) for name in values))


def unresolved_cross_sku(reason: str, *failure_codes: str) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "status": "unresolved",
        "score_100": None,
        "cross_cell_ranking_allowed": False,
        "deployability_100": None,
        "scoring_authority": "infraswe-scoring-v0.4",
        "formula": "ada-sm89-cross-sku-diagnostic-v0.1",
        "reason": reason,
        "failure_codes": sorted(set(failure_codes)),
    }


def score_cross_sku_reuse(
    cells: Mapping[str, Mapping[str, Any]], *, shape_weights: Mapping[str, float]
) -> dict[str, Any]:
    """Evaluate the RFC v0.1 cross-SKU diagnostic without replacing v0.4 C/U/M.

    Ratios are board-local reference_time / candidate_time values. Missing a board or shape
    makes the diagnostic unresolved; it is never converted to a numeric zero.
    """

    expected_cells = set(CANONICAL_PLATFORM_CELLS)
    if set(cells) != expected_cells:
        return unresolved_cross_sku(
            "both canonical L40S and L20 cells are required",
            "ADA_CROSS_SKU_CELL_MISSING",
        )
    if not shape_weights:
        raise ValueError("shape_weights cannot be empty")

    source_hashes = {str(cell.get("source_subtree_sha256", "")) for cell in cells.values()}
    semantic_hashes = {str(cell.get("semantic_artifact_sha256", "")) for cell in cells.values()}
    codegen_keys = {str(cell.get("codegen_cache_key", "")) for cell in cells.values()}
    tuning_keys = {str(cell.get("board_tuning_cache_key", "")) for cell in cells.values()}
    failures = []
    if "" in source_hashes or len(source_hashes) != 1:
        failures.append("ADA_SHARED_SOURCE_IDENTITY_MISMATCH")
    if "" in semantic_hashes or len(semantic_hashes) != 1:
        failures.append("ADA_SEMANTIC_ARTIFACT_IDENTITY_MISMATCH")
    if "" in codegen_keys or len(codegen_keys) != 1:
        failures.append("ADA_CODEGEN_CACHE_NOT_REUSED")
    if "" in tuning_keys or len(tuning_keys) != len(expected_cells):
        failures.append("ADA_BOARD_TUNING_CACHE_NOT_SEPARATED")
    if failures:
        return unresolved_cross_sku(
            "shared codegen identity or board-local tuning isolation was not established",
            *failures,
        )

    aggregates: dict[str, float] = {}
    ratios_by_cell: dict[str, dict[str, float]] = {}
    production_regressions: list[str] = []
    for cell_name in CANONICAL_PLATFORM_CELLS:
        raw_ratios = cells[cell_name].get("realized_ratios")
        if not isinstance(raw_ratios, Mapping) or set(raw_ratios) != set(shape_weights):
            return unresolved_cross_sku(
                f"{cell_name} is missing the frozen shape portfolio",
                "ADA_CROSS_SKU_SHAPE_PORTFOLIO_MISMATCH",
            )
        ratios = {
            shape: _positive(raw_ratios[shape], f"{cell_name}.{shape}") for shape in shape_weights
        }
        ratios_by_cell[cell_name] = ratios
        aggregates[cell_name] = _weighted_geometric(ratios, shape_weights)
        production_regressions.extend(
            f"{cell_name}:{shape}" for shape, ratio in ratios.items() if ratio < 0.98
        )

    l40s = aggregates["l40s-48gb-pcie"]
    l20 = aggregates["l20-48gb-pcie"]
    geometric = math.sqrt(l40s * l20)
    floor = min(l40s, l20)
    diagnostic = 0.5 * geometric + 0.5 * floor
    deployable = not production_regressions
    return {
        "schema_version": "0.1",
        "status": "diagnostic" if deployable else "not_deployable",
        "score_100": 100 * min(1.0, diagnostic),
        "cross_cell_ranking_allowed": False,
        "deployability_100": None,
        "scoring_authority": "infraswe-scoring-v0.4",
        "formula": "ada-sm89-cross-sku-diagnostic-v0.1",
        "cell_aggregates": aggregates,
        "geometric_mean": geometric,
        "worst_cell": floor,
        "unclamped_realized_ratio": diagnostic,
        "realized_ratios": ratios_by_cell,
        "production_ratio_floor": 0.98,
        "production_regressions": production_regressions,
        "failure_codes": (
            ["ADA_CROSS_SKU_PRODUCTION_REGRESSION"] if production_regressions else []
        ),
        "reason": (
            "architecture-local reuse diagnostic; global Deployability-100 remains v0.4 C/U/M"
        ),
    }


def architecture_overlay_score(components: Mapping[str, float | None]) -> dict[str, Any]:
    """Render the Ada RFC overlay strictly as a non-global cell diagnostic."""

    weights = {
        "concurrency_stability": 0.30,
        "cross_sku_reuse": 0.20,
        "maintainability": 0.15,
        "production_realization": 0.15,
        "local_efficiency": 0.10,
        "compile_runtime": 0.05,
        "evidence": 0.05,
    }
    if set(components) != set(weights):
        raise ValueError("Ada overlay components must match the frozen diagnostic template")
    missing = sorted(name for name, value in components.items() if value is None)
    if missing:
        return {
            "status": "unresolved",
            "score_100": None,
            "missing_components": missing,
            "cross_cell_ranking_allowed": False,
            "deployability_100": None,
            "formula": "ada-artifact-overlay-v0.1-diagnostic",
        }
    values = {name: float(value) for name, value in components.items() if value is not None}
    for name, value in values.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    return {
        "status": "diagnostic",
        "score_100": 100 * sum(weights[name] * values[name] for name in weights),
        "components": values,
        "weights": weights,
        "cross_cell_ranking_allowed": False,
        "deployability_100": None,
        "formula": "ada-artifact-overlay-v0.1-diagnostic",
    }
