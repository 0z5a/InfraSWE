from __future__ import annotations

from typing import Any

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    BenchmarkCellManifest,
    BenchmarkCellPolicy,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def _lookup_path(material: dict[str, Any], dotted_path: str) -> Any:
    value: Any = material
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError("BenchmarkCell comparison field is missing: " + dotted_path)
        value = value[part]
    return value


def build_benchmark_cell(
    *,
    policy: BenchmarkCellPolicy,
    task: dict[str, Any],
    runner: dict[str, Any],
    hardware: dict[str, Any],
    software: dict[str, Any],
    execution: dict[str, Any],
    benchmark: dict[str, Any],
) -> BenchmarkCellManifest:
    policy_material = policy.model_dump(mode="json", exclude={"policy_sha256"})
    if policy.policy_sha256 != canonical_sha256(policy_material):
        raise ValueError("BenchmarkCellPolicy digest mismatch")
    full_material = {
        "task": task,
        "runner": runner,
        "hardware": hardware,
        "software": software,
        "execution": execution,
        "benchmark": benchmark,
    }
    overlap = set(policy.comparison_included_fields) & set(policy.comparison_excluded_fields)
    if overlap:
        raise ValueError("BenchmarkCell comparison fields cannot be both included and excluded")
    comparison_material = {
        path: _lookup_path(full_material, path)
        for path in sorted(policy.comparison_included_fields)
    }
    full_digest = canonical_sha256(full_material)
    comparison_digest = canonical_sha256(
        {
            "policy_sha256": policy.policy_sha256,
            "fields": comparison_material,
        }
    )
    cell_id = "cell-" + comparison_digest.removeprefix("sha256:")[:20]
    preliminary = BenchmarkCellManifest(
        cell_id=cell_id,
        policy_sha256=policy.policy_sha256,
        task=task,
        runner=runner,
        hardware=hardware,
        software=software,
        execution=execution,
        benchmark=benchmark,
        full_environment_digest=full_digest,
        comparison_cell_digest=comparison_digest,
        cell_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"cell_sha256"})
    return preliminary.model_copy(update={"cell_sha256": canonical_sha256(material)})


def audit_benchmark_cell(
    cell: BenchmarkCellManifest,
    policy: BenchmarkCellPolicy,
) -> list[str]:
    failures: list[str] = []
    policy_material = policy.model_dump(mode="json", exclude={"policy_sha256"})
    if policy.policy_sha256 != canonical_sha256(policy_material):
        failures.append("BENCHMARK_CELL_POLICY_DIGEST_MISMATCH")
    if cell.policy_sha256 != policy.policy_sha256:
        failures.append("BENCHMARK_CELL_POLICY_BINDING_MISMATCH")
    full_material = {
        "task": cell.task,
        "runner": cell.runner,
        "hardware": cell.hardware,
        "software": cell.software,
        "execution": cell.execution,
        "benchmark": cell.benchmark,
    }
    if cell.full_environment_digest != canonical_sha256(full_material):
        failures.append("BENCHMARK_CELL_FULL_ENVIRONMENT_DIGEST_MISMATCH")
    try:
        comparison_material = {
            path: _lookup_path(full_material, path)
            for path in sorted(policy.comparison_included_fields)
        }
    except ValueError:
        failures.append("BENCHMARK_CELL_COMPARISON_FIELD_MISSING")
    else:
        expected = canonical_sha256(
            {"policy_sha256": policy.policy_sha256, "fields": comparison_material}
        )
        if cell.comparison_cell_digest != expected:
            failures.append("BENCHMARK_CELL_COMPARISON_DIGEST_MISMATCH")
    material = cell.model_dump(mode="json", exclude={"cell_sha256"})
    if cell.cell_sha256 != canonical_sha256(material):
        failures.append("BENCHMARK_CELL_DIGEST_MISMATCH")
    return sorted(set(failures))


def assert_raw_performance_comparable(
    left: BenchmarkCellManifest,
    right: BenchmarkCellManifest,
) -> None:
    if left.policy_sha256 != right.policy_sha256:
        raise ValueError("raw performance comparison requires the same BenchmarkCellPolicy")
    if left.comparison_cell_digest != right.comparison_cell_digest:
        raise ValueError("cross-cell raw performance comparison is forbidden")
