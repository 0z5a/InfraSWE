from __future__ import annotations

from infraswe.schema import schema_documents, stale_schema_names


def test_checked_in_protocol_schemas_match_models(project_root) -> None:
    assert stale_schema_names(project_root / "schemas") == []
    task_defs = schema_documents()["task.schema.json"]["$defs"]
    result_defs = schema_documents()["result.schema.json"]["$defs"]
    assert "KernelScoringConfig" in task_defs
    assert "ConcurrencyConfig" in task_defs
    assert "DeployabilityScore" in result_defs
    assert "CellEfficiencyScore" in result_defs
    assert "TrainingWorkloadConfig" in task_defs
    assert "TrainingProfilingConfig" in task_defs
    assert "training-evidence.schema.json" in schema_documents()
    assert "training-cert.schema.json" in schema_documents()
    assert "training-result.schema.json" in schema_documents()
    assert "target-project-profile-v0.5.schema.json" in schema_documents()
    assert "draft-v0.5.schema.json" in schema_documents()
    assert "sealed-draft-v0.5.schema.json" in schema_documents()
    assert "project-comparison-cell-v0.5.schema.json" in schema_documents()
    assert "affected-case-plan-v0.5.schema.json" in schema_documents()
    assert "default-draft-catalog-v0.5.schema.json" in schema_documents()
    assert "draft-source-resolution-v0.5.schema.json" in schema_documents()
    assert "triton-purity-audit-v0.5.schema.json" in schema_documents()
    assert "project-score-v0.5.schema.json" in schema_documents()
    assert "judge-profile-v0.5.3.schema.json" in schema_documents()
    assert "judge-rubric-v0.5.3.schema.json" in schema_documents()
    assert "judge-input-pack-v0.5.3.schema.json" in schema_documents()
    assert "judge-output-v0.5.3.schema.json" in schema_documents()
    assert "judge-aggregation-v0.5.3.schema.json" in schema_documents()
    assert "judge-score-projection-v0.5.3.schema.json" in schema_documents()
    assert "judge-trust-v0.5.3.schema.json" in schema_documents()
