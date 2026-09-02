from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdaSM89Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "fail", "unresolved", "not_applicable"]
    evidence: dict[str, Any] = Field(default_factory=dict)
    failure_codes: list[str] = Field(default_factory=list)


class AdaSM89CapabilityManifest(BaseModel):
    """Fail-closed L40S/L20 capability record.

    Board-varying values intentionally remain evidence dictionaries: their exact NVML/CUDA
    availability changes with driver and virtualization mode, while the top-level identity and
    gate semantics stay strict and schema checked.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    generated_at: str
    probe_version: str
    profile_id: str
    status: Literal["ready", "compile_only", "partial", "not_ready"]
    platform_cell: Literal["l40s-48gb-pcie", "l20-48gb-pcie", "generic-sm89"] | None
    capability_fingerprint: str
    platform: dict[str, Any]
    memory: dict[str, Any]
    power_thermal: dict[str, Any]
    interconnect: dict[str, Any]
    virtualization: dict[str, Any]
    software: dict[str, Any]
    features: dict[str, dict[str, Any]]
    calibration: dict[str, Any]
    gates: dict[str, AdaSM89Gate]
    contract_sha256: str
    contract: dict[str, Any]
    failure_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ready_requires_canonical_platform_and_compile_gates(self) -> AdaSM89CapabilityManifest:
        if self.status == "ready":
            if self.platform_cell not in {"l40s-48gb-pcie", "l20-48gb-pcie"}:
                raise ValueError("ready manifests require a canonical L40S or L20 cell")
            if self.gates.get("platform") is None or self.gates["platform"].status != "pass":
                raise ValueError("ready manifests require a passing platform gate")
            if self.gates.get("compile") is None or self.gates["compile"].status != "pass":
                raise ValueError("ready manifests require a passing compile gate")
        if (
            self.status == "compile_only"
            and self.gates.get("compile", None)
            and self.gates["compile"].status != "pass"
        ):
            raise ValueError("compile_only manifests require a passing compile gate")
        return self


class AdaSM89NativeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    verifier: Literal["infraswe-native-sm89"]
    matcher_version: str
    feature_id: Literal[
        "SM89-TARGET-001",
        "SM89-FP8-MMA-001",
        "SM89-FP8-CVT-001",
        "SM89-CPASYNC-001",
    ]
    title: str
    namespace: str
    artifact_set_sha256: str
    capability_fingerprint: str | None
    status: Literal["certified", "static_only", "failed"]
    certified: bool
    gates: dict[str, Any]
    failure_codes: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    tool_commands: list[dict[str, Any]] = Field(default_factory=list)
    replay_count: int | None = Field(default=None, ge=0)
    binary_path: str | None = None
    replays_path: str | None = None
    runtime_reason: str | None = None

    @model_validator(mode="after")
    def certification_matches_status(self) -> AdaSM89NativeResult:
        if self.certified != (self.status == "certified"):
            raise ValueError("only certified status may set certified=true")
        return self


class AdaSM89CrossSKUResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    status: Literal["unresolved", "diagnostic", "not_deployable"]
    score_100: float | None = Field(default=None, ge=0, le=100)
    cross_cell_ranking_allowed: Literal[False] = False
    deployability_100: None = None
    scoring_authority: Literal["infraswe-scoring-v0.4"] = "infraswe-scoring-v0.4"
    formula: Literal["ada-sm89-cross-sku-diagnostic-v0.1"]
    reason: str
    failure_codes: list[str] = Field(default_factory=list)
    cell_aggregates: dict[str, float] | None = None
    geometric_mean: float | None = Field(default=None, ge=0)
    worst_cell: float | None = Field(default=None, ge=0)
    unclamped_realized_ratio: float | None = Field(default=None, ge=0)
    realized_ratios: dict[str, dict[str, float]] | None = None
    production_ratio_floor: float | None = Field(default=None, ge=0)
    production_regressions: list[str] | None = None

    @model_validator(mode="after")
    def score_matches_status(self) -> AdaSM89CrossSKUResult:
        if self.status == "unresolved" and self.score_100 is not None:
            raise ValueError("unresolved cross-SKU evidence cannot carry a score")
        if self.status != "unresolved" and self.score_100 is None:
            raise ValueError(f"{self.status} cross-SKU evidence requires a diagnostic score")
        return self
