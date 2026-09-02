from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GB10Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "fail", "unresolved", "not_applicable"]
    evidence: dict[str, Any] = Field(default_factory=dict)
    failure_codes: list[str] = Field(default_factory=list)


class GB10CapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    generated_at: str
    probe_version: str
    profile_id: str
    status: Literal["ready", "partial", "not_ready"]
    capability_fingerprint: str
    hardware: dict[str, Any]
    toolchain: dict[str, Any]
    runtime_attributes: dict[str, Any]
    topology: dict[str, Any]
    features: dict[str, dict[str, Any]]
    gates: dict[str, GB10Gate]
    contract_sha256: str
    contract: dict[str, Any]
    failure_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
