"""Loss-aware import of native framework traces, without manufacturing evidence.

Megatron event traces and slime trainer timelines are not interchangeable with
kernel captures. The import manifest supplies study annotations and separately
captured run/lifecycle metadata; it cannot replace native timestamps. Aggregate
verl sweep summaries are recognized but cannot reconstruct per-rank records.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from infraswe.models.communication_phase import (
    CommunicationPhaseModel,
    CommunicationPhaseTraceRecord,
    CommunicationPhaseTraceSet,
    CommunicationResourceLifecycleEvent,
)
from infraswe.models.draft import Digest


class NativeTraceImportManifest(CommunicationPhaseModel):
    """Content-bound study annotations, never a source of replacement GPU times."""

    schema_version: Literal["0.1"] = "0.1"
    source_artifacts: list[Digest] = Field(min_length=1)
    companion_artifacts: list[Digest] = Field(min_length=1)
    trace_set_metadata: dict[str, Any]
    record_bindings: dict[str, dict[str, Any]]


class NativeTraceImportIssue(CommunicationPhaseModel):
    source_ref: str
    code: str
    detail: str


class NativeTraceImportReport(CommunicationPhaseModel):
    """An unresolved import is useful diagnostics, not a scorable trace set."""

    schema_version: Literal["0.1"] = "0.1"
    framework: Literal["megatron", "slime", "verl"]
    status: Literal["ready", "unresolved"] = "unresolved"
    source_artifacts: list[Digest]
    companion_artifacts: list[Digest]
    manifest_sha256: Digest | None = None
    selected_policy_id: str | None = None
    observed_record_count: int = 0
    auxiliary_operations: dict[str, int] = Field(default_factory=dict)
    issues: list[NativeTraceImportIssue] = Field(default_factory=list)
    partial_records: list[dict[str, Any]] = Field(default_factory=list)
    source_resource_lifecycle_events: list[CommunicationResourceLifecycleEvent] = Field(
        default_factory=list
    )
    trace_set: CommunicationPhaseTraceSet | None = None


_BINDING_FIELDS = {
    "pair_id",
    "pair_role",
    "process_group_ranks",
    "process_group_id",
    "communicator_sequence_id",
    "direction",
    "topology_class",
}
_NATIVE_FIELDS = {
    "pair_id",
    "pair_role",
    "run_id",
    "rank",
    "world_size",
    "local_rank",
    "microbatch",
    "layer",
    "direction",
    "operation",
    "logical_operation_id",
    "process_group_id",
    "process_group_ranks",
    "communicator_sequence_id",
    "message_bytes",
    "api_launch_timestamp_ns",
    "api_return_timestamp_ns",
    "gpu_start_timestamp_ns",
    "gpu_end_timestamp_ns",
    "completion_timestamp_ns",
    "consumer_timestamp_ns",
    "transport",
    "topology_class",
}


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_artifacts(paths: list[Path]) -> dict[str, bytes]:
    artifacts = {}
    for path in paths:
        payload = path.read_bytes()
        digest = _digest(payload)
        if digest in artifacts:
            raise ValueError(f"duplicate artifact content: {path}")
        artifacts[digest] = payload
    return artifacts


def import_native_communication_trace(
    framework: Literal["megatron", "slime", "verl"],
    sources: list[Path],
    *,
    manifest_path: Path | None = None,
    companion_paths: list[Path] | None = None,
    policy_id: str | None = None,
) -> NativeTraceImportReport:
    """Normalize source fields and explain every missing certification boundary.

    JSONL references are ``sha256:<digest>:<one-based-line>``. Native compute,
    conversion and engine-ack spans remain auxiliary observations; they never
    become GPU collective completion or resource-release evidence. "ready"
    means importable, not a passing regression or automatic-policy approval.
    """
    if framework not in {"megatron", "slime", "verl"} or not sources:
        raise ValueError("a supported framework and at least one source are required")
    if policy_id is not None and (framework != "verl" or not policy_id):
        raise ValueError("policy selection requires a nonempty verl policy ID")
    artifacts = _read_artifacts(sources)
    companions = _read_artifacts(companion_paths or [])
    if set(artifacts) & set(companions):
        raise ValueError("companion evidence must be distinct from native source artifacts")
    report = NativeTraceImportReport(
        framework=framework,
        source_artifacts=sorted(artifacts),
        companion_artifacts=sorted(companions),
        selected_policy_id=policy_id,
    )

    def issue(ref: str, code: str, detail: str) -> None:
        report.issues.append(NativeTraceImportIssue(source_ref=ref, code=code, detail=detail))

    manifest = None
    if manifest_path is not None:
        raw_manifest = manifest_path.read_bytes()
        report.manifest_sha256 = _digest(raw_manifest)
        manifest = NativeTraceImportManifest.model_validate_json(raw_manifest)
        if sorted(manifest.source_artifacts) != sorted(artifacts):
            raise ValueError("manifest source digests do not match exact input artifacts")
        if sorted(manifest.companion_artifacts) != sorted(companions):
            raise ValueError("manifest companion digests do not match loaded evidence artifacts")
    else:
        issue(
            "run",
            "missing_manifest",
            "Need content-bound study, clock, lifecycle and run identity evidence",
        )

    auxiliary: Counter[str] = Counter()
    consumed_bindings = set()
    normalized = []
    native_domains = set()
    native_policies = set()
    native_launches = set()
    for digest, payload in artifacts.items():
        if framework == "verl":
            try:
                summary = json.loads(payload)
            except json.JSONDecodeError:
                summary = None  # Multiple native JSONL records are parsed below.
            if isinstance(summary, dict) and summary.get("schema_version") == 2:
                if summary.get("framework") != "verl":
                    raise ValueError("unsupported verl phase-sweep summary identity")
                report.observed_record_count += 1
                issue(
                    digest,
                    "aggregate_only",
                    "Phase-sweep summaries cannot reconstruct per-rank evidence",
                )
                continue
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            ref = f"{digest}:{line_number}"
            raw = json.loads(line)
            report.observed_record_count += 1
            if (
                not isinstance(raw, dict)
                or raw.get("framework") != framework
                or type(raw.get("schema_version")) is not int
                or raw.get("schema_version") not in ({3} if framework == "verl" else {1, 2})
            ):
                raise ValueError(f"unsupported native trace identity/schema at {ref}")
            operation = raw.get("operation")
            if not isinstance(operation, str) or not operation:
                raise ValueError(f"missing native operation at {ref}")
            if raw.get("status", "ok") != "ok":
                issue(ref, "failed_native_span", str(raw.get("status")))
            if framework == "verl":
                if raw.get("record_type") != "collective" or raw.get("sample_phase") not in {
                    "warmup",
                    "measurement",
                }:
                    raise ValueError(f"unsupported verl raw record type or sample phase at {ref}")
                if not isinstance(raw.get("policy_id"), str) or not raw["policy_id"]:
                    raise ValueError(f"missing verl policy ID at {ref}")
                if raw["sample_phase"] == "warmup":
                    auxiliary[f"warmup/{raw['policy_id']}/{operation}"] += 1
                    continue
                if policy_id is not None and raw["policy_id"] != policy_id:
                    auxiliary[f"unselected/{raw['policy_id']}/{operation}"] += 1
                    continue
                native_policies.add(raw["policy_id"])
                launch_id = raw.get("process_launch_id")
                if not isinstance(launch_id, str) or not launch_id:
                    issue(
                        ref,
                        "missing_process_launch_id",
                        "Native process launch identity is required",
                    )
                else:
                    native_launches.add(launch_id)
            collective = framework == "verl" or (
                raw.get("process_group_id") is not None
                if framework in {"megatron", "verl"}
                else operation == "weight_bucket_send" and raw.get("transport") == "nccl"
            )
            if not collective:
                auxiliary[operation] += 1
                continue
            native_domains.add(raw.get("timestamp_domain"))
            if raw.get("gpu_timestamp_semantics") != "event-bracket":
                issue(
                    ref,
                    "unknown_timing_semantics",
                    "Native adapter only supports explicitly declared CUDA event brackets",
                )
            record = {name: raw[name] for name in _NATIVE_FIELDS if raw.get(name) is not None}
            record.update(
                framework=framework,
                step=raw.get(
                    {"megatron": "iteration", "slime": "global_step", "verl": "step"}[framework]
                ),
            )
            if raw.get("stream_id") is not None:
                record["stream_id"] = str(raw["stream_id"])
            if raw.get("requested_offset_us") is not None:
                record["requested_offset_us"] = raw["requested_offset_us"]
            binding = manifest.record_bindings.get(ref, {}) if manifest else {}
            if binding:
                consumed_bindings.add(ref)
            if set(binding) - _BINDING_FIELDS:
                raise ValueError(f"binding attempts to replace native evidence at {ref}")
            for key, value in binding.items():
                if key in record and record[key] != value:
                    raise ValueError(f"binding conflicts with native {key} at {ref}")
                record[key] = value
            record["attributes"] = {
                "native_source_ref": ref,
                "native_timestamp_domain": raw.get("timestamp_domain"),
                "native_metadata": raw.get("metadata", {}),
                "native_observer_role": raw.get("role"),
                "native_policy_id": raw.get("policy_id"),
                "native_resource_scope": raw.get("resource_scope"),
                "native_process_launch_id": raw.get("process_launch_id"),
                "native_hostname": raw.get("hostname"),
            }
            report.partial_records.append(record)
            if framework == "verl":
                try:
                    if raw.get("resource_scope") != "persistent-buffer-transfer-lease":
                        raise ValueError("missing native buffer-reuse lease scope")
                    acquire = raw["buffer_reuse_acquire_timestamp_ns"]
                    release = raw["buffer_reuse_release_timestamp_ns"]
                    if type(acquire) is not int or type(release) is not int:
                        raise ValueError("native lease timestamps must be integer observations")
                    if (
                        acquire > raw["api_launch_timestamp_ns"]
                        or release < raw["completion_timestamp_ns"]
                    ):
                        raise ValueError("native buffer lease does not enclose the transfer")
                    if (
                        raw.get("consumer_timestamp_ns") is not None
                        and release < raw["consumer_timestamp_ns"]
                    ):
                        raise ValueError("native buffer lease released before consumer")
                    lifecycle = [
                        CommunicationResourceLifecycleEvent(
                            process_group_id=record["process_group_id"],
                            logical_operation_id=record["logical_operation_id"],
                            rank=record["rank"],
                            event=event,
                            timestamp_ns=timestamp,
                            message_bytes=record["message_bytes"],
                        )
                        for event, timestamp in (("acquire", acquire), ("release", release))
                    ]
                    report.source_resource_lifecycle_events.extend(lifecycle)
                except (KeyError, ValueError, TypeError) as exc:
                    issue(ref, "invalid_native_lifecycle", str(exc))
            try:
                normalized.append(CommunicationPhaseTraceRecord.model_validate(record))
            except ValidationError as exc:
                for error in exc.errors():
                    issue(
                        ref,
                        "missing_or_invalid_record_field",
                        f"{'.'.join(map(str, error['loc']))}: {error['msg']}",
                    )
    report.auxiliary_operations = dict(sorted(auxiliary.items()))
    if len(native_policies) > 1:
        issue(
            "run",
            "mixed_policy_cells",
            "Select one --policy-id; different candidates cannot form one trace set",
        )
    if len(native_launches) > 1:
        issue(
            "run",
            "mixed_process_launches",
            "One native trace set must retain one process invocation",
        )
    if manifest:
        extra = set(manifest.record_bindings) - consumed_bindings
        if extra:
            raise ValueError(f"bindings refer to absent or auxiliary records: {sorted(extra)}")
    if not normalized:
        issue(
            "run",
            "no_complete_collective_records",
            "No complete native collective record is available to score",
        )
    if manifest and not report.issues:
        metadata = dict(manifest.trace_set_metadata)
        if framework == "verl":
            if native_policies != {metadata.get("policy")}:
                raise ValueError("manifest policy differs from the selected native policy cell")
            claimed_launches = metadata.get("experiment_provenance", {}).get(
                "independent_process_run_ids", []
            )
            if set(claimed_launches) != native_launches:
                raise ValueError(
                    "manifest independent process identities differ from native launch IDs"
                )
            observed_lifecycle = [
                event.model_dump(mode="json") for event in report.source_resource_lifecycle_events
            ]
            if (
                "resource_lifecycle_events" in metadata
                and metadata["resource_lifecycle_events"] != observed_lifecycle
            ):
                raise ValueError("manifest cannot replace observed native lifecycle events")
            metadata["resource_lifecycle_events"] = observed_lifecycle
        if {"records", "framework", "gpu_timestamp_semantics"} & metadata.keys():
            raise ValueError(
                "trace-set metadata cannot replace native records, framework or timing semantics"
            )
        if (
            len(native_domains) != 1
            or None in native_domains
            or metadata.get("timestamp_domain") not in native_domains
        ):
            raise ValueError("trace-set timestamp domain must match every native source")
        evidence = metadata.get("evidence_digests", [])
        required = set(artifacts) | set(companions) | {report.manifest_sha256}
        # Bind the manifest without requiring its self-referential digest in its own JSON.
        metadata["evidence_digests"] = sorted(set(evidence) | required)
        loaded = set(artifacts) | set(companions)
        referenced = {
            metadata.get("gpu_timing_provenance", {}).get("artifact_sha256"),
            metadata.get("artifact_coverage", {}).get("manifest_sha256"),
            *metadata.get("experiment_provenance", {}).get(
                "independent_process_artifact_sha256", []
            ),
            *evidence,
        }
        if not referenced <= loaded:
            raise ValueError("trace-set provenance references evidence that was not loaded")
        try:
            trace = CommunicationPhaseTraceSet.model_validate(
                {
                    **metadata,
                    "framework": framework,
                    "gpu_timestamp_semantics": "event-bracket",
                    "records": normalized,
                }
            )
            if {record.rank for record in normalized} != set(range(trace.world_size)):
                raise ValueError("native records do not cover every declared rank")
            report.trace_set = trace
            report.status = "ready"
        except (ValidationError, ValueError) as exc:
            issue("run", "incomplete_run_evidence", str(exc))
    return report
