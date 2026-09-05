"""Native-shaped fixtures are schema tests, not measured hardware evidence."""

from __future__ import annotations

import hashlib
import json

import pytest
from test_communication_phase import _trace
from typer.testing import CliRunner

from infraswe.cli import app
from infraswe.telemetry.communication_native import import_native_communication_trace


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def native_fixture(tmp_path, world, framework="megatron"):
    trace = _trace(
        world_size=world,
        run_id="native",
        framework=framework,
        policy="ordered",
        requested_offset_us=0,
        duration_scale=1,
        step_scale=1,
    )
    native = []
    bindings = []
    for source in trace.records:
        row = source.model_dump(mode="json")
        binding = {name: row.pop(name) for name in ("pair_id", "pair_role", "process_group_ranks")}
        row.pop("attributes")
        row["schema_version"] = 2
        row["iteration" if framework == "megatron" else "global_step"] = row.pop("step")
        row["stream_id"] = 0
        row["gpu_timestamp_semantics"] = "event-bracket"
        row["timestamp_domain"] = (
            "process-monotonic-projected-cuda-event"
            if framework == "megatron"
            else "process-realtime-projected-cuda-event"
        )
        row["clock_sync_error_bound_us"] = None
        if framework == "slime":
            row["operation"] = "weight_bucket_send"
            row["record_type"] = "span"
            row["role"] = "trainer"
            row["sequence_id"] = len(native)  # Deliberately not a communicator sequence.
            for name in (
                "process_group_id",
                "communicator_sequence_id",
                "direction",
                "topology_class",
            ):
                binding[name] = row.pop(name)
        native.append(row)
        bindings.append(binding)
    source_path = tmp_path / "native.jsonl"
    source_bytes = ("\n".join(json.dumps(row) for row in native) + "\n").encode()
    source_path.write_bytes(source_bytes)
    source_digest = digest(source_bytes)
    companion = tmp_path / "measured-companion.json"
    companion.write_text(
        json.dumps(
            {
                "fixture_only": True,
                "clock_error_us": 5,
                "lifecycle": trace.model_dump(mode="json")["resource_lifecycle_events"],
            }
        )
    )
    companion_digest = digest(companion.read_bytes())
    metadata = trace.model_dump(
        mode="json", exclude={"records", "framework", "gpu_timestamp_semantics"}
    )
    metadata["timestamp_domain"] = native[0]["timestamp_domain"]
    metadata["gpu_timing_provenance"] = {
        "capture_kind": "cuda-event-bracket",
        "adapter": f"{framework}-native-v2",
        "artifact_sha256": source_digest,
    }
    metadata["artifact_coverage"]["manifest_sha256"] = companion_digest
    metadata["experiment_provenance"] = {
        "phase": "confirmation",
        "independent_process_run_ids": ["native-process"],
        "independent_process_artifact_sha256": [companion_digest],
    }
    metadata["evidence_digests"] = [source_digest, companion_digest]
    manifest_data = {
        "source_artifacts": [source_digest],
        "companion_artifacts": [companion_digest],
        "trace_set_metadata": metadata,
        "record_bindings": {
            f"{source_digest}:{index + 1}": value for index, value in enumerate(bindings)
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_data))
    return source_path, companion, manifest, native


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("framework", ["megatron", "slime"])
def test_native_fields_and_bound_annotations_reach_trace_schema(tmp_path, world, framework):
    source, companion, manifest, raw = native_fixture(tmp_path, world, framework)
    report = import_native_communication_trace(
        framework, [source], manifest_path=manifest, companion_paths=[companion]
    )
    assert report.status == "ready"
    assert report.trace_set.gpu_timestamp_semantics == "event-bracket"
    assert len(report.trace_set.records) == len(raw)
    for record, original in zip(report.trace_set.records, raw, strict=True):
        assert record.gpu_start_timestamp_ns == original["gpu_start_timestamp_ns"]
        assert record.gpu_end_timestamp_ns == original["gpu_end_timestamp_ns"]
        assert record.consumer_timestamp_ns == original["consumer_timestamp_ns"]
    assert report.manifest_sha256 in report.trace_set.evidence_digests


def test_missing_manifest_returns_actionable_partial_records(tmp_path):
    source, _, _, _ = native_fixture(tmp_path, 2)
    report = import_native_communication_trace("megatron", [source])
    assert report.status == "unresolved"
    assert report.trace_set is None
    assert report.partial_records
    assert any("process_group_ranks" in issue.detail for issue in report.issues)
    assert any(issue.code == "missing_manifest" for issue in report.issues)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_start_timestamp_ns", 0),
        ("consumer_timestamp_ns", 1),
        ("completion_timestamp_ns", 2),
        ("message_bytes", 100),
    ],
)
def test_bindings_cannot_manufacture_timing_or_payload(tmp_path, field, value):
    source, companion, manifest, _ = native_fixture(tmp_path, 2)
    data = json.loads(manifest.read_text())
    next(iter(data["record_bindings"].values()))[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="replace native evidence"):
        import_native_communication_trace(
            "megatron", [source], manifest_path=manifest, companion_paths=[companion]
        )


@pytest.mark.parametrize(
    "mutation",
    ["source", "companion", "domain", "kernel_upgrade", "clock_missing", "unknown_binding"],
)
def test_evidence_corruption_or_missing_clock_never_becomes_ready(tmp_path, mutation):
    source, companion, manifest, _ = native_fixture(tmp_path, 4)
    data = json.loads(manifest.read_text())
    if mutation == "source":
        source.write_text(source.read_text() + "\n")
    elif mutation == "companion":
        companion.write_text("different evidence")
    elif mutation == "domain":
        data["trace_set_metadata"]["timestamp_domain"] = "fake shared clock"
    elif mutation == "kernel_upgrade":
        data["trace_set_metadata"]["gpu_timing_provenance"]["capture_kind"] = "profiler-kernel"
    elif mutation == "clock_missing":
        del data["trace_set_metadata"]["clock_sync_error_bound_us"]
    else:
        data["record_bindings"]["absent:1"] = {"pair_id": "invented"}
    manifest.write_text(json.dumps(data))
    if mutation in {"kernel_upgrade", "clock_missing"}:
        report = import_native_communication_trace(
            "megatron", [source], manifest_path=manifest, companion_paths=[companion]
        )
        assert report.status == "unresolved"
    else:
        with pytest.raises(ValueError):
            import_native_communication_trace(
                "megatron", [source], manifest_path=manifest, companion_paths=[companion]
            )


@pytest.mark.parametrize(
    "operation",
    ["weight_convert", "engine_receive", "engine_load_weights", "train_forward_backward"],
)
def test_slime_ack_and_compute_spans_never_become_collective_completion(tmp_path, operation):
    source = tmp_path / "slime.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "framework": "slime",
                "operation": operation,
                "world_size": 1,
                "completion_timestamp_ns": 123,
            }
        )
        + "\n"
    )
    report = import_native_communication_trace("slime", [source])
    assert report.auxiliary_operations == {operation: 1}
    assert report.partial_records == []
    assert report.trace_set is None


def test_verl_percentiles_cannot_reconstruct_per_rank_timing(tmp_path):
    source = tmp_path / "sweep.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "framework": "verl",
                "world_size": 4,
                "results": [{"policy": "concurrent", "pair_completion_p95_ms": 1.2}],
            }
        )
    )
    report = import_native_communication_trace("verl", [source])
    assert any(issue.code == "aggregate_only" for issue in report.issues)
    assert report.trace_set is None


def test_cli_writes_unresolved_report_but_no_scorable_trace(tmp_path):
    source, _, _, _ = native_fixture(tmp_path, 2)
    report = tmp_path / "report.json"
    trace = tmp_path / "trace.json"
    result = CliRunner().invoke(
        app,
        [
            "communication",
            "import-native",
            "--framework",
            "megatron",
            "--source",
            str(source),
            "--output",
            str(report),
            "--trace-output",
            str(trace),
        ],
    )
    assert result.exit_code == 2, result.output
    assert json.loads(report.read_text())["status"] == "unresolved"
    assert not trace.exists()
