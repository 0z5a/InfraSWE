# Native communication trace import

`infraswe communication import-native` imports native fields without silently
manufacturing the evidence required by `communication phase-regression`.

```bash
infraswe communication import-native --framework megatron \
  --source rank-0.jsonl --source rank-1.jsonl \
  --manifest import-manifest.json --companion measured-run-evidence.json \
  --output import-report.json --trace-output normalized-trace.json
```

Without a manifest, the command still writes an actionable report with partial
records, missing fields and auxiliary operation counts, then exits 2. It does
not emit a scorable trace. An existing `--trace-output` is never refreshed on
an unresolved import; consumers must check the current report and exit code.

## Native adapters

| Input | Preserved observations | Never inferred |
| --- | --- | --- |
| Megatron JSONL v1/v2 | iteration→step, real group identity/membership when exported, communicator sequence, native timestamps, stream, bytes, semantic context | Pair design, absent membership, clock calibration, release events |
| slime JSONL v1/v2 | global_step→step, NCCL weight-send spans and their observer role | Global span sequence as communicator sequence; engine ACK as GPU completion |
| verl phase-sweep JSON v2 | Recognized aggregate artifact, exact content digest | Per-rank trace reconstructed from percentiles; nonexistent consumers or lifetimes |
| verl native JSONL v3 | Exact per-rank group/sequence, API and physical completion, payload consumer, persistent-buffer transfer leases | Gloo host time as GPU time; warmup or another offset policy as confirmation |

Compute, conversion, engine receive/load observations are auxiliary, not
collective records. Old single-GPU timelines remain useful forensic evidence
but cannot satisfy a distributed comparison. Unknown timing semantics or failed
spans remain unresolved. CUDA event brackets are never upgraded to observed
NCCL kernels.

## Manifest contract

The manifest contains `schema_version: "0.1"`, exact `source_artifacts` and
`companion_artifacts` SHA-256 lists, `trace_set_metadata` using the existing
trace-set schema, and `record_bindings` keyed by
`sha256:<source-content-hash>:<one-based-JSONL-line>`.

Bindings may only supply pair ID/role, process-group membership/identity,
communicator sequence, direction and topology class. They cannot replace a
present native value or supply missing payload sizes, GPU/completion/consumer
timestamps. These annotations must come from the separately captured run
evidence, not guesses based on group names or row order. The importer binds
provided evidence by content; it does not independently attest its truth.

Run metadata must retain the native timestamp domain and provide measured clock
uncertainty, workload/model/checkpoint/policy/topology identities, artifact
coverage, independent experiment provenance, step/isolated references and real
resource acquire/release events. Every referenced provenance artifact must be
loaded as a source or companion. The manifest's own digest is automatically
added to the resulting evidence set, avoiding a self-reference in its JSON.

For verl native captures, pass `--policy-id concurrent` (or an exact offset ID,
such as `offset/-1000us`). Warmup and unselected candidates are counted as
auxiliary observations, not silently mixed into a scoring cell. Mixed selected
policies remain unresolved. Native buffer-reuse leases are imported directly;
the manifest cannot replace their observed acquire/release timestamps. The
lease scope is retained in record attributes and is not physical allocation/free
or total CUDA memory-residency evidence. Gloo captures retain null GPU times and
remain unresolved for GPU scoring, even when their CPU lifecycle is complete.
The manifest's independent-process IDs must exactly match the observed native
launch ID. A sweep's policy candidates cannot be relabeled as separate process
invocations; mixed or missing launch identities remain unresolved.

`ready` means **schema-importable**, not a passing regression or automatic
policy recommendation. The existing scorer still applies independent-run,
clock, event-vs-kernel, lifecycle, coverage and consumer-deadline gates. Independent
cross-framework GPU/hardware confirmation remains follow-up work; aggregate
summaries are explicitly refused.
