# Communication-phase regression

This module normalizes collective traces from training and train/serve frameworks, compares a
trusted baseline with a candidate inside one exact benchmark cell, and emits one
`SystemPathLoadCell` for the existing concurrent-stability scorer. It does not create a new global
communication score. The result is explicitly marked `within-cell-regression-only`, and
cross-cell ranking is forbidden.

## Evidence contract

Each input is a `CommunicationPhaseTraceSet` containing rank-local records plus step-time and
isolated-latency references. It binds the exact model revision, checkpoint, behavior-policy
state, and topology used by the controlled A/B run. The scheduling-policy label may differ, but
the execution identity may not. `framework` is open-ended so adapters can emit `verl`,
`megatron-core`, `slime`, or another implementation without changing the schema. A comparison is
allowed only when both traces bind the same `cell_identity_sha256`, `workload_sha256`,
`world_size`, and controlled execution identity. Each `step_time_ms` sample must already be
reduced to the slowest-rank/global completion time; rank-local averages are not valid step
evidence.

The trace set also declares a shared `timestamp_domain`, whether GPU starts are
`kernel-observed` or only an `event-bracket`, and a measured clock-synchronization error bound.
Event brackets can be summarized for diagnostics, but only kernel-observed starts can produce a
regression load cell; eligibility events must never be described as exact NCCL kernel starts.
The label is checked against content-addressed timing provenance: profiler-kernel evidence names
the observed kernels, while a CUDA-event adapter remains an event bracket.

Every collective record declares:

- its semantic `operation`, stable `logical_operation_id`, and pair `a`/`b` role;
- a stable `process_group_id`, sorted `process_group_ranks`, and communicator sequence ID;
- API launch, GPU start/end, optional completion, and consumer timestamps;
- bytes, stream, transport, topology class, and requested phase coordinate.

Each rank-local record also has one explicit resource acquisition and one final release event.
The release cannot precede transport completion. Duplicate, missing, orphaned, early, or
byte-mismatched events fail the lifecycle gate. Outstanding bytes and counts are computed from
these events, so a staging resource that remains resident after the GPU interval is not silently
treated as reusable.

`requested_offset_us` is the operation's requested phase coordinate relative to the pair anchor.
Normally A records use zero and B records use the requested B-minus-A offset. The scorer compares
that difference with the median GPU-realized B-minus-A offset. Host timestamps are diagnostic;
the GPU timestamps must come from framework instrumentation or a profiler adapter. InfraSWE does
not synthesize GPU timing on the CPU. A `pair_id` binds one requested A/B phase relationship; an
engine-wave adapter should emit a distinct pair for each distinct requested wave coordinate.

Rank count and topology are data, not policy. Any `world_size >= 2` is accepted. For example, a
four-rank overlap mesh can declare A groups `[0, 1]`, `[2, 3]` and B groups `[0, 2]`, `[1, 3]`.
There is no rank-count-specific branch in validation or scoring.

## Regression outputs

The frozen result exposes six named components from the communication-phase plan:

| Component | Within-cell meaning |
|---|---|
| `comm_phase_sweep` | p95 pair-completion retention |
| `comm_contention_stretch` | p95 isolated-to-contended stretch retention |
| `realized_offset_stability` | GPU-realized offset error against the sealed tolerance |
| `collective_order_safety` | communicator-order safety combined with rank-skew bounds |
| `windowed_scheduler_gain` | step-time retention combined with in-flight window safety |
| `consumer_slack_utilization` | consumer-deadline retention; raw utilization is reported separately |

The component values are regression inputs, not a cross-hardware leaderboard. They map to one
communication `SystemPathLoadCell`; callers must still provide the complete frozen load ladder
and fresh-process replay count to `score_system_path_concurrent_stability`. Fewer than 1,000
completed pairs keep p99 evidence exploratory. A phase sweep is represented as a collection of
per-candidate within-cell results, rather than selecting a best point across incomparable cells.
Candidate-selection evidence is distinct from independent confirmation evidence. A configurable
minimum number of unique process-run IDs may be required; selection samples cannot publish a
load cell as if they were confirmation runs.

Collective order, pair coverage, explicit outstanding-byte/count budgets, step/pair p95,
contention stretch, realized offset error, rank skew, and consumer deadlines are fail-closed.
An A/B pair with no shared rank is also rejected because it does not represent overlapping
process-group contention.
Missing isolated latency, step timing, or required consumer timestamps produces `unresolved`, not
a numeric zero. Fixed millisecond offsets are accepted as benchmark inputs only; this contract
does not promote them into a runtime scheduling abstraction.

The evidence envelope also declares full-run artifact coverage: expected, verified, and
reconstructed unit counts, exact ordering, and a manifest digest. Partial-shard verification is a
valid intermediate state but cannot satisfy a full ordered reconstruction claim.

## Source-annotated hard negatives

`hard-negative-mutations-v0.1.json` records the reviewer follow-up source and the named invariant
for each mutation. The executable tests cover requested-versus-realized offset confusion,
event-bracket provenance relabeling, locally faster A with worse pair/consumer outcomes, missing
rank or communicator sequence, cross-topology/policy/checkpoint identity, independent count and
byte budgets, duplicate/missing resource release, partial-shard coverage, and selection-only
performance evidence.

The suite uses only logical 2-rank and 4-rank fixtures. Killing these mutants demonstrates
coverage of those declared checks only; it does not estimate overall agent precision/recall,
production reliability, GPU/network performance, or arbitrary topology support.

## CLI

```bash
uv run infraswe communication phase-regression \
  --baseline baseline-trace-set.json \
  --candidate candidate-trace-set.json \
  --policy regression-policy.json \
  --regime normal \
  --load-ratio 0.5 \
  --output communication-phase-regression.json
```

The command exits 0 for pass, 1 for a measured regression, and 2 for invalid or unresolved
evidence. The trace record, trace set, policy, and result schemas are checked into `schemas/`.
