# InfraSWE Kernel Frontier experiments

This directory contains the evaluator-owned measurement programs used for the
InfraSWE v0.3 kernel scoring calibration.  It deliberately keeps three outputs
separate:

- certification evidence (correctness, provenance, runtime call graph, fresh replay);
- artifact quality (raw latency, speedup, calibrated-target AnchorScore);
- search efficiency (not applicable to these artifact-only upstream references).

## Frozen upstream implementations

| Label | Source | Revision | Intended cell |
|---|---|---|---|
| FA1 | Dao-AILab/flash-attention | `v1.0.9` / `6d48e14` | SM80 |
| FA2 | Dao-AILab/flash-attention | `ce088ab9` | SM80 and SM120 |
| FA3 | `hopper/` in the same repository | `ce088ab9` | SM80 |
| FA4 | `flash-attn-4` | `4.0.0b28` | SM80 and SM120 (measured; explicit b28 dispatch) |

The reproducible container recipes pin the PyTorch base image to digest
`sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6`;
the recorded runs used each instance's `/venv/main` environment, whose full
package, driver, and hardware state is captured in `provenance.json`.
An implementation that cannot execute in a declared hardware cell is reported
as not applicable; scores from different cells are never directly ranked.

## Measurement protocol

- BF16 forward attention and classic-kernel portfolios;
- evaluator-owned CUDA events and completion synchronization;
- randomized ABBA/BAAB matched blocks;
- at least 30 blocks per fresh process replay;
- repetition factors calibrated to at least 50 ms per timed position;
- three fresh-process replays;
- separate profiler run and module/source digest;
- calibrated BF16 GEMM, HBM-copy and launch-floor anchor manifest.

Raw JSON keeps every matched block so the paired estimator and confidence
intervals can be recomputed without trusting the rendered report.

## Negative controls

`remote_run_negative_controls_a100.sh` runs four deliberately bad attention
backends in a result cell that is kept separate from the official FA1–FA4 table:

- `garbage-slow-fa4-waste64` returns the correct FA4 result after 64 useless
  Triton streaming passes, so certification should pass while Artifact-100 falls;
- `garbage-zero-triton` launches a real native kernel but writes only zeros, so
  correctness must force the leaderboard-effective score to zero;
- `garbage-cache-copy` caches the first correct answer per shape and replays it,
  so the evaluator's dynamic-input probe must reject it;
- `fa-garbage-math-fallback` reports an FA-like backend name but executes PyTorch
  SDPA math, so the native Flash trace gate must keep it unscored.

These controls use the same three-replay, matched-block, profiler, and provenance
protocol as the scored candidates. They are evaluator tests, not leaderboard
submissions.
