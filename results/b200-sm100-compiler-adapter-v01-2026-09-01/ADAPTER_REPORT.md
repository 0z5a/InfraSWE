# InfraSWE B200 / SM100 compiler-feature adapter v0.1

Date: 2026-09-01

## Outcome

The initial B200 compiler-feature adapter is implemented against the frozen
CUDA 13.3 / PTX 9.3 baseline. It adds:

- one experimental B200/SM100 hardware profile;
- separate `sm_100`, `sm_100f`, and `sm_100a` compiler probes;
- five MVP task contracts for TMEM/TCGen05, Cluster Launch Control, irregular
  TMA, native low-bit block scaling, and CTA-pair TMA;
- optional multimem/fabric and disabled PTX preview lanes;
- a versioned PTX/SASS verifier with entry-body reachability screening,
  native-binary checks, fallback rejection, artifact/capability binding,
  watchdog, profiler, correctness, and mutation gates;
- exactly three fresh-process replay records and separated Core, Scheduler,
  Fabric, and Preview report namespaces;
- remote setup, runner, supervisor, JSON/Markdown summary, and ZIP generation.

## Local verification

- `ruff check .`: passed.
- `pytest -q`: 89 passed.
- Three new shell entry points: `bash -n` passed.
- Ten repository JSON/schema documents parsed; all new artifact schemas are
  exercised by tests.

## Evidence boundary

No B200 lease was available while this adapter was authored. There are no
fabricated hardware measurements in this package. Native conformance and all
performance scores remain pending until evaluator-owned kernels and dynamic
evidence are executed on a CUDA 13.3 B200 cell. Certification coverage is not a
performance score, and every `leaderboard_score_100` remains `null` in the
initial adapter.

The attached RFC is retained as reference material only; its embedded prose was
not treated as an independent instruction source. The RFC's PTX 9.4 preview is
not enabled because the stable NVIDIA documentation used for this checkpoint is
PTX ISA 9.3.
