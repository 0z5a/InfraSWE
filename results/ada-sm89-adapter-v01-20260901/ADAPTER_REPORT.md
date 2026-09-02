# InfraSWE NVIDIA L40S/L20 Ada SM89 adapter v0.1

## Release status

```text
implementation: INITIAL_COMPLETE
local_protocol_validation: PASS
sm89_compile_validation: UNRESOLVED_REMOTE_INFRASTRUCTURE
l40s_native_runtime: UNRESOLVED_HARDWARE_NOT_AVAILABLE
l20_native_runtime: UNRESOLVED_HARDWARE_NOT_AVAILABLE
deployability_100: UNRESOLVED_NO_V04_C/U/M_EVIDENCE
```

This is an executable initial adapter, not an L40S/L20 performance certification. The supplied
SSH endpoint accepted TCP connections on port 40665 but closed every connection before sending an
SSH banner / completing key exchange on 2026-09-01. The supplied public key exactly matches the
local private key (`SHA256:svx7m9C/ScoHhIMnsVqJqQKPhi2WeZ0L1dE0x2OtqPU`), so this is recorded as
remote infrastructure unavailability rather than an authentication or candidate failure.

## Implemented architecture contract

- One shared `ada_sm89` codegen/kernel contract.
- Separate `l40s-48gb-pcie` and `l20-48gb-pcie` calibration/autotune cells.
- Exact native target `sm_89` plus `compute_89` PTX fallback.
- Shared codegen cache identity and distinct board-tuning cache identity.
- Fail-closed platform identity based on CC 8.9, recognized product, and 48 GB framebuffer sanity.
- Hard rejection of TMA/bulk tensor copy, WGMMA, CTA clusters/DSM, TMEM, TCGen05, multimem/fabric,
  architecture-only `sm_90a`/`sm_100a`/`sm_120a`, and native FP4/E2M1 paths.

## Initial release feature matrix

| Feature | Initial implementation | Current evidence state |
|---|---|---|
| `SM89-TARGET-001` | native cubin + PTX fallback build, native/JIT dispatch, cold/warm JIT records | local contract PASS; CUDA compile unresolved |
| `SM89-FP8-MMA-001` | E4M3 and E5M2 warp MMA smoke with FP32 accumulators, PTX+HMMA verifier | local contract PASS; native runtime unresolved |
| `SM89-FP8-CVT-001` | packed E4M3x2/E5M2x2 `satfinite` conversion, odd tail guard, amax | local contract PASS; native runtime unresolved |
| `SM89-CPASYNC-001` | 16-byte `cp.async`, commit/wait group, legal tail source size | local contract PASS; native runtime unresolved |
| `SM89-L2-001` | runtime-sized working set, persisting access window, cold/warm measurement | local contract PASS; native runtime unresolved |
| `ADA-CONCURRENCY-001` | v0.4 load-ladder contract and trusted external-evidence validator | workload evidence unresolved |
| `ADA-CROSS-SKU-001` | same-artifact/cache isolation gate, worst-side + geometric diagnostic | second-board evidence unresolved |
| `ADA-TORCHCOMPILE-001` | specialization budget and cross-SKU cache validator | framework evidence unresolved |

## Scoring conflict resolution

The scoring RFC v0.4 is authoritative wherever the architecture RFC conflicts:

```text
Deployability-100 = 100 * C^0.45 * U^0.30 * M^0.25
```

- Ada architecture weights are retained only as a non-global diagnostic overlay.
- Correctness, native-path, liveness, silent fallback, and provenance are hard gates.
- Seven fresh-process replays are used for initial official adapter evidence; fewer than five can
  never produce an official v0.4 stability component, and this adapter does not certify below 7.
- Missing L40S, L20, E2, E3, or task evidence is `UNRESOLVED`, never numeric zero.
- SOL, memory bandwidth, raw latency, and L40S-vs-L20 measurements are local cell scorecards only.

## Local verification

Run at `2026-09-01T09:06:21Z`:

```text
pytest: 126 passed
ruff: all checks passed
schema freshness: 13 schemas fresh
new Ada-focused tests: 12 passed
```

The four initial CUDA native features also have synthetic PTX/SASS negative-control tests for:

- wrong architecture target;
- missing HMMA evidence;
- fewer than seven replays;
- WGMMA, TMA, TCGen05/TMEM, multimem, and FP4/E2M1 contamination;
- dynamic artifact/capability/entry binding;
- L40S/L20 codegen sharing and board-tuning isolation.

## Required hardware closure

Before publishing `KernelCert` or any score, run the checked-in probe and suite on both canonical
cells. At minimum preserve:

1. one L40S and one L20 capability manifest;
2. `sm_89` cubin and `compute_89` PTX/JIT evidence;
3. seven fresh-process replays for native tasks;
4. frozen v0.4 load cells with at least 1,000 completed requests per cell for formal tail scoring;
5. E2 representative concurrency traces and targeted E3 counters;
6. local anchors, power/thermal state, PCIe topology, and independent board autotune fingerprints.

Until those exist, the adapter is correctly reported as implemented but hardware-unresolved.
