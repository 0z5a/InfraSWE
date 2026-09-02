# InfraSWE

InfraSWE is an executable benchmark harness for AI infrastructure agents. A run is not
considered complete when an agent merely emits a patch: the patch is collected as a declared
artifact, the agent environment is destroyed, and the artifact is applied in fresh verifier
environments where functional, SLO, fault-recovery, policy, and evidence checks run.

This repository implements the v0.1 protocol core, the original 16-package release matrix, an
experimental cross-framework training package governed by the v0.4 scoring envelope, and the
first executable reference slice of the v0.1 task/artifact/capability trust boundary.

## Implemented task packages

| Task | Track | Execution scope |
|---|---|---|
| `cuda-artifact-capability-selection` | Build & packaging | 1×SM80 capability/ABI selection with real CUDA evidence |
| `container-image-dependency-lock` | Build & packaging | Hermetic CPU image/dependency lock resolution |
| `gpu-resource-request-contract` | Deploy & configuration | Hermetic CPU GPU admission contract |
| `gpu-service-rollout-regression` | Deploy & configuration | Hermetic CPU control-plane model |
| `health-probe-drain-regression` | Deploy & configuration | Hermetic CPU readiness/drain rollout contract |
| `telemetry-root-cause-correlation` | Observability | Hermetic CPU multi-signal root-cause correlation |
| `cuda-extension-arch-target` | Build & packaging | 1×SM80 native extension architecture targeting |
| `dynamic-batching-slo-collapse` | Inference performance | 1×SM80 deadline-aware batching with real CUDA probes |
| `numa-cpu-affinity-regression` | Inference performance | 1×SM80 PCI/NUMA CPU affinity with a pinned CUDA probe |
| `gpu-oom-worker-recovery` | Reliability | 1×SM80 real CUDA OOM injection and worker recovery |
| `nccl-topology-silent-fallback-2gpu` | Distributed communication | Experimental 2×SM80 task with a real NCCL collective |
| `kv-aware-routing-cache-collapse-2gpu` | Inference performance | Experimental 2×SM80 task with real CUDA probes and a deterministic routing workload |
| `tensor-parallel-shard-contract` | Inference performance | Experimental 2×SM120 NCCL shard reconstruction |
| `collective-order-rank-divergence` | Distributed communication | Experimental 2×SM120 canonical collective scheduling and NCCL execution |
| `collective-compute-overlap-regression` | Distributed communication | Experimental 2×SM120 serial/async-stream NCCL comparison |
| `rank-exit-collective-recovery` | Reliability | Experimental 2×SM120 real rank-exit injection and group rebuild |
| `training-sft-cross-framework-v1` | Training | Hermetic SFT adapter contract; runtime/cell evidence remains external |

Every package has a noop base case, a trusted solution, hidden behavioral checks, fault evidence,
and three fresh verifier replays. GPU packages require a host satisfying their declared hardware
profile and pinned agent/verifier images.

The rollout task remains a hermetic control-plane model rather than live k3s, the original NCCL
task uses the available 2×A100 profile rather than the design draft's formal 4×A100 topology, and
the KV-routing task uses a deterministic semantic workload rather than a deployed inference
service. Production agent adapters and provider-backed lease lifecycle certification remain
outside this checkpoint.

The kernel-frontier evaluator also includes an experimental one-device AMD profile for
MI300X (`gfx942`) with PyTorch 2.4.0 / ROCm 6.1. Its initial AOTriton and portable Triton
adapter is documented in
[`benchmarks/kernel_frontier/MI300X_ROCM61.md`](benchmarks/kernel_frontier/MI300X_ROCM61.md);
it remains outside the default leaderboard until real-hardware evidence is collected.

A second experimental adapter registers NVIDIA B200 (`sm100`, compute capability 10.0)
with a CUDA 13.3 / PTX 9.3 compiler contract. It separates generic, family-specific,
and architecture-specific targets and fail-closes TMEM/TCGen05, Cluster Launch Control,
and Blackwell TMA evidence. See
[`benchmarks/kernel_frontier/BLACKWELL_B200.md`](benchmarks/kernel_frontier/BLACKWELL_B200.md).
Its initial release reports native evidence coverage only; performance scores remain N/A.

The experimental NVIDIA Ada adapter implements one shared `sm_89` codegen/kernel core for
L40S and L20, while keeping `l40s-48gb-pcie` and `l20-48gb-pcie` as separate local calibration
and autotune cells. It emits native cubin plus `compute_89` PTX evidence and fail-closes Hopper
or Blackwell-only TMA, WGMMA, cluster, TMEM/TCGen05, multimem, and FP4 paths. See
[`platforms/nvidia-ada-sm89/README.md`](platforms/nvidia-ada-sm89/README.md). Its architecture
overlay is diagnostic only; global scoring continues to use the v0.4 C/U/M formula and seven
fresh-process replays for official adapter evidence.

The initial cross-framework training track adds a framework-neutral `TrainingAdapter`, layered
task/evidence/result schemas, a lazy native-PyTorch reference adapter, and hermetic semantic
oracles for SFT, fixed-rollout GRPO, the full DAPO component contract, and Muon parameter
grouping. Its negative controls cover packing leakage, loss normalization, RNG resume, rollout
grouping, policy staleness, incomplete DAPO, wrong Muon groups, silent fallback, deadlock, and
resource leaks. See
[`benchmarks/training_cross_framework/README.md`](benchmarks/training_cross_framework/README.md).
Frameworks without an installed and verified adapter remain `protocol-supported`; no local
fixture is promoted to a hardware-cell certification.

Training scoring resolves draft conflicts in favor of v0.4: the global score is always
`100*C^0.45*U^0.30*M^0.25`, at least five fresh processes are required (seven recommended),
missing evidence is unresolved rather than zero, and SOL/memory data stays inside one hardware
cell. Training G2 maps to v0.4 E1 framework evidence; an official Deployability-100 therefore
requires at least a G3 system trace (v0.4 E2).

The v0.5 Draft layer is implemented as a separate versioned envelope and does not rewrite v0.4
scores. Draft sources resolve in the frozen order `local > remote Git > built-in defaults`. When
neither explicit source is configured, the default catalog provides independent proposed profiles
for vLLM, SGLang, FlashAttention, FlashInfer, CUTLASS/CuTe, Liger-Kernel, DeepGEMM,
Megatron-Core, TorchTitan, and verl. Exact project aliases select a profile; an
otherwise ambiguous candidate falls back to vLLM by frozen catalog order and emits an audit flag.
All upstream source revisions are pinned, cross-project ProjectFit ranking is forbidden, missing
official evidence remains unresolved, and a human project-maintainer review is required before
Seal. See [`catalog/default-drafts-v0.5/README.md`](catalog/default-drafts-v0.5/README.md).

Default references are role-separated in
[`catalog/default-candidates-v0.5/README.md`](catalog/default-candidates-v0.5/README.md): oracle,
peer implementation, host, workload source, and coverage target. Resolution is metadata-only and
uses frozen first-match rules without weights. No candidate is imported or compiled during
selection; only an explicitly activated selected peer may enter the Draft precompile phase, which
is measured outside cold-start and steady-state benchmark timing.

The v0.5.1 precedent layer builds a deterministic SQLite/FTS5 index over immutable repository and
history snapshots. Candidate footprints bind symbols, build/config anchors, lifecycle surfaces,
and communication or memory-tier semantics to a versioned exact/graph/failure/negative query
plan. Leakage, conflicts, coverage, proposed rules, and RetrievalTrust are sealed into an audited
bundle; retrieval rank and trust never affect the candidate score.

Communication and memory-tiering use system-path Drafts rather than new global score axes. Their
goodput, tail, jitter, progress, resource, and fairness evidence feeds the single concurrent
stability component; Operational Fit is an identity projection of that evidence. Raw bandwidth
and transfer metrics remain cell-local cards. The catalog includes ten concrete communication
profiles, five concrete memory-object profiles, and one deliberately unsealable memory-tier parent.

The v0.5.3 Judge layer implements a fail-closed offline trust core for LLM-as-a-Judge. It seals
exact model identities, human-reviewed criterion ownership, calibration and drift evidence, a
multi-family panel policy, and a content-addressed input pack with identity/score blindness,
secret scanning, and explicit untrusted-candidate boundaries. Strict structured outputs must
resolve authoritative evidence references before weighted-median aggregation. Only pre-sealed
semantic residual criteria inside `P`, `M`, or `U` can be projected, under component weight caps;
there is intentionally no `LLMJudge-100`, and Judge evidence cannot change InfraCert, `C`, `R`,
`O`, `X`, or cell-local performance. This checkpoint supplies no hosted-model caller and cannot
claim an official Judge score without a real pinned, calibrated two-family panel.

The v0.1 trust-boundary slice adds TaskSpecification/AcceptanceContract/FeasibilityWitness
qualification, content-addressed candidate artifact transport into pristine verification,
evidence-authority tracing, capability proof resolution, resource/topology contracts, leases, and
Benchmark Cell comparability. It is a reference protocol/controller implementation, not yet a
production cloud scheduler or container-enforcement plane. See
[`TRUST_BOUNDARY_RFC_IMPLEMENTATION_STATUS_20260902_ZH.md`](TRUST_BOUNDARY_RFC_IMPLEMENTATION_STATUS_20260902_ZH.md).

Formal mergeability still requires an official, sealed ProjectFit before an `>=85` acceptance
claim. Historical PR calibration does not synthesize that score from coarse static features: the
R8 polarized 30-PR cohort is retained as a negative control after producing the same score for all
30 cases and matching only 13 outcomes. Generic historical evaluation now defaults to the R4
ordered, case-specific contract route without a numeric score; R5 remains an explicit replay mode
for old locks only. The five-case R9 follow-up is documented in
[`results/historical-pr-blind-20260901/supplemental-r9/REPORT_20260902_ZH.md`](results/historical-pr-blind-20260901/supplemental-r9/REPORT_20260902_ZH.md).
The post-release ten-case R10 follow-up is documented in
[`results/historical-pr-blind-20260901/supplemental-r10/REPORT_20260902_ZH.md`](results/historical-pr-blind-20260901/supplemental-r10/REPORT_20260902_ZH.md).
The twenty-case R11 repairability follow-up is documented in
[`results/historical-pr-blind-20260901/supplemental-r11/REPORT_20260902_ZH.md`](results/historical-pr-blind-20260901/supplemental-r11/REPORT_20260902_ZH.md).
The twelve-case R12 communication follow-up, including dual-A100 NCCL and exact TorchTitan DDP
evidence, is documented in
[`results/historical-pr-blind-20260901/supplemental-r12/REPORT_20260902_ZH.md`](results/historical-pr-blind-20260901/supplemental-r12/REPORT_20260902_ZH.md).

New decision artifacts use `check` for the narrow state that needs additional verification or a
bounded correction. The former `revise` spelling is accepted only as a legacy input and remains
unchanged inside already hash-frozen historical artifacts.

## Quick start

Python 3.12 is required.

```bash
uv sync --extra dev
uv run infraswe task validate tasks/gpu-service-rollout-regression
uv run infraswe task certify tasks/gpu-service-rollout-regression --executor docker
uv run infraswe report runs/<run-id>
uv run infraswe training probe --output training-capabilities.json
uv run python benchmarks/training_cross_framework/run_minimum_suite.py
uv run infraswe draft defaults --output catalog/default-drafts-v0.5
uv run infraswe draft system-profiles --output catalog/system-drafts-v0.5.2
uv run infraswe draft resolve --candidate candidate.json --output draft-resolution.json
uv run infraswe precedent footprint --request footprint-request.json --source-root .
uv run infraswe precedent index --snapshot snapshot.json --records precedents.jsonl
uv run infraswe precedent plan --snapshot snapshot.json --footprint candidate-footprint.json
uv run infraswe precedent retrieve --index .infraswe/index.sqlite \
  --footprint candidate-footprint.json --plan query-plan.json --output retrieval
uv run infraswe precedent review-rules --rules retrieval/rule-candidates.json \
  --decisions human-rule-decisions.jsonl
uv run infraswe precedent audit-bundle retrieval/retrieval-bundle.json
uv run infraswe judge profile validate judge-profile.yaml \
  --calibration calibration-report.json --drift drift-sentinel.json
uv run infraswe judge cell seal --profile judge-profile.yaml --rubric judge-rubric.yaml \
  --calibration calibration-report.json --drift drift-sentinel.json
uv run infraswe judge pack build --spec judge-input-spec.yaml \
  --source-root trial --output trial/judge/input-pack
uv run infraswe judge validate-output raw-judge-output.json --profile judge-profile.yaml \
  --cell judge-cell.json --rubric judge-rubric.yaml \
  --pack trial/judge/input-pack/manifest.json --member-id member-a \
  --repetition 1 --decoding-seed 1
uv run infraswe judge aggregate --runs trial/judge/runs --profile judge-profile.yaml \
  --cell judge-cell.json --rubric judge-rubric.yaml \
  --pack trial/judge/input-pack/manifest.json
uv run infraswe task audit-contract --specification task-spec.json \
  --contract acceptance-contract.json --witness-set witness-set.json
uv run infraswe artifact lint-policy artifact-policy.json
uv run infraswe evidence verify evidence-pack.json --trial-seal trial-seal.json
uv run infraswe capability registry-validate capability-registry.json
uv run infraswe capability audit-resolution capability-resolution-v0.1.json
uv run infraswe cell compare cell-a.json cell-b.json
```

The `oracle` baseline should pass three fresh replays. The `noop` baseline should fail the target
assertions and is useful for checking that the task is not vacuous.

## Security boundary

The sample runner uses separate temporary workspaces for the agent and every verifier replay. The
hidden `tests/` and `solution/` directories are never copied into the agent workspace. For release
certification, use `--executor docker`; Docker runs commands with no network, dropped Linux
capabilities, and `no-new-privileges`.

The local executor exists for SDK development and tests. It is logical isolation, not a production
sandbox.
