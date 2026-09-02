# Cross-framework training adapter v0.1

This directory provides architecture-neutral capability, semantic, and scoring-boundary probes
for SFT, fixed-rollout GRPO, DAPO component contracts, and Muon parameter grouping.

The minimum suite is hermetic and uses synthetic tiny fixtures. A passing suite proves that the
protocol accepts valid fixtures and rejects its negative controls; it does **not** constitute a
framework or hardware-cell certification. Real evidence is validated with
`validate_external_evidence.py`.

Training profiler grades map to the v0.4 authority as follows: G0→E0, G1/G2→E1, G3→E2, and
G4→E3. Consequently G2 alone cannot issue Deployability-100, even though the training RFC draft
describes it as a minimum. Cell SOL/memory evidence still requires G4 and remains cell-local.
