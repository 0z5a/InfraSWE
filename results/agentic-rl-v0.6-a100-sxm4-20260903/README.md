# Agentic RL v0.6 dual-A100 smoke

This directory preserves the fail-closed v0.6 reference-slice smoke run performed on
2026-09-03 with two visible NVIDIA A100-SXM4-40GB devices.

The staged working tree ran `tests/test_agentic_v06.py` with `33 passed`. The capability
preflight was then invoked with `--gpu-count 2` and intentionally exited with protocol code 5.
GPU visibility alone is insufficient for production readiness: topology attestation, enforced
rootless sandboxing, an exact-token gateway, a trainer adapter, and distributed gang enforcement
were all absent. The content-addressed report records those negative capabilities instead of
promoting the node.

The report is a protocol/runtime capability observation, not an official performance result and
not proof of a production Rollout Fabric.
