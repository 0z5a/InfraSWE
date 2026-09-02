# Make NCCL transport fallback explicit and topology-aware

The launcher currently requests a preferred transport without checking whether every participating
GPU pair supports it. NCCL can still complete the collective by choosing another path, so the job
looks healthy while its claimed mechanism is false.

Repair `launch_policy.py` so that it:

- chooses a direct peer transport only when the complete peer-access matrix supports it;
- emits a safe shared-memory plan when any required peer edge is unavailable;
- records when and why a fallback was selected;
- emits deterministic NCCL environment settings for the chosen transport;
- handles asymmetric and malformed topology input conservatively.

Preserve the `build_plan(peer_access)` API. Do not bypass the collective or alter verifier files.

