# Default candidate registry v0.5

This proposed registry separates five roles: correctness oracle, peer implementation, host
project, workload source, and coverage target. It complements the ten project-conditioned
contract profiles; it does not flatten those projects into one cross-project score.

Selection is an ordered, deterministic metadata operation. It uses no learned model and no
weighted candidate score. Loading the registry never imports or compiles a candidate. After a
Draft selects references, activation defaults to the single primary peer. An adapter may
precompile only that explicitly activated candidate, and the compilation phase stays outside
cold-start and steady-state benchmark timing. Source, compiler, runtime, driver, hardware,
adapter, and environment identity remain part of the evidence-cache boundary.

The executable architecture and timing invariants are documented in
[`ARCHITECTURE_AND_PRECOMPILE_POLICY_zh.md`](ARCHITECTURE_AND_PRECOMPILE_POLICY_zh.md).

Regenerate the registry and schemas with:

```bash
PYTHONPATH=src uv run python -m infraswe draft candidates \
  --output catalog/default-candidates-v0.5
PYTHONPATH=src uv run python -m infraswe schema export --output schemas
```
