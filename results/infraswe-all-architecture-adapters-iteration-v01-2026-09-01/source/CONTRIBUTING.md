# Contributing tasks

A publishable task must demonstrate a stable base failure, a trusted solution passing every fresh
replay, a non-empty regression oracle, and a complete artifact/evidence manifest. Run:

```bash
uv run infraswe task validate tasks/<task-id>
uv run infraswe task certify tasks/<task-id> --executor docker
```

Keep `tests/` and `solution/` out of the agent image. Pin base commits, container images, workloads,
and datasets by digest. Verifiers must score externally observable behavior and must not compare a
candidate patch to the reference patch.

