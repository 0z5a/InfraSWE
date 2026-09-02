# Restore the tensor-parallel shard contract

The planner currently shards every parameter on axis 0, truncates uneven dimensions, and does not
preserve replicated parameters. Repair `shard_policy.py` and `tp_config.json`, preserving
`build_shard_plan(parameters, world_size, config) -> dict`.

Every declared parameter must be covered exactly once. Column-parallel parameters split axis 0,
row-parallel parameters split axis 1, and replicated parameters remain complete on every rank.
Shard intervals must be contiguous, non-overlapping, and cover the full dimension. Require even
partitioning, deterministic output independent of input order, immutable inputs, and fail closed on
malformed, unclassified, duplicated, or indivisible parameters. Do not silently drop tensor data.
