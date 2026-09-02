# Prevent collective-order rank divergence

Ranks currently submit collectives in their local discovery order. When that order differs, the
same sequence number refers to different tensors and NCCL can hang or corrupt results. Repair
`collective_policy.py` and `order_config.json`, preserving
`build_collective_schedule(rank_steps, world_size, config) -> dict`.

Validate that every rank declares the same operation IDs and identical kind/element metadata. If
they do, emit the configured canonical order for every rank, record whether divergence was
detected, and include a deterministic fingerprint. If an operation is missing or metadata differs,
return an explicit blocked plan. Reject malformed inputs, do not mutate them, and never silently
continue with rank-local order.
