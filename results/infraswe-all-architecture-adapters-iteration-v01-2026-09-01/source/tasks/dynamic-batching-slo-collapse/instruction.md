# Restore deadline-aware dynamic batching

The scheduler waits too long, forms oversized batches, and mixes incompatible models. Throughput
looks high while deadline goodput and tail latency collapse. Repair `batch_policy.py` and
`serving_config.json`, preserving `schedule_batches(requests, config) -> list[list[str]]`.

Every request ID must appear exactly once. Use earliest-deadline-first ordering, group only the same
model, enforce maximum batch size, aggregate token budget, and arrival wait span, and reject a
single request that cannot fit the token budget. Results must be deterministic under request order,
inputs must not be mutated, and no request may be silently dropped.
