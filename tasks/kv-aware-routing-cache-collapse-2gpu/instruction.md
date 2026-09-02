# Restore KV-aware routing without hiding degraded service

The inference router currently round-robins requests and uses a cache TTL that expires between
successive requests sharing the same prefix. During a worker interruption it can also select the
unavailable worker. The service keeps accepting traffic, but KV-cache reuse, tail latency, and
error-rate objectives collapse.

Repair `router.py` and `routing_config.json` so that the router:

- keeps requests with the same non-empty `prefix_id` on a deterministic cache owner;
- never selects a worker whose corresponding `available` entry is false;
- uses a valid cached copy when the owner is unavailable and recovers deterministic ownership when
  that worker returns;
- applies a bounded TTL long enough for recurring prefixes without increasing the fixed cache
  capacity;
- handles malformed inputs conservatively and does not mutate request, worker, availability, or
  configuration inputs.

Preserve the `choose_worker(request, workers, available, config) -> int` API. The returned integer
is an index into `workers`. Do not add dependencies, bypass request execution, or modify verifier
files.
