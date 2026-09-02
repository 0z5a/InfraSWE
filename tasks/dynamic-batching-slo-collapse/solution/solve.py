from __future__ import annotations

import json
from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def schedule_batches(
    requests: list[dict[str, Any]], config: dict[str, Any]
) -> list[list[str]]:
    if not isinstance(requests, list) or any(
        not isinstance(request, dict) for request in requests
    ) or not isinstance(config, dict):
        raise ValueError("requests must be a list of objects and config must be an object")
    if config.get("deadline_aware") is not True or config.get("group_by_model") is not True:
        raise ValueError("deadline and model-aware scheduling must be enabled")
    max_size = _integer(config.get("max_batch_size"), "max_batch_size", minimum=1)
    max_tokens = _integer(config.get("max_batch_tokens"), "max_batch_tokens", minimum=1)
    max_wait = _integer(config.get("max_wait_ms"), "max_wait_ms")
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for request in requests:
        identifier = request.get("id")
        model = request.get("model")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("request IDs must be unique non-empty strings")
        if not isinstance(model, str) or not model:
            raise ValueError("request.model must be a non-empty string")
        arrival = _integer(request.get("arrival_ms"), "arrival_ms")
        deadline = _integer(request.get("deadline_ms"), "deadline_ms")
        tokens = _integer(request.get("tokens"), "tokens", minimum=1)
        if deadline < arrival or tokens > max_tokens:
            raise ValueError("request deadline or token budget is infeasible")
        identifiers.add(identifier)
        normalized.append(
            {
                "arrival_ms": arrival,
                "deadline_ms": deadline,
                "id": identifier,
                "model": model,
                "tokens": tokens,
            }
        )
    ordered = sorted(
        normalized,
        key=lambda request: (
            request["deadline_ms"],
            request["arrival_ms"],
            request["id"],
        ),
    )
    batches: list[list[dict[str, Any]]] = []
    for request in ordered:
        placed = False
        for batch in batches:
            arrivals = [item["arrival_ms"] for item in batch]
            if (
                batch[0]["model"] == request["model"]
                and len(batch) < max_size
                and sum(item["tokens"] for item in batch) + request["tokens"] <= max_tokens
                and max([*arrivals, request["arrival_ms"]])
                - min([*arrivals, request["arrival_ms"]])
                <= max_wait
            ):
                batch.append(request)
                placed = True
                break
        if not placed:
            batches.append([request])
    return [[request["id"] for request in batch] for batch in batches]
"""

config = {
    "deadline_aware": True,
    "group_by_model": True,
    "max_batch_size": 4,
    "max_batch_tokens": 1024,
    "max_wait_ms": 5,
}

Path("batch_policy.py").write_text(source, encoding="utf-8")
Path("serving_config.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
