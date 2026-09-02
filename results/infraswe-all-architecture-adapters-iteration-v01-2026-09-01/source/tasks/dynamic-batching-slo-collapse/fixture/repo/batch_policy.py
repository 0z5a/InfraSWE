from __future__ import annotations

from typing import Any


def schedule_batches(requests: list[dict[str, Any]], config: dict[str, Any]) -> list[list[str]]:
    size = int(config.get("max_batch_size", 32))
    return [
        [str(request.get("id")) for request in requests[index : index + size]]
        for index in range(0, len(requests), size)
    ]
