from __future__ import annotations


def parse_prometheus_scalar(text: str, metric: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, raw = line.partition(" ")
        if name == metric and raw:
            return float(raw)
    return None
