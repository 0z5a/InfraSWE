from __future__ import annotations

import argparse
import json
from pathlib import Path

from infraswe.io import atomic_write_json
from infraswe.training.probe import probe_capabilities


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe cross-framework training capabilities")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = probe_capabilities().model_dump(mode="json")
    if args.output:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"ready", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
