from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    payload = {
        "schema_version": "0.1",
        "adapter_id": "org.infraswe/reference-adapter",
        "algorithm": "sft",
        "loss_reduction": "valid-target-token-mean",
        "packing_boundary_attention": "forbidden",
        "optimizer_step_boundary": "exact",
        "checkpoint_components": [
            "weights",
            "optimizer",
            "scheduler",
            "data_cursor",
            "data_rng",
            "dropout_rng",
        ],
        "silent_fallback_count": 0,
        "fresh_process_resume": True,
    }
    output = Path("adapter_manifest.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
