from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("adapter_manifest.json")
    if not path.is_file():
        print("target.adapter-manifest: fail")
        print("regression.semantic-contract: pass")
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "algorithm": "sft",
        "loss_reduction": "valid-target-token-mean",
        "packing_boundary_attention": "forbidden",
        "optimizer_step_boundary": "exact",
        "silent_fallback_count": 0,
        "fresh_process_resume": True,
    }
    passed = all(payload.get(name) == value for name, value in expected.items())
    required_checkpoint = {
        "weights",
        "optimizer",
        "scheduler",
        "data_cursor",
        "data_rng",
        "dropout_rng",
    }
    passed = passed and required_checkpoint <= set(payload.get("checkpoint_components", []))
    print(f"target.adapter-manifest: {'pass' if passed else 'fail'}")
    print("regression.semantic-contract: pass")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
