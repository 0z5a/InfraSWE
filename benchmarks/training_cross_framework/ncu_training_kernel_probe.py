from __future__ import annotations

import json


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "unresolved", "reason": "CUDA unavailable"}))
        return 2
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    left = torch.randn((512, 512), device="cuda", requires_grad=True)
    right = torch.randn((512, 512), device="cuda", requires_grad=True)
    for _ in range(3):
        loss = torch.nn.functional.silu(left @ right).square().mean()
        loss.backward()
        left.grad = None
        right.grad = None
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "pass",
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "loss": float(loss.detach().cpu()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
