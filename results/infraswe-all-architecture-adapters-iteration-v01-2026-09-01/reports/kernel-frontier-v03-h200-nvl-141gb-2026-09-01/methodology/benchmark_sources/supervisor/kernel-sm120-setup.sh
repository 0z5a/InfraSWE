#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail

export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10

python=/venv/main/bin/python
run_root=/workspace/infraswe/runs/kernel-sm120-setup
mkdir -p "${run_root}"

uv pip install \
  --python "${python}" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.8.0'
uv pip install \
  --python "${python}" \
  'flash-attn-4==4.0.0b28'
uv pip install \
  --python "${python}" \
  --target /workspace/infraswe/envs/fa4 \
  --no-deps \
  'flash-attn-4==4.0.0b28'

CUDA_VISIBLE_DEVICES=0 "${python}" - <<'PY'
import json
from pathlib import Path

import torch
import triton

x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
result = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "gpu": torch.cuda.get_device_name(),
    "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
    "matmul_finite": bool(torch.isfinite(y).all().item()),
}
Path("/workspace/infraswe/runs/kernel-sm120-setup/torch-smoke.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, sort_keys=True))
PY

uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
