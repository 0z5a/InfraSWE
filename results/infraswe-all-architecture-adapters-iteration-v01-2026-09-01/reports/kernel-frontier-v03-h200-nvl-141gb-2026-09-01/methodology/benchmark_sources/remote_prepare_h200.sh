#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
max_jobs="${INFRASWE_MAX_JOBS:-24}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
stage_root="${workspace}/runs/kernel-h200-setup"

mkdir -p "${workspace}/envs" "${workspace}/runs" "${stage_root}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/started-at.txt"
nvidia-smi -q -i 0 > "${stage_root}/nvidia-smi-before.txt"

uv pip install --python "${python}" \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${python}" \
  flash-attn-4==4.0.0b28 \
  einops==0.8.2 \
  ninja==1.13.2 \
  numpy==2.5.2 \
  packaging==25.0 \
  psutil==7.2.2 \
  wheel==0.48.0
uv pip install --python "${python}" \
  --target "${workspace}/envs/fa4" \
  --no-deps \
  flash-attn-4==4.0.0b28

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${workspace}/envs/fa4" "${python}" - <<'PY'
import torch
from flash_attn.cute import flash_attn_func

q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
output = flash_attn_func(q, q, q, causal=True)
if isinstance(output, tuple):
    output = output[0]
torch.cuda.synchronize()
print("H200_FA4_SMOKE_OK", tuple(output.shape), torch.cuda.get_device_capability(0))
PY

cd "${benchmark_root}"
if ! PYTHONPATH="${workspace}/envs/fa1" "${python}" -c \
  "import torch, flash_attn, flash_attn_cuda" >/dev/null 2>&1; then
  INFRASWE_REMOTE_ROOT="${workspace}" \
  INFRASWE_PYTHON="${python}" \
  INFRASWE_MAX_JOBS="${max_jobs}" \
  INFRASWE_RESUME_BUILD=1 \
  INFRASWE_FA1_ARCHS=9.0 \
    bash ./remote_build_fa.sh fa1
fi

if ! PYTHONPATH="${workspace}/envs/fa2" "${python}" -c \
  "import torch, flash_attn, flash_attn_2_cuda" >/dev/null 2>&1; then
  INFRASWE_REMOTE_ROOT="${workspace}" \
  INFRASWE_PYTHON="${python}" \
  INFRASWE_MAX_JOBS="${max_jobs}" \
  INFRASWE_RESUME_BUILD=1 \
  INFRASWE_FA2_ARCHS=90 \
    bash ./remote_build_fa.sh fa2
fi

if ! PYTHONPATH="${workspace}/envs/fa3" "${python}" -c \
  "import torch; from flash_attn_3 import _C, flash_attn_interface" >/dev/null 2>&1; then
  INFRASWE_REMOTE_ROOT="${workspace}" \
  INFRASWE_PYTHON="${python}" \
  INFRASWE_MAX_JOBS="${max_jobs}" \
  INFRASWE_RESUME_BUILD=1 \
    bash ./remote_build_fa.sh fa3
fi

for variant in fa1 fa2 fa3 fa4; do
  find "${workspace}/envs/${variant}" -type f -name '*.so' -print \
    > "${stage_root}/${variant}-binaries.txt"
done
uv pip freeze --python "${python}" > "${stage_root}/environment-freeze.txt"
nvidia-smi -q -i 0 > "${stage_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/completed-at.txt"
echo "H200 environment and FA1-FA4 builds complete"
