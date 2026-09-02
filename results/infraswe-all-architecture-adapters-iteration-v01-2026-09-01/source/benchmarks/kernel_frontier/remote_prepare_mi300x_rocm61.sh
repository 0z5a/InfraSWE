#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
stage_root="${workspace}/runs/kernel-mi300x-rocm61-torch240-setup"
export PATH="/opt/rocm/bin:${PATH}"

mkdir -p "${workspace}/runs" "${stage_root}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/started-at.txt"

command -v rocm-smi >/dev/null
command -v rocminfo >/dev/null
command -v uv >/dev/null
rocm-smi > "${stage_root}/rocm-smi-before.txt"
rocminfo > "${stage_root}/rocminfo.txt"
if command -v hipcc >/dev/null; then
  hipcc --version > "${stage_root}/hipcc-version.txt"
fi

cd "${benchmark_root}"
"${python}" rocm_lease_guard.py \
  --device-index "${gpu}" \
  --output "${stage_root}/exclusive-device.json"

uv pip install --python "${python}" \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/rocm6.1
uv pip install --python "${python}" \
  einops==0.8.0 \
  numpy==1.26.4 \
  packaging==24.1 \
  psutil==6.0.0
uv pip install --python "${python}" --editable "${workspace}"

PYTHONPATH="${workspace}/src" "${python}" -m infraswe lease preflight \
  --profile gpu-1x-gfx942-mi300x-rocm61 \
  --profiles-dir "${workspace}/profiles" \
  --output "${stage_root}/hardware-manifest.json"

ROCR_VISIBLE_DEVICES="${gpu}" HIP_VISIBLE_DEVICES="${gpu}" \
PYTHONPATH="${benchmark_root}" "${python}" - "${stage_root}/stack-smoke.json" <<'PY'
import json
import re
import sys
from pathlib import Path

import torch
import triton
from attention_bench import load_adapter
from bench_utils import atomic_write_json, hardware_manifest, profiler_evidence, utc_now
import triton_kernels

output = Path(sys.argv[1])
version = torch.__version__.split("+")[0]
hip = str(getattr(torch.version, "hip", "") or "")
properties = torch.cuda.get_device_properties(0)
match = re.search(r"gfx[0-9a-z]+", str(getattr(properties, "gcnArchName", "")).lower())
architecture = match.group(0) if match else None
assert version == "2.4.0", f"expected torch 2.4.0, got {torch.__version__}"
assert hip.startswith("6.1"), f"expected ROCm/HIP 6.1, got {hip or 'none'}"
assert architecture == "gfx942", f"expected gfx942, got {architecture}"
assert "MI300X" in properties.name.upper(), f"expected MI300X, got {properties.name}"

left = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
right = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
gemm = left @ right

vector = torch.randn(1 << 20, device="cuda", dtype=torch.bfloat16)
triton_output = triton_kernels.vector_add(vector, vector)

q = torch.randn((1, 512, 8, 64), device="cuda", dtype=torch.bfloat16)
adapter = load_adapter("torch-sdpa-aotriton")
attention = adapter.prepare(q, q, q, True)
attention_output = attention()
torch.cuda.synchronize()
profile = profiler_evidence(attention)
names = [str(event.get("name", "")).lower() for event in profile.get("device_events", [])]
tokens = ("aotriton", "attn_fwd", "fmha_fwd", "flash")
native_trace = any(any(token in name for token in tokens) for name in names)
assert native_trace, f"AOTriton native trace not observed: {names[:12]}"

atomic_write_json(
    output,
    {
        "schema_version": "0.3",
        "evidence_kind": "mi300x-rocm61-stack-smoke",
        "generated_at": utc_now(),
        "status": "passed",
        "hardware": hardware_manifest(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton": triton.__version__,
        "architecture": architecture,
        "checks": {
            "bf16_gemm": list(gemm.shape),
            "triton_vector_add": list(triton_output.shape),
            "aotriton_attention": list(attention_output.shape),
            "aotriton_native_trace": native_trace,
        },
        "profiler": profile,
        "implementation_provenance": adapter.provenance,
    },
)
PY

uv pip freeze --python "${python}" > "${stage_root}/environment-freeze.txt"
rocm-smi > "${stage_root}/rocm-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/completed-at.txt"
echo "MI300X PyTorch 2.4.0 / ROCm 6.1 setup complete: ${stage_root}"
