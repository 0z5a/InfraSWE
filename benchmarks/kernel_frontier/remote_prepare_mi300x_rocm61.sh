#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
diagnostic_only="${INFRASWE_DIAGNOSTIC_ONLY:-0}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
stage_root="${workspace}/runs/kernel-mi300x-rocm61-torch240-setup"
export PATH="/opt/rocm/bin:${PATH}"

if [[ "${diagnostic_only}" != "0" && "${diagnostic_only}" != "1" ]]; then
  echo "INFRASWE_DIAGNOSTIC_ONLY must be 0 or 1" >&2
  exit 64
fi

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
lease_guard_status=0
"${python}" rocm_lease_guard.py \
  --device-index "${gpu}" \
  --output "${stage_root}/exclusive-device.json" || lease_guard_status=$?
if [[ "${lease_guard_status}" -ne 0 && "${diagnostic_only}" != "1" ]]; then
  echo "exclusive-device precheck failed; set INFRASWE_DIAGNOSTIC_ONLY=1 only for non-scoring qualification" >&2
  exit "${lease_guard_status}"
fi
export INFRASWE_SETUP_LEASE_STATUS="${lease_guard_status}"
export INFRASWE_DIAGNOSTIC_ONLY="${diagnostic_only}"

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
import os
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
native_trace = any(
    not name.startswith("aten::") and any(token in name for token in tokens)
    for name in names
)
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
        "qualification": {
            "mode": (
                "diagnostic-only"
                if os.environ["INFRASWE_DIAGNOSTIC_ONLY"] == "1"
                else "official-eligible-setup"
            ),
            "exclusive_device_precheck": (
                int(os.environ["INFRASWE_SETUP_LEASE_STATUS"]) == 0
            ),
            "official_measurement_eligible": (
                os.environ["INFRASWE_DIAGNOSTIC_ONLY"] == "0"
                and int(os.environ["INFRASWE_SETUP_LEASE_STATUS"]) == 0
            ),
        },
    },
)
PY

uv pip freeze --python "${python}" > "${stage_root}/environment-freeze.txt"
rocm-smi > "${stage_root}/rocm-smi-after.txt"
post_lease_guard_status=0
"${python}" rocm_lease_guard.py \
  --device-index "${gpu}" \
  --output "${stage_root}/exclusive-device-after.json" || post_lease_guard_status=$?
"${python}" - \
  "${stage_root}/setup-qualification.json" \
  "${diagnostic_only}" \
  "${lease_guard_status}" \
  "${post_lease_guard_status}" <<'PY'
import sys
from pathlib import Path

from bench_utils import atomic_write_json, utc_now

output = Path(sys.argv[1])
diagnostic_only = sys.argv[2] == "1"
pre_status = int(sys.argv[3])
post_status = int(sys.argv[4])
atomic_write_json(
    output,
    {
        "schema_version": "0.3",
        "evidence_kind": "mi300x-rocm61-setup-qualification",
        "generated_at": utc_now(),
        "mode": "diagnostic-only" if diagnostic_only else "official-eligible-setup",
        "stack_smoke_passed": True,
        "exclusive_device_precheck": pre_status == 0,
        "exclusive_device_postcheck": post_status == 0,
        "official_measurement_eligible": (
            not diagnostic_only and pre_status == 0 and post_status == 0
        ),
        "failure_codes": sorted(
            {
                (
                    "LIVENESS_DEVICE_NOT_EXCLUSIVE"
                    if status == 3
                    else "LIVENESS_EXCLUSIVE_LEASE_UNVERIFIABLE"
                )
                for status in (pre_status, post_status)
                if status != 0
            }
        ),
    },
)
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "${stage_root}/completed-at.txt"
if [[ "${post_lease_guard_status}" -ne 0 && "${diagnostic_only}" != "1" ]]; then
  echo "exclusive-device postcheck failed; setup evidence is not measurement-eligible" >&2
  exit "${post_lease_guard_status}"
fi
echo "MI300X PyTorch 2.4.0 / ROCm 6.1 setup complete: ${stage_root}"
