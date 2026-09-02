#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/kernel-frontier-v03-mi300x-rocm61-torch240}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
attention_backend="torch-sdpa-aotriton"
classic_backend="triton-gfx942-initial"
export PATH="/opt/rocm/bin:${PATH}"

mkdir -p \
  "${run_root}/lease" \
  "${run_root}/raw/calibration" \
  "${run_root}/raw/attention/${attention_backend}" \
  "${run_root}/raw/classic" \
  "${run_root}/profiles/${attention_backend}" \
  "${run_root}/profiles/${classic_backend}"

cd "${benchmark_root}"
"${python}" rocm_lease_guard.py \
  --device-index "${gpu}" \
  --output "${run_root}/lease/exclusive-device.json"

export ROCR_VISIBLE_DEVICES="${gpu}"
export HIP_VISIBLE_DEVICES="${gpu}"

"${python}" - <<'PY'
import re
import torch

properties = torch.cuda.get_device_properties(0)
match = re.search(r"gfx[0-9a-z]+", str(getattr(properties, "gcnArchName", "")).lower())
architecture = match.group(0) if match else None
assert torch.__version__.split("+")[0] == "2.4.0", torch.__version__
assert str(getattr(torch.version, "hip", "") or "").startswith("6.1"), torch.version.hip
assert architecture == "gfx942", architecture
assert "MI300X" in properties.name.upper(), properties.name
PY

PYTHONPATH="${workspace}/src" "${python}" -m infraswe lease preflight \
  --profile gpu-1x-gfx942-mi300x-rocm61 \
  --profiles-dir "${workspace}/profiles" \
  --output "${run_root}/hardware-manifest.json"

if [[ ! -s "${run_root}/rocm-smi-before.txt" ]]; then
  rocm-smi > "${run_root}/rocm-smi-before.txt"
fi

for replay in 1 2 3; do
  output="${run_root}/raw/calibration/replay-${replay}.json"
  if [[ ! -s "${output}" ]]; then
    "${python}" calibrate_gpu.py \
      --output "${output}" \
      --replay-index "${replay}" \
      --samples 30
  fi
done

for replay in 1 2 3; do
  output="${run_root}/raw/attention/${attention_backend}/replay-${replay}.json"
  if [[ ! -s "${output}" ]]; then
    "${python}" attention_bench.py \
      --backend "${attention_backend}" \
      --output "${output}" \
      --replay-index "${replay}" \
      --blocks 30 \
      --min-timed-span-ms 50 \
      --implementation-commit pytorch-2.4.0-rocm6.1-embedded-aotriton
  fi
done

attention_cases=(
  common-b4-s512-h16-d64-noncausal
  common-b2-s1024-h16-d64-causal
  common-b1-s2048-h16-d128-causal
  boundary-b3-s1000-h12-d64-causal
  stress-b1-s4096-h8-d128-causal
)
for case_id in "${attention_cases[@]}"; do
  output="${run_root}/profiles/${attention_backend}/${case_id}.json"
  if [[ ! -s "${output}" ]]; then
    "${python}" profile_case.py \
      --suite attention \
      --backend "${attention_backend}" \
      --case-id "${case_id}" \
      --output "${output}"
  fi
done

for replay in 1 2 3; do
  output="${run_root}/raw/classic/replay-${replay}.json"
  if [[ ! -s "${output}" ]]; then
    "${python}" classic_bench.py \
      --backend "${classic_backend}" \
      --implementation-commit infraswe-gfx942-portable-fixed-v1 \
      --output "${output}" \
      --replay-index "${replay}" \
      --blocks 30 \
      --min-timed-span-ms 50
  fi
done

classic_cases=(
  vector-add-bf16-16m
  softmax-bf16-4096x4096
  layernorm-bf16-4096x4096
  rmsnorm-bf16-4096x4096
  swiglu-bf16-8192x4096
  rope-bf16-b4-s2048-h16-d128
  gemm-bf16-4096-cube
)
for case_id in "${classic_cases[@]}"; do
  output="${run_root}/profiles/${classic_backend}/${case_id}.json"
  if [[ ! -s "${output}" ]]; then
    "${python}" profile_case.py \
      --suite classic \
      --backend "${classic_backend}" \
      --case-id "${case_id}" \
      --output "${output}"
  fi
done

"${python}" collect_provenance.py \
  --root "${workspace}" \
  --output "${run_root}/provenance.json" \
  --platform-only
if command -v uv >/dev/null; then
  uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
else
  "${python}" -m pip freeze > "${run_root}/environment-freeze.txt"
fi
rocm-smi > "${run_root}/rocm-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"

"${python}" score_results.py \
  --root "${run_root}" \
  --zip "${run_root}.zip"
echo "MI300X ROCm 6.1 kernel-frontier run complete: ${run_root}.zip"
