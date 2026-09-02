#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${workspace}/runs/kernel-score-gradient-pilot-v03-sm120"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p \
  "${run_root}/raw/calibration" \
  "${run_root}/raw/negative/garbage-slow-fa4-waste64" \
  "${run_root}/profiles/garbage-slow-fa4-waste64"
nvidia-smi -q > "${run_root}/nvidia-smi-before.txt"

cd "${benchmark_root}"
for replay in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python}" calibrate_gpu.py \
    --output "${run_root}/raw/calibration/replay-${replay}.json" \
    --replay-index "${replay}" \
    --samples 30
done

for replay in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
    --backend garbage-slow-fa4-waste64 \
    --output "${run_root}/raw/negative/garbage-slow-fa4-waste64/replay-${replay}.json" \
    --replay-index "${replay}" \
    --blocks 30 \
    --min-timed-span-ms 50 \
    --implementation-commit controlled-degradation-v1
done

attention_cases=(
  common-b4-s512-h16-d64-noncausal
  common-b2-s1024-h16-d64-causal
  common-b1-s2048-h16-d128-causal
  boundary-b3-s1000-h12-d64-causal
  stress-b1-s4096-h8-d128-causal
)
for case_id in "${attention_cases[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" profile_case.py \
    --suite attention \
    --backend garbage-slow-fa4-waste64 \
    --case-id "${case_id}" \
    --output "${run_root}/profiles/garbage-slow-fa4-waste64/${case_id}.json"
done

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" sweep_mediocre.py \
  --evidence-root "${run_root}" \
  --output "${run_root}/sweep.json" \
  --passes 0 2 4 8 12 16 24 32 48 64 96 128 \
  --samples 7

PYTHONPATH="${python_path}" "${python}" collect_provenance.py \
  --root "${workspace}" \
  --output "${run_root}/provenance.json" \
  --implementations fa4
uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
nvidia-smi -q > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "SM120 score-gradient pilot complete"

