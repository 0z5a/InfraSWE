#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/kernel-frontier-v03-a100}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi
for replay in 1 2 3; do
  test -s "${run_root}/raw/calibration/replay-${replay}.json"
done

cd "${benchmark_root}"
mkdir -p "${run_root}/raw/attention/fa4" "${run_root}/profiles/fa4"
for replay in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
    --backend fa4 \
    --output "${run_root}/raw/attention/fa4/replay-${replay}.json" \
    --replay-index "${replay}" \
    --blocks 30 \
    --min-timed-span-ms 50 \
    --implementation-commit pypi-flash-attn-4-4.0.0b28
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
    --backend fa4 \
    --case-id "${case_id}" \
    --output "${run_root}/profiles/fa4/${case_id}.json"
done

echo "A100 FA4 formal run and per-case profiles complete"
