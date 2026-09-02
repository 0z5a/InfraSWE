#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
pilot_root="${workspace}/runs/kernel-score-gradient-pilot-v03-sm120"
run_root="${workspace}/runs/kernel-score-gradient-v03-sm120"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p \
  "${run_root}/raw/calibration" \
  "${run_root}/raw/gradient" \
  "${run_root}/profiles" \
  "${run_root}/pilot"
cp -a "${pilot_root}/raw/calibration/." "${run_root}/raw/calibration/"
cp -a \
  "${pilot_root}/raw/negative/garbage-slow-fa4-waste64" \
  "${run_root}/raw/gradient/"
cp -a \
  "${pilot_root}/profiles/garbage-slow-fa4-waste64" \
  "${run_root}/profiles/"
cp -a "${pilot_root}/sweep-full.json" "${run_root}/pilot/sweep.json"
nvidia-smi -q > "${run_root}/nvidia-smi-before.txt"

backends=(
  mediocre-fa4-waste0
  mediocre-fa4-waste32
  mediocre-fa4-waste96
  mediocre-fa4-waste192
  mediocre-fa4-waste384
  mediocre-fa4-waste1024
)
attention_cases=(
  common-b4-s512-h16-d64-noncausal
  common-b2-s1024-h16-d64-causal
  common-b1-s2048-h16-d128-causal
  boundary-b3-s1000-h12-d64-causal
  stress-b1-s4096-h8-d128-causal
)

cd "${benchmark_root}"
for backend in "${backends[@]}"; do
  mkdir -p "${run_root}/raw/gradient/${backend}" "${run_root}/profiles/${backend}"
  for replay in 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
      --backend "${backend}" \
      --output "${run_root}/raw/gradient/${backend}/replay-${replay}.json" \
      --replay-index "${replay}" \
      --blocks 30 \
      --min-timed-span-ms 50 \
      --implementation-commit controlled-degradation-v1
  done

  for case_id in "${attention_cases[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" profile_case.py \
      --suite attention \
      --backend "${backend}" \
      --case-id "${case_id}" \
      --output "${run_root}/profiles/${backend}/${case_id}.json"
  done
done

PYTHONPATH="${python_path}" "${python}" collect_provenance.py \
  --root "${workspace}" \
  --output "${run_root}/provenance.json" \
  --implementations fa4
uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
nvidia-smi -q > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "SM120 score-gradient formal run complete"

