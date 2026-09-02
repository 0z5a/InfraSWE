#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-/venv/main/bin/python}"
gpu="${INFRASWE_GPU:-0}"
run_root="${workspace}/runs/kernel-negative-controls-v03-a100"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p "${run_root}/raw/calibration" "${run_root}/raw/negative" "${run_root}/profiles"
nvidia-smi -q > "${run_root}/nvidia-smi-before.txt"

backends=(
  garbage-slow-fa4-waste64
  garbage-zero-triton
  garbage-cache-copy
  fa-garbage-math-fallback
)
attention_cases=(
  common-b4-s512-h16-d64-noncausal
  common-b2-s1024-h16-d64-causal
  common-b1-s2048-h16-d128-causal
  boundary-b3-s1000-h12-d64-causal
  stress-b1-s4096-h8-d128-causal
)

cd "${benchmark_root}"
for replay in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python}" calibrate_gpu.py \
    --output "${run_root}/raw/calibration/replay-${replay}.json" \
    --replay-index "${replay}" \
    --samples 30
done

for backend in "${backends[@]}"; do
  mkdir -p "${run_root}/raw/negative/${backend}" "${run_root}/profiles/${backend}"
  for replay in 1 2 3; do
    output="${run_root}/raw/negative/${backend}/replay-${replay}.json"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
      --backend "${backend}" \
      --output "${output}" \
      --replay-index "${replay}" \
      --blocks 30 \
      --min-timed-span-ms 50 \
      --implementation-commit negative-control-v1
    status=$?
    set -e
    test -s "${output}"
    case "${backend}" in
      garbage-zero-triton|garbage-cache-copy)
        if [[ "${status}" -eq 0 ]]; then
          echo "${backend} unexpectedly passed correctness in replay ${replay}" >&2
          exit 4
        fi
        ;;
      *)
        if [[ "${status}" -ne 0 ]]; then
          echo "${backend} unexpectedly failed execution in replay ${replay}" >&2
          exit "${status}"
        fi
        ;;
    esac
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
nvidia-smi -q > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "A100 negative-control run complete"
