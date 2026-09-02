#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: remote_run_dense_score_gradient_a100.sh PASS [PASS ...]" >&2
  exit 2
fi

workspace="${INFRASWE_REMOTE_ROOT:-/root/infraswe}"
python="${INFRASWE_PYTHON:-${workspace}/.venv/bin/python}"
gpu="${INFRASWE_GPU:-0}"
pilot_root="${INFRASWE_PILOT_ROOT:-${workspace}/runs/kernel-score-gradient-dense-pilot-v03-a100}"
run_root="${INFRASWE_RUN_ROOT:-${workspace}/runs/kernel-score-gradient-dense-v03-a100}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
python_path="${workspace}/envs/fa4"
passes=("$@")

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d')"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

for replay in 1 2 3; do
  test -s "${pilot_root}/raw/calibration/replay-${replay}.json"
  test -s "${pilot_root}/raw/negative/garbage-slow-fa4-waste64/replay-${replay}.json"
done
test -s "${pilot_root}/pilot/sweep.json"

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
cp -a "${pilot_root}/pilot/sweep.json" "${run_root}/pilot/sweep.json"
printf "%s\n" "${passes[@]}" > "${run_root}/selected-passes.txt"
nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-before.txt"

attention_cases=(
  common-b4-s512-h16-d64-noncausal
  common-b2-s1024-h16-d64-causal
  common-b1-s2048-h16-d128-causal
  boundary-b3-s1000-h12-d64-causal
  stress-b1-s4096-h8-d128-causal
)

cd "${benchmark_root}"
for pass_count in "${passes[@]}"; do
  if [[ "${pass_count}" == "64" ]]; then
    continue
  fi
  backend="mediocre-fa4-waste${pass_count}"
  mkdir -p "${run_root}/raw/gradient/${backend}" "${run_root}/profiles/${backend}"
  for replay in 1 2 3; do
    output="${run_root}/raw/gradient/${backend}/replay-${replay}.json"
    if [[ ! -s "${output}" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" attention_bench.py \
        --backend "${backend}" \
        --output "${output}" \
        --replay-index "${replay}" \
        --blocks 30 \
        --min-timed-span-ms 50 \
        --implementation-commit controlled-degradation-v1
    fi
  done

  for case_id in "${attention_cases[@]}"; do
    output="${run_root}/profiles/${backend}/${case_id}.json"
    if [[ ! -s "${output}" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" "${python}" profile_case.py \
        --suite attention \
        --backend "${backend}" \
        --case-id "${case_id}" \
        --output "${output}"
    fi
  done
done

PATH="${python%/bin/python}/bin:${PATH}" PYTHONPATH="${python_path}" "${python}" \
  collect_provenance.py \
  --root "${workspace}" \
  --output "${run_root}/provenance.json" \
  --implementations fa4
uv pip freeze --python "${python}" > "${run_root}/environment-freeze.txt"
nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"
echo "A100 dense score-gradient formal run complete: ${run_root}"
