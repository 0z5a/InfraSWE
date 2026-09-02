#!/usr/bin/env bash
set -euo pipefail

workspace="${INFRASWE_REMOTE_ROOT:-/workspace/infraswe}"
python="${INFRASWE_PYTHON:-python3}"
gpu="${INFRASWE_GPU:-0}"
run_root="${1:-${workspace}/runs/b200-sm100-feature-score-v02}"
benchmark_root="${workspace}/benchmarks/kernel_frontier"
formal_root="${run_root}/formal"
cuda_root="${INFRASWE_CUDA_ROOT:-/usr/local/cuda-13.3}"

export PATH="${cuda_root}/bin:${PATH}"

active_pids="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" | sed '/^$/d'
)"
if [[ -n "${active_pids}" ]]; then
  echo "GPU ${gpu} is not exclusively available; active compute PIDs: ${active_pids}" >&2
  exit 3
fi

mkdir -p \
  "${run_root}/tma-irregular/keep" \
  "${formal_root}" \
  "${run_root}/methodology"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/started-at.txt"
nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-before.txt"
env | sort > "${run_root}/environment.txt"
"${cuda_root}/bin/nvcc" --version > "${run_root}/nvcc-version.txt"
"${cuda_root}/bin/ptxas" --version > "${run_root}/ptxas-version.txt"

{
  time -p "${cuda_root}/bin/nvcc" \
    -std=c++20 \
    -O3 \
    -lineinfo \
    -arch=sm_100a \
    --keep \
    --keep-dir "${run_root}/tma-irregular/keep" \
    "${benchmark_root}/b200_tma_gather_scatter.cu" \
    -lcuda \
    -o "${run_root}/tma-irregular/b200_tma_gather_scatter"
} 2>&1 | tee "${run_root}/tma-irregular/compile.log"

for replay in 1 2 3; do
  mkdir -p \
    "${formal_root}/replay-${replay}/artifacts" \
    "${formal_root}/replay-${replay}/dump"
  timeout 600s "${python}" "${benchmark_root}/b200_feature_bench.py" \
    --replay-index "${replay}" \
    --output "${formal_root}/replay-${replay}.json" \
    --artifact-root "${formal_root}/replay-${replay}/artifacts" \
    --dump-root "${formal_root}/replay-${replay}/dump" \
    --tma-binary "${run_root}/tma-irregular/b200_tma_gather_scatter" \
    --tma-artifact-root "${run_root}/tma-irregular" \
    --warmups 5 \
    --iterations 30 \
    --samples 3 \
    --lifecycle-iterations 3000 \
    --tma-iterations 10000 \
    2>&1 | tee "${formal_root}/replay-${replay}.log"
done

"${python}" "${benchmark_root}/summarize_b200_feature_scores.py" \
  --root "${formal_root}" \
  --json-output "${run_root}/score.json" \
  --markdown-output "${run_root}/report.md" \
  --require-passed

cp "${benchmark_root}/b200_feature_bench.py" "${run_root}/methodology/"
cp "${benchmark_root}/b200_tma_gather_scatter.cu" "${run_root}/methodology/"
cp "${benchmark_root}/summarize_b200_feature_scores.py" "${run_root}/methodology/"
cp "${benchmark_root}/remote_run_b200_feature_scores.sh" "${run_root}/methodology/"
nvidia-smi -q -i "${gpu}" > "${run_root}/nvidia-smi-after.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${run_root}/completed-at.txt"

"${python}" -m zipfile -c "${run_root}.zip" "${run_root}"
sha256sum "${run_root}.zip" > "${run_root}.zip.sha256"
echo "B200 SM100 feature score complete: ${run_root}.zip"
