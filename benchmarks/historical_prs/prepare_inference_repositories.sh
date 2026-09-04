#!/usr/bin/env bash
# Prepare isolated partial clones for the four inference PR corpora.

set -uo pipefail

corpus_root="${1:-/workspace/inference-pr-corpus}"
repository_root="${corpus_root}/repos"
venv_root="${corpus_root}/venvs"
log_root="${corpus_root}/logs"
mkdir -p "${repository_root}" "${venv_root}" "${log_root}"

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x /venv/main/bin/uv ]]; then
  uv_bin=/venv/main/bin/uv
else
  echo "uv is required" >&2
  exit 2
fi

specifications=(
  "flashinfer|https://github.com/flashinfer-ai/flashinfer.git"
  "sglang|https://github.com/sgl-project/sglang.git"
  "tensorrt-llm|https://github.com/NVIDIA/TensorRT-LLM.git"
  "vllm|https://github.com/vllm-project/vllm.git"
)

prepare_one() {
  local name="$1"
  local url="$2"
  local repository="${repository_root}/${name}"
  local venv="${venv_root}/${name}"
  local clone_rc=0
  local venv_rc=0

  echo "phase=clone_start project=${name} url=${url}"
  if [[ -d "${repository}/.git" ]]; then
    git -C "${repository}" remote set-url origin "${url}"
    GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0 git -C "${repository}" fetch \
      --filter=blob:none --no-tags --prune origin || clone_rc=$?
  else
    GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0 \
      git clone --filter=blob:none --no-tags --no-checkout \
      "${url}" "${repository}" || clone_rc=$?
  fi
  if [[ ${clone_rc} -ne 0 ]]; then
    echo "phase=clone_failed project=${name} rc=${clone_rc}"
    return "${clone_rc}"
  fi
  default_branch="$(
    git -C "${repository}" symbolic-ref --quiet --short refs/remotes/origin/HEAD \
      2>/dev/null || true
  )"
  if [[ -n "${default_branch}" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "${repository}" switch \
      --discard-changes --detach "${default_branch}" \
      >/dev/null || clone_rc=$?
  fi
  if [[ ${clone_rc} -ne 0 ]]; then
    echo "phase=checkout_failed project=${name} rc=${clone_rc}"
    return "${clone_rc}"
  fi
  echo "phase=clone_complete project=${name} head=$(git -C "${repository}" rev-parse HEAD)"

  echo "phase=venv_start project=${name}"
  if [[ -x "${venv}/bin/python" ]]; then
    echo "phase=venv_reuse project=${name}"
  else
    "${uv_bin}" venv --python /venv/main/bin/python --system-site-packages \
      "${venv}" || venv_rc=$?
  fi
  if [[ ${venv_rc} -eq 0 ]]; then
    echo "phase=venv_complete project=${name}"
  else
    echo "phase=venv_failed project=${name} rc=${venv_rc}"
  fi
  echo "phase=done project=${name} clone_rc=${clone_rc} venv_rc=${venv_rc}"
  return "${venv_rc}"
}

project_names=()
job_pids=()
for specification in "${specifications[@]}"; do
  name="${specification%%|*}"
  url="${specification#*|}"
  project_names+=("${name}")
  prepare_one "${name}" "${url}" >"${log_root}/${name}.log" 2>&1 &
  job_pids+=("$!")
  echo "project=${name} pid=${job_pids[-1]} log=${log_root}/${name}.log"
done

failure_count=0
for index in "${!job_pids[@]}"; do
  if wait "${job_pids[$index]}"; then
    echo "project=${project_names[$index]} status=ready"
  else
    rc=$?
    echo "project=${project_names[$index]} status=failed rc=${rc}"
    failure_count=$((failure_count + 1))
  fi
done

echo "prepared_count=$((${#project_names[@]} - failure_count))"
echo "failure_count=${failure_count}"
exit "${failure_count}"
