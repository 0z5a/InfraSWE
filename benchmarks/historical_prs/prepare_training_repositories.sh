#!/usr/bin/env bash
# Parallel repository and isolated-environment preparation for training PR runs.

set -uo pipefail

training_root="${1:-/workspace/training-pr-corpus}"
repository_root="${training_root}/repos"
venv_root="${training_root}/venvs"
log_root="${training_root}/logs"
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
  "megatron-core|https://github.com/NVIDIA/Megatron-LM.git"
  "slime|https://github.com/THUDM/slime.git"
  "verl|https://github.com/verl-project/verl.git"
  "verl-omni|https://github.com/verl-project/verl-omni.git"
)

prepare_one() {
  local name="$1"
  local url="$2"
  local repository="${repository_root}/${name}"
  local venv="${venv_root}/${name}"
  local clone_rc=0
  local build_rc=0
  local compile_rc=0

  echo "phase=clone_start project=${name} url=${url}"
  if [[ -d "${repository}/.git" ]]; then
    git -C "${repository}" remote set-url origin "${url}"
    GIT_TERMINAL_PROMPT=0 git -C "${repository}" fetch \
      --filter=blob:none --no-tags --prune origin || clone_rc=$?
    if [[ ${clone_rc} -eq 0 ]]; then
      default_branch="$(git -C "${repository}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)"
      if [[ -n "${default_branch}" ]]; then
        git -C "${repository}" switch --detach "${default_branch}" || clone_rc=$?
      fi
    fi
  else
    GIT_TERMINAL_PROMPT=0 git clone --filter=blob:none --no-tags \
      "${url}" "${repository}" || clone_rc=$?
  fi
  if [[ ${clone_rc} -ne 0 ]]; then
    echo "phase=clone_failed project=${name} rc=${clone_rc}"
    return "${clone_rc}"
  fi
  echo "phase=clone_complete project=${name} head=$(git -C "${repository}" rev-parse HEAD)"

  echo "phase=build_start project=${name}"
  "${uv_bin}" venv --python /venv/main/bin/python --system-site-packages \
    "${venv}" || build_rc=$?
  if [[ ${build_rc} -eq 0 ]]; then
    "${uv_bin}" pip install --python "${venv}/bin/python" --no-deps \
      --editable "${repository}" || build_rc=$?
  fi
  if [[ ${build_rc} -eq 0 ]]; then
    echo "phase=build_complete project=${name}"
  else
    echo "phase=build_failed project=${name} rc=${build_rc}"
  fi

  echo "phase=compile_start project=${name}"
  if [[ -x "${venv}/bin/python" ]]; then
    "${venv}/bin/python" -m compileall -q \
      -x '(^|/)(\.git|build|dist|\.venv)(/|$)' "${repository}" || compile_rc=$?
  else
    /venv/main/bin/python -m compileall -q \
      -x '(^|/)(\.git|build|dist|\.venv)(/|$)' "${repository}" || compile_rc=$?
  fi
  if [[ ${compile_rc} -eq 0 ]]; then
    echo "phase=compile_complete project=${name}"
  else
    echo "phase=compile_failed project=${name} rc=${compile_rc}"
  fi

  echo "phase=done project=${name} clone_rc=${clone_rc} build_rc=${build_rc} compile_rc=${compile_rc}"
  if [[ ${build_rc} -ne 0 ]]; then
    return "${build_rc}"
  fi
  return "${compile_rc}"
}

project_names=()
job_pids=()
for specification in "${specifications[@]}"; do
  name="${specification%%|*}"
  url="${specification#*|}"
  project_names+=("${name}")
  prepare_one "${name}" "${url}" >"${log_root}/${name}.log" 2>&1 &
  pid="$!"
  job_pids+=("${pid}")
  echo "project=${name} pid=${pid} log=${log_root}/${name}.log"
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
