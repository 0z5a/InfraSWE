#!/usr/bin/env bash
set -euo pipefail

root="${INFRASWE_COMMUNICATION_CORPUS_ROOT:-/workspace/communication-pr-corpus}"
mkdir -p "$root/repos" "$root/logs" "$root/tmp"

repositories=(
  "nccl|https://github.com/NVIDIA/nccl.git"
  "nccl-extensions|https://github.com/NVIDIA/nccl-extensions.git"
  "nccl-tests|https://github.com/NVIDIA/nccl-tests.git"
  "deepep|https://github.com/deepseek-ai/DeepEP.git"
  "nvshmem|https://github.com/NVIDIA/nvshmem.git"
  "rccl|https://github.com/ROCm/rccl.git"
  "ucx|https://github.com/openucx/ucx.git"
  "ucc|https://github.com/openucx/ucc.git"
  "uccl|https://github.com/uccl-project/uccl.git"
  "pytorch|https://github.com/pytorch/pytorch.git"
  "vllm|https://github.com/vllm-project/vllm.git"
  "sglang|https://github.com/sgl-project/sglang.git"
)

clone_one() {
  local name="$1"
  local url="$2"
  local destination="$root/repos/$name"
  local log="$root/logs/$name-clone.log"
  if [[ -d "$destination/.git" ]]; then
    if [[ "$(git -C "$destination" remote get-url origin)" != "$url" ]]; then
      echo "$name: existing origin mismatch" >&2
      return 1
    fi
    echo "$name: already present"
    return 0
  fi
  if [[ -e "$destination" ]]; then
    echo "$name: destination exists without a git repository" >&2
    return 1
  fi
  local temporary
  temporary="$(mktemp -d "$root/tmp/$name.XXXXXX")"
  if timeout 900s git -c protocol.version=2 clone \
    --filter=tree:0 \
    --no-checkout \
    --single-branch \
    "$url" "$temporary/repository" >"$log" 2>&1; then
    mv "$temporary/repository" "$destination"
    rmdir "$temporary"
    echo "$name: cloned"
    return 0
  fi
  echo "$name: clone failed; see $log" >&2
  return 1
}

pids=()
names=()
for entry in "${repositories[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  clone_one "$name" "$url" &
  pids+=("$!")
  names+=("$name")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${names[$index]}: bounded clone failure" >&2
    status=1
  fi
done

megatron_link="$root/repos/megatron-core"
if [[ ! -e "$megatron_link" ]]; then
  ln -s /workspace/training-pr-corpus/repos/megatron-core "$megatron_link"
fi

echo "communication repositories present:"
find "$root/repos" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort
exit "$status"
