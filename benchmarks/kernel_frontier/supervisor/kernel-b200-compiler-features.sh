#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail

cd /workspace/infraswe/benchmarks/kernel_frontier
env \
  INFRASWE_REMOTE_ROOT=/workspace/infraswe \
  INFRASWE_PYTHON=/venv/main/bin/python \
  INFRASWE_GPU=0 \
  bash ./remote_prepare_b200_cuda133.sh

exec env \
  INFRASWE_REMOTE_ROOT=/workspace/infraswe \
  INFRASWE_PYTHON=/venv/main/bin/python \
  INFRASWE_GPU=0 \
  bash ./remote_run_b200_compiler_features.sh
