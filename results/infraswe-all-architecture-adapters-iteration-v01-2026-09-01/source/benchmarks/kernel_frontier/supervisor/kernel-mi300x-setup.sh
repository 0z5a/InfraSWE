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
  bash ./remote_prepare_mi300x_rocm61.sh

exec env \
  INFRASWE_REMOTE_ROOT=/workspace/infraswe \
  INFRASWE_PYTHON=/venv/main/bin/python \
  INFRASWE_GPU=0 \
  bash ./remote_run_frontier_mi300x_rocm61.sh
