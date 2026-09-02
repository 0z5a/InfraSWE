#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
set -euo pipefail

uv pip install \
  --python /venv/main/bin/python \
  einops ninja packaging psutil
cd /workspace/infraswe/benchmarks/kernel_frontier
INFRASWE_REMOTE_ROOT=/workspace/infraswe \
INFRASWE_PYTHON=/venv/main/bin/python \
INFRASWE_FA2_ARCHS=120 \
INFRASWE_MAX_JOBS=32 \
  bash ./remote_build_fa.sh fa2
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/infraswe/runs/kernel-sm120-fa2-build-completed-at.txt
