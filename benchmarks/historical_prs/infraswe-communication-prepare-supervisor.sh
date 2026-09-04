#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/infraswe
exec env \
  PATH=/workspace/tools:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  stdbuf -oL -eL bash benchmarks/historical_prs/prepare_communication_repositories.sh 2>&1
