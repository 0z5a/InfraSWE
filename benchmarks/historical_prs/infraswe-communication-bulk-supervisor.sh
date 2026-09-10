#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/infraswe
exec env \
  PATH=/workspace/tools:/opt/instance-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  GH_CONFIG_DIR=/workspace/infraswe-secrets/gh \
  INFRASWE_GITHUB_CREDENTIAL_COPY=/workspace/infraswe-secrets/gh/hosts.yml \
  INFRASWE_EXECUTION_MODE=local \
  INFRASWE_LOCAL_PYTHON=/venv/main/bin/python \
  INFRASWE_COMMUNICATION_RESULT_ROOT=results/historical-pr-blind-20260901/communication-bulk-95pct \
  INFRASWE_COMMUNICATION_QUEUE_LOCK=results/historical-pr-blind-20260901/communication-bulk-95pct/queue-lock-groups3000.json \
  INFRASWE_COMMUNICATION_WORKERS=16 \
  INFRASWE_COMMUNICATION_LANES_PER_PROJECT=16 \
  INFRASWE_GITHUB_WORKERS=4 \
  INFRASWE_GITHUB_REVEAL_WORKERS=4 \
  INFRASWE_GITHUB_BATCH_SIZE=10 \
  INFRASWE_GITHUB_REVEAL_BATCH_SIZE=50 \
  INFRASWE_TEST_TIMEOUT=45 \
  INFRASWE_DECISION_V061_SHADOW=1 \
  INFRASWE_PUBLISH_ON_COMPLETE=0 \
  INFRASWE_STOP_INSTANCE_ON_COMPLETE=0 \
  stdbuf -oL -eL bash benchmarks/historical_prs/run_communication_bulk_campaign.sh 2>&1
