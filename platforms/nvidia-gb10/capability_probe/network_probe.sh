#!/bin/sh
set -eu

run_optional() {
  label=$1
  shift
  printf '%s\n' "[$label]"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 || true
  else
    printf '%s\n' "unavailable"
  fi
}

run_optional ibdev2netdev ibdev2netdev
run_optional ibv_devinfo ibv_devinfo
run_optional links ip -br link
