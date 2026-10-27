#!/usr/bin/env bash
# Host-side deadline and cleanup around the container-held GPU lock.
# Usage on the iteration node: bash h100_trial.sh NAME OUTER_S INNER_S COMMAND [ARG...]
set -euo pipefail
if (( $# < 4 )); then echo 'Expected NAME OUTER_S INNER_S COMMAND [ARG...]' >&2; exit 64; fi
container=$1
outer_seconds=$2
inner_seconds=$3
shift 3
[[ $container =~ ^okc-proj-[a-z0-9-]+$ ]] || exit 64
[[ $outer_seconds =~ ^[0-9]+$ && $inner_seconds =~ ^[0-9]+$ ]] || exit 64
(( inner_seconds > 0 && outer_seconds > inner_seconds + 20 )) || exit 64
if docker inspect "$container" >/dev/null 2>&1; then
  echo "Refusing existing container $container" >&2
  exit 73
fi
cleanup() {
  result=$?
  trap - EXIT
  if docker inspect "$container" >/dev/null 2>&1; then
    label=$(docker inspect --format '{{index .Config.Labels "okc.scheduler"}}' "$container")
    if [[ $label != container-flock ]]; then
      echo "Unexpected owner label on $container; refusing cleanup" >&2
      exit 70
    fi
    timeout -k 2s 12s docker stop --time 5 "$container" >/dev/null 2>&1 || true
    if docker inspect "$container" >/dev/null 2>&1; then
      timeout -k 2s 8s docker kill "$container" >/dev/null 2>&1 || true
    fi
    if docker inspect "$container" >/dev/null 2>&1; then
      echo "TRIAL_CLEANUP_FAILED $container" >&2
      exit 70
    fi
  fi
  echo "TRIAL_EXIT $container $result"
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -u +'%Y-%m-%dT%H:%M:%SZ'
set +e
timeout --signal=TERM --kill-after=15s "${outer_seconds}s" \
  env PROJ_GPU_CONTAINER="$container" /root/okc_run_sglang.sh \
  timeout --signal=TERM --kill-after=10s "${inner_seconds}s" "$@"
result=$?
set -e
exit "$result"
