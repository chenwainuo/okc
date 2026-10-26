#!/bin/bash
# P1 runner: compile-only outside the GPU lock (no bake on a shared GPU), then the harness under the lock.
# Usage (on the box): /root/okc/proj_rsi/candidates/pw27h/p1/run.sh <tag> <submission> [h100_bench args]
tag=$1; sub=$2; shift 2
/root/okc_run.sh python proj_rsi/candidates/pw27h/p1/build.py $sub > /root/okc/logs/$tag.build.log 2>&1
/root/gpu_lock.sh /root/okc_run.sh python proj_rsi/tools/h100_bench.py $sub --json logs/$tag.json "$@" > /root/okc/logs/$tag.log 2>&1
echo BENCH_DONE >> /root/okc/logs/$tag.log
