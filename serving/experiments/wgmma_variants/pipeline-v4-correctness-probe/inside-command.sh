#!/usr/bin/env bash
set -euo pipefail
# Source/runner mounts are read-only; host already checked exact bytes/tree.
python3 /probe/run_routed_probe.py > /output/routed.log 2>&1
python3 /probe/check_full_expert.py --bundle-root /baseline --runner-root /runner \
  --output /output/full-correctness.json > /output/full.log 2>&1
python3 /control/summarize.py
