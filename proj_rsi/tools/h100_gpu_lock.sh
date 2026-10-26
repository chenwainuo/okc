#!/usr/bin/env bash
# Compatibility entrypoint: okc_run.sh owns the lock inside the container.
# Taking the same lock here would deadlock the child container.
set -euo pipefail
exec "$@"
