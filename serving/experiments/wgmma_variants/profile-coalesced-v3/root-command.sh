#!/usr/bin/env bash
# ROOT EXECUTES ONLY under serialized observational profiling lease.
# No invocation has been performed by the preparing agent.
set -euo pipefail
cd /root/okc
bundle=90e3f42727f5e84932eeaf3d0c1c85bc941cb1dd3fe93b6a5152d483e910e087
candidate="/root/okc/results/rsi/immutable-trees/candidate-$bundle"
probe=/root/okc/results/rsi/candidates/wgmma-profile-coalesced-v3
output=/root/okc/results/rsi/wgmma-coalesced-profile-v3
image=lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a
name=okc-wgmma-coalesced-profile-v3
mkdir "$output" # refuse any evidence-directory overwrite
exec 9>/root/okc/.gpu-cache/rsi-device0.lock
flock -n 9
[[ -z $(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader) ]] || exit 2
[[ -d "$candidate" ]] || exit 3
# Verify entire immutable tree against external frozen manifest before GPU launch.
python3 - "$candidate" "$probe" <<'VERIFY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/root/okc/benchmarks')
pins=json.loads((Path(sys.argv[2])/'source-pins.json').read_text())
assert hashlib.sha256(Path('/root/okc/benchmarks/rsi_bundle.py').read_bytes()).hexdigest()==pins['bundle_verifier_sha256']
from rsi_bundle import verify_tree
manifest=Path('/root/okc/results/rsi/bundles/wgmma-coalesced-v3.tar.gz.manifest.json')
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==pins['manifest_sha256']
assert json.loads(manifest.read_text())['archive_sha256']==pins['bundle_sha256']
verify_tree(Path(sys.argv[1]),json.loads(manifest.read_text()))
VERIFY
cleanup() {
    original_code=$?
    trap - EXIT
    trap '' TERM INT
    python3 "$probe/teardown.py" "$name" "$output" "$original_code" || exit 42
    exit "$original_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Existing role/hash-specific coalesced-v3 cache; never the baseline/other-candidate cache.
timeout --signal=TERM --kill-after=15s 240s docker run --rm --init --name "$name" \
  --gpus '"device=0"' --cap-add SYS_ADMIN --shm-size=16g --network=none \
  -v "$candidate:/candidate:ro" -v /root/okc/benchmarks:/runner:ro \
  -v "$probe:/probe:ro" -v "$output:/output" \
  -v "/root/okc/.gpu-cache/rsi/candidate/$bundle:/cache/candidate/$bundle" \
  -e PYTHONPATH= -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e XDG_CACHE_HOME="/cache/candidate/$bundle" -e SGLANG_JIT_CACHE_DIR="/cache/candidate/$bundle/sglang" \
  -e TRITON_CACHE_DIR="/cache/candidate/$bundle/triton" -e TORCH_EXTENSIONS_DIR="/cache/candidate/$bundle/torch_extensions" \
  --entrypoint bash "$image" /probe/inside-command.sh > "$output/profile.log" 2>&1
