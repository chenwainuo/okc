# Native allocation traffic diagnostic

`state=CPU_METADATA_DRIVER_PARSER_READY; GPU=UNRUN; purpose=TRAFFIC_CAUSE_NOT_SPEED; parent_and_helpers=FROZEN`

Question: does GENERIC reduce aggregate device-memory read traffic for native-v1's exact unpooler128 Lt under unchanged D2D population? Native-v1 is numerically/lifetime valid but flat versus its native allocation controls and about2.6us behind actual RQ. That failed version does not determine whether compression was effective or exhaust the family. See [counter audit](../../audits/hw-compression-counters.md).

Installed Nsight Compute2025.1.1 metadata was queried in CPU-only runc containers, `NVIDIA_VISIBLE_DEVICES=void`, while another root-owned GPU job ran. `ncu-gh100-base.txt`, the full canonical catalog in `ncu-gh100-all.txt.gz`, launch catalog, CLI help, and installed MemoryWorkloadAnalysis/MemoryL2Compression definitions are preserved. `metadata-provenance.json` records source hashes. The all-metrics query ignored its metrics filter; the full output is archived compressed and the selected excerpt is verbatim. Every one of21requested profiling/launch names is checked against the actual GH100 catalogs; no speculative metric name is sent to the GPU.

DRAM read/write bytes and sectors measure memory-unit traffic, not W.numel or an inferred bandwidth×latency. The byte/sector identity is checked at32B. Hopper's supported compression counters are `lrc__ilc_input_sectors`, `lrc__ilc_input_sectors_compressible`, `lrc__ilc_output_sectors` and their advertised ratios. There is no decompression-named base counter in this catalog. The installed rule also discusses writes/access patterns; zero ILC events during a read-only Lt launch cannot prove that population never compressed W. Raw counters/units and physical traffic must be interpreted together; no allocation flag or absent denominator is a compression ratio.

One driver, two selected NVTX graph nodes: GENERIC then VMM NONE. It reconstructs exact layout2 config`[66,18,1,0,0,4,35,0,2]` through frozen public `lt_add`, checks `lt_cfg`, and never calls prepare/autotune or measures an algorithm grid. It requires the exact native-v1 Torch/cuBLAS wheel and WtSHA`09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120`; original raw/Wt are each64MiB, X is native-v1 seed8851/1MiB. The same frozen native Case/validation/graph/lifetime code and helper7f2/manifest8f32 are reused. No shader, shape, precision, workspace, arithmetic or population change. This avoids the prior Nsight injection changing a fresh prepare's selected algorithm.

Both allocations must grant/query exact flags1/0, roundtrip all bytes, pass all4096basis columns plus FP64/official normal/outlier checks, actual captured five-input mutations/NaN poison and fresh-output capture. Each profiled replay gets an immediate complete output oracle before another replay can repair it. Postprofile actual graphs and original/raw/cloned bytes remain exact. Checked graph-reset→view/state-drop→Region0 cleanup is mandatory. No locals()/frame snapshot in cleanup.

CPU-only gate:

```bash
python3 proj_rsi/candidates/hw_compression_counters/preflight.py
```

Checks canonical supported suffixes and launch attributes, fixed-source/helper hashes, no prepare/autotune/timing calls, AST/bash syntax, raw base units, exact NVTX/shader/launch identity, corrupt/missing/duplicate report rejection and honest zero-ILC/undefined-ratio handling. It invokes no CUDA/helper API and adds no new C++/CUDA code to compile.

After independent review, root alone may run one bounded exclusive diagnostic with existing profiling capability:

```bash
bash /okc/proj_rsi/candidates/hw_compression_counters/run.sh
```

Proposed inner180s/outer240s. `run.sh` profiles only `compression_generic/` and `compression_none/`, launch-count2, kernel/node replay, cache-control all, clock-control none, kill no. Setup/copy/queries/flush/oracles/cleanup stay outside selected ranges. NCU replay cache control is diagnostic instrumentation, not a performance benchmark; readonly W is not intentionally restored/repopulated by our code. Record actual replay passes and any tool limitations.

CPU report import after capture (CLI's actual option is `--print-units`, never the unsupported `--units`):

```bash
/usr/local/cuda/bin/ncu --import /okc/logs/hw-native-counters01.ncu-rep \
  --csv --page raw --print-units base > /okc/logs/hw-native-counters01.raw.csv
python3 proj_rsi/candidates/hw_compression_counters/parse_report.py \
  --csv /okc/logs/hw-native-counters01.raw.csv \
  --run-json /okc/logs/hw-native-counters01.json \
  --json /okc/logs/hw-native-counters01.analysis.json
```

Parser requires complete successful driver/cleanup, exactly two case-specific NVTX results, exact expected Nvjet, matching launch/resource dimensions and same fixed config. It reports signed aggregate DRAM-byte difference and L2 request/hit changes, with optional ILC availability; no latency verdict or weight-only compression ratio. One pair has no confidence interval. Missing metrics, changed shader, failed oracle or retained resources invalidate attribution and inform the next diagnostic. No GPU, profiler capture, speed claim, official slot, population variant or automatic rerun is authorized by this source plan alone.
