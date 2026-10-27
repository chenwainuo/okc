# Six-node calibration for the frozen copy reader

`state=CPU_IMPLEMENTATION_FROZEN_FOR_REVIEW; GPU_UNRUN; new_CUDA_CPP=NONE; speed_claim=NONE`

This measurement-control revision supersedes the **unrun** four-node sign-low GO. `hw_compression_signlow/` manifest11fff and its reviewed source remain unchanged. Run this one six-node matrix after new critic/root GO; do not run four nodes then a separate zero pair.

Calibration gap: the responsive known-zero control used **Lt**, while raw/byteplanes01 used a new volatile byte-copy reader and stayed flat. A positive Lt result does not establish copy-reader observability. This version adds only known-zero calibration to the same raw/sign-low payloads:

|payload|GENERIC NVTX node|NONE NVTX node|
|---|---|---|
|actual original native Wt64MiB|raw_generic|raw_none|
|exact sign-low XOR16, BASE0x3c00|signlow_generic|signlow_none|
|known-zero64MiB **bytes**|zero_generic|zero_none|

Known-zero bytes are a diagnostic source, **not** XOR-encoding original BF16 weight0 (which would yield different bits). The entire zero source is CPU-snapshotted/hash-checked as all0; its output complement is all0xff. Both flags use unchanged helper7f2/8f32 D2D+streamsync. No foreign-memory SM rewrite, shader, geometry or other format is added.

Raw still requires actual GPU seed5151/Torch2.8.0+cu128 and nativeWtSHA09d1920b…168120, with no seed/value specialization. Sign-low encoding/decoding and full65536-pattern CPU dependency780f are unchanged. **Before cloning**, decoding the complete GPU-created64MiB sign-low source must recover original native bytes, with CPU/GPU encodings equal. No arithmetic precision is changed; this reader only copies uint8 storage.

All six cases import frozen8e3d's exact CUDA d5b5/bindings e391/Case/copy_extension: `okc_hwc_copy64m`,128blocks×256threads, volatile128bit loads/stores over every64MiB byte. Runtime JIT/resource/occupancy gates remain32regs/0stack-spill-local-shared and occupancy≥1. No new CUDA/C++ source exists here. Six source owners, six ordinary main outputs and six retained fresh outputs give **18disjoint full spans**, checked in driver and reconstructed from recorded pointers in parser. Ordinary-output compression stays UNKNOWN if query unsupported.

Each case gets its own full CPU snapshot, eager/main-graph/fresh-output checks, expected-byte-complement poison, and immediate full-byte checks on the selected actual replay before any repair. No graph replay follows selected profiling. Full original/prepared/foreign sources remain immutable. Reset all main/fresh graphs, drop aliases, checked/idempotent close all six owners to Region0/undefined tensors; retained-resource warnings invalidate the version. Cleanup finally remains AST-identical to the valid frozen four-node protocol.

CPU gate:

```bash
python3 proj_rsi/candidates/hw_compression_signlow_zero/preflight.py
```

No CUDA API or extension build locally. Verifies all65536XOR bit patterns against frozen NumPy proof, known-zero64MiB hash/0xffpoison, exact frozen dependencies, same21metrics/cache/clock method, six-case schema/18spans, and byte/CSV/lifetime/format/zero negative controls. Parser accepts a legitimate0read-byte GENERIC observation; each NONE comparison must have a positive denominator. The missing calibration pair, duplicate labels, invalid units, failed GPU inverse, poisoned zero metadata, overlapping fresh/source spans or only four cleanup records are rejected. Runtime results remain unrun.

After independent GO, root alone runs one exclusive bounded matrix, proposed inner180s/outer240s with existing profiling capability:

```bash
bash /okc/proj_rsi/candidates/hw_compression_signlow_zero/run.sh
```

Same21supported DRAM/L2/ILC/launch metrics, kernel/node replay, cache-controlall/clock-controlnone/killno; six NVTX ranges and launch-count6. Preparation, snapshots/D2D, poison,256MiBflush, full byte oracles and cleanup stay outside selected nodes. Capture actual replay passes/warnings and node cleanup. No timer, GEMM, decode backend, precision relaxation, other format or performance claim.

CPU import after root capture:

```bash
/usr/local/cuda/bin/ncu --import /okc/logs/hw-signlow-zero01.ncu-rep \
  --csv --page raw --print-units base > /okc/logs/hw-signlow-zero01.raw.csv
python3 proj_rsi/candidates/hw_compression_signlow_zero/parse_report.py \
  --csv /okc/logs/hw-signlow-zero01.raw.csv \
  --run-json /okc/logs/hw-signlow-zero01.json \
  --json /okc/logs/hw-signlow-zero01.analysis.json
```

Interpret each GENERIC/NONE pair first; compare sign-low/raw allocation ratios with same-reader zero as calibration. If zero responds while sign-low is flat, disposition of this specific format/path is stronger. If zero is also flat, **reader/observer responsiveness remains unestablished**, and neither this flat sign-low result nor old byte-plane flatness proves data incompressibility. A reader/policy question then precedes any codec conclusion. Zero response is an observation, not a required positive result or family-closure gate. A sign-low response still supplies no serving speed, transferable WGMMA-layout ratio, certified wire-byte count or whole-W factor. ILC success0 remains insufficient to deny read benefit; aggregate counters include output/cache/metadata and one ordered matrix has no CI. Root chooses the next evidence-driven question; no automatic matmul or rerun follows.
