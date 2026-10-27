# Four-node byte-plane / allocation counter diagnostic

`state=CPU_IMPLEMENTATION_FROZEN_FOR_REVIEW; GPU_UNRUN; speed_claim=NONE; formats=raw+byteplanes64_only`

This **supersedes the earlier conditional two-node transformed-copy proposal**. Changing from Lt to a byte reader requires its own raw-weight controls: four selected nodes `raw_generic`, `raw_none`, `planes_generic`, `planes_none`. This is a causal matrix, not a format/geometry autotune. Native-normal Lt and zero Lt records and all previous CPU proofs remain frozen.

`raw` means the actual original contiguous prepared Wt, BF16[4096,8192], viewed as67,108,864bytes; it is not W's untransposed layout. Reproduce only the existing GPU test seed5151, Torch2.8.0+cu128/reference generation and exact WtSHA`09d1920b4f42bd5ec1fbaa26f2ac936d688a81574c605b5ec474f0ee42168120`. No value/seed specialization exists in the copy kernel. `byteplanes64` bijectively maps each128B/64weight group from interleaved low/high bytes to64low then64high bytes. Exact CPU/GPU-prepared full-byte inverse checks precede capture. No arithmetic decode, GEMM, quantization, other format, population change, clock change or benchmark is included.

Four **independent** foreign weight owners use the identical frozen lifetime-proven helper7f2/manifest8f32, D2D+stream-sync, with properties queried as GENERIC1/NONE0. Each source has its own complete CPU snapshot and disjoint64MiB output. Outputs are ordinary Torch CUDA allocations; queried flags/retain errors are recorded, and unqueryable compression remains UNKNOWN. Fresh-output captures are separate and retained too. Do not assume ordinary means NONE.

The new `okc_hwc_copy64m` kernel is identical for all four: fixed grid[128,1,1], block[256,1,1], source/output64MiB and16B alignment/range guards. Each thread copies128uint4s at stride32768vectors. Every vector uses explicit **volatile** PTX128bit load and store; no checksum or dead load/output path. Local nvcc12.8.93 SM90a build shows32registers,0stack/spills/local/shared and actual LDG/STG128 STRONG.SYS. That is this reader's emitted instruction policy; it differs from Lt. Runtime checks CUDA attributes/occupancy and actual JIT cuobjdump REG32/STACK0/SHARED0/LOCAL0 before any data capture. No flags/geometry variant is selected.

Each actual destination is initialized to its expected payload's bytewise complement, guaranteeing every byte differs, before eager validation, the main graph validation, a fresh-output graph and the selected profiled replay. Exact full64MiB comparisons immediately follow the selected actual replay **before any later replay**; there are no postprofile graph replays. Sources/original raw/Wt/planes/foreign bytes stay immutable. All main/fresh graphs reset before aliases drop; every owner must close twice safely with Region0, and any retained-resource warning invalidates the trial. These are byte-transport gates, not GEMM/full-basis correctness claims.

CPU gates and frozen evidence:

```bash
python3 proj_rsi/candidates/hw_compression_byteplanes/preflight.py
```

This invokes **no CUDA API**: all65536bit pattern/byte-plane inverses, index bijection, all4194304vector-address coverage, byte oracle corruption controls, four-case CSV/schema/unit/NVTX/resource negatives, nvcc device build, host C++ syntax, PTX/SASS/resource proof and manifest. `preflight.json`, `build-proof.json`, `copy.ptx`, `copy.sass`, `copy.sm90a.cubin` retain the evidence. Local host headers are CPU Torch2.13; actual Torch2.8 JIT/build and launch are still first GPU-side gates.

After independent critic GO, root alone may run one exclusive bounded trial, proposed inner180s/outer240s, using existing profiling capability:

```bash
bash /okc/proj_rsi/candidates/hw_compression_byteplanes/run.sh
```

Same21 supported GH100 metrics as native/zero counters, four NVTX ranges/launch-count4, kernel replay/node graph, cache-controlall/clock-controlnone/killno. Setup/JIT, transforms/D2D, queries, poison,256MiBflush and all oracles are outside selected ranges. NCU memory replay/cache behavior is diagnostic instrumentation, not serving timing. Capture actual pass counts/warnings, exit status and process/node cleanup.

CPU import/parser after capture:

```bash
/usr/local/cuda/bin/ncu --import /okc/logs/hw-byteplanes01.ncu-rep \
  --csv --page raw --print-units base > /okc/logs/hw-byteplanes01.raw.csv
python3 proj_rsi/candidates/hw_compression_byteplanes/parse_report.py \
  --csv /okc/logs/hw-byteplanes01.raw.csv \
  --run-json /okc/logs/hw-byteplanes01.json \
  --json /okc/logs/hw-byteplanes01.analysis.json
```

New parser requires exactly four unique labeled nodes, new copy shader identity, same matching launch/resources and base units; it cannot reuse fixed-Nvjet assumptions. It validates complete driver/oracle/ownership metadata, records raw units and unavailable ILC ratios honestly, and checks DRAMbytes=32×sectors. Kernel shared memory0 is distinct from NCU's possible1KiB Hopper driver reserve.

Interpret **GENERIC/NONE within each payload first**, then compare the two allocation ratios. Absolute counts versus prior Lt do not isolate format because the reader changed. A stronger planes allocation response supports representation responsiveness in this copy path, with no serving gain yet; equal raw/planes responses could reflect the new reader/policy; flat results inform the next version. Aggregate DRAM-unit counters include output/cache/metadata; they are not certified wire counts, a whole-W factor, a confidence interval, or a kernel latency comparison. Same instruction/resource identity plus fixed volatile address coverage is substantially stronger work control than shader-name equality alone. ILC success0 remains insufficient to deny a read benefit.

No automatic rerun, new transform, matmul backend, official submission or timing claim is authorized by this package. Root owns the next evidence-driven decision after this one reviewed matrix.
