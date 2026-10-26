# rs256full — one geometry, port-model diagnostic

`state=CLOSED_PORT01_RUNTIME_PREFLIGHT_INVALID; full_matmul=NOT_IMPLEMENTED; probe_kernels=UNRUN; pw36=UNTOUCHED`

**port01 outcome:** exact frozen source `cc46c06739966806e4bd86ca078824e6d768068e8bac9426937b0f2b5fbf1774` built successfully on the H100, then failed its runtime occupancy check before launching a probe kernel: mode1 regs168/local0, requested dynamic shared164928B, occupancy0. The harness calls `info<1>()` before `launch<1>()`, while only `launch` sets the opt-in dynamic shared-memory attribute. This initialization-order bug invalidates the occupancy rejection as geometry evidence. The actual default attribute value was not logged. All numeric/snapshot/write/overlap tests are NOT_RUN; no timing samples exist; both port models remain unresolved. RootPID348717 and container are terminal. `port01.json` preserves exact logs/hashes. Per the preregistered runtime-failure rule this run is closed: no source fix, retry, or full-matmul extension. The source remains frozen; commands/interpretation below describe the attempted experiment, not authorization to rerun it.

The proposed kernel is materially different from the closed `rq` NT256/two-pass m256 paths and pw15's one-math-WG full-K path: channels128 × tokens64, two math WGs, full K, four token tiles at M256, pooler grid32×4=128 CTAs, direct output stores. However, its shared-memory budget has two incompatible interpretations. A valid NO-GO requires distinguishing them first. The initial aggregate-port rejection was withdrawn; neither a wall-time ceiling nor a ≥1µs speedup is established.

Evidence read: `handover_folder/04-track-rs128-m128.md`, `05-track-rs32.md`; `rs32/submission.py` RQ/RS32 HALF; `audits/pipeline-diff.md` §§3/4c; `audits/m128-plan.md` §4.1; `audits/p4-rs128-anatomy.md`. The 128B/clk aggregate port is a repository cost-model assumption. Existing ladders also vary pipeline/ingress, so they are insufficient to prove shared read/write arbitration. P4's measured effective tensor clock also differs from nvidia-smi; cycle counts must not be converted using the reported1980MHz. Existing in-prepare/graph timings must not be compared directly to a CUPTI kernel floor.

| Per CTA/K256 round | Bytes |
|---|---:|
| W bulk fill | 49,152 |
| four SW128 X TMA boxes, each64×64 BF16 | 32,768 |
| packed W LDS | 49,152 |
| X WGMMA reads, two WGs | 65,536 |
| all reads | 114,688 |
| all writes | 81,920 |
| aggregate | 196,608 |

Tensor floor=1024clk. Aggregate128B read+write model floor=1536clk; independent128B read/write model floor=max(896,640)=896clk. The disagreement is large enough to warrant one bounded microprobe. Two80KiB slots plus conservative alignment/control=164,928B. A double buffer128regs/thread + D32regs + streamed Q12regs is feasible only with producer40/math232 redistribution; compile proof below confirms this synthetic code has no spills.

`port_probe.cu` compiles exactly three controls, one geometry:

- `RS_LDS_DEC`: two cooperative math WGs; original RQ dec1; 48KiB LDS + 64KiB WGMMA B reads per K256 round. Fixed resident slot0, two A register buffers,16 m64n64k16 per WG/round, one commit, wait_group1. No TMA traffic in the measured window.
- `TMA`: one producer lane fills disjoint slot1 with one48KiB W bulk + four8KiB 2D SW128 X TMAs per round. Source has only80KiB unique bytes, repeatedly read without an HBM flush. Producer-only measures ingress/issue/barrier costs independently.
- `COMBINED`: same math and producer, producer paced at most two rounds ahead of WG0. Completion count at each consumer's end and temporal overlap are recorded, so slow producers cannot silently masquerade as overlapping one80KiB fill per math round.

Both slots initially contain synthetic BF16 ones (lossless12bit W, zero escapes). All1,048,576 FP32 accumulator results are written and checked exactly against256×iters. All128 producer80KiB snapshots are checked byte-for-byte. Writes/progress, uniqueSM count, runtime spills/occupancy are checked. Result/snapshot output stores start only after both measured windows end. GPU allocations and binary belong to this process; no shared torch scratch, counters, or submission state. There is no matmul API, fallback, prepare, judge dispatch, or production escape/probe handling.

CPU gate (no GPU API):

```bash
python3 proj_rsi/candidates/rs256full/preflight.py --build --json /tmp/rs256full-preflight.json
```

This proves LDS byte coverage, SW128 bijection, every synthetic packed register pair, disjoint buffers, compiles/links with `sm_90a`, rejects C75 diagnostics and any stack/spills, and checks SASS. Local CUDA12.8 result: all3 controls168regs,0stack,0spill; math controls32 HGMMA/2 WARPGROUP.ARRIVE/36 LDS.128 across two unrolled rounds plus initial load. Combined4 UTMALDG +2 UBLKCP (one initialization bulk plus one W bulk). `preflight.json` records current source SHA256 and exact build evidence.

Root-only launch, inside the root-owned exclusive container/lock. No child has launched or remotely compiled this:

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v proj_rsi/candidates/rs256full/port_probe.cu -o /tmp/rs256full-port-probe
timeout 30s /tmp/rs256full-port-probe 128 9
```

Nine rotated/interleaved samples/control; two warmups excluded. Every sample prints cycles per K256, producer cycles per80KiB, overlap, writes per consumer round, exact-check status, and uniqueSM count. No wall-time speedup conversion.

Falsifiable interpretation, set before GPU execution:

1. Any numeric/byte/state/occupancy failure, C75 warning, stack/spill, or source mismatch → INVALID; no performance inference.
2. Combined overlap<0.95 or writes_per_consumer_round<0.95 → INCONCLUSIVE on port sharing; producer did not supply the intended simultaneous traffic. Producer-only cycles then identify the ingress-side problem.
3. If resident consumer≤1200clk/round and combined stays within1.10× with adequate writes/overlap, aggregate128B read+write model is refuted for this workload. A full-K prototype remains conditional: measured hot-L2 result excludes full-K source delivery, real escape lists, cold start, and direct-store cost.
4. If combined≥1.35× resident, adequate writes/overlap, and approaches/exceeds1536clk, result supports material read/write contention. It does not uniquely prove physical bank-port sharing: TMA issue, LDS scheduling, and producer bookkeeping are residual confounders. Park this geometry unless the measured same-unit tuned r6 kernel baseline leaves ≥1µs after plausible start/store costs.
5. If resident itself>1200clk or 1.10–1.35× slowdown, the two simple port models are insufficient. Record the result and stop; no geometry sweep or implementation under this task.

Even a favorable result is not a speedup claim. The proposed full-K plan replays packed W four times through L2 (201.3MB) and X across row blocks (134.2MB), while avoiding the old8.4MB split-K partial writes. Those delivery/tail tradeoffs need a later same-run, same-unit r6TUNED comparison. The original task's ≥1µs pooler-m256 kill bar remains unchanged.
