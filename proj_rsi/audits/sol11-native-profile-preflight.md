# Native-X versus DP counter diagnostic: CPU preflight

Ready for independent root review; GPU UNRUN by this worker. This diagnostic
asks why the native-X lineage retains a gap after removing packed-X preparation,
not for a timing decomposition or another benchmark. Existing native-x-01 paired
medians are native31.856us, DP34.096us, DS35.312us, actual incumbent23.968us;
native wins all32 against DP/DS and loses all32 against incumbent. Those results
remain attached to native-x-01. This two-arm capture cannot attribute the whole
incumbent gap or establish speed causality from counter differences.

Frozen new driver: `tools/native_x_compare_ncu.py`, SHA256
`16e7502719383460c833b1319de5764f455ab78bfecc8cccb8ec745a697c5a1b`.
CPU result: `audits/sol11-native-profile-cpu.json`, SHA256
`d5673504990704a08bbcf3b56069dffcb979f8d4c818e42886fb2653e0c1d691`.

CPU command (executed successfully):

```sh
python3 proj_rsi/tools/native_x_compare_ncu.py --cpu-check \
  --json proj_rsi/audits/sol11-native-profile-cpu.json
```

CPU mode replaces the extension-build importer with a rejecting stub and uses a
call-profile tripwire against execution of torch.cuda or cpp_extension functions;
CUDA initialization remains false before/after. No CUDA API, extension build,
prepare, profiler, benchmark, remote GPU job or GPU context was executed here.
Full frozen binding, physical/canonical gates, maps and four negative controls
pass for each arm: PC binding, failed-predicate inversion, role-branch mutation,
and mapper predicate inversion independently of canonical equality.

Pins checked on every run:

| Input | Manifest SHA256 | Generated source SHA256 |
|---|---|---|
| DP | `281ff5a3ccc787ddeec466722d899f4b217c4d44248de0493188fb9b04718241` | `97dd5613f9dfe859c87bb62f21e6d9543c10cdc7fc7310c2e87e40246fb1cf3b` |
| Native-X | `025f4a2de43f7c14a4ee4779dcb6a929974311b0380b433710ed4022beef50fc` | `ffd30a15ec5c1b5cc6fc1d550f7da512783f006f8a34cc37093ea18269d7e377` |

All manifest files are pinned, plus exact group8 parent driver
`245b78898a16c2087b00924cd809cd33fb74bfd7c19c9012bd7d040005485702`
and native trial manifest
`a73cff33a3d87ca925f9bd025c70ad1bfeeac1c94799e4fcd23de45a9e8f4821`.
Trial verify_lab/verify_frozen and exact canonical PC+predicate+instruction equality
are reused. Each actual JIT .so SASS is saved and hashed before first candidate
math. Physical A4 retirement/fence gates run on both actual binaries; native
layout/protocol/physical proofs and DP scalar-store/proxy/owner-barrier gates run.
The reused native proof contains fixed canonical PCs; its applicability to actual
JIT is established by exact semantic equality. The new role and counter PC maps
independently derive actual PCs/CFG, with no historical PC constants.

The source region sizes derive Wfull=102400, Bfull=102448, Bempty=102512,
Xfull=136192. Actual wait operand immediates plus disjoint CFG roles classify all
fast/outlined waits into math Xfull/Bfull or producer Wfull/Bempty. Exact frozen
source/canonical equality is required for these address interpretations. Each
fast failed predicate branches to a matching outlined TRYWAIT with identical
operands, retry predicate and a return to immediately after its entry branch.
Math startup Xfull and loop Xfull are retained separately by PC placement;
no Xfull/Bfull distinction is inferred from an old PC list.

Maps retain math bar5, LDSM, elected TMA, proxy, WGMMA fence/wait1/finalwait0,
producer owner barriers and every unclassified instruction. Native32LDSM,
8bar5,8mathTMA,0mathLDG; DP16loop LDG128 plus16 predicated common-setup LDG128
are verified. Setup instructions are retained in the common partition rather
than silently attributed to producers. Native raw X TMA/LDSM hardware storage-bit
roundtrip remains UNPROBED; no hardware-bit pass is claimed.

Runtime exact resources are122registers,0local,512threads,65536rounded registers,
1CTA after opt-in for each main; dynamic shared137232native versus103600DP.
DP pack has22regs/0local. There is no incumbent prepare/autotune. Fresh separate
X/output/counters/packed-weight/escape allocations per arm; DP alone allocates
1MiB XP, native XP argument is empty and native host source maps current x.data_ptr.
One NVTX profile range replays DPpack+DPmain, then native main: three selected
nodes. Immediately after this range, each actual selected output passes the task
oracle and finite/poison checks before any replay/workspace mutation. DP full1MiB
XP equals current-X byte permutation; currentX/W/packedW/escape are byte immutable,
and counters zero. Graphs drain/reset in finally. Existing same-kernel basis,
FP64, normal/outlier/probe and graph-mutation evidence is in native-x-01; this
counter driver does not rerun those broad validators under NCU.

Proposed normal profiling cap only: inner240/outer300 seconds including compile
and cleanup. Root alone executes after independent GO in its exclusive GPU slot.
Use a fresh report/JSON path. Timeout or signal may bypass Python finally; such
an attempt is invalid/incomplete until root confirms process/context drain.

```sh
timeout --signal=TERM --kill-after=10s 300s ncu \
  --replay-mode kernel --graph-profiling node --target-processes all \
  --cache-control all --clock-control none --nvtx --nvtx-include profile/ \
  --launch-count 3 --kill no \
  --section LaunchStats --section SourceCounters \
  --section MemoryWorkloadAnalysis_Tables --section SchedulerStats \
  --section WarpStateStats --section SpeedOfLight \
  --export /okc/logs/native-x-compare-ncu01 -f \
  timeout --signal=TERM --kill-after=10s 240s \
  python3 /okc/proj_rsi/tools/native_x_compare_ncu.py \
  --json /okc/logs/native-x-compare-ncu01-driver.json
```

Retain raw/source CSV, report, session/host logs, driver JSON and both JIT SASS.
Require >=200 samples per math/producer role and >=30 in a leading phase before
qualitative phase ranking; otherwise report insufficiency. Keep full denominators,
all common/setup and unclassified residuals, and per-replay raw/source execution
totals independently. Failed predicate counts are failed lanes, not initial
readiness or exact warp retries. TRYWAIT can block on its first successful
execution; division by32 gives only a warp-equivalent normalization without a
uniformity proof. PC fractions are not microseconds. Cache-control all per-node
replay differs from normal pack-to-main consumption. NCU durations are not a
pipeline benchmark. Report missing/unsupported metrics rather than inventing zero.
