# Source-track takeover audit

Snapshot: 2026-10-26T19:35:25.689201+00:00. Actor: `/root/proj_tracks`. Scope: readonly source/log audit; no GPU launches, remote mutations, kernel edits, or commits. Only this audit file was written.

## Immediate decision

Recover the existing H100 queue before launching any additional track work. The highest-value completed design is the `pw36` merge, followed by the already-queued RS128 stream-K and P4 patch-table experiments. Every queued measurement already has a concrete process handle; resuming these tracks means harvesting those handles, not submitting duplicates.

This projection challenge targets `<3000 us`. It is a distinct workload from earlier serving/MoE work; its wins do not establish that older 3x objective.

## Verified H100 queue snapshot

Observed on `root@107.170.6.60`, 2026-10-26 19:31 UTC. All jobs use `/tmp/gpu.lock`; no new job was started by this auditor.

| order observed | handle | command / output | state at inspection |
|---|---:|---|---|
| active | host `308855/308858/308859`, Docker client `312157`, GPU Python `312199`, container `a28b13f4ee6b` | tensorcore `run_box.sh ladder9 3600 ... ladder.py ... qkv_gate,router 16,128,256 3,8,4,5,0 10`; `logs/tensorcore_ladder9.log` | same GPU job for >24 minutes, GPU 100%, ~1703 MiB; last log update 19:08 |
| waiting | `311438` | `timeout 900 python proj_rsi/candidates/rs128/lab.py --plan sk --layers pooler,qkv_gate,unpooler,o --reps 40 --graph`; `logs/rs128/sk1.log` | output absent, not yet executed |
| waiting | `313704` | bandwidth `lab/batch5.sh`; `logs/bandwidth/batch5.log` | empty output; final official `v9.json` pending |
| waiting | `316870` | latency `batch.sh`; `logs/latency/b7.log` | empty output |
| waiting | `318362/318363` | P4 `rs128p/lab.py --plan anat ...`, then official targeted harness | `pt-lab.log` / `pt-bench.log` absent |
| waiting | `325294/325297/325298` | tensorcore `cmp8`, variants v22/v23/v24/current, shapes qkv16/128 + router16/128 | `logs/tensorcore_cmp8.log` empty |
| waiting | `334428/334430` | pw36 parent comparison script | `logs/pw36/run3.log` empty; script runs pw35, rs32, pw36 in order |

The active ladder's final printed rows cover all qkv modes at m16/128/256, each ending `ok`. The next loop is router m16; the hang is somewhere before that entire row is printed (possibly torch baseline or one of modes3/8/4/5/0). Exact kernel/mode is not identifiable from the current log. This is a stalled experimental measurement, not a correctness verdict for a submitted candidate. Preserve partial logs before root performs any bounded termination. A timeout on the Docker client alone does not prove daemon-child termination.

The following full qkv durations are already enough to rule out shipping this tensorcore file for those shapes: m16 `26.35` vs torch `22.38`, m128 `32.35` vs `22.43`, m256 `41.78` vs `26.75` us. These are same-process H100 observations, not judge timing.

## Priority and stop criteria

1. **pw36 integration:** combine two judge-supported changes, then harvest its existing three-parent comparison. Integration owner and root handle it. Prefer this over additional speculative variants. No scored claim before its own bake/prepare/correctness evidence.
2. **RS128 stream-K:** harvest `sk1.log` before any source change. `plan_sk` compares regular-grid s3s5/s4s5hz/s4s5z against corresponding `KS=-132` variants, with cuBLAS in the same run. Require >=0.3 us in-prepare against the shipped actual choice before porting; require official checks. No evidence of multicast implementation or a queued multicast run was found: it remains an unimplemented hypothesis, not live work.
3. **P4 producer-written patch table:** harvest the existing queued lab plus official harness. Require >=0.3 us in the same run against shipped `s4s5hz`, then port one mechanism only. Do not reopen the closed escape prewalk/switch/polling experiments.
4. **Latency loader:** harvest b7 warm/nowarm comparison and correctness first. Microbenchmarks already show bulk throughput near 22–23 GB/s/SM for large streams; the end-to-end gap persists. New uncommitted dry-loop instruction-warming change is an unvalidated hypothesis, not a win. Stop if it remains behind the incumbent's actual picks; comparison against torch alone is insufficient.
5. **Bandwidth school:** finish queued batch5 / v9 official harness, then compare mechanisms with incumbent picks. Its large-m mma.sync path is already closed by measured issue rate; current file routes m>=33 to torch. Avoid re-opening that large-m design.
6. **Tensorcore school:** retain partial ladder9 and terminal cmp7 data; diagnose any hang before allowing cmp8 or further probes to occupy the queue. Existing complete BF16 kernels lose while raw mainloop probes can look fast. Router improvements must be compared against shipped pw35 rtk, not two-call torch alone.
7. **Contrarian:** no live queue was found. Existing bench5 finished at 18:43; official correctness passed earlier. Current evidence does not beat the compressed incumbent: most small-shape gains are relative to torch; large shapes route to torch. Treat as mechanism inventory pending one fair verifier run, not as a candidate ready to submit.

## Track-specific findings

### RS128 / P4

`rs128/NOTES.md` is absent. Existing kernel bytes are already committed; lab contains a new stream-K plan. m256 remains closed for this rq design: 100% resident tensor rate does not compensate operand delivery plus reduction tail. The root should reconstruct any missing notes from the existing logs, without retesting the closed variants.

Dirty P4 files implement FL bit11 (`2048`) patch-table paths. Slot tables contain one lane mask per math warp and up to two patch words per lane, written by producer warp11. Patch-table paths add shared memory and include `ecap<=64` validation. Build-m recorded oversized `r8t` / `s4hzt` allocations of 249312 bytes; dirty source/build-n corrects resident r8t depth8->7, giving 218536 bytes. Production m128 table path `s4s5hzt` uses 156984 bytes vs baseline144520; 168 registers, no local/spill in build-n. Correcting resource fit is not numerical validation or a timing win.

The dirty lab intentionally narrows ANAT to shipped/resident/table counterparts. Its retained noescape controls make the table hypothesis distinguishable from earlier escape-list experiments. Queued official harness covers verify16 and verify32 on pooler/qkv/unpooler/o; m256 test is diagnostic only and must not reopen automatic m256 shipping.

### Latency

Latest terminal b6 shows default qkv m1 ~18.8 us and router m1 ~20.9 us kernel-only plus ~0.7 us memset. Larger router split cuts router m1 to9.74 us, still far from the small-shape incumbent. The loader microbenchmark is better than those end-to-end numbers: at50.3MB, `ld unroll16` / `cpa depth8 stage8K` reach2.89TB/s (~22.5GB/s/SM), `cpa depth4 stage16K`2.90TB/s (~22.7GB/s/SM). Thus do not continue treating raw bulk bandwidth alone as the unresolved bottleneck.

Current dirty source adds a consumer dry iteration before producer traffic to warm instruction cache, resets accumulators, and gates producer start on a CTA barrier; `LATENCY_DBG=512` disables it. This change needs the b7 paired warm/nowarm data and original odd/probe correctness before any port. No edit was made by this auditor.

### Clean-room evidence quality

The school DESIGN tables acknowledge early contention / replay floors around12–14us. Their blanket H100→H200 predictions are hypotheses, not validated transfer. Existing comparisons against torch often omit the incumbent's tuned cuBLAS/packed choice. The proposed verifier/synthesizer artifacts `proj_rsi/audits/cleanroom-measure.md` and `cleanroom-plan.md` do not exist, so that stage remains unfinished. A single locked comparative verifier run, with only valid nonhanging schools and pw36/actual incumbent as reference, is required before any port from these schools.

## Source provenance at inspection

All following local hashes equal the remote `/root/okc` bytes at inspection. Pending tasks read these files at execution time; changes after this snapshot invalidate that correspondence.

| source | sha256 |
|---|---|
| `proj_rsi/candidates/rs128/submission.py` | `afe34ccd1a09cde358a32f9272dcaa8c8010af88b28ff1ac6629de9c255d2f02` |
| `proj_rsi/candidates/rs128/lab.py` | `70bb751379b5834287fc712f8b41a0302b9632a3446953ae450049ad1525f516` |
| `proj_rsi/candidates/rs128p/submission.py` | `24c517b809d50fb82b59e35c90990d3a87ef7a623219f37dede50a377a005af4` |
| `proj_rsi/candidates/rs128p/lab.py` | `bf50a0442999dcc9d77413b9ea112f48333f4b8485177572622a532d1ba1ede6` |
| `proj_rsi/cleanroom/latency/submission.py` | `5dc2891de47e04d573dc930e9ebe4c9a2a595ead6dbfd57f33fa4398e1972af7` |
| `proj_rsi/cleanroom/latency/batch.sh` | `4a81c0fd8831cf69a87ddc9aa411f3f3e1e82d8d7636e6cf6713ee1bbfb67918` |
| `proj_rsi/cleanroom/bandwidth/submission.py` | `9ac20ff1a8711db694a7ab1cc70066f33dee402bc2328da520221e119f641fe6` |
| `proj_rsi/cleanroom/tensorcore/submission.py` | `b92f012bf96a168fda296073e2e1fb3030289caa258a54a09a0ff6b07295b522` |
| `proj_rsi/cleanroom/tensorcore/tools/var_v23.py` | `76eef6a1f82f17ffbb5ed6a6283088c2c6d5e75a60628c48fd1f2c5fa0c0f321` |
| `proj_rsi/cleanroom/tensorcore/tools/var_v24.py` | `9757828dc951ba2a22fec16fa04366f2c3d85806ee9be50de9cd1c2b2820ddda` |
| `proj_rsi/cleanroom/contrarian/submission.py` | `47027d0dbeff674d95f4ee80a37f0e6f1e8c8919f112e7528a26ce10d031c094` |

No source patch is justified yet: the relevant new variants are already compiled and queued, and their decisive measurements have not run. Root should consume existing evidence before spending new GPU time.
