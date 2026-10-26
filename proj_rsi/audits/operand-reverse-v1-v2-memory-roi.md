# Reverse v1/v2: local load-work amplification, stable downstream traffic

`state=RAW_COUNTERS_IDENTITIES_ORACLES_TERMINAL_VERIFIED; performance_admission=NONE; source_edits=NONE; V3_speedup=UNPROVEN`

V2's scalar direct-A loader generates~4x global load requests and~2x L1 sector instances, with EXACTLY the same L1 miss count/L2-to-L1 return sectors and slightly fewer DRAM read bytes. This supports a frontend/request/dependency question rather than an off-chip byte explosion. It does not prove a saturated unit or assign the timing regression to one cause. V3's coalesced8B pair-lane loader is a specific way to test the local-work trade while retaining fewer transpose instructions; its correctness, resources and unprofiled performance require their own evidence.

Artifacts are in `/home/ubuntu/okc/results/proj-rsi/takeover/20261026/reverse-v1-v2/`: `analysis.json` retains28finite relevant metrics, exact raw units/values, derived count identities and all4raw artifact hashes. Original files are unchanged. CSV has a Docker banner; parsing begins at the first quoted ID header, then one units row/two result rows/933columns. `analysis.json` is a compact extraction, not a replacement for the raw CSV/binary report.

## Identity and collection gates

Host log independently verifies start23:27:34Z/end23:27:50Z, exit0, two selected kernels with32metric-replay passes EACH. These are not32independent benchmark observations. Root separately verified no remaining container/process/GPU allocation. The log warns about unmodified clocks and six unavailable CTC RX/TX-user metrics; these stay unavailable and are not used as zeros. All28selected memory/instruction/scheduler fields are finite; relevant units and DRAM bytes=32×sectors checks pass.

| Field | v1 | v2 |
|---|---|---|
| Generated source SHA |ca92a4fb0d3a488768c4531ba315e8d0327b0a4c4a994147b12c61a929ae85f8|97664bdee7593d46e0493b9599062c37af150262d597002a722016f060548faf|
| Exact demangled kernel |`rq<128,8,1,0,0,0,20>`|`rq_v2<128,8,1,0,0,0,20>`|
| Grid/block |128x1x1 /384x1x1|same|
| Registers/thread |168|162|
| Dynamic/shared-driver/total bytes |103600/1024/104624|same|
| Register/shared occupancy limits |1CTA/SM each|same|

Source file hashes equal the driver's frozen identities. Dynamic103600B is the source allocation; total104624B includes1024B driver allocation, avoiding a false shared-memory mismatch. Both immediate profiled-output task oracles pass (relative L2`1.421e-4`, worst-row`2.633e-4`,12probe rows exact), poisoned outputs overwritten, X immutable, counters reset. Driver packs/unpacks all original W bits before capture and resets both actual graphs at completion. It does **not** perform a postprofile W/pack/escape-byte snapshot; those comprehensive checks belong to the separate valid performance lab.

Kernel/node replay, `cache-control all` and application256MiB read flush are explicit diagnostic conditions. Both kernels share the same fixed W/X/pack/escape arrays, use separate output/counters and have no prepare/autotuner selection drift. Collected profile duration is deliberately excluded from performance evidence here. Separate unprofiled performance already established v2 F1`57.023998us` versus conservative v1`47.231998us`, all21pairs slower (median loss`9.567998us`); this report tests causes, not that speed claim again.

## Counts and byte accounting

| Exact counter / raw units | v1 | v2 | Interpretation |
|---|---:|---:|---|
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` / blank |262656|1049088|+786432requests,3.994x including shared overhead|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` /sector |4196352|8390656|+4194304sector instances,1.9995x|
| same `_lookup_hit.sum` /sector |0|4194304|V2 duplicate sector instances are caught locally|
| same `_lookup_miss.sum` /sector |4196352|4196352|exactly equal|
| `l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum` /sector |4196352|4196352|same returned downstream sector volume|
| global LD hit-rate /% |0|49.99|not a49.99% DRAM traffic saving|
| global useful-data/sector ratio /byte/sector |32|16|half-use sector instances, with reuse in L1|
| `lts__t_requests_srcunit_tex_op_read.sum` /request |2332432|4108438|more L2 request work too; not4x read bytes|
| `lts__t_sectors_srcunit_tex_op_read.sum` /sector |5458382|5554136|+1.7543%, source-unit TEX READ scope|
| `lts__t_sectors.sum` /sector |6404010|6535628|+2.0552%, ALL-client/operation scope|
| `dram__bytes_read.sum` /byte |52152320|51472896|−679424B,−1.3028%|
| `dram__sectors_read.sum` /sector |1629760|1608528|consistent32Bsector conversion|
| `dram__bytes_write.sum` /byte |2434560|2362880|−71680B, also no downstream write explosion|

L2 read-source and all-client totals have different scopes;5.458M is not the all-client total. Individual L2 hit/miss/sector counters can also come from different replay passes and do not sum exactly in this capture; do not force conservation by silently replacing fields. Preserve the raw values and collection conditions.

The instruction mapping gives unusually exact local count checks. There are`64stages*8mathwarps*128CTAs=65536warp-stage instances`. V1 uses4vector128bit/thread load instructions per warp-stage; v2 uses16scalar32bit/thread instructions, carrying the same128MiB logical X per launch. The measured global LD instruction/request counts are math`262144`vs`1048576`, plus512common global-load requests. Their exact delta is786432. NCU's global LD warp-instruction counter independently equals those request counts. The512extra requests/2048sector instances are consistent with the unchanged producer escape-window loads in this particular seed/run; do not hardcode that overhead for arbitrary escape data.

V1's math sector instances are`64*65536=4194304`; v2's are`128*65536=8388608`. Adding2048common sectors gives both measured totals. V2's additional4194304instances all show up as hits, leaving misses unchanged. That is the source expectation for two16B halves requested by separate scalar instructions. It is local work amplification, not evidence that X's1MiB unique contents or128MiB requested SM payload doubled off-chip.

Both measured DRAM read totals are near48MiB unique packed W+1MiB unique X, despite128MiB logical repeated X requests. That is consistent with substantial cache reuse. DRAM counters are aggregate and also include escape/metadata/re-read effects, so they do not assign an exact W-versus-X byte split. Source/W-producer work also remains consistent at98304shared-load,540672shared-store,8192TMA-load and128TMA-store warp instructions in BOTH. These equal counts do not establish equal producer waiting/overlap.

## Scheduler evidence and confounds

| Metric (`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`; raw unit`inst`) | v1 | v2 |
|---|---:|---:|
| long_scoreboard |2.87|5.70|
| short_scoreboard |2.15|0.26|
| mio_throttle |0.96|0.09|
| lg_throttle |0.34|0.23|
| dispatch_stall |0.11|0.63|

These are normalized issue/stall ratios, **not absolute microseconds, exclusive percentages or PC-local critical-path time**. Long-scoreboard rising while short-scoreboard/MIO falls is consistent with a changed load/dependency pattern; it does not prove that the new scalar LDG itself causes the entire slowdown. Issue-active is30.03→25.93%of active cycles and tensor-pipe activity19.14→14.87%of elapsed peak. Percentages can fall while absolute request counts rise; differing denominators, instruction types and latency exposure prevent a direct saturation/ceiling conclusion.

This capture has NO `pcsamp` fields or PC/source sampling section. Earlier v1 SHFL/SEL load-dependency samples support why a loader revision was reasonable, but those sample counts are observations, not executed instruction counts or causal microseconds, and cannot be transplanted into this v2 report.

V1 versus v2 is not a clean one-variable experiment: v1 uses`ld.global.nc.L1::no_allocate.v4.u32`; v2 uses`ld.global.ca.u32`/SASS STRONG.SM. Cache allocation/coherence changed along with vector/scalar delivery. V2 directly refills retired A after wait_group1, while v1 refills separate raw Q before that wait; load overlap and A-register lifetimes differ. V1 redistributes producer64/math216 registers under static168; v2 removes that redistribution and compiles static162. Occupancy remains one CTA by the recorded limits, but code scheduling/register liveness differs. Aggregate metrics mix producer escape/global loads and W-TMA/shared work with X consumers.32replay passes and unforced dynamic clocks are further reasons not to treat normalized rates as independent causal time.

## What this supports for V3

The currently written V3 hybrid returns to v1's nc/no_allocate policy and raw-Q prefetch/A2 lifetime structure, uses8coalesced64bit/thread loads, then a two-lane demux with16SHFL+16SEL per thread/stage. It targets the newly measured scalar-request amplification while halving v1's fixed transpose work. This is a concrete trade with plausible information ROI, not a demonstrated win or isolated cache-policy fix.

If that exact loader compiles as intended and common metadata work remains unchanged, falsifiable counts are:

- Math LDG requests`8*65536=524288`; including the same512overhead, total524800, between v1/v2.
- Coalesced math sector instances4194304; including2048overhead, total4196352 like v1, rather than v2's8390656.
- Math SHFL+SEL warp instructions2097152, half v1's4194304; v2 had removed them entirely at the scalar-request cost.
- No hypothesis of a large DRAM byte reduction is needed: backend transfers were already similar. The question is whether request/conversion/dependency balance and prefetch overlap improve complete unprofiled time.

V3 still needs independent mapping/bit/resource/lifetime review and same-run correctness+performance versus v1/v2 AND actual incumbent. A gain versus a failed version is learning; adoption requires a gain versus the serving baseline. A nonwin should identify which of these count/overlap predictions failed, not reject the whole operand-reverse family or trigger an unmodeled sweep. No source edit, GPU launch, new counter run or promotion was performed for this audit.
