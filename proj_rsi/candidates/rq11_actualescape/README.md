# RQ11 escape accounting on exact CPU reference weights

**CPU proof PASS; actual CUDA fixture distribution remains UNKNOWN.** This uses the unchanged `reference.generate_weights(5151, device='cpu')` function, not NumPy synthetic normals, sampled matrices or a reset RNG per layer. CPU and CUDA RNGs differ, and this local environment is Torch2.13.0+cpu rather than the GPU runner's environment. These numbers must not be relabelled as actual H100/H200 weight compression ratios or serving speed measurements.

Owned tool: `proj_rsi/tools/rq11_actualescape.py`. Final v2 run:5.536s, one CPU compute thread, nice10, internal180s bound and external190s timeout/5s kill grace. No remote command, CUDA context, extension, prepare or GPU measurement occurred. The script verifies CUDA remains uninitialized. All94,371,840 BF16 values from the four actual CPU reference matrices reconstruct bit-exact through both formats. The generated2,097,152 FP32 router values are hashed and explicitly excluded; no FP32 truncation is allowed. All matrix hashes, exact128-bin h7 distributions, every candidate window's coverage, group lengths and warp-unit/lane-tail histograms are in `actual-cpu-reference.json`.

## Exact matrix results at incumbent-relevant geometries

The KS references come from frozen official rs32 source/log job fb548b5dc6f04696: qkv16/32/128 useKS7, pooler32/128 useKS4, unpooler32 ando32 useKS2. Their counts below are evaluated on **CPU reference weight bytes**, not observed GPU bytes from that job. Every KS in1/2/4/6/7/8 is also retained in the full JSON; this covers the later local qkvKS6 comparison without pretending its CPU weights are CUDA-identical.

| Matrix | KS | RQ12/RQ11 best B | Escape count RQ12/RQ11 | Escape rate RQ12/RQ11 | Max list RQ12/RQ11 | CAP RQ12/RQ11 |
|---|---:|---|---|---|---|---|
| qkv_gate2304×8192 |7|54 /57|2605 /255822|.01380% /1.35539%|8 /320|32 /352|
| o8192×1024 |2|54 /58|1009 /104766|.01203% /1.24891%|5 /141|32 /160|
| pooler4096×8192 |4|54 /57|4668 /454200|.01391% /1.35362%|13 /517|32 /544|
| unpooler8192×4096 |2|54 /58|3240 /834321|.00966% /2.48647%|11 /902|32 /928|

A warp-unit is the exact pinned `_pack_nat`16×64 fragment, not1024 arbitrary adjacent flat elements. Mean RQ11 escape-walk entries per unit are13.8792/12.7888/13.8611/25.4615 for qkv/o/pooler/unpooler, with maxima31/26/32/47. RQ12 means are.1413/.1232/.1425/.0989. These are counts, not decoder latency estimates. All escape fields are checked across all six KS values, including unequal split boundaries. An independent coordinate-to-group path agrees with count-matrix grouping for every physical list.

## Full payload, header, ESC and padding ledger

| Matrix / KS | RQ12 allocated packed+ESC bytes | RQ11 allocated packed+ESC bytes | Net allocation saving | Source-requested load saving* |
|---|---:|---:|---:|---:|
| qkv /7 |28,440,704|27,371,648|3.75889%|4.92052%|
| o /2 |12,714,112|12,189,824|4.12367%|5.40633%|
| pooler /4 |50,462,848|48,365,696|4.15583%|4.83716%|
| unpooler /2 |50,462,848|49,938,560|1.03896%|1.82249%|

*Source-requested loads are exact accounting for payload and the unchanged32-entry stock escape prefetch walker; they are **not measured HBM/L2 traffic**, whose cache behavior has not been probed. See `window-costs.json`. Allocation slack is not silently counted as if every padding byte were loaded.

RQ12 stores1024 low bytes plus512 selector/header bytes per warp-unit; RQ11 stores1024 plus384. All bits are occupied, so selector padding is0. Full128-row stages are12288/11264 bytes; HALF64 stages6144/5632; all plane/stride constraints are maintained and no extra stage-alignment padding is charged. A four-byte logical decode table is passed through the unchanged eight-byte host ABI argument; there is no separate per-weight/device table allocation. Shared-memory rings and protocol counters are separate equal-format costs, not disguised payload savings.

Every exact ESC list uses `CAP =32*ceil((max_list+1)/32)`, G=KS×NB×8 lists, four bytes per entry, and32 trailing sentinel words. Thus device-format allocation is:

```text
payload + 4*G*CAP +128
```

The result separately gives actual entry bytes, globally fixed-stride unused words (including each list's sentinel/padding), and128 guard bytes. It does not truncate at32 or infer tails from averages. Source `_rqesc` also returns a four-byte dummy tensor, but callers discard it; it is not charged as a retained serving allocation. Common full/HALF counter arenas are8×NB/16×NB bytes and are listed separately.

For example, qkvKS7 has RQ11 low bytes18,874,368, header bytes7,077,888 and ESC allocation1,419,392. Of that ESC allocation,1,023,288 are actual entries,395,976 fixed-stride unused bytes and128 guard bytes. The corresponding RQ12 ESC allocation is129,152. RQ11's large global max-CAP materially erodes its nominal8.333% payload reduction. UnpoolerKS8 has **exactly zero allocation saving** on this CPU fixture; this is a geometry-specific result, not family retirement.

The separate walker proof charges an initial32-word lane load for every group plus a new32-word window after every32 consumed entries, including a sentinel-only window when list length is a multiple32. Requested ESC words per group are `32*(floor(length/32)+1)`. This includes empty lists and boundary counts0/31/32/33/63/64; rounding down or using ceil(length/32) would undercount.

## Prepare-cache residency is a separate material cost

Pinned `_rqbake` retains `(lay,tab,esc6)` in `state['rq_pk']`; esc6 has six int64 fields,48 bytes per escaped value. A straightforward RQ11 implementation reusing that prepare/cache structure would therefore retain much more metadata even though the main kernel never reads esc6 directly. This conditional source-style cache ledger must not be hidden as generic tensor overhead:

| Matrix / KS | RQ12 packed+ESC+esc6 bytes | RQ11 with same cache policy | Additional RQ11 residency |
|---|---:|---:|---:|
| qkv /7 |28,565,744|39,651,104|11,085,360|
| o /2 |12,762,544|17,218,592|4,456,048|
| pooler /4 |50,686,912|70,167,296|19,480,384|
| unpooler /2 |50,618,368|89,985,968|39,367,600|

Across one listed geometry per layer, packed+ESC allocation falls142,080,512→137,865,728 bytes, while retaining stock-style esc6 changes residency142,633,568→217,022,960 bytes. This is not a complete prepare peak-memory measurement, allocator-reservation measurement or traffic claim. It excludes common original weights/other family packs and assumes one retained KS-specific ESC allocation per layer. Multiple selected KS plans require additional ESC arrays but share the packed layer/cache. The11-bit serving integration does not exist, so its cache retention is explicitly conditional; a new design could drop or compact that staging cache and would need separate source accounting.

## Proofs, preserved failures and next decision

`test_actualescape.py`:9 CPU tests PASS in.891s. It checks all65,536 storage patterns at all125 legal RQ11 windows and all121 legal RQ12 windows (16,121,856 pattern/window cases), exact frozen RQ11 byte layout over all storage patterns at three representative windows, real artifact hashes/cost identities, unequal KS split grouping, CAP/sentinel limits, nonfinite/mutation rejection and exact fixture-byte reproduction across accounting revisions. Full actual matrix payload and escape roundtrips pass separately. Payload SHA is ordered as real NB/KC/warp packing.

Attempt01 failed because the AST fixture extractor omitted FP32_LAYERS; no result was admitted. Its log, source bytes and incomplete payload are retained. Attempt02's CPU data were valid but charged a discarded4B dummy as retained and omitted the explicit esc6 residency ledger; exact source/results/tests remain attached to that revision. Final corrected output supersedes those accounting fields. Matrix hashes reproduce exactly between the two completed passes.

Reproduce locally without overwriting this frozen result:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 190s nice -n 10 python3 proj_rsi/tools/rq11_actualescape.py --seconds 180 --output /tmp/rq11-reference-proof.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 40s nice -n 10 python3 proj_rsi/candidates/rq11_actualescape/test_actualescape.py
```

This closes the exact **CPU reference fixture** escape/window/allocation question without synthetic samples. It does not close the actual CUDA fixture-distribution question. On this fixture, a1–4% allocation opportunity and substantial serial-escape/cache costs do not establish a serving speed gain. Retain this version for source-qualified producer/decode isolation or an exact saved-CUDA-weight accounting pass if those data become available; do not integrate/advertise a GPU codec win from these numbers. The lossless-codec family remains available for other representations and cache policies.


## v2 metadata correction and preserved v1 identity

Independent reviewer18 found `max_prefetch_windows_per_list` incorrectly used `ceil(max_list/32)` rather than `floor(max_list/32)+1`. This missed the initial/sentinel-only window for empty or exact-multiple lists (for example qkvKS7 max320 requires11, not10 windows). The separate requested-byte ledger already used the correct formula; payload, ESC allocation, residency and all byte-saving totals were unaffected. The field is corrected in the tool and full result, with a dedicated regression for0/1/31/32/33/63/64/256/320/352 and every saved geometry.

The entire prior18-pin v1 package, original manifest/SHA inventory and exact tool bytes are preserved under `history/cpu-reference-v1/`. `archive-map.json` binds each original repository path to its preserved snapshot path/hash. New v2 manifest binds these historical bytes as well as current results. The corrected full reference run reproduces every old matrix SHA; CUDA stays uninitialized. This is a new accounting revision, not a rewritten historical result or a new performance measurement.
