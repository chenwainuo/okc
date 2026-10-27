# Exact weight layouts after the responsive zero control

`state=CPU_PATTERN_AND_BIJECTION_PROOF; GPU_UNRUN; serving_gain=UNMEASURED; family=ITERATING`

**Next question: local byte representation, not D2D repair.** Zero64MiB Wt under the existing D2D helper reduced aggregate observed DRAM reads73.827%; normal native BF16 stayed flat under the same fixed Lt allocation comparison. The feature/path/observer can respond, while the normal payload lacks a useful response in its current representation. Zero's result is not a normal-weight ratio. Preserve both observations and packed-v1's timing/cleanup history.

## CPU evidence, with exact scope

Reproducible CPU-only [proof](../candidates/hw_compression_patterns/cpu_patterns.py) and [numbers](../candidates/hw_compression_patterns/cpu_patterns.json). Torch2.13CPU/Python3.10;4,194,304 Gaussian BF16 weights, std1/64, CPU seed5151. This matches unpooler's marginal distribution, **not the actual H100 RNG payload**. No GPU/Driver API, software compression library or hardware-ratio estimator is called.

Both proposed integer maps are bijections on **all65,536bit patterns**, including infinities, every NaN payload and signed zeros. Both directions pass; encoded storage is uint16, not numerically interpreted as BF16. Byte-plane layout inverses pass. Packed32 inverse tests cover all values in each half through a bijective partner permutation; that is not an exhaustive2^32pair test. The arithmetic bound below separately excludes inter-half carry.

Fixed BASE=0x3c00, independent of activations. For original u16 bits `u`:

```text
centered-zigzag:
  d=((u&0x7fff)-BASE)&0x7fff
  zz=((d<<1)^-(d>>14))&0x7fff
  e=(zz<<1)|(u>>15)
inverse:
  zz=e>>1; d=(zz>>1)^-(zz&1)
  u=((d+BASE)&0x7fff)|((e&1)<<15)

cheaper sign-low XOR:
  e=((((u&0x7fff)^BASE)<<1)|(u>>15))&0xffff
  u=((e>>1)^BASE)|((e&1)<<15)
```

Neither map changes any floating-point value when decoded; no float rescaling/overflow route is introduced. A later implementation must decode **before** math and retain every original numerical/probe gate.

| Physical representation | Bytes/weight | Zero bytes | Common high-zero bits/u16 per32B sector, mean | Distinct bytes/32B sector, mean |
|---|---:|---:|---:|---:|
| native16 |2|0.2045%|0.00004|20.9048|
| centered-zigzag16 |2|12.9561%|4.7220|20.7646|
| sign-low XOR16 |2|15.1682%|3.9765|20.6477|
| native16, low/high byte planes |2|0.2045%|0.00002|18.6080|
| centered-zigzag16, byte planes |2|12.9561%|2.3607|18.4498|

All have **zero all-zero32B sectors and zero constant-byte32B sectors** in this sample. Zigzag encoded values fit10bits85.46%,11bits97.50%,12bits99.844%,13bits99.9993%. Those are local-pattern clues, not documented NVIDIA format requirements. Byte-plane common-zero counts interpret the physical bytes as u16; compare them only as a byte-pattern summary, not decoded weight magnitudes. Whole-weight symbol entropy is unchanged10.4648569bits by a bijection. Byte entropy barely shifts; a transform rearranges patterns instead of deleting information. No guarantee that ILC recognizes any of these patterns follows.

## Decode / payload accounting before any device prototype

Actual frozen [rq_dec](../candidates/pw36/submission.py:4052) assembles **eight BF16 weights** from12bytes: two mask/add pairs, one rotate, four PRMT operations, about9source-level32bit operations/8weights, plus inherited exact escape repair. This is not7operations per weight. Compiler scheduling/SASS may fuse or change the source count; no new SASS was measured here.

| Format | Logical W payload at8192×4096 | Inverse cost to8 BF16 values, source-level model | Main risk |
|---|---:|---|---|
| shipped packed12 |50,331,648B+small sidebands|~9ops+escape handling|Current practical comparator |
| native two8bit planes |67,108,864B|4PRMT, from two low-byte/two high-byte words; no variable boundaries or escapes|Needs hardware reduction exceeding extra logical bytes; two plane streams/layout mapping |
| sign-low XOR16 |67,108,864B|~6logical ops/packed2values, ~24/8; possible compiler fusion unproved|Extra integer issue despite simpler bijection |
| centered-zigzag16 |67,108,864B|~12logical ops/packed2values, ~48/8; possible compiler fusion unproved|Much higher inverse issue than current codec |

For packed zigzag, let `z=(e>>1)&0x7fff7fff`; form `d=((z>>1)&0x3fff3fff)^((z&0x00010001)*0x7fff)`, then `mag=(d+BASE*0x00010001)&0x7fff7fff`, and OR shifted sign bits. Each unsigned15bit half obeys `d<=0x7fff`, `d+0x3c00<=0xbbff<0x10000`, so the add cannot cross a halfword. The multiplication creates independent15bit masks. This gives a valid packed inverse, not a claim that it beats rq_dec. Scalar~6–8ops and rq_dec~9ops use different denominators.

Pure byte-plane recovery is cheaper and does not repeat codecE's variable-length parsing: four PRMTs create four packed-BF16 output registers from fixed-width low/high planes. Existing codecE/bitplane criticism concerned data-dependent symbol boundaries and serial issue; hardware decompressing fixed streams changes that mechanism. It still requires new legal load/layout mapping, resource checks, and numerical validation. Stock Lt cannot consume recoded/planed bytes: feeding them as native BF16 would compute the wrong GEMM. A prepare-time decode into an ordinary native allocation would erase the transport benefit. No whole-matmul rewrite is justified yet.

## Next discriminating measurement, if root authorizes

**Rank first: native low/high byte planes**, because inverse4PRMT/8 is below current9/8 and removes escapes; do not pick zigzag solely because its zero-byte count rises. ILC's formats/granularity remain undocumented, so this is an information-ROI choice, not a forecast. Use plane spans large/aligned enough to contain many full cache lines within each prepared W tile; CPU64value/128B grouping above proves byte bijection only, not the final TMA geometry.

Before touching matmul, one bounded diagnostic could clone a **single chosen exact transformed payload** into GENERIC/NONE with frozen helper7f2 and use one identical full-span uint8 copy consumer per allocation. Reuse one documented Torch copy kernel/launch, retain every output, initialize the actual destination to the source's bytewise complement before each selected replay, compare all bytes immediately, and require immutable source and graph-reset→Region0. Capture only those two nodes with the existing memory-counter policy. This tests whether the representation yields an observed read response, without conflating it with decode issue/math scheduling. It does **not** rerun a transform grid, alter population, claim a serving gain, or require a new GEMM. Same-copy GEN/NONE is the valid contrast; absolute counts cannot be directly compared with the prior Lt kernel's aggregate counters.

The64MiB representation must save **more than25%** of its W traffic just to match packed12's48MiB payload before metadata, X, overlap and decode. Historical entropy10.46bits suggests only~6.46MB ideal residual headroom beyond packed12; not a hardware lower bound or realized saving. A weak response informs another version; it is not a family-wide ban. A strong response permits one fused decode design/resource proof against actual pw36/RQ, followed by full oracles and paired graph/F1 comparison. No GPU, transform-copy driver or matmul implementation is authorized by this brief.
