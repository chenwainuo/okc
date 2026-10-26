# Codec study: exponent bits after r6's 12-bit format

Date: 2026-10-26. Scope: codec only. There is no gv integration and no okc run.

- Prototype: `proj_rsi/tools/codec_proto.py`. It is pure numpy plus python ints. Pack and unpack round-trip bit-exactly for every design on 1M bf16 at four distribution phases. Run it with `python3 -m proj_rsi.tools.codec_proto`; it takes about 10 s.
- Decode-cost probes: `/tmp/claude-1000/-home-ubuntu-okc/8dda88b9-22f0-4310-b863-aa591883e787/scratchpad/codec/dec.cu`. They were built with nvcc 12.8 `-arch=sm_90a -cubin -O3`, with 0 spills in every kernel, and counted on `cuobjdump -sass`.

## Verdict

**Build E.** E is a pair Huffman code over the full 8-bit exponent. It uses one 256-entry LUT lookup per A register and per-lane substreams in a segment layout.

| Measure | E | r6 | Analyst's 11-bit + escape route |
|---|---|---|---|
| Bits/weight | **10.79–10.83** at every phase | 12 | 11.40–11.68 |
| Bytes vs r6 | −10% | — | −5…−3% |
| Decode cost (lane-instr/weight) | **5.8** | ≈ 1.1 | — |

- E's 5.8 lane-instructions per weight are inside the ≤ 6 budget.
- The entropy bound for this split is 10.63–10.71, so E sits 0.1–0.2 bits above it.
- The only cheaper fixed-length option is R′ at 11.20–11.34. It is the 11-bit route with escape entries compacted to 16 bits. Choose R′ only if E's decode cannot be hidden.

## What is coded

- Weights are randn/√K in bf16. The exponent histogram repeats when K is multiplied by 4, so four phases cover every case:
  - K = 8192: the pooler phase. H(h7) = 1.625.
  - K = 1024: the o-projection phase. H(h7) = 1.708.
  - K = 2048: H(h7) = 1.700.
  - K = 3072: H(h7) = 1.650.
- Sign and mantissa are incompressible, at 8 bits.
- Designs A–D keep r6's low byte (exponent LSB plus 7 mantissa bits) and a raw sign bit, and code only h7. Their total is 9 + code.
- Design E moves the exponent LSB into the code and the sign into the byte plane. The plane byte becomes sign | mant7. The bits are no worse, because the LSB is not uniform given h7: H(exp) = 2.54 versus H(h7) + 1 = 2.63.
- Geometry is r6's unit: 16 rows × 64 k, 32 weights per lane, 8 per k16 step = 4 A registers = 4 weight pairs.

## Results

Bits are total bits/weight, measured with the prototype on 1M weights, all overheads included. Decode is lane-instructions per weight, from the SASS probes.

| Design | K=8192 | K=1024 | K=2048 | K=3072 | Decode | Verdict |
|---|---|---|---|---|---|---|
| Entropy bound (9 + H(h7)) | 10.625 | 10.708 | 10.700 | 10.650 | — | |
| **E**: pair Huffman ≤ 8 bits on exp8, 256-LUT | **10.818** | **10.788** | **10.818** | **10.796** | **5.78** per-lane / 6.47 interleaved | **build** |
| D: pair Huffman ≤ 7 bits on h7, 128-LUT, sign plane | 10.838 | 10.879 | 10.869 | 10.822 | 5.94 / 6.47 | dominated by E |
| C: single-symbol Huffman ≤ 4 bits, 16-LUT (brief's c) | 10.941 | 10.919 | 10.926 | 10.864 | 9.59 | fails decode budget |
| B: per-8 header, width 1/2/3 bits | 11.110 | 11.214 | 11.287 | 11.215 | 8.47 | loses to R′ |
| B16: per-16 header, width 2/3/4 bits (brief's b, analytic) | 11.323 | 11.307 | 11.371 | 11.418 | — | ≥ 2 + header by construction, dead |
| R: 2-bit window + r6 32-bit XOR escape list (the 11-bit route) | 11.431 | 11.395 | 11.556 | 11.680 | 1.80 (probe), ≈ 1.1 r6 dec1 | reference |
| R′: R with 16-bit escape entries (hypothetical) | 11.216 | 11.197 | 11.278 | 11.340 | ≈ R | fallback |
| A: '0'+2 bits or '1'+4 bits (brief's a) | 12.066 | 12.065 | 12.072 | 12.083 | not built | worse than r6's 12, dead |

Notes on the table:

- **Overheads.** The bit counts include stream words with tail padding, escapes (32 bits each), one 32-bit stream offset per warp, and 16 units per warp stream. The per-lane layout adds 32 × 16-bit lane starts per segment and costs +0.000–0.009 bits; the prototype prints it for each variable-length design. Per-matrix tables (256 × 4 B LUT, rank map) are excluded; they cost < 0.01 bits on any real matrix.
- **Why R's escapes are expensive.** A 4-value h7 window misses 1.2–2.1% of weights, and r6's list entry costs 32 bits. That costs 0.4–0.7 bits/weight.
- **Where variable length helps.** The gain is concentrated in the top-2 h7 values, which cover 85% of weights at about 1 bit of information. A fixed 2-bit code wastes about 0.37 bits there.
- **Other codes tried.** Enumerative and ternary block codes (top-3 ranks, 5 trits/byte) and length-limited enumerative tuples come out at ≥ 11.2 bits once their escapes are paid for. rANS was excluded by the brief.

## E in detail

**Code.**
- Rank the exponents per matrix and keep the top R = 10.
- Use pair symbols (rank of w0, rank of w1): 100 symbols, canonical Huffman limited to 8 bits by package-merge.
- The stream is read LSB first.
- If either weight of a pair is outside the top 10 (0.3–0.4% of weights), its exponent is coded as rank 0 and fixed from r6's XOR escape list (32-bit entries). That costs 0.10–0.14 bits; 16-bit entries would recover half of it.

**LUT entry (32 bits).**
- Bits 0–4: code length.
- Bits 7–14: exp(w0).
- Bits 23–30: exp(w1).
- The table is 1 KB of smem per matrix.

**Per A register (pair), 6.5 lane-instructions.**

```
e   = lut[lo & 0xff]                        // LOP3 + LEA + LDS
lo  = shf.r.wrap(lo, hi, e); hi = shf.r.wrap(hi, 0, e)   // wrap uses e & 31 = length
V  -= e                                      // IADD3 per 2 pairs; garbage lands at bit 7+ only
A   = lop3(e, 0x807f807f, prmt(plane, 0, 0x1100 | 0x3322), 0xb8)   // (0xe4 in v1 was wrong)  PRMT + LOP3: sign|mant7 dup -> bits 0-6, 15
```

**Per step (8 weights), refill.**
- The 64-bit buffer is refilled with one word when `(V & 0x60) == 0`, that is when V < 32. The test is immune to the exponent garbage in V.
- Insertion is `lo |= w << V; hi |= shf.l(0, w, V)`.
- Per-lane layout: a predicated LDS from the lane's own pointer, about 10 instructions.
- Interleaved layout: ballot + popc order, about 16 instructions. That is the 6.47 variant.
- With the 8-bit limit, 4 pairs never consume more than 32 bits, so one refill per step is always enough. The prototype asserts this and replays the exact refill schedule.

**Layout.**
- A segment is U units of one warp. Each segment is stored as 32 contiguous lane substreams with a 32 × 16-bit start table.
- The segment is variable-length, at about 1.8 bits × 1024 weights × U ≈ 3.7 KB for U = 16. It moves as one bulk copy into smem, so the producer side keeps whole-segment transfers. It needs a per-segment offset table (one word per segment).
- The 8-bit planes (sign|mant7) stay fixed-size, exactly as r6 streams its low bytes.

## Cost against the HBM stream

**m ≤ 8.**
- At 3.35 TB/s and 10.8 bits/weight, an SM must retire about 10.7 weights/clk (132 SMs, 1.755 GHz).
- E's 5.8 lane-instructions per weight come to about 62 lane-instructions per clk, about 48% of the 128/clk issue rate.
- r6's existing per-weight work is about 1.5 lane-instructions per weight, about 12%.
- That is feasible on issue, but it is no longer hidden for free. Note that the 5.8 includes E's own sign and plane merge, which replaces r6's dec1 entirely.

**m = 32.** HMMA per weight rises fourfold, to 0.5 lane-HMMA per weight, which is still small next to decode. The decode cost per weight is unchanged, so E is acceptable at m = 32.

## Risks the analyst should price in

1. **Latency chain.** Each lane's 16 pairs per unit form one dependent chain: LOP3 → LDS (about 30 clk) → SHF → … That is about 600 clk per unit per warp. Sustaining one unit per about 96 clk per SM needs at least 6–7 decoding warps per SM. Alternatively, decode two units per lane interleaved (two buffers, +5 registers).
2. **LUT bank conflicts.** 32 lanes peek effectively random bytes into a 256-entry table (8 entries per bank). Expect about 3–4-way conflicts, which comes to about 0.6 smem wavefronts per clk per SM at full HBM rate. If that matters, replicate the LUT 4× with a lane-dependent base (4 KB, +1 IADD per pair).
3. **Format change.** E changes the 8-bit plane to sign|mant7. r6's dec1 and nibble plane disappear, and the escape list XORs the 8-bit exponent instead of h7.
4. **No GPU here.** All decode numbers are static SASS counts. The device decoders were compiled, not executed. The bit-exact round trip is in numpy, using the same LUT, length and refill rules as the device code.

## SASS counting method

- Each probe kernel streams units per warp. Per unit, it decodes the 16 A registers per lane and feeds them to 4 `mma.sync.m16n8k16`.
- The count covers the main-loop body minus HMMA, LDG, BRA, LDC and S2R, divided by 32 weights per lane per unit.
- Per-design counts:

  | Kernel | Design | Lane-instr/weight |
  |---|---|---|
  | `kE8<true>` | E, per-lane | 5.78 |
  | `kE8<false>` | E, interleaved | 6.47 |
  | `kDL` | D, per-lane | 5.94 |
  | `kD2` | D, interleaved | 6.47 |
  | `kC` | C | 9.59 |
  | `kB` | B | 8.47 |
  | `kR` | R | 1.80 |

- The D and E probes also carry r6's low-byte loads and their address math (about 0.3 lane-instructions per weight).
- `kD`, `kD2` and `kDL` were compiled with a 0x3fc mask and an 8-bit limit. The limit-7 D in the table changes only constants, not instruction count.

## pw17 post-mortem (check 9f6de54218a447d5): why E ran at 1.45–2.1× the stream

- **Cycles.** ge-p512r took 26.24 µs in-prepare, about 21.9 kernel-µs after the 4.3 µs offset (r6: 18.30 in-prepare against 13.95 judge). Less about 1.5 µs of start, that is **≈ 2,500 clk per unit per warp** at 16 warps/SM. The stream needs ≈ 1,560.
- **Chain estimate.** 16 dependent pair lookups (3 ALU ≈ 13 clk + LDS) plus 4 chained refill LDS give ≈ 800 clk at a 30-clk LDS. Matching 2,500 needs **≈ 110 clk per chained LDS**. Unpooler n=1024 runs 4 warps/SM, with no contention, and still implies ≈ 65.
- **Cause: bank conflicts.** Replaying the real pooler pack, wavefronts per warp are:

  | LDS | Wavefronts |
  |---|---|
  | v2 ring refill, ED 4 / ED 8 | **12.0 / 22.0** |
  | Single / replicated LUT | 3.7 / 1.0 |
  | `[slot][lane]` ring + word rotation | 2.2 |

  Lanes advance in lockstep, so v2's 64-B lane stride puts them on the same banks; the spec's "2–3-way" was wrong. This fits d8 (36.3) > single (29.4) > rep (26.2).
- **v3.** Code lengths come from ALU ops, so no LDS sits on the chain. The ring is rotated `[slot][lane]` and refill words are prefetched.

## Closing: instruction budget of the pw18 pooler m=1 E path (check b15f30a832ab4169)

Lane-instructions per unit (1,024 weights, 32 per lane), K1 single LUT, from SASS (total 256):

| Part | Instr / unit | Per weight |
|---|---|---|
| Length chain: LOP3 `lo&7`, PRMT, 2 SHF, V update | 72 | 2.25 |
| Value lookups: LOP3 `lo&0xff`, address, LDS | 48 | 1.50 |
| Plane merge and select: PRMT dup, LOP3 | 32 | 1.00 |
| Ring, refill and prefetch: 4 refills and 4 issues with predicates and addresses | ≈ 60 | 1.9 |
| Rest of r6's 49: 2 plane + 2 x LDG, 4 HMMA, escape test, loop (its 36-op decode is gone) | ≈ 13 | 0.4 |
| Address and move overhead | ≈ 31 | 1.0 |

- **Floor for a per-symbol variable-length decode on this register layout.** Each pair needs at least a length (1–2 ops), a 2-op 64-bit shift, a value lookup (address + LDS) and a 2-op merge into the A register. That is about 7 per pair, plus about 0.5 per weight for refills: **≈ 4 lane-instr/weight**. r6 spends 1.5.
- **Could a bit-plane-transposed layout with warp-wide integer ops reach ≤ 2 lane-instr/weight? No.** The mma A fragments must still be assembled per lane, 2 weights per register. r6 already spends 1.1/weight doing that from fixed-width fields. An entropy code adds per-symbol boundary resolution (≥ 1 op/weight) that transposition cannot remove, because the boundaries depend on the data.
