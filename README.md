# Open Kernel Challenge: `muse_bf16_proj`

A single-file CUDA/PyTorch kernel for the dense projections of Muse Spark Open
(`qkv_gate`, `o`, `pooler`, `unpooler` in BF16, the `router` in FP32) as one H200
runs them while serving: `out = x @ W.T` for 1 to 256 tokens per step.

Latest judged result: **3133.1 µs per decode step, 1.542× SGLang's kernels**
(score = a fixed reference time divided by the paired speedup over SGLang).

The kernel is `muse_bf16_proj/submission.py` (one file, ~6.6k lines: the CUDA
sources are embedded strings compiled with `torch.utils.cpp_extension.load_inline`
during the untimed `prepare()` call). `reference.py`, `checks.py` and `baseline.py`
are the challenge's reference, correctness spec and SGLang baseline.

## Results (judge, H200, µs per layer; speedup is paired against SGLang)

| load | qkv_gate | o | pooler | unpooler | router | load sum | speedup |
|---|---|---|---|---|---|---|---|
| decode, m=1 | 9.06 | 5.38 | 13.83 | 4.13 | 3.58 | 35.97 | 1.66× |
| decode, m=16 | 9.79 | 5.81 | 14.27 | 14.57 | 4.62 | 49.05 | 1.56× |
| decode, m=32 | 10.14 | 6.36 | 15.30 | 15.30 | 5.18 | 52.28 | 1.48× |
| verify, m=8 | 9.17 | 5.18 | 13.66 | 4.03 | 4.31 | 36.35 | 2.08× |
| verify, m=128 | 12.31 | 7.35 | 17.11 | 17.38 | 7.13 | 61.27 | 1.42× |
| verify, m=256 | 17.53 | 8.57 | 24.51 | 23.82 | 9.08 | 83.52 | 1.19× |

Weights: qkv_gate 2304×8192, o 8192×1024, pooler 4096×8192, unpooler 8192×4096
(rows 0-1023 only when m ≤ 8), router 256×8192 FP32. Every shape is timed as a
replayed CUDA graph with a cold L2, so the small shapes are dominated by the
HBM stream of the weights plus fixed per-kernel latency.

## How it works

### 1. Lossless 12-bit weight packing (prepare time)

Every BF16 weight is stored in 12 bits instead of 16: sign + 7 mantissa bits +
a 4-bit exponent offset inside a per-group exponent window. Values whose
exponent falls outside the window are rare (about one per CTA per K-stage) and
are recorded as *escapes*: a sorted list of (stage, word, XOR patch) entries that
the kernel applies after decoding. The pack is bit-exact (verified by an
independent unpacker before any kernel runs) and cuts the HBM traffic by 25%,
which is the whole game for the memory-bound shapes. The router is stored as a
BF16 hi + lo pair, both 12-bit packed, with FP32 accumulation.

### 2. One kernel family per regime

- **m ≤ 32, BF16 layers (`gv` / `gv_predxor`)**: register-direct GEMV. Each
  thread streams packed words straight from global memory, decodes them with a
  predicated-XOR trick (~1.1 integer ops per weight), and accumulates in FP32;
  `x` is read once per 128 rows. Launch geometry (rows per CTA, K split 1/2/4,
  x in L1 or a shared-memory slab) is chosen per shape at prepare time.
- **m = 16/32 and m = 128 (`rq`, "RS128")**: swap-AB split-K GEMM on the
  tensor cores. Each CTA owns 128 weight rows; two math warpgroups decode the
  12-bit pack (12 KB per 128-row k64 stage, delivered by `cp.async.bulk` into a
  5-slot ring) directly into the **A registers of `wgmma`**, while `x` arrives as
  the shared-memory B operand through a 2D TMA. K is split over a one-wave grid;
  partials are summed with `cp.reduce.async.bulk.tensor` BF16 adds into the
  output, which the kernel zeroes itself (no memset node, no cluster). Tile
  widths NT = 16 / 32 / 128 cover the three token counts.
- **router, m = 128/256 (`rtk`)**: a one-node `wgmma` kernel over the hi + lo
  pair with FP32 accumulation and a split-K reduction; m = 1 uses a 1024-thread
  single-block GEMV (`rr`).
- **cuBLASLt** remains the fallback and is the pick where it still wins
  (`o` at m = 128/256, pooler/unpooler at m = 256).

### 3. The escape path

The 12-bit pack needs a few patches per stage, and in a tensor-core mainloop the
cost is not the patches but the instructions that sit between the decode and
`wgmma.fence`. The shipped design keeps a **per-thread staged cursor**: each
thread holds only its own escape entries sorted by stage, a warp vote skips the
whole apply block when nobody in the warp has an entry for this stage, and the
apply block itself is kept short so the taken branch over it is cheap. This
removed about 1 µs per shape at m = 128 compared with a warp-wide list walk, and
proportionally less at m = 32 and m = 16 where the kernels are closer to the
HBM bound.

### 4. Per-shape selection in `prepare()`

`prepare()` builds every candidate path for each (layer, m) shape, validates each
one against the reference (one-hot probe rows must be bit-exact; random inputs
within tolerance), and then times the candidates as captured CUDA-graph replays
against a reference pick, in the same paired-ratio style the judge uses. A
candidate replaces the reference only if it beats a margin; a handful of picks
whose gain is below what a single timing run can resolve are fixed from
offline measurements. The whole prepare runs in about 200 s on the judge
(limit 300 s); the timed `kernel_fn` is one graph-captured launch with no host
synchronisation.

## Running it

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh   # the okc CLI
python3 bootstrap.py                                         # challenge container
okc check muse_bf16_proj/submission.py                       # correctness + timing on the judge
```

`muse_bf16_proj/TASK.md` describes the problem, the loads and the judge's timing
protocol.

## Layout

- `muse_bf16_proj/` — the kernel (`submission.py`), reference, checks, baseline, task description.
- `muse_moe_w4a16/`, `serving/` — the other challenge problems (not the focus of this file).
- `docs/`, `audits/`, `reports/` — design notes and measurement records.

## License

See `LICENSE`.
