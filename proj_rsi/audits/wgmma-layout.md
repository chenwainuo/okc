# wgmma (register-A, smem-B) layout reference for pw9

Sources:
- **PTX ISA 9.4 HTML.** Downloaded 2026-10-25 from docs.nvidia.com/cuda/parallel-thread-execution. The text and the figure PNGs (`_images/wgmma-64N16-A.png`, `wgmma-64N16-D.png`) were read directly. Section numbers below are from that version.
- **CUTLASS** `cute/arch/mma_sm90_gmma.hpp`, at `/tmp/okc-wgmma-research/flashinfer/data/cutlass/include/cute/arch/`.
- **DeepGEMM** `sm90_bf16_gemm.cuh`, at `/tmp/okc-wgmma-research/deep_gemm/include/deep_gemm/impls/`.

Conventions:
- Notation:
  - `T` = thread index in the warpgroup (0..127).
  - `w = T / 32` is the warp within the warpgroup.
  - `l = T % 32` is the lane.
  - `g = l >> 2` is the groupID.
  - `t = l & 3` is the threadID_in_group.
- **Our use (swap-AB):**
  - A = decoded W: M = 64 W rows per warpgroup, K = 16 per instruction.
  - B = x: N = tokens (128 or 256), K-major.
  - D[row = W row][col = token].

## 1. A fragment in registers: `.m64nNk16 .bf16`

Reference: §9.7.17.5.1.1.1, "Matrix Fragments for wgmma.mma_async.m64nNk16", Figure 151 ("WGMMA .m64nNk16 register fragment layout for matrix A").

Each thread holds 4 × .b32 = 8 bf16, `a0..a7`, with the low element in the low 16 bits of each register.

| register | elements | row of A (0..63) | k (0..15) |
|---|---|---|---|
| R0 | a0, a1 | 16w + g | 2t, 2t+1 |
| R1 | a2, a3 | 16w + g + 8 | 2t, 2t+1 |
| R2 | a4, a5 | 16w + g | 8 + 2t, 8 + 2t + 1 |
| R3 | a6, a7 | 16w + g + 8 | 8 + 2t, 8 + 2t + 1 |

This is identical to the mma.sync.m16n8k16 .bf16 A layout applied to rows 16w..16w+15.

The mma.sync formula is from §9.7.16.5.8, Figure 82:
- `row = groupID` for a_i with i ∈ {0, 1, 4, 5}, otherwise `groupID + 8`.
- `col = 2·threadID_in_group + (i & 1)`, plus 8 for i ≥ 4.

**Correction (found by the implementer, verified in the pw9 review):** the register slot order matches, but the r6/pw `_pack` assigns permuted k to those slots. Lane t at step q holds k = 16t + 4q + 0..3 of the 64-chunk. That was valid for mma.sync only because x was loaded into B registers with the same permutation. wgmma reads x from smem in natural k order, so the wgmma paths need `_pack_nat`. In it, slot (q, lane g·4+t, v = (kh, half, e)) holds W[16T + 8·half + g][64c + 16q + 8·kh + 2t + e], which gives exactly the table above per k16 step q. The +32 B descriptor step q walks the same natural k = 16q..16q+15.

There is no register permutation. The packer's register order {row g k-lo, row g+8 k-lo, row g k-hi, row g+8 k-hi} is the judge-proven pw_dec output `A[4q..4q+3]` for k16 step q. It feeds wgmma unchanged:
- The 4 registers of step q are the `{a0..a3}` operand of the q-th k16 wgmma in a 64-wide chunk.
- Warp w of the warpgroup must decode W row tile `16w..16w+15` of the warpgroup's 64-row block.

Warpgroup definition (§9.7.17.1): 4 contiguous warps whose first warp rank is a multiple of 4. Any producer or epilogue warp must therefore sit outside the consumer warpgroups' aligned ranges. For example, consumers are warps 0..4k-1 and the producer is warp 4k (pwc already does this when NWC is a multiple of 4).

## 2. Accumulator D: `.m64nNk16 .f32`

Reference: §9.7.17.5.1.1.1, Figure 152.

Each thread holds N/2 f32 registers `d0..d(N/2-1)`. For n8 column block `i = 0 .. N/8-1`:

| register | row (W row) | col (token) |
|---|---|---|
| d[4i+0] | 16w + g | 8i + 2t |
| d[4i+1] | 16w + g | 8i + 2t + 1 |
| d[4i+2] | 16w + g + 8 | 8i + 2t |
| d[4i+3] | 16w + g + 8 | 8i + 2t + 1 |

So `(d[4i], d[4i+1], d[4i+2], d[4i+3])` is exactly the mma.sync m16n8 C fragment `(c0, c1, c2, c3)` of token tile i for warp w's 16-row tile. The existing `pw_put(out, N, m, n = nb·BN + 16·w' + g, tok = 8i + 2t, float4(d[4i..4i+3]))` and the fragment-order staging index carry over unchanged:
- `w'` = the warpgroup's row-block offset/16 + w.
- out gets `x→(n, tok)`, `y→(n, tok+1)`, `z→(n+8, tok)`, `w→(n+8, tok+1)`.

Register cost per thread: N = 128 needs 64 accumulator regs; N = 256 needs 128, plus 16 A regs per chunk in flight.

## 3. Shared-memory descriptor for B: K-major, 128B swizzle

### Descriptor format

Reference: §9.7.17.5.1.2.2, "Matrix Descriptor Format". The descriptor is 64-bit, with `encode(x) = (x & 0x3FFFF) >> 4`.

| bits | field | value for our B tile |
|---|---|---|
| 13–0 | start address, encoded | `(smem_addr & 0x3FFFF) >> 4` (shared::cta address of the tile) |
| 29–16 | leading-dimension byte offset, encoded | not used for swizzled K-major: "assumed to be 1" (§9.7.17.5.1.2.1.1). Write 1. |
| 45–32 | stride-dimension byte offset, encoded | K-major SBO = "offset from the first 8 rows to the next 8 rows" (§9.7.17.5.1.2.1.2). Here that is 8 token rows × 128 B = 1024 B, so encode 1024 >> 4 = 64. |
| 51–49 | matrix base offset | 0 when the tile base is 1024-B aligned. Otherwise `(pattern start addr >> 7) & 7`. |
| 63–62 | swizzle mode | 1 = 128-Byte swizzle (2 = 64B, 3 = 32B, 0 = none) |

```
desc = ((uint64_t)((addr & 0x3FFFF) >> 4))
     | ((uint64_t)1  << 16)      // LBO (ignored for swizzled K-major)
     | ((uint64_t)64 << 32)      // SBO = 1024 B
     | ((uint64_t)0  << 49)      // base offset (tile 1024-aligned)
     | ((uint64_t)1  << 62);     // 128B swizzle
```

### Layout and alignment

- **Canonical layout** (§9.7.17.5.1.2.1.3, K-major, 128B swizzling): `Swizzle<3,4,3> o ((8,m),(T,2k)):((8T,SBO),(1,T))` with T = 8 for bf16, shown in Figure 160 ("K major 128B swizzling").
  - The swizzle atom is 8 rows × 128 B (Table 47: 128B / K = 8×8 in 128-bit units).
  - `Swizzle<3,4,3>`: address bits [6:4] ^= bits [9:7]. That is, the 16-B chunk c of row r sits at `r·128 + ((c ^ (r & 7)) << 4)` relative to a 1024-B-aligned base.
- **Alignment:**
  - Any shared-memory matrix must be 16-B aligned (§9.7.17.5.1).
  - For base offset 0 with 128B swizzle, the repeating pattern must start on a 1024-B boundary (§9.7.17.5.1.2.2 table).
  - So align each B tile, i.e. each stage's x tile, to 1024 B, with stage strides a multiple of 1024. The pw3/pw4 `tm` kernel already does this (`sbase = (sraw + 1023) & ~1023`, SB a multiple of 1024). DeepGEMM declares `extern __shared__ __align__(1024)` and static-asserts every stage size `% 1024 == 0` (sm90_bf16_gemm.cuh:93-96).
- **Tile:** N tokens × 64 k is N/8 swizzle atoms stacked along the stride dimension: 16 KB at N = 128, 32 KB at N = 256.
- **The descriptor must be identical in all 4 warps of the warpgroup** (§9.7.17.5.2: "contents of a matrix descriptor must be same across all the warps").

### Advancing per k16 step inside the 64-wide chunk

- Step q (q = 0..3) of the chunk uses k = 16q..16q+15, which is bytes 32q..32q+31 of every 128-B row. Add 32·q bytes to the start address, i.e. **+2·q in the encoded low field**. Leave LBO, SBO, base offset and swizzle unchanged.
- This is legal with 128B swizzle. The swizzle is applied to the absolute address bits, so a start address of base + 32q inside a 1024-aligned atom still resolves to the swizzled chunk positions.
- CUTLASS and DeepGEMM do exactly this. DeepGEMM `sm90_bf16_gemm.cuh:266-271`: `advance_gmma_desc_lo(desc_base_lo, 0, (k*WGMMA::K) % BLOCK_ATOM_K, atom_k_idx*BLOCK_N*BLOCK_ATOM_K)` with BLOCK_ATOM_K = 64 for bf16 SW128.
  - The in-atom k offset becomes a start-address offset; base offset stays 0.
  - Per stage, the descriptor low word advances by `stage_bytes / 16` (line 252).
- Do not carry past 128 B per row (q ≤ 3). The next 64-k chunk is a different tile or stage, with its own 1024-aligned base.

## 4. TMA `CU_TENSOR_MAP_SWIZZLE_128B` vs the descriptor layout

- **Map:** a 2D map on x[token][k] (dims {K, m}, stride K·2 B, box {64, N}, element strides {1,1}, interleave NONE, SWIZZLE_128B) writes each token row as 128 B. 16-B chunk c of row r goes to `dst + r·128 + ((c ^ (r & 7)) << 4)` when dst is 1024-B aligned. This is the `Swizzle<3,4,3>` K-major SW128 atom of §3, with consecutive 8-row groups 1024 B apart, which matches SBO = 1024.
- **Proven on the judge:** pw3/pw4 `tm` (check 5d0ab1df4e61480e and later) read this exact TMA layout with the XOR rule above and validated bit-exact.
- **Constraints:**
  - For SWIZZLE_128B the inner box extent in bytes must be ≤ 128. We use exactly 128 (64 bf16), so the box is one atom wide.
  - Box dims must be ≤ 256, so N = 256 tokens is the maximum box height.
  - Global address 16-B aligned; global strides multiples of 16 B.
  - Destination smem 1024-B aligned. Here that is required for the swizzle phase to match base offset 0 in the descriptor.
- **Rows ≥ m (odd m):** with dims {K, m} and OOB fill NONE, out-of-range rows are zero-filled. The transaction byte count is still the full box.

## 5. Fences, commit and wait

Reference: §9.7.17.7.1–3 and §9.7.17.4.

### wgmma.fence

- `wgmma.fence.sync.aligned` is required, executed by all 4 warps:
  - (a) before the first `wgmma.mma_async` of the warpgroup;
  - (b) between any ordinary register access and a wgmma that uses the same registers as accumulator or as A fragment.
- The exception: accumulator accesses across consecutive wgmmas of the same shape are ordered by default.
- Therefore, in our loop:
  - **Every chunk:** decode A registers (prmt + escape XOR), then `wgmma.fence`, then the 4 k16 wgmmas, then `wgmma.commit_group`.
  - The decode writes A registers with normal instructions, so the fence is needed every chunk, not just once.
  - Zeroing accumulators also needs a fence before the next wgmma, unless the first wgmma uses scale-d = 0.

### commit_group and wait_group

- `wgmma.commit_group.sync.aligned` batches all uncommitted wgmmas of the warp into a group.
- `wgmma.wait_group.sync.aligned N` waits until at most N most-recent groups are pending.
- "Accessing the accumulator register or the input register containing the fragments of matrix A of a wgmma.mma_async instruction without first performing a wgmma.wait_group … that waits on a wgmma-group including that wgmma.mma_async instruction is undefined behavior" (§9.7.17.7.3).

### Trap 1: A registers are read asynchronously

- The decoded A registers for chunk i+1 must not physically alias chunk i's A registers until a `wait_group` covering chunk i has executed. Use double buffers `A[2][4 steps][4]` and `wait_group 1` before writing buffer (i & 1) again, or `wait_group 0` with a single buffer.
- **The compiler does not know the read is asynchronous.** After the asm statement it considers A dead and may reuse the physical registers for the next decode, even with double-buffered C++ arrays. Prevent this by keeping each A register live until after the covering wait. After `wgmma.wait_group`, apply `asm volatile("" : "+r"(a) :: "memory")` to every A register of the waited chunk; this is CUTLASS's `warpgroup_fence_operand`, mma_sm90_gmma.hpp:86-102. Apply it to the accumulators before `wgmma.fence` and after `commit_group`, as DeepGEMM does at sm90_bf16_gemm.cuh:259-277.
- After compiling, check the `ptxas -v` output for "wgmma.mma_async instructions are serialized" warnings. They mean ptxas inserted waits after every wgmma: correct, but the overlap is lost.

### Trap 2: B is also read asynchronously from smem

The stage holding the x tile, and W if W is staged in the same slot, must not be released to the producer until `wait_group` confirms completion of every wgmma that read it. DeepGEMM arrives on the empty barrier only after `warpgroup_wait<0>()` (sm90_bf16_gemm.cuh:278-281).

This differs from pw7, which arrives right after the W decode. That is still fine for W bytes consumed by `ld.shared` into registers, but not for an x tile read by wgmma. If x and W share a stage slot, release the slot only after the wait that covers that stage's wgmmas.

### Trap 3: proxies

- wgmma runs in the async proxy (§9.7.17.4).
- **x tile written by TMA:** that is also async proxy. The mbarrier complete_tx wait suffices.
- **x tile written by generic stores:** `st.shared`, `cp.async` (non-bulk), or zero-filling rows ≥ m by hand all count. Then `fence.proxy.async.shared::cta` is required after the writes and before the CTA barrier that precedes the first wgmma reading them (§9.7.17.7.1: "An async proxy fence must be used to establish an ordering between prior writes to shared memory matrices and subsequent reads").
- **Reusing B smem afterwards:** the completion of a wgmma implies a generic-async proxy fence. After `wait_group 0`, smem that wgmma read may be overwritten by generic stores, for example the DSMEM-reduce staging in the ring area. That reuse is safe only after `wait_group 0` in every consumer warpgroup, followed by the CTA barrier.

### `.aligned`

wgmma, fence, commit and wait must be executed by all 128 threads of the warpgroup, convergently. The per-lane escape XOR is fine, but the wgmma itself must not sit inside lane-divergent code.

## 6. Operands: bf16 with register A

Syntax (§9.7.17.5.2):
```
wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16  d, {a0,a1,a2,a3}, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;
```
- **scale-d:** a predicate (or 0/1). With false, D = A·B and the old D is ignored; with true, D = A·B + D. CUTLASS passes it through `setp.ne.b32 p, %r, 0` (mma_sm90_gmma.hpp, `MMA_64x8x16_F32BF16BF16_RS::fma`, around lines 2312-2325).
- **imm-scale-a, imm-scale-b ∈ {1, -1}:** -1 negates that operand. **Use 1, 1.** The PTX ISA examples use `-1, -1` (e.g. `scaleD, -1, -1, 0`), so copying an example verbatim negates the product.
- **imm-trans-b ∈ {0, 1}:** 0 means B is K-major, i.e. "B stored column-major": KxN with each token's 16 k contiguous. That matches x[token][k] loaded as above. **Use 0.**
- **There is no imm-trans-a operand in the register-A form.** "The transpose operation is only supported for … matrices accessed from shared memory using matrix descriptors" (§9.7.17.5.2). CUTLASS enforces `static_assert(tnspA == GMMA::Major::K, "Register source operand A must have K major layout.")` (mma_sm90_gmma.hpp, `MMA_64x8x16_F32BF16BF16_RS`). So A must arrive in registers exactly in the §1 orientation, and it does: the packer's fragments are K-major per row.
- **Shapes:** `.m64n128k16` / `.m64n256k16` with `.dtype .f32` (bf16 has an f32 accumulator only). N = 8·i for i = 1..32.
- Requires `sm_90a` and PTX ISA ≥ 8.0.

## 7. Numerics notes for validation

- §9.7.17.5.2: "The accumulation order, rounding, and handling of subnormal inputs are unspecified."
- **One-hot rows** (power of two times W, all other products zero) stay bit-exact for normal bf16 weights.
- **Subnormal bf16 weights** (|w| < 2^-126; these are escape values in the codec) could be flushed by the tensor core. The prepare one-hot check would then report INVALID rather than crash. Expect this only if the weights contain bf16 subnormals; mma.sync has the same unspecified clause and has passed.
- **Random rows:** accumulation order differs from mma.sync, so results agree within tolerance but are not bit-identical to pw7.

## Checklist for the pw9 review

1. A registers: warp w of the warpgroup decodes W rows `blk64 + 16w .. +15`. The register order per k16 step q is `A[4q+0..3]` from pw_dec, unchanged.
2. D epilogue: `float4(d[4i..4i+3])` goes to `pw_put(n = row0 + 16w + g, tok = 8i + 2t)`.
3. Descriptor:
   - `lo = (addr >> 4) & 0x3FFF` (or `(addr & 0x3FFFF) >> 4`), plus 2q per k16 step.
   - LBO field = 1, SBO field = 64, base offset 0, bits 63:62 = 1.
   - Tile base 1024-aligned, stage stride a multiple of 1024.
4. TMA: box {64, N}, SWIZZLE_128B, dims {K, m}, expect_tx = N·128 per x tile.
5. Every chunk: wgmma.fence after the decode, then 4 wgmmas with scale-d (0 for the first k-step of a fresh accumulator, else 1), then commit.
6. A registers double-buffered and kept live by `"+r"` fences until the covering `wait_group`. Accumulators fenced by `"+f"` around the batch.
7. Stage release (empty arrive) only after the `wait_group` covering that stage's wgmmas. Before the DSMEM staging writes: `wait_group 0` in every warpgroup, then the CTA barrier.
8. If any part of the B tile is written by generic stores: `fence.proxy.async.shared::cta` before the barrier.
9. Immediates: `1, 1, 0` (scale-a, scale-b, trans-b). No trans-a.
10. ptxas: no "wgmma serialization" warning, 0 spills.
