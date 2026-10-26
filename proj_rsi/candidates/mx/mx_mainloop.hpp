// mx: CUTLASS 3.9.2 sm90 warp-specialised mainloop with a two-plane 12-bit packed A operand (W) converted to bf16 in
// registers before an RS wgmma. Forked from cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:
// same pipeline (PipelineTmaAsync, producer warp + one consumer warpgroup, KernelTmaWarpSpecialized), same B path
// (bf16 x via TMA, K-major, SW128), but A is not a CuTe tensor: it is r6's 12-bit code packed in wgmma fragment order
// (proj_rsi/tools/plane_pack.py) and streamed with cp.async.bulk (multicast across the cluster's N CTAs) into a plain
// byte ring that shares the A/B full/empty barriers. Consumer: per k-tile 3 x LDS.128 per 64-row unit, dec1 merge
// (9 ops / 8 weights) into the RS fragment registers. Escapes (weights outside the 16-exponent window, ~1e-4) are
// added exactly in fp32 to the accumulators before the first wgmma (ESC = 1).
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#ifndef MX_SCHED
#define MX_SCHED 1  // 0: stock mixed-input schedule (fence+wgmma+commit+wait<3> per k-block); 1: batched, double-buffered A fragments
#endif
#ifndef MX_SKIP
#define MX_SKIP 0  // probes only: 1 = no B loads, 2 = no A loads, 4 = no decode
#endif
namespace mx {
using namespace cute;
// MX_SKIP & 8: escapes: skip the header/x copy; & 16: skip the apply
// MX_SKIP & 64: debug timestamps (globaltimer ns) into flags[8192 + 8 * cta + slot]:
// 0 aux start, 1 escape list done, 2 apply start, 3 apply done, 4 consumer mainloop done, 5 consumer start,
// 6 first wgmma issued, 7 epilogue done
CUTLASS_DEVICE void dbg_ts(int* flags, int slot) {
#if (MX_SKIP & 64)
  uint64_t t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  reinterpret_cast<unsigned long long*>(flags + 8192)[8 * cta + slot] = t;
#endif
}

static constexpr int UNIT = 64;          // rows per packed unit (one wgmma M)
static constexpr int PLANES = 6144;      // plane bytes per (unit, 64-k tile): 4096 plane8 + 2048 plane4
static constexpr int CHUNK = 6208;       // + 64-byte escape header: u32 n | 15 x u32 (row << 22 | (k % 64) << 16 | xor16)

template <int Stages_, class ClusterShape_, class TileShape_, int ESC_, int AS_ = 0, int NWG_ = 1>
struct Mainloop {
  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;
  static constexpr int Stages = Stages_;
  static constexpr int ESC = ESC_;
  using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput<Stages, ClusterShape, cutlass::gemm::KernelTmaWarpSpecialized>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using ArchTag = cutlass::arch::Sm90;
  using ElementA = cutlass::bfloat16_t;  // nominal (A is packed bytes)
  using ElementB = cutlass::bfloat16_t;
  using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;  // nominal
  using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;  // x [tok][K]: K-major
  static constexpr int TM = size<0>(TileShape{});
  static constexpr int TN = size<1>(TileShape{});
  static constexpr int TK = size<2>(TileShape{});
  static_assert(TK == 64 && TM % UNIT == 0, "mx: TK must be 64 and TM a multiple of 64");
  static constexpr int MU = TM / UNIT;   // 64-row units per tile
  static constexpr int NWG = NWG_;       // math warpgroups: 1, or 2 splitting the rows (cooperative-style, setmaxnreg)
  static constexpr int MUT = MU / NWG;   // units per math warpgroup (= its MMA_M)
  static constexpr int NCONS = 128 * NWG;
  static_assert(MU % NWG == 0);
  static constexpr int CM = size<0>(ClusterShape{});
  static constexpr int CN = size<1>(ClusterShape{});
  static_assert(CHUNK % (16 * CN) == 0);
  // AS > 0: A (packed W) has its own ring of AS stages with its own full/empty mbarriers, deeper than the B (x) ring
  // (Stages): W streams from HBM and needs the bytes in flight, x comes from L2. A stages are released right after the
  // consumer's LDS + decode (not after the wgmma). Requires CN == 1 (no A multicast).
  static constexpr int AS = AS_;
  static constexpr bool SPLIT = AS > 0;
  static_assert(!SPLIT || CN == 1, "mx: separate A ring only without A multicast");

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
      TileShape, GMMA::Major::K, GMMA::Major::K>(), Layout<Shape<Int<NWG>, _1, _1>>{}));
  using ElementAccumulator = float;

  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::rs_smem_selector<GMMA::Major::K, cutlass::bfloat16_t,
      decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<TN>{}, Int<TK>{}, Int<Stages>{}), Step<_1, _2, _3>{}));

  static constexpr uint32_t A_STAGE = MU * CHUNK;
  static constexpr uint32_t B_STAGE = TN * TK * 2;
  static constexpr uint32_t TmaTransactionBytes = ((MX_SKIP & 2) || SPLIT ? 0 : A_STAGE) + ((MX_SKIP & 1) ? 0 : B_STAGE);

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  // ESC = escape capacity per CTA (entries of this tile's rows within its k range); 0 = no escape correction
  // ESC != 0: escapes (weights outside the window, ~1e-4) are patched exactly in the decoded A fragments from the
  // 64-byte header of each (unit, k tile) chunk: the owning thread XORs the true bf16 bits into its register.
  static constexpr int ESC_BYTES = 16;
  struct SharedStorage {
    struct TensorStorage {
      CUTE_ALIGNAS(1024) cute::ArrayEngine<cutlass::bfloat16_t, cute::cosize_v<SmemLayoutB>> smem_B;
      CUTE_ALIGNAS(128) uint8_t smem_A[(SPLIT ? AS : Stages) * A_STAGE];
      CUTE_ALIGNAS(8) uint64_t a_full[SPLIT ? AS : 1];
      CUTE_ALIGNAS(8) uint64_t a_empty[SPLIT ? AS : 1];
      CUTE_ALIGNAS(16) uint8_t esc[ESC ? ESC_BYTES : 16];  // escape list of this CTA: n, r[ESC], d[ESC], x[ESC][TN] (bf16)
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  struct Arguments {
    uint8_t const* ptr_A = nullptr;     // packed planes [M/64][K/64][6144]
    StrideA dA{};                       // unused
    cutlass::bfloat16_t const* ptr_B = nullptr;  // x [N tok][K]
    StrideB dB{};
    int base = 0;                       // window base (h7 code)
    int const* esc_off = nullptr;       // [M/64 + 1]
    int const* esc_rk = nullptr;        // r << 16 | k
    float const* esc_d = nullptr;       // w - w_dec
    int splits = 1;                     // K splits (grid.z, mx::Kernel)
    cutlass::bfloat16_t* ptr_D = nullptr; // out [tok][M] (zeroed in-kernel by split 0 when splits > 1)
    int64_t ldd = 0;
    int* flags = nullptr;               // [2 * tiles] zero-initialised once; self-cleaning
  };

  struct Params {
    using TMA_B = decltype(make_tma_copy_B_sm90(GmemTiledCopyB{},
        make_tensor(static_cast<cutlass::bfloat16_t const*>(nullptr), make_layout(make_shape(int32_t(0), int32_t(0), int32_t(0)), StrideB{})),
        SmemLayoutB{}(_, _, cute::Int<0>{}), TileShape{}, ClusterShape{}));
    TMA_B tma_load_b;
    uint8_t const* ptr_A;
    cutlass::bfloat16_t const* ptr_B;
    int64_t ldb;
    int k_tiles;
    uint32_t tb;                        // base * 0x01010101
    int const* esc_off;
    int const* esc_rk;
    float const* esc_d;
    int splits;
    cutlass::bfloat16_t* ptr_D;
    int64_t ldd;
    int* flags;
    int n_tok;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  // set by mx::Kernel before load / mma: this CTA's tile and k-tile range
  int tile_m_ = 0, tile_n_ = 0, kt0_ = 0, kt1_ = 0;
  int* flags_dbg_ = nullptr;
  int esc_n_ = 0;   // escapes collected by this consumer thread's CTA (uniform)
  CUTLASS_DEVICE void set_tile(int m, int n, int k0, int k1) { tile_m_ = m; tile_n_ = n; kt0_ = k0; kt1_ = k1; }
  // one thread, before the kernel's post-init CTA/cluster sync
  CUTLASS_DEVICE static void init_a_ring(TensorStorage& ts) {
    if constexpr (SPLIT) {
      for (int i = 0; i < AS; ++i) {
        cutlass::arch::ClusterBarrier::init(&ts.a_full[i], 1);
        cutlass::arch::ClusterBarrier::init(&ts.a_empty[i], 4 * NWG);  // one arrival per consumer warp
      }
      cutlass::arch::fence_barrier_init();
    }
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void*) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N, K, L), args.dB));
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}), TileShape{}, ClusterShape{});
    Params p{tma_load_b, args.ptr_A, args.ptr_B, (int64_t)get<0>(args.dB), (int)(K / TK), uint32_t(args.base) * 0x01010101u,
             args.esc_off, args.esc_rk, args.esc_d, args.splits, args.ptr_D, args.ldd, args.flags, int(N)};
    return p;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    bool ok = (M % TM == 0) && (K % TK == 0) && (N % 8 == 0) && args.ptr_A != nullptr && (reinterpret_cast<uintptr_t>(args.ptr_A) % 16 == 0)
              && args.base >= 0 && args.base <= 120 && L == 1 && args.splits >= 1 && args.splits <= K / TK
              && (args.splits == 1 || (args.ptr_D && args.flags && args.ldd >= M));
    if (ESC) ok = ok && args.esc_off && args.esc_rk && args.esc_d;
    return ok;
  }

  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& p) {
    cute::prefetch_tma_descriptor(p.tma_load_b.get_tma_descriptor());
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& p) const {
    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mA_mkl = make_identity_tensor(make_shape(M, K, L));  // shape carrier only
    Tensor mB_nkl = p.tma_load_b.get_tma_tensor(make_shape(N, K, L));
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, Underscore, _1>{});
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _), Step<Underscore, _1, _1>{});
    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  CUTLASS_DEVICE static void bulk_g2s(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar, uint16_t mask) {
    if constexpr (CN > 1) {
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
                   :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory");
    } else {
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
    }
  }

  template <class... Ts, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(Params const& p, MainloopPipeline pipeline, PipelineState smem_pipe_write,
                           cute::tuple<Ts...> const& load_inputs, BlockCoord const& blk_coord, KTileIterator k_tile_iter,
                           int k_tile_count, int thread_idx, uint32_t block_rank_in_cluster, TensorStorage& shared_tensors) {
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    Tensor sB = as_position_independent_swizzle_tensor(sB_);
    uint2 cid = {block_rank_in_cluster % CM, block_rank_in_cluster / CM};
    Tensor gB_nkl = get<1>(load_inputs);
    auto block_tma_b = p.tma_load_b.get_slice(cid.x);
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);
    Tensor tBgB = block_tma_b.partition_S(gB);
    Tensor tBsB = block_tma_b.partition_D(sB);
    uint16_t mcast_mask_a = 0, mcast_mask_b = 0;
    auto block_layout = Layout<ClusterShape>{};
    for (int n = 0; n < CN; ++n) mcast_mask_a |= uint16_t(1) << block_layout(cid.x, n, Int<0>{});
    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      for (int m = 0; m < CM; ++m) mcast_mask_b |= uint16_t(1) << block_layout(m, cid.y, Int<0>{});
    }
    // this CTA's slice of every unit chunk (multicast to the CN CTAs sharing the W tile)
    constexpr uint32_t SL = CHUNK / CN;
    const uint32_t part = (CN > 1 ? cid.y : 0) * SL;
    const uint32_t sA0 = cute::cast_smem_ptr_to_uint(shared_tensors.smem_A);
    uint8_t const* gA0 = p.ptr_A + ((int64_t)m_coord * MU * p.k_tiles) * CHUNK + part;

    if constexpr (SPLIT) {
      // A runs up to AS tiles ahead of the consumer, independently of the B ring (non-blocking empty checks while B
      // is being issued; the remainder is issued blocking after the last B tile)
      const int kt0 = *k_tile_iter;
      const int n_k = k_tile_count;
      int a_issued = 0, a_idx = 0;
      uint32_t a_ph = 1;
      const bool leader = cute::elect_one_sync();
      auto issue_a = [&]() {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&shared_tensors.a_full[a_idx], A_STAGE);
        const uint32_t bar = cute::cast_smem_ptr_to_uint(&shared_tensors.a_full[a_idx]);
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < MU; ++u)
          bulk_g2s(sA0 + a_idx * A_STAGE + u * CHUNK, gA0 + ((int64_t)u * p.k_tiles + kt0 + a_issued) * CHUNK, CHUNK, bar, 0);
        ++a_issued;
        if (++a_idx == AS) { a_idx = 0; a_ph ^= 1; }
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (int j = 0; j < n_k; ++j) {
        if (leader) {
          while (a_issued < n_k && cutlass::arch::ClusterBarrier::test_wait(&shared_tensors.a_empty[a_idx], a_ph, 1)) issue_a();
        }
        __syncwarp();
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        int write_stage = smem_pipe_write.index();
        if (leader && !(MX_SKIP & 1)) copy(p.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_, _, _, kt0 + j), tBsB(_, _, _, write_stage));
        ++smem_pipe_write;
      }
      if (leader) {
        while (a_issued < n_k) {
          cutlass::arch::ClusterBarrier::wait(&shared_tensors.a_empty[a_idx], a_ph);
          issue_a();
        }
      }
      __syncwarp();
      return;
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      pipeline.producer_acquire(smem_pipe_write);
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        const int kt = *k_tile_iter;
        const uint32_t bar = cute::cast_smem_ptr_to_uint(tma_barrier);
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < MU; ++u)
          if (!(MX_SKIP & 2)) bulk_g2s(sA0 + write_stage * A_STAGE + u * CHUNK + part, gA0 + ((int64_t)u * p.k_tiles + kt) * CHUNK, SL, bar, mcast_mask_a);
        if (!(MX_SKIP & 1)) copy(p.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_, _, _, kt), tBsB(_, _, _, write_stage));
      }
      ++k_tile_iter;
      ++smem_pipe_write;
    }
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    if (cute::elect_one_sync()) pipeline.producer_tail(smem_pipe_write);
  }

  // dec1 merge of one k-block (8 values) into 4 bf16x2 fragment registers
  CUTLASS_DEVICE static void dec1(uint32_t L0, uint32_t L1, uint32_t C, uint32_t tb, uint32_t* a) {
    const uint32_t hl = (C & 0x87878787u) + tb;
    const uint32_t hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + tb;
    a[0] = __byte_perm(L0, hl, 0x5140); a[1] = __byte_perm(L0, hl, 0x7362);
    a[2] = __byte_perm(L1, hh, 0x5140); a[3] = __byte_perm(L1, hh, 0x7362);
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read, FrgTensorC& accum, int k_tile_count,
                          int thread_idx, TensorStorage& shared_tensors, Params const& p) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    const int t = thread_idx % 128;
    const int wg = thread_idx / 128;
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    TiledMma tiled_mma;
    auto thr = tiled_mma.get_thread_slice(thread_idx);
    Tensor fakeA = make_tensor(make_smem_ptr(static_cast<cutlass::bfloat16_t*>(nullptr)), make_layout(make_shape(Int<TM>{}, Int<TK>{}), LayoutRight{}));
    Tensor tCrA = thr.partition_fragment_A(fakeA);                                // (MMA, MMA_M, MMA_K)
    static_assert(size<0>(decltype(tCrA){}) == 8 && size<1>(decltype(tCrA){}) == MUT && size<2>(decltype(tCrA){}) == 4);
    auto wg_slice = tiled_mma.get_slice(wg * 128);
    Tensor tCsB = wg_slice.partition_B(sB);
    Tensor tCrB = wg_slice.make_fragment_B(tCsB);                                // (MMA, MMA_N, MMA_K, PIPE)
    const uint32_t sA0 = cute::cast_smem_ptr_to_uint(shared_tensors.smem_A) + wg * MUT * CHUNK + t * 16;
    const uint32_t tb = p.tb;

    uint4 R8[MUT][2], R4[MUT];
    auto load_raw = [&](int stage) {
      CUTLASS_PRAGMA_UNROLL
      for (int u = 0; u < MUT; ++u) {
        const uint32_t a = sA0 + stage * A_STAGE + u * CHUNK;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(R8[u][0].x), "=r"(R8[u][0].y), "=r"(R8[u][0].z), "=r"(R8[u][0].w) : "r"(a));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(R8[u][1].x), "=r"(R8[u][1].y), "=r"(R8[u][1].z), "=r"(R8[u][1].w) : "r"(a + 2048));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(R4[u].x), "=r"(R4[u].y), "=r"(R4[u].z), "=r"(R4[u].w) : "r"(a + 4096));
      }
    };
    auto dequant = [&](auto kb_) {
      constexpr int kb = decltype(kb_)::value;
      if constexpr ((MX_SKIP & 4) != 0) { if (kb != 0) return; }
      CUTLASS_PRAGMA_UNROLL
      for (int u = 0; u < MUT; ++u) {
        Tensor r = recast<uint32_t>(tCrA(_, u, kb));
        const uint4 q = R8[u][kb / 2];
        const uint32_t L0 = (kb & 1) ? q.z : q.x, L1 = (kb & 1) ? q.w : q.y;
        const uint32_t C = kb == 0 ? R4[u].x : kb == 1 ? R4[u].y : kb == 2 ? R4[u].z : R4[u].w;
        uint32_t a[4];
        dec1(L0, L1, C, tb, a);
        r(0) = a[0]; r(1) = a[1]; r(2) = a[2]; r(3) = a[3];
      }
    };
    auto gemm_kb = [&](auto kb_, int stage) {
      constexpr int kb = decltype(kb_)::value;
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb, stage), accum);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      warpgroup_commit_batch();
    };

    PipelineState smem_pipe_release = smem_pipe_read;
    if constexpr (MX_SCHED >= 1) {
      // One wgmma group per k-tile (4 x k16, one fence, one commit), one group left in flight. The next k-tile's A
      // fragments are decoded into the other register buffer while the current group runs (DeepGEMM / CUTLASS SS
      // schedule with the decode in the shadow of 4 wgmmas). k_tile_count must be even (two buffers, static indices).
      Tensor tCrA1 = make_fragment_like(tCrA);
      auto decode_all = [&](auto& frag) {
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < MUT; ++u) {
          CUTLASS_PRAGMA_UNROLL
          for (int kb = 0; kb < 4; ++kb) {
            if constexpr ((MX_SKIP & 4) != 0) { if (kb != 0) continue; }
            Tensor r = recast<uint32_t>(frag(_, u, kb));
            const uint4 q = R8[u][kb / 2];
            const uint32_t L0 = (kb & 1) ? q.z : q.x, L1 = (kb & 1) ? q.w : q.y;
            const uint32_t C = kb == 0 ? R4[u].x : kb == 1 ? R4[u].y : kb == 2 ? R4[u].z : R4[u].w;
            uint32_t a[4];
            dec1(L0, L1, C, tb, a);
            r(0) = a[0]; r(1) = a[1]; r(2) = a[2]; r(3) = a[3];
          }
        }
      };
      int a_idx = 0;
      uint32_t a_ph = 0;
      const uint32_t sAh = cute::cast_smem_ptr_to_uint(shared_tensors.smem_A) + wg * MUT * CHUNK + PLANES;  // stage 0 headers
      uint32_t hdr_n[MUT];   // escape counts of the fetched k tile (loaded with the planes, used after the decode)
      auto load_hdr = [&](int a_stage) {
        if constexpr (ESC > 0 && !(MX_SKIP & 8)) {
          CUTLASS_PRAGMA_UNROLL
          for (int u = 0; u < MUT; ++u)
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(hdr_n[u]) : "r"(sAh + a_stage * A_STAGE + u * CHUNK));
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int u = 0; u < MUT; ++u) hdr_n[u] = 0;
        }
      };
      // owner of escape (r, kk) in the wgmma m64k16 A fragment: thread 32 (r / 16) + 4 (r % 8) + (kk % 8) / 2,
      // k-block kk / 16, register (r / 8) % 2 + 2 ((kk / 8) % 2), half kk % 2
      auto patch = [&](auto& frag, int a_stage) {
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < MUT; ++u) {
          const uint32_t h = sAh + a_stage * A_STAGE + u * CHUNK;
          for (uint32_t i = 0; i < hdr_n[u]; ++i) {
            uint32_t e;
            asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(e) : "r"(h + 4 + 4 * i));
            const int r = (e >> 22) & 63, kk = (e >> 16) & 63;
            const int owner = (r >> 4) * 32 + (r & 7) * 4 + ((kk & 7) >> 1);
            if (owner == t) {
              const int idx = (kk >> 4) * 4 + ((r >> 3) & 1) + 2 * ((kk >> 3) & 1);
              const uint32_t x = (e & 0xffffu) << ((kk & 1) * 16);
              CUTLASS_PRAGMA_UNROLL
              for (int kb = 0; kb < 4; ++kb) {
                Tensor rr = recast<uint32_t>(frag(_, u, kb));
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < 4; ++q) rr(q) ^= (idx == kb * 4 + q) ? x : 0u;
              }
            }
          }
        }
      };
      auto any_hdr = [&]() { uint32_t a = 0; for (int u = 0; u < MUT; ++u) a |= hdr_n[u]; return a; };
      auto fetch_decode = [&](auto& frag, int b_stage) {
        if constexpr (SPLIT) {
          cutlass::arch::ClusterBarrier::wait(&shared_tensors.a_full[a_idx], a_ph);
          load_raw(a_idx);
          load_hdr(a_idx);
          decode_all(frag);
          if (any_hdr()) patch(frag, a_idx);
          __syncwarp();
          if ((t & 31) == 0) cutlass::arch::ClusterBarrier::arrive(&shared_tensors.a_empty[a_idx]);
          if (++a_idx == AS) { a_idx = 0; a_ph ^= 1; }
        } else {
          load_raw(b_stage);
          load_hdr(b_stage);
          decode_all(frag);
          if (any_hdr()) patch(frag, b_stage);
        }
      };
      auto issue = [&](auto& frag, int stage) {
        warpgroup_fence_operand(accum);
        warpgroup_arrive();
        CUTLASS_PRAGMA_UNROLL
        for (int kb = 0; kb < 4; ++kb) {
          cute::gemm(tiled_mma, frag(_, _, kb), tCrB(_, _, kb, stage), accum);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
        warpgroup_commit_batch();
        warpgroup_fence_operand(accum);
      };
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      if (thread_idx == 0) dbg_ts(p.flags, 5);
      flags_dbg_ = p.flags;
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      fetch_decode(tCrA, smem_pipe_read.index());
      if constexpr (MX_SCHED == 2) {
        // two wgmma groups in flight (wait<2>), three A fragment buffers: tile t uses buffer t % 3 and decodes tile t + 1
        // into the buffer of tile t - 2 (complete after the wait); tile t - 2's stage is released there too
        Tensor tCrA2 = make_fragment_like(tCrA);
        auto step = [&](auto& cur, auto& nxt, int kt) {
          const int stage = smem_pipe_read.index();
          ++smem_pipe_read;
          issue(cur, stage);
          if (kt == 0 && thread_idx == 0) dbg_ts(p.flags, 6);
          if (kt + 1 < k_tile_count) barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
          warpgroup_wait<2>();
          if (kt >= 2) { pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release; }
          if (kt + 1 < k_tile_count) {
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
            fetch_decode(nxt, smem_pipe_read.index());
          }
        };
        CUTLASS_PRAGMA_NO_UNROLL
        for (int kt = 0; kt < k_tile_count; kt += 3) {
          step(tCrA, tCrA1, kt);
          if (kt + 1 < k_tile_count) step(tCrA1, tCrA2, kt + 1);
          if (kt + 2 < k_tile_count) step(tCrA2, tCrA, kt + 2);
        }
        warpgroup_fence_operand(accum);
        if (thread_idx == 0) dbg_ts(p.flags, 4);
        return;
      }
      // tiles 0, 1 (no release before tile 1's issue... release of tile t-1 follows the wait<1> after issuing tile t)
      CUTLASS_PRAGMA_NO_UNROLL
      for (int kt = 0; kt < k_tile_count; kt += 2) {
        {  // even tile: A in tCrA
          const int stage = smem_pipe_read.index();
          ++smem_pipe_read;
          issue(tCrA, stage);
          if (kt == 0 && thread_idx == 0) dbg_ts(p.flags, 6);
          if (kt + 1 < k_tile_count) barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
          warpgroup_wait<1>();
          if (kt > 0) { pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release; }
          if (kt + 1 < k_tile_count) {
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
            fetch_decode(tCrA1, smem_pipe_read.index());
          }
        }
        if (kt + 1 < k_tile_count) {  // odd tile: A in tCrA1
          const int stage = smem_pipe_read.index();
          ++smem_pipe_read;
          issue(tCrA1, stage);
          if (kt + 2 < k_tile_count) barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
          warpgroup_wait<1>();
          pipeline.consumer_release(smem_pipe_release); ++smem_pipe_release;
          if (kt + 2 < k_tile_count) {
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
            fetch_decode(tCrA, smem_pipe_read.index());
          }
        }
      }
      warpgroup_fence_operand(accum);
      if (thread_idx == 0) dbg_ts(p.flags, 4);
      return;
    }
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    constexpr int K_WAIT_MAX = 3;
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    // first k tile
    {
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      load_raw(read_stage);
      dequant(Int<0>{});
      gemm_kb(Int<0>{}, read_stage); dequant(Int<1>{});
      gemm_kb(Int<1>{}, read_stage); dequant(Int<2>{});
      gemm_kb(Int<2>{}, read_stage); dequant(Int<3>{});
      gemm_kb(Int<3>{}, read_stage);
      --k_tile_count;
      if (k_tile_count > 0) {
        pipeline.consumer_wait(smem_pipe_read, barrier_token);
        load_raw(smem_pipe_read.index());
        warpgroup_wait<K_WAIT_MAX>();
        dequant(Int<0>{});
      }
    }
    if (k_tile_count == 0) return;
    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 1; --k_tile_count) {
      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;
      warpgroup_fence_operand(accum);
      gemm_kb(Int<0>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      dequant(Int<1>{});
      gemm_kb(Int<1>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      dequant(Int<2>{});
      gemm_kb(Int<2>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      dequant(Int<3>{});
      gemm_kb(Int<3>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      load_raw(smem_pipe_read.index());
      dequant(Int<0>{});
      warpgroup_fence_operand(accum);
    }
    warpgroup_fence_operand(accum);
    {
      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      gemm_kb(Int<0>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      dequant(Int<1>{});
      gemm_kb(Int<1>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      dequant(Int<2>{});
      gemm_kb(Int<2>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      dequant(Int<3>{});
      gemm_kb(Int<3>{}, read_stage);
      warpgroup_wait<K_WAIT_MAX>();
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
    }
    warpgroup_fence_operand(accum);
  }

  // Work of the producer warpgroup's three idle warps (ptid 0..95), at kernel entry, off the critical path: with the
  // reduce-add epilogue, split 0 zeroes this tile of out and publishes flags[2 tile] = 1 (every split's epilogue waits).
  CUTLASS_DEVICE void aux_work(Params const& p, TensorStorage& ts, int ptid, int split, int tile_id) {
    if (ptid == 0) dbg_ts(p.flags, 0);
    if (p.ptr_D != nullptr && split == 0) {
      constexpr int CPR = TM / 8;  // 16-byte chunks per token row of the tile
      const uint4 z = make_uint4(0, 0, 0, 0);
      for (int i = ptid; i < TN * CPR; i += 96) {
        const int tok = tile_n_ * TN + i / CPR;
        if (tok < p.n_tok) *reinterpret_cast<uint4*>(p.ptr_D + (int64_t)tok * p.ldd + tile_m_ * TM + (i % CPR) * 8) = z;
      }
      __threadfence();
      asm volatile("bar.sync 12, 96;\n" ::: "memory");
      if (ptid == 0) asm volatile("st.release.gpu.global.b32 [%0], %1;\n" :: "l"(p.flags + 2 * tile_id), "r"(1) : "memory");
    }
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE void escape_apply(FrgTensorC&, TensorStorage&, int) {}  // escapes are patched into the A fragments

  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    constexpr int LAG = MX_SCHED == 2 ? 2 : 1;  // stages still held by the consumer after its loop
    const int held = k_tile_count < LAG ? k_tile_count : LAG;
    smem_pipe_release.advance(k_tile_count - held);
    warpgroup_wait<0>();
    for (int i = 0; i < held; ++i) {
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
    }
  }
};

}  // namespace mx
