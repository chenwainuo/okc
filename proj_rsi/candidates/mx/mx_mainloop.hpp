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

namespace mx {
using namespace cute;

static constexpr int UNIT = 64;          // rows per packed unit (one wgmma M)
static constexpr int CHUNK = 6144;       // bytes per (unit, 64-k tile): 4096 plane8 + 2048 plane4

template <int Stages_, class ClusterShape_, class TileShape_, int ESC_>
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
  static constexpr int MU = TM / UNIT;   // 64-row units per tile (= MMA_M of the single consumer warpgroup)
  static constexpr int CM = size<0>(ClusterShape{});
  static constexpr int CN = size<1>(ClusterShape{});
  static_assert(CHUNK % (16 * CN) == 0);

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
      TileShape, GMMA::Major::K, GMMA::Major::K>(), Layout<Shape<_1, _1, _1>>{}));
  using ElementAccumulator = float;

  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::rs_smem_selector<GMMA::Major::K, cutlass::bfloat16_t,
      decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{}, make_shape(Int<TN>{}, Int<TK>{}, Int<Stages>{}), Step<_1, _2, _3>{}));

  static constexpr uint32_t A_STAGE = MU * CHUNK;
  static constexpr uint32_t B_STAGE = TN * TK * 2;
  static constexpr uint32_t TmaTransactionBytes = A_STAGE + B_STAGE;

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  struct SharedStorage {
    struct TensorStorage {
      CUTE_ALIGNAS(1024) cute::ArrayEngine<cutlass::bfloat16_t, cute::cosize_v<SmemLayoutB>> smem_B;
      CUTE_ALIGNAS(128) uint8_t smem_A[Stages * A_STAGE];
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
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void*) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N, K, L), args.dB));
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}), TileShape{}, ClusterShape{});
    Params p{tma_load_b, args.ptr_A, args.ptr_B, (int64_t)get<0>(args.dB), (int)(K / TK), uint32_t(args.base) * 0x01010101u,
             args.esc_off, args.esc_rk, args.esc_d};
    return p;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = append<4>(problem_shape, 1);
    bool ok = (M % TM == 0) && (K % TK == 0) && (N % 8 == 0) && args.ptr_A != nullptr && (reinterpret_cast<uintptr_t>(args.ptr_A) % 16 == 0)
              && args.base >= 0 && args.base <= 120 && L == 1;
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
          bulk_g2s(sA0 + write_stage * A_STAGE + u * CHUNK + part, gA0 + ((int64_t)u * p.k_tiles + kt) * CHUNK, SL, bar, mcast_mask_a);
        copy(p.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_, _, _, kt), tBsB(_, _, _, write_stage));
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
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});
    TiledMma tiled_mma;
    auto thr = tiled_mma.get_thread_slice(t);
    Tensor fakeA = make_tensor(make_smem_ptr(static_cast<cutlass::bfloat16_t*>(nullptr)), make_layout(make_shape(Int<TM>{}, Int<TK>{}), LayoutRight{}));
    Tensor tCrA = thr.partition_fragment_A(fakeA);                                // (MMA, MMA_M, MMA_K)
    static_assert(size<0>(decltype(tCrA){}) == 8 && size<1>(decltype(tCrA){}) == MU && size<2>(decltype(tCrA){}) == 4);
    auto wg = tiled_mma.get_slice(0);
    Tensor tCsB = wg.partition_B(sB);
    Tensor tCrB = wg.make_fragment_B(tCsB);                                      // (MMA, MMA_N, MMA_K, PIPE)
    const uint32_t sA0 = cute::cast_smem_ptr_to_uint(shared_tensors.smem_A) + t * 16;
    const uint32_t tb = p.tb;

    uint4 R8[MU][2], R4[MU];
    auto load_raw = [&](int stage) {
      CUTLASS_PRAGMA_UNROLL
      for (int u = 0; u < MU; ++u) {
        const uint32_t a = sA0 + stage * A_STAGE + u * CHUNK;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(R8[u][0].x), "=r"(R8[u][0].y), "=r"(R8[u][0].z), "=r"(R8[u][0].w) : "r"(a));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(R8[u][1].x), "=r"(R8[u][1].y), "=r"(R8[u][1].z), "=r"(R8[u][1].w) : "r"(a + 2048));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(R4[u].x), "=r"(R4[u].y), "=r"(R4[u].z), "=r"(R4[u].w) : "r"(a + 4096));
      }
    };
    auto dequant = [&](auto kb_) {
      constexpr int kb = decltype(kb_)::value;
      CUTLASS_PRAGMA_UNROLL
      for (int u = 0; u < MU; ++u) {
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
    if constexpr (ESC) {
      clear(accum);
      escape_prologue(accum, tiled_mma, t, p);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    } else {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
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

  // accum[r, tok] += d * x[n0 + tok, k] for every escape (r, k, d) of this tile's rows; runs before the first wgmma,
  // while the pipeline fills. The thread's rows per unit are found once (64-bit mask); the list is read 32 entries at a
  // time (one coalesced load per lane, broadcast by shfl); matches gather x through L2 (__ldg, issued together per entry).
  template <class FrgTensorC>
  CUTLASS_DEVICE void escape_prologue(FrgTensorC& accum, TiledMma const& tiled_mma, int t, Params const& p) {
    auto thr = tiled_mma.get_thread_slice(t);
    Tensor cC = make_identity_tensor(make_shape(Int<TM>{}, Int<TN>{}));
    Tensor tCcC = thr.partition_C(cC);                                    // (MMA, MMA_M, MMA_N) -> (m, n)
    static_assert(size<1>(decltype(tCcC){}) == MU);
    const int m_tile = blockIdx.x, n_tile = blockIdx.y;  // ws kernel: grid (m tiles, n tiles) (cluster-shaped)
    const int lane = t & 31;
    cutlass::bfloat16_t const* xb = p.ptr_B + (int64_t)n_tile * TN * p.ldb;
    CUTLASS_PRAGMA_UNROLL
    for (int u = 0; u < MU; ++u) {
      uint64_t mask = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<0>(tCcC); ++i)
        CUTLASS_PRAGMA_UNROLL
        for (int nn = 0; nn < size<2>(tCcC); ++nn) mask |= uint64_t(1) << (get<0>(tCcC(i, u, nn)) - u * UNIT);
      const int unit = m_tile * MU + u;
      const int e0 = __ldg(p.esc_off + unit), e1 = __ldg(p.esc_off + unit + 1);
      for (int eb = e0; eb < e1; eb += 32) {
        const int ne = min(32, e1 - eb);
        int rk = lane < ne ? __ldg(p.esc_rk + eb + lane) : 0;
        float d = lane < ne ? __ldg(p.esc_d + eb + lane) : 0.f;
        for (int j = 0; j < ne; ++j) {
          const int rkj = __shfl_sync(0xffffffffu, rk, j);
          const float dj = __shfl_sync(0xffffffffu, d, j);
          const int r = rkj >> 16, k = rkj & 0xffff;
          if ((mask >> r) & 1) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size<0>(tCcC); ++i)
              CUTLASS_PRAGMA_UNROLL
              for (int nn = 0; nn < size<2>(tCcC); ++nn) {
                auto c = tCcC(i, u, nn);
                if (get<0>(c) - u * UNIT == r)
                  accum(i, u, nn) += dj * float(xb[(int64_t)get<1>(c) * p.ldb + k]);
              }
          }
        }
      }
    }
  }

  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    k_tile_count -= 1;
    smem_pipe_release.advance(k_tile_count);
    warpgroup_wait<0>();
    pipeline.consumer_release(smem_pipe_release);
    ++smem_pipe_release;
  }
};

}  // namespace mx
