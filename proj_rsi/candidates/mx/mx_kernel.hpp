// mx kernel: CUTLASS 3.9.2's sm90 KernelTmaWarpSpecialized GemmUniversal (non-persistent, one producer + one consumer
// warpgroup), with operator() copied from cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp and two changes:
// grid.z = mainloop.splits K-splits (split s owns k tiles [s T / S, (s + 1) T / S)), and the mainloop is told its tile
// (set_tile) so the escape correction can be filtered by rows and k range. With splits > 1 the epilogue must reduce-add
// (TMA cp.reduce.async.bulk .add) into a zeroed output (mx::WithReduceAdd).
#pragma once
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

namespace mx {
// CUTLASS's TMA epilogue issues the store from a whole warp (issue_tma_store = warp 0): 32 identical bulk stores per
// subtile. Harmless for a plain store, 32x for a reduce-add. OneLane<Op> executes Op only on lane 0 of that warp.
template <class Op> struct OneLane {
  template <class... A> CUTE_HOST_DEVICE static void copy(A const&... a) {
#if defined(__CUDA_ARCH__)
    if ((threadIdx.x & 31) == 0) Op::copy(a...);
#endif
  }
};
}  // namespace mx

namespace cute {
template <class Op, class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<mx::OneLane<Op>, NumBitsPerTMA, AuxParams_> {
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using RefLayout = SrcLayout;
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  CUTE_HOST_DEVICE constexpr TmaDescriptor const* get_tma_descriptor() const { return &tma_desc_; }
  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto get_tma_tensor(GShape const& g_shape) const {
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }
  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr void copy_unpack_(void const* const src_ptr, Coord const& dst_coord, seq<Is...>) const {
    mx::OneLane<Op>::copy(&tma_desc_, src_ptr, get<Is>(dst_coord)...);
  }
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(Copy_Traits const& traits, Tensor<TS, SLayout> const& src, Tensor<TD, DLayout>& dst) {
    traits.copy_unpack_(cute::raw_pointer_cast(src.data()), dst.data().coord_, tuple_seq<decltype(dst.data().coord_)>{});
  }
};
}  // namespace cute

namespace mx {

template <class E, class Op = cute::SM90_TMA_REDUCE_ADD> struct WithReduceAdd { using type = E; };
template <class DP, class TS, class ET, class EC, class SC, class ED, class SD, class FC, class G2S, class SLC, class S2R, class S2G,
          class SLD, class R2S, class CAC, class R2R, class Op>
struct WithReduceAdd<cutlass::epilogue::collective::CollectiveEpilogue<DP, TS, ET, EC, SC, ED, SD, FC, G2S, SLC, S2R, S2G, SLD, R2S, CAC, R2R>, Op> {
  using type = cutlass::epilogue::collective::CollectiveEpilogue<DP, TS, ET, EC, SC, ED, SD, FC, G2S, SLC, S2R, OneLane<Op>, SLD, R2S, CAC, R2R>;
};

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_>
struct Kernel : cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_> {
  using Base = cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_>;
  using typename Base::Params;
  using typename Base::Arguments;
  using typename Base::CollectiveMainloop;
  using typename Base::CollectiveEpilogue;
  using typename Base::TileShape;
  using typename Base::TiledMma;
  using typename Base::ClusterShape;
  using typename Base::SharedStorage;
  using typename Base::StrideA;
  using typename Base::StrideB;
  using typename Base::StrideC;
  using typename Base::StrideD;
  using Base::IsConvProblemShape;
  using ProblemShape = ProblemShape_;
  static constexpr int NumMma = int(cute::size(TiledMma{}));  // consumer threads (128: one math warpgroup, 256: two)

  static dim3 get_grid_shape(Params const& params) {
    dim3 g = Base::get_grid_shape(params);
    g.z = params.mainloop.splits;
    return g;
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using namespace cutlass;
    using X = Underscore;

#if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200))
#  define ENABLE_SM90_KERNEL_LEVEL 1
#endif
// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };
    enum class ProducerWarpRole {
      MainloopEpilogue = 0,
      Warp1 = 1,
      Warp2 = 2,
      Warp3 = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();


    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (int(warp_group_role) >= 1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumMma;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (int(warp_group_role) >= 1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumMma;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    if (warp_idx == 0 && lane_predicate) CollectiveMainloop::init_a_ring(shared_storage.tensors.mainloop);

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();
  
    // Preconditions only valid for Gemm
    static_assert(IsConvProblemShape || cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(IsConvProblemShape || cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(IsConvProblemShape || cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(IsConvProblemShape || cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
    TiledMma tiled_mma;

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    // Using constexpr if (C++17 and later)
    auto problem_shape_MNKL = append<4>(params.problem_shape, cute::Int<1>{});
    
    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Prepare and partition the input tensors. 
    // Expects a tuple of tensors for conv where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");
    
    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
    auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
    // handles the difference between the rank of Tensor returned by load_input in case they do not have a batch mode
    auto l_coord = 0;
    const int split = int(blockIdx.z);
    auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

    // Get pipeline iterators and increments from tensor shapes
    const int k_tiles_total = int(size<3>(gA_mkl));
    const int k_begin = split * k_tiles_total / params.mainloop.splits;
    const int k_end = (split + 1) * k_tiles_total / params.mainloop.splits;
    auto k_tile_iter  = cute::make_coord_iterator(idx2crd(k_begin, shape<3>(gA_mkl)), shape<3>(gA_mkl));
    int k_tile_count = k_end - k_begin;
    collective_mainloop.set_tile(int(m_coord), int(n_coord), k_begin, k_end);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      if constexpr (NumMma > 128) cutlass::arch::warpgroup_reg_dealloc<40>();
      if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        collective_mainloop.load(
          params.mainloop,
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          load_inputs,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
        // Update starting mainloop pipeline state for the pipeline drain
        mainloop_pipe_producer_state.advance(k_tile_count);
        // Make sure mainloop consumer has been waited upon before issuing epilogue load
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if (collective_epilogue.is_producer_load_needed()) {
          // Ensure warp is converged before issuing epilogue loads
          __syncwarp();
          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );
          collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
        }
      }
      else {  // mx: warps 1-3 of the producer warpgroup: in-kernel zeroing for split-K and the escape gather
        collective_mainloop.aux_work(params.mainloop, shared_storage.tensors.mainloop, warp_group_thread_idx - NumThreadsPerWarp,
                                     split, int(blockIdx.x * gridDim.y + blockIdx.y));
      }
    }
    else if (int(warp_group_role) >= 1) {
      if constexpr (NumMma > 128) cutlass::arch::warpgroup_reg_alloc<232>();
      const int mma_thread_idx = thread_idx - NumThreadsPerWarpGroup;  // 0 .. NumMma - 1
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                 // (MMA,MMA_M,MMA_N)

      collective_mainloop.mma(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        accumulators,
        k_tile_count,
        mma_thread_idx,
        shared_storage.tensors.mainloop,
        params.mainloop
      );

      // Make sure the math instructions are done and free buffers before entering the epilogue
      collective_mainloop.mma_tail(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        k_tile_count
      );
      collective_mainloop.escape_apply(accumulators, shared_storage.tensors.mainloop, mma_thread_idx);

      const int tile_id = int(blockIdx.x * gridDim.y + blockIdx.y);
      if (params.mainloop.ptr_D != nullptr) {  // reduce-add epilogue: wait until split 0's aux warps have zeroed this tile
        if (mma_thread_idx == 0) {
          int* f = params.mainloop.flags + 2 * tile_id;
          int v = 0;
          do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(f) : "memory"); } while (v == 0);
          if (atomicAdd(f + 1, 1) == params.mainloop.splits - 1) { f[1] = 0; f[0] = 0; }  // last of the S waiters resets (next launch is stream-ordered)
        }
        asm volatile("bar.sync 14, %0;\n" :: "n"(NumMma) : "memory");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Epilogue and write to gD
      auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
      collective_epilogue.store(
        epi_load_pipeline,
        epi_load_pipe_consumer_state,
        epi_store_pipeline,
        epi_store_pipe_producer_state,
        problem_shape_MNKL,
        blk_shape,
        blk_coord,
        accumulators,
        tiled_mma,
        mma_thread_idx,
        shared_storage.tensors.epilogue
      );

      collective_epilogue.store_tail(
        epi_load_pipeline,
        epi_load_pipe_consumer_state_next,
        epi_store_pipeline,
        epi_store_pipe_producer_state_next
      );
      if (mma_thread_idx == 0) mx::dbg_ts(params.mainloop.flags, 7);
    }
#endif
  }
};

}  // namespace mx
