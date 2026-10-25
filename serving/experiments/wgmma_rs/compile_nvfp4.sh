#!/usr/bin/env bash
set -euo pipefail
/usr/local/cuda/bin/nvcc -std=c++20 --expt-relaxed-constexpr \
  -gencode arch=compute_90a,code=sm_90a -DSGL_CUDA_ARCH=900 -Xptxas=-v \
  -I/sglang/python/sglang/kernels/jit/include \
  -I/sglang/python/sglang/kernels/jit/csrc \
  -I/opt/sglang/lib/python3.12/site-packages/tvm_ffi/include \
  -I/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
  /probe/rs_nvfp4_oracle.cu -o /probe/rs_nvfp4_oracle
/probe/rs_nvfp4_oracle --cpu-only
