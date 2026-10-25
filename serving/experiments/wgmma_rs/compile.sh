#!/usr/bin/env bash
# Run inside pinned image with /sglang read-only source and /probe candidate dir.
# No GPU required; no launch occurs.
set -euo pipefail
/usr/local/cuda/bin/nvcc -std=c++20 --expt-relaxed-constexpr \
  -gencode arch=compute_90a,code=sm_90a -DSGL_CUDA_ARCH=900 -dc -Xptxas=-v \
  -I/sglang/python/sglang/kernels/jit/include \
  -I/sglang/python/sglang/kernels/jit/csrc \
  -I/opt/sglang/lib/python3.12/site-packages/tvm_ffi/include \
  -I/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
  /probe/rs_tile_compile.cu -o /probe/rs_tile_compile.o
/usr/local/cuda/bin/cuobjdump -sass /probe/rs_tile_compile.o > /probe/rs_tile_compile.sass
