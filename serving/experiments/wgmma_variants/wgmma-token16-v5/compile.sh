#!/usr/bin/env bash
set -euo pipefail
/usr/local/cuda/bin/nvcc -std=c++20 -O3 --expt-relaxed-constexpr \
 -gencode arch=compute_90a,code=sm_90a -DSGL_CUDA_ARCH=900 -Xptxas=-v \
 -I/probe -I/baseline/serving/sglang/python/sglang/kernels/jit/include \
 -I/baseline/serving/sglang/python/sglang/kernels/jit/csrc \
 -I/opt/sglang/lib/python3.12/site-packages/tvm_ffi/include \
 -I/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
 -c /probe/compile.cu -o /probe/routed_down.o
/usr/local/cuda/bin/cuobjdump --dump-sass /probe/routed_down.o > /probe/routed_down.sass
