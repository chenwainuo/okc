#!/usr/bin/env bash
# Inside pinned image; /probe contains this isolated prototype. No GPU needed.
set -euo pipefail
/usr/local/cuda/bin/nvcc -std=c++20 --expt-relaxed-constexpr \
  -gencode arch=compute_90a,code=sm_90a -Xptxas=-v \
  -I/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
  /probe/rs_bf16_oracle.cu -o /probe/rs_bf16_oracle
/probe/rs_bf16_oracle --cpu-only
