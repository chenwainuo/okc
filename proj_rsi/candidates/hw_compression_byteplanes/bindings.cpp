#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" cudaError_t okc_hwc_launch(const void*, void*, cudaStream_t);
extern "C" cudaError_t okc_hwc_attributes(cudaFuncAttributes*, int*);

static void checked(cudaError_t e) {
  TORCH_CHECK(e == cudaSuccess, "copy CUDA error: ", cudaGetErrorString(e));
}

static void copy64m(const at::Tensor& src, const at::Tensor& dst, uint64_t stream) {
  TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "same CUDA device required");
  TORCH_CHECK(src.scalar_type() == at::kByte && dst.scalar_type() == at::kByte, "uint8 only");
  TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "contiguous only");
  TORCH_CHECK(src.numel() == 67108864 && dst.numel() == 67108864, "exact64MiB required");
  auto s = reinterpret_cast<uintptr_t>(src.data_ptr()), d = reinterpret_cast<uintptr_t>(dst.data_ptr());
  TORCH_CHECK(s % 16 == 0 && d % 16 == 0, "16B alignment required");
  TORCH_CHECK(s + 67108864 <= d || d + 67108864 <= s, "source/output overlap");
  int current = -1; checked(cudaGetDevice(&current));
  TORCH_CHECK(current == src.get_device(), "current device mismatch");
  checked(okc_hwc_launch(src.data_ptr(), dst.data_ptr(), reinterpret_cast<cudaStream_t>(stream)));
}

static pybind11::dict info() {
  cudaFuncAttributes a{}; int occupancy = 0;
  checked(okc_hwc_attributes(&a, &occupancy));
  pybind11::dict d;
  d["numRegs"] = a.numRegs; d["localSizeBytes"] = a.localSizeBytes;
  d["sharedSizeBytes"] = a.sharedSizeBytes; d["maxThreadsPerBlock"] = a.maxThreadsPerBlock;
  d["occupancy_blocks_per_SM"] = occupancy; d["binaryVersion"] = a.binaryVersion;
  d["grid"] = pybind11::make_tuple(128, 1, 1);
  d["block"] = pybind11::make_tuple(256, 1, 1);
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy64m", &copy64m); m.def("info", &info);
}
