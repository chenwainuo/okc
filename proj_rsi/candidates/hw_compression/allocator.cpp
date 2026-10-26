// Host-only allocation adapter. No device kernel, decoder, cache policy or precision change.
#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
#define HC_STR1(x) #x
#define HC_STR(x) HC_STR1(x)

struct Driver {
  void* lib = nullptr;
#define HC_DECL(f) decltype(&f) p_##f = nullptr
  HC_DECL(cuGetErrorName); HC_DECL(cuGetErrorString); HC_DECL(cuCtxGetCurrent);
  HC_DECL(cuCtxGetDevice); HC_DECL(cuCtxPushCurrent); HC_DECL(cuCtxPopCurrent);
  HC_DECL(cuCtxSynchronize); HC_DECL(cuDeviceGetAttribute); HC_DECL(cuStreamIsCapturing);
  HC_DECL(cuMemGetAllocationGranularity); HC_DECL(cuMemCreate);
  HC_DECL(cuMemGetAllocationPropertiesFromHandle); HC_DECL(cuMemAddressReserve);
  HC_DECL(cuMemMap); HC_DECL(cuMemSetAccess); HC_DECL(cuMemcpyDtoDAsync);
  HC_DECL(cuStreamSynchronize); HC_DECL(cuMemUnmap); HC_DECL(cuMemRelease);
  HC_DECL(cuMemAddressFree); HC_DECL(cuMemRetainAllocationHandle);
#undef HC_DECL
  Driver() {
    lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error("libcuda.so.1 unavailable");
#define HC_LOAD(f) p_##f = reinterpret_cast<decltype(&f)>(dlsym(lib, HC_STR(f))); \
    if (!p_##f) throw std::runtime_error("driver symbol unavailable: " HC_STR(f))
    HC_LOAD(cuGetErrorName); HC_LOAD(cuGetErrorString); HC_LOAD(cuCtxGetCurrent);
    HC_LOAD(cuCtxGetDevice); HC_LOAD(cuCtxPushCurrent); HC_LOAD(cuCtxPopCurrent);
    HC_LOAD(cuCtxSynchronize); HC_LOAD(cuDeviceGetAttribute); HC_LOAD(cuStreamIsCapturing);
    HC_LOAD(cuMemGetAllocationGranularity); HC_LOAD(cuMemCreate);
    HC_LOAD(cuMemGetAllocationPropertiesFromHandle); HC_LOAD(cuMemAddressReserve);
    HC_LOAD(cuMemMap); HC_LOAD(cuMemSetAccess); HC_LOAD(cuMemcpyDtoDAsync);
    HC_LOAD(cuStreamSynchronize); HC_LOAD(cuMemUnmap); HC_LOAD(cuMemRelease);
    HC_LOAD(cuMemAddressFree); HC_LOAD(cuMemRetainAllocationHandle);
#undef HC_LOAD
  }
  // Driver stays loaded until process exit so storage deleters cannot call unloaded code.
  void check(CUresult r, const char* op) const {
    if (r == CUDA_SUCCESS) return;
    const char *name = nullptr, *detail = nullptr;
    p_cuGetErrorName(r, &name); p_cuGetErrorString(r, &detail);
    throw std::runtime_error(std::string(op) + ": " + (name ? name : "unknown") +
                             " (" + std::to_string(int(r)) + ") " + (detail ? detail : ""));
  }
};
static Driver& driver() { static Driver* d = new Driver; return *d; }
#define HC_CK(f, ...) driver().check(driver().p_##f(__VA_ARGS__), HC_STR(f))

struct Region {
  CUcontext context = nullptr;
  CUmemGenericAllocationHandle handle = 0;
  CUdeviceptr ptr = 0;
  size_t bytes = 0, mapped_bytes = 0, granularity = 0;
  bool mapped = false, cleanup_failed = false;
  int device = -1, requested = -1, actual = -1;
  void release() {
    if (!handle && !ptr) return;
    // The graph owner must already have destroyed every graph exec referencing this region.
    // Synchronization drains submitted work; it cannot make a future graph replay safe.
    HC_CK(cuCtxPushCurrent, context);
    try {
      HC_CK(cuCtxSynchronize);
      if (mapped) { HC_CK(cuMemUnmap, ptr, mapped_bytes); mapped = false; }
      if (handle) { HC_CK(cuMemRelease, handle); handle = 0; }
      if (ptr) { HC_CK(cuMemAddressFree, ptr, mapped_bytes); ptr = 0; }
    } catch (...) {
      cleanup_failed = true;
      CUcontext old; driver().p_cuCtxPopCurrent(&old);
      throw; // Retain remaining resources rather than free potentially live/failed work.
    }
    CUcontext old; HC_CK(cuCtxPopCurrent, &old);
  }
  ~Region() noexcept {
    if (cleanup_failed) return;
    try { release(); }
    catch (const std::exception& e) { std::fprintf(stderr, "HWCOMP cleanup retained resources: %s\n", e.what()); }
  }
};

static py::dict capabilities() {
  CUcontext ctx = nullptr; CUdevice dev = -1;
  HC_CK(cuCtxGetCurrent, &ctx);
  TORCH_CHECK(ctx, "initialize Torch CUDA outside capture before querying allocation support");
  HC_CK(cuCtxGetDevice, &dev);
  int vmm = 0, generic = 0;
  HC_CK(cuDeviceGetAttribute, &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
  HC_CK(cuDeviceGetAttribute, &generic, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, dev);
  py::dict out; out["device"] = int(dev); out["vmm_supported"] = vmm;
  out["generic_compression_supported"] = generic; return out;
}

static py::dict pointer_properties(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "query expects a contiguous CUDA tensor");
  CUmemGenericAllocationHandle h = 0;
  CUresult r = driver().p_cuMemRetainAllocationHandle(&h, t.data_ptr());
  py::dict out; out["retain_result"] = int(r);
  if (r != CUDA_SUCCESS) {
    out["compression"] = py::none();
    out["meaning"] = "not queryable through VMM retain; compression state UNKNOWN, not NONE";
    return out;
  }
  CUmemAllocationProp p = {};
  CUresult q = driver().p_cuMemGetAllocationPropertiesFromHandle(&p, h);
  CUresult release = driver().p_cuMemRelease(h);
  driver().check(q, "cuMemGetAllocationPropertiesFromHandle");
  driver().check(release, "cuMemRelease retained query handle");
  out["compression"] = int(p.allocFlags.compressionType);
  out["meaning"] = "allocation attribute, not a measured compression ratio";
  return out;
}

struct WeightClone {
  std::shared_ptr<Region> region;
  at::Tensor tensor;
  WeightClone(const at::Tensor& src, uint64_t stream_address, bool compress) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous() && src.scalar_type() == at::kByte,
                "only exact contiguous CUDA uint8 RQ packed weights are admitted");
    TORCH_CHECK(src.numel() > 0 && !src.requires_grad(), "empty/trainable weights not admitted");
    auto cap = capabilities();
    TORCH_CHECK(cap["vmm_supported"].cast<int>(), "UNSUPPORTED: VMM unavailable");
    TORCH_CHECK(!compress || cap["generic_compression_supported"].cast<int>(),
                "UNSUPPORTED: generic compression unavailable");
    auto r = std::make_shared<Region>();
    HC_CK(cuCtxGetCurrent, &r->context);
    HC_CK(cuCtxGetDevice, &r->device);
    TORCH_CHECK(r->device == src.get_device(), "current context/source device mismatch");
    CUstream stream = reinterpret_cast<CUstream>(stream_address);
    CUstreamCaptureStatus capture;
    HC_CK(cuStreamIsCapturing, stream, &capture);
    TORCH_CHECK(capture == CU_STREAM_CAPTURE_STATUS_NONE, "allocation/copy is prepare-only, outside capture");
    r->bytes = size_t(src.numel());
    r->requested = compress ? CU_MEM_ALLOCATION_COMP_GENERIC : CU_MEM_ALLOCATION_COMP_NONE;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = r->device;
    prop.allocFlags.compressionType = static_cast<unsigned char>(r->requested);
    HC_CK(cuMemGetAllocationGranularity, &r->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    TORCH_CHECK(r->granularity && r->bytes <= std::numeric_limits<size_t>::max() - r->granularity,
                "allocation size/granularity overflow");
    r->mapped_bytes = ((r->bytes + r->granularity - 1) / r->granularity) * r->granularity;
    HC_CK(cuMemCreate, &r->handle, r->mapped_bytes, &prop, 0);
    CUmemAllocationProp actual = {};
    HC_CK(cuMemGetAllocationPropertiesFromHandle, &actual, r->handle);
    r->actual = actual.allocFlags.compressionType;
    TORCH_CHECK(r->actual == r->requested, "UNSUPPORTED: allocation did not grant requested compression attribute");
    HC_CK(cuMemAddressReserve, &r->ptr, r->mapped_bytes, r->granularity, 0, 0);
    HC_CK(cuMemMap, r->ptr, r->mapped_bytes, 0, r->handle, 0); r->mapped = true;
    CUmemAccessDesc access = {}; access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    HC_CK(cuMemSetAccess, r->ptr, r->mapped_bytes, &access, 1);
    // Driver D2D copy, including all packed bits, on caller's current Torch stream.
    HC_CK(cuMemcpyDtoDAsync, r->ptr, reinterpret_cast<CUdeviceptr>(src.data_ptr()), r->bytes, stream);
    HC_CK(cuStreamSynchronize, stream);
    tensor = at::from_blob(reinterpret_cast<void*>(r->ptr), src.sizes(), src.strides(),
                          [keep = r](void*) {}, src.options());
    region = std::move(r);
  }
  py::dict info() const {
    TORCH_CHECK(region, "clone closed");
    py::dict d; d["bytes"] = region->bytes; d["mapped_bytes"] = region->mapped_bytes;
    d["granularity"] = region->granularity; d["requested_compression"] = region->requested;
    d["actual_compression"] = region->actual; d["device"] = region->device;
    d["ptr"] = uint64_t(region->ptr); d["lifetime"] = "Tensor storage owns Region; graphs additionally need explicit owner";
    return d;
  }
  void close() {
    TORCH_CHECK(!tensor.defined() || (tensor.use_count() == 1 && tensor.storage().use_count() == 1),
                "cannot close: external tensor/alias references remain; destroy graph owners and state copies first");
    tensor = at::Tensor();
    TORCH_CHECK(!region || region.use_count() == 1, "cannot close: storage owners remain");
    if (region) region->release(); // Explicit lab close propagates cleanup failures.
    region.reset();
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("abi", [] { py::dict d; d["CUDA_VERSION"] = CUDA_VERSION;
      d["CUmemAllocationProp_size"] = sizeof(CUmemAllocationProp);
      d["COMP_NONE"] = int(CU_MEM_ALLOCATION_COMP_NONE);
      d["COMP_GENERIC"] = int(CU_MEM_ALLOCATION_COMP_GENERIC); return d; });
  m.def("capabilities", &capabilities);
  m.def("pointer_properties", &pointer_properties);
  py::class_<WeightClone>(m, "WeightClone")
      .def(py::init<const at::Tensor&, uint64_t, bool>())
      .def_property_readonly("tensor", [](const WeightClone& c) { return c.tensor; })
      .def("info", &WeightClone::info).def("close", &WeightClone::close);
}
