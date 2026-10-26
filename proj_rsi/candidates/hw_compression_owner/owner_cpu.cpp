// CPU-only model of frozen v1's from_blob + pybind Tensor property + close guard.
#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;
static int released = 0;
struct Region {
  std::unique_ptr<uint8_t[]> bytes{new uint8_t[64]()};
  ~Region() { ++released; }
};
struct Owner {
  std::shared_ptr<Region> region;
  at::Tensor tensor;
  Owner() {
    auto r = std::make_shared<Region>();
    tensor = at::from_blob(r->bytes.get(), {64}, [keep = r](void*) {},
                          at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    region = std::move(r);
  }
  py::dict counts() const {
    py::dict d;
    d["defined"] = tensor.defined(); d["Region_owners"] = region.use_count();
    d["released_total"] = released;
    if (tensor.defined()) {
      d["TensorImpl_owners"] = tensor.use_count();
      d["Storage_direct_owners"] = tensor.storage().use_count();
      auto copied_storage = tensor.storage();
      d["Storage_with_explicit_copy_owners"] = copied_storage.use_count();
    }
    return d;
  }
  void v1_close() {
    TORCH_CHECK(!tensor.defined() || (tensor.use_count() == 1 && tensor.storage().use_count() == 1),
                "v1 guard rejected TensorImpl or Storage aliases");
    tensor = at::Tensor();
    TORCH_CHECK(!region || region.use_count() == 1, "v1 guard rejected remaining Region owner");
    region.reset();
  }
  void storage_owner_close() {
    // Dispose of our Tensor handle before asking whether any actual storage owner remains.
    // This also releases a preserved, otherwise unreferenced THPVariable wrapper.
    tensor = at::Tensor();
    TORCH_CHECK(!region || region.use_count() == 1,
                "storage-owner guard rejected remaining external Tensor/Storage alias");
    region.reset();
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Owner>(m, "Owner")
      .def(py::init<>())
      .def_property_readonly("tensor", [](const Owner& o) { return o.tensor; })
      .def("counts", &Owner::counts).def("v1_close", &Owner::v1_close)
      .def("storage_owner_close", &Owner::storage_owner_close);
}
