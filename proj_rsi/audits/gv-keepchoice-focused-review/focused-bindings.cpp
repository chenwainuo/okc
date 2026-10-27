#include <torch/extension.h>
int64_t gvpredrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvocc(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvregs(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rsrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t prrun(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor);
PYBIND11_MODULE(review_keepchoice,m){
m.def("gvpredrun",&gvpredrun);
m.def("gvrun",&gvrun);
m.def("gvocc",&gvocc);
m.def("gvregs",&gvregs);
m.def("rrrun",&rrrun);
m.def("rsrun",&rsrun);
m.def("prrun",&prrun);
}
