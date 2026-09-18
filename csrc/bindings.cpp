#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Native C++20 core for cid-engine";
#ifdef CID_ENGINE_WITH_CUDA
  m.attr("cuda_backend_built") = pybind11::bool_(true);
#else
  m.attr("cuda_backend_built") = pybind11::bool_(false);
#endif
}
