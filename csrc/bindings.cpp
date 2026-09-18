#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Native C++20 core for cid-engine";
}
