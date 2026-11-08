#include <pybind11/stl.h>

#include "bindings_scan_internal.hpp"

namespace flexscanx_pybind {

void register_scan_bindings(py::module_ &m) {
  m.def(
      "ScanSlice",
      [](py::capsule cap, std::vector<int> observed, std::vector<double> expected,
         py::dict settings) {
        auto *handle = cap.get_pointer<PreHandle>();
        return execute_scan_slice(handle, observed, expected, settings);
      },
      py::arg("handle"), py::arg("observed"), py::arg("expected"), py::arg("settings"));
}

} // namespace flexscanx_pybind
