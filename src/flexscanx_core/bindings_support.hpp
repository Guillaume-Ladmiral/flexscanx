#pragma once

#include <memory>
#include <vector>

#include <pybind11/pybind11.h>

#include "static_precompute.hpp"

namespace py = pybind11;

namespace flexscanx_pybind {

struct PreHandle {
  std::unique_ptr<flexscanx_core::Precompute> pre;
};

void capsule_destruct(PyObject *capsule);

void validate_buildstatic_csr(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                              int N);

double fs_trigamma(double x);

void register_static_bindings(py::module_ &m);
void register_scan_bindings(py::module_ &m);

} // namespace flexscanx_pybind
