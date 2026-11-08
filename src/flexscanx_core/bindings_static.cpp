#include "bindings_support.hpp"

#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "algorithms/universe.hpp"

using namespace flexscanx_core;

namespace flexscanx_pybind {

void register_static_bindings(py::module_ &m) {
  m.def(
      "BuildStatic",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> coords,
         std::vector<int> row_ptr, std::vector<int> col_idx, int K, double radius,
         std::string knn_mode, bool knn_refill_force) {
        if (coords.ndim() != 2 || coords.shape(1) != 2)
          throw std::runtime_error("coords must be Nx2 array");
        if (coords.shape(0) < 1)
          throw py::value_error("coords must contain at least one point");
        if (coords.shape(0) > std::numeric_limits<int>::max())
          throw py::value_error("coords contains too many points");
        int N = static_cast<int>(coords.shape(0));
        if (K < 1 || K > N)
          throw py::value_error("K must be in [1, N]");
        if (!(radius >= 0.0))
          throw py::value_error("radius must be finite and non-negative or +inf");
        validate_buildstatic_csr(row_ptr, col_idx, N);
        std::vector<double> x(N), y(N);
        auto c = coords.unchecked<2>();
        for (int i = 0; i < N; ++i) {
          x[i] = c(i, 0);
          y[i] = c(i, 1);
          if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
            throw py::value_error("coordinates must be finite");
        }
        CSR csr;
        csr.row_ptr = std::move(row_ptr);
        csr.col_idx = std::move(col_idx);
        KNNMode mode = KNNMode::K_FIXED;
        for (auto &ch : knn_mode)
          ch = (char)std::toupper((unsigned char)ch);
        if (knn_mode == "K_REFILL")
          mode = KNNMode::K_REFILL;
        else if (knn_mode == "K_FIXED")
          mode = KNNMode::K_FIXED;
        else
          throw py::value_error("knn_mode must be one of {'K_FIXED', 'K_REFILL'}");
        auto h = std::make_unique<PreHandle>();
        h->pre = build_static(x, y, csr, K, radius, mode, knn_refill_force);
        if (mode == KNNMode::K_REFILL && h->pre && !h->pre->refill_enabled && !knn_refill_force) {
          if (PyErr_WarnEx(PyExc_UserWarning,
                           "knn_mode='K_REFILL' requested but disabled due to guardrail (N>5000). "
                           "Proceeding with K_FIXED. Pass knn_refill_force=True to override.",
                           1) < 0) {
            throw py::error_already_set();
          }
        }
        try {
          std::string eff = (h->pre->knn_mode == KNNMode::K_REFILL && h->pre->refill_enabled)
                                ? "K_REFILL"
                                : "K_FIXED";
          auto logging = py::module_::import("logging");
          auto logger = logging.attr("getLogger")("flexscanx");
          std::string msg = std::string("BuildStatic effective knn_mode=") + eff +
                            ", requested=" + knn_mode +
                            ", refill_enabled=" + (h->pre->refill_enabled ? "true" : "false") +
                            ", N=" + std::to_string(h->pre->N) + ", K=" + std::to_string(h->pre->K);
          logger.attr("debug")(msg);
        } catch (...) {
        }
        return py::capsule(h.release(), "flexscanx_pre", capsule_destruct);
      },
      py::arg("coords"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("K"),
      py::arg("radius") = std::numeric_limits<double>::infinity(),
      py::arg("knn_mode") = std::string("K_REFILL"), py::arg("knn_refill_force") = false);

  m.def(
      "DescribeStatic",
      [](py::capsule cap) {
        auto *h = cap.get_pointer<PreHandle>();
        if (!h || !h->pre)
          throw std::runtime_error("invalid precompute handle");
        py::dict info;
        info["N"] = h->pre->N;
        info["K"] = h->pre->K;
        info["radius"] = h->pre->radius;
        std::string mode = (h->pre->knn_mode == KNNMode::K_REFILL ? "K_REFILL" : "K_FIXED");
        info["knn_mode"] = mode;
        info["refill_enabled"] = h->pre->refill_enabled;
        info["has_refill_data"] = h->pre->refill_enabled && !h->pre->knn_idx_refill.empty();
        // Centers whose K-th and (K+1)-th neighbours are exactly equidistant
        // (FXA-187): their universe is resolved by area index, so results can
        // depend on the input row order.
        info["knn_tie_boundary_centers"] = h->pre->knn_tie_boundary_count();
        return info;
      },
      py::arg("handle"));

  // Debug/verification binding (U-6, 2026-09-07): the effective universe the
  // scan cores see for one center, after the K limit, radius, refill and the
  // optional exclusion mask -- exactly what build_universe (flexible) and
  // build_knn_order (circular) produce.  Indices are 0-based.  Not part of the
  // public inference surface; used to compare the engine against independent
  // enumeration references on tie-laden geometry.
  m.def(
      "DescribeUniverse",
      [](py::capsule cap, int center, std::optional<std::vector<int>> excluded) {
        auto *h = cap.get_pointer<PreHandle>();
        if (!h || !h->pre)
          throw std::runtime_error("invalid precompute handle");
        const Precompute &pre = *h->pre;
        if (center < 0 || center >= pre.N)
          throw py::value_error("center must be in [0, N)");
        std::vector<char> mask;
        const std::vector<char> *mask_ptr = nullptr;
        if (excluded.has_value()) {
          mask.assign(pre.N, 0);
          for (int idx : *excluded) {
            if (idx < 0 || idx >= pre.N)
              throw py::value_error("excluded indices must be in [0, N)");
            mask[idx] = 1;
          }
          mask_ptr = &mask;
        }
        std::vector<char> in_universe;
        build_universe(pre, center, mask_ptr, in_universe);
        std::vector<int> flexible;
        for (int i = 0; i < pre.N; ++i)
          if (in_universe[i])
            flexible.push_back(i);
        std::vector<int> order;
        std::vector<double> order_dist2;
        build_knn_order(pre, center, mask_ptr, order, order_dist2);
        py::dict out;
        out["center"] = center;
        out["flexible"] = flexible;
        out["circular_order"] = order;
        out["circular_dist2"] = order_dist2;
        return out;
      },
      py::arg("handle"), py::arg("center"), py::arg("excluded") = py::none());
}

} // namespace flexscanx_pybind
