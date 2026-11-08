#include <pybind11/pybind11.h>

#include "bindings_support.hpp"
#include "middle_p_binomial.hpp"
#include "middle_p_poisson.hpp"
#include "version.hpp"

PYBIND11_MODULE(_core, m) {
  m.doc() = "flexscanx core";

  m.def(
      "version", []() { return std::string(FLEXSCANX_VERSION) + "-core"; },
      "Return core version string (auto-generated from pyproject.toml)");
  m.def(
      "source_digest", []() { return std::string(FLEXSCANX_SOURCE_DIGEST); },
      "Return the SHA-256 digest of the native sources this extension was built from "
      "(sorted '<relative path>\\n<sha256>\\n' records over src/flexscanx_core/*.cpp|hpp|in "
      "and CMakeLists.txt); a mismatch against the working tree means a stale binary");

  auto validate_poisson = [](double mean) {
    if (!(mean >= 0.0) || !std::isfinite(mean))
      throw py::value_error("mean must be finite and non-negative");
  };
  m.def(
      "poisson_middle_p_upper",
      [validate_poisson](int ax, double mean) {
        validate_poisson(mean);
        return flexscanx_core::poisson_middle_p_upper(ax, mean);
      },
      py::arg("ax"), py::arg("mean"));
  m.def(
      "poisson_middle_p_lower",
      [validate_poisson](int ax, double mean) {
        validate_poisson(mean);
        return flexscanx_core::poisson_middle_p_lower(ax, mean);
      },
      py::arg("ax"), py::arg("mean"));
  auto validate_binomial = [](int n, double p) {
    if (n < 0)
      throw py::value_error("n must be non-negative");
    if (!(p >= 0.0 && p <= 1.0) || !std::isfinite(p))
      throw py::value_error("p must be finite and lie in [0, 1]");
  };
  m.def(
      "binomial_middle_p_upper",
      [validate_binomial](int ax, int n, double p) {
        validate_binomial(n, p);
        return flexscanx_core::binomial_middle_p_upper(ax, n, p);
      },
      py::arg("ax"), py::arg("n"), py::arg("p"));
  m.def(
      "binomial_middle_p_lower",
      [validate_binomial](int ax, int n, double p) {
        validate_binomial(n, p);
        return flexscanx_core::binomial_middle_p_lower(ax, n, p);
      },
      py::arg("ax"), py::arg("n"), py::arg("p"));

  flexscanx_pybind::register_static_bindings(m);
  flexscanx_pybind::register_scan_bindings(m);
}
