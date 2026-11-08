#include "bindings_scan_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <pybind11/stl.h>

using namespace flexscanx_core;

namespace flexscanx_pybind {

namespace {

py::object safe_ratio(double num_in, double den_in, double num_out, double den_out) {
  bool ok_in = (den_in > 0.0) && std::isfinite(num_in) && std::isfinite(den_in);
  bool ok_out = (den_out > 0.0) && std::isfinite(num_out) && std::isfinite(den_out);
  if (!ok_in || !ok_out)
    return py::none();

  double p_in = num_in / den_in;
  double p_out = num_out / den_out;
  if (!(p_out > 0.0) || !std::isfinite(p_out)) {
    if ((p_in > 0.0) && std::isfinite(p_in))
      return py::float_(std::numeric_limits<double>::infinity());
    return py::none();
  }

  return py::float_(p_in / p_out);
}

} // namespace

py::dict assemble_scan_result(const ScanExecutionContext &ctx, const ClusterSelection &selection,
                              const std::vector<double> &simdist) {
  const auto &observed = *ctx.observed;
  const auto &expected = *ctx.expected;

  py::dict out;
  std::vector<int> area0 = selection.area;
  std::sort(area0.begin(), area0.end());
  std::vector<int> area1b;
  area1b.reserve(area0.size());
  for (int value : area0)
    area1b.push_back(value + 1);

  out["area"] = area1b;
  out["n_case"] = selection.nZ;
  out["expected"] = selection.mZ;

  double nG = 0.0;
  double mG = 0.0;
  for (size_t i = 0; i < observed.size(); ++i) {
    nG += observed[i];
    mG += expected[i];
  }

  double nOut = nG - selection.nZ;
  double mOut = mG - selection.mZ;
  out["ratio_smoothing_applied"] =
      ctx.use_eb ? (ctx.eb_ok ? "EB" : "NONE") : (ctx.use_haldane ? "HALDANE" : "NONE");

  if (ctx.model == "POISSON") {
    py::object rate_ratio;
    if (ctx.use_eb && ctx.eb_ok) {
      double lin = (ctx.eb_a + selection.nZ) / (ctx.eb_b + selection.mZ);
      double lout = (ctx.eb_a + nOut) / (ctx.eb_b + mOut);
      rate_ratio = safe_ratio(lin, 1.0, lout, 1.0);

      double var_log_lin = fs_trigamma(ctx.eb_a + selection.nZ);
      double var_log_lout = fs_trigamma(ctx.eb_a + nOut);
      double se = std::sqrt(var_log_lin + var_log_lout);
      double lr = std::log((lin > 0.0 && std::isfinite(lin)) ? lin : 1.0) -
                  std::log((lout > 0.0 && std::isfinite(lout)) ? lout : 1.0);
      double z = 1.96;
      out["Rate_Ratio_CI"] = py::make_tuple(std::exp(lr - z * se), std::exp(lr + z * se));
    } else {
      double k = ctx.use_haldane ? 0.5 : 0.0;
      rate_ratio = safe_ratio(selection.nZ + k, selection.mZ, nOut + k, mOut);
      out["Rate_Ratio_CI"] = py::none();
    }
    out["Rate_Ratio"] = rate_ratio;
  } else {
    py::object risk_ratio;
    if (ctx.use_eb && ctx.eb_ok) {
      double pin = (ctx.eb_alpha + selection.nZ) / (ctx.eb_alpha + ctx.eb_beta + selection.mZ);
      double pout = (ctx.eb_alpha + nOut) / (ctx.eb_alpha + ctx.eb_beta + mOut);
      risk_ratio = safe_ratio(pin, 1.0, pout, 1.0);

      double a_in = ctx.eb_alpha + selection.nZ;
      double b_in = ctx.eb_beta + (selection.mZ - selection.nZ);
      double a_out = ctx.eb_alpha + nOut;
      double b_out = ctx.eb_beta + (mOut - nOut);
      auto var_logp = [](double a, double b) { return fs_trigamma(a) - fs_trigamma(a + b); };
      double var_log_pin = var_logp(a_in, b_in);
      double var_log_pout = var_logp(a_out, b_out);
      double se = std::sqrt(std::max(0.0, var_log_pin) + std::max(0.0, var_log_pout));
      double lr = std::log((pin > 0.0 && std::isfinite(pin)) ? pin : 1.0) -
                  std::log((pout > 0.0 && std::isfinite(pout)) ? pout : 1.0);
      double z = 1.96;
      out["Risk_Ratio_CI"] = py::make_tuple(std::exp(lr - z * se), std::exp(lr + z * se));
    } else {
      double k = ctx.use_haldane ? 0.5 : 0.0;
      risk_ratio = safe_ratio(selection.nZ + k, selection.mZ + 2.0 * k, nOut + k, mOut + 2.0 * k);
      out["Risk_Ratio_CI"] = py::none();
    }
    out["Risk_Ratio"] = risk_ratio;
  }

  out["stats"] = selection.stats;

  double maxd = 0.0;
  int from = -1;
  int to = -1;
  for (size_t i = 0; i < area0.size(); ++i) {
    for (size_t j = i + 1; j < area0.size(); ++j) {
      double d = euclid(ctx.handle->pre->x[area0[i]], ctx.handle->pre->y[area0[i]],
                        ctx.handle->pre->x[area0[j]], ctx.handle->pre->y[area0[j]]);
      if (d > maxd) {
        maxd = d;
        from = area0[i];
        to = area0[j];
      }
    }
  }

  out["max_dist"] = maxd;
  if (from >= 0 && to >= 0) {
    out["from_area"] = from + 1;
    out["to_area"] = to + 1;
  }

  if (!simdist.empty()) {
    int64_t rank = 1;
    for (double value : simdist) {
      if (value >= selection.stats)
        ++rank;
    }
    out["rank"] = (int)rank;
    out["pval"] = double(rank) / double(simdist.size() + 1);
    // MC-3: expose the per-direction full-region null-max array so the BOTH
    // envelope can build the combined two-sided null in Python. Additive and
    // gated -- only emitted for the BOTH sub-scans (emit_sim_max), so the
    // single-direction HOT/COLD result dict is unchanged.
    if (ctx.emit_sim_max)
      out["sim_max"] = simdist;
  } else {
    out["rank"] = 1;
    out["pval"] = py::none();
  }
  // ``simdist`` is resized to the completed prefix when sequential Monte Carlo
  // stopping fires.  Report its size rather than the requested budget so every
  // downstream p-value/provenance consumer can identify the true denominator.
  out["simcount_actual"] = simdist.size();

  return out;
}

} // namespace flexscanx_pybind
