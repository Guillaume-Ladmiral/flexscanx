#include "algorithms/scheduler.hpp"
#include "bindings_scan_internal.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include <pybind11/stl.h>

#include "middle_p_binomial.hpp"
#include "middle_p_poisson.hpp"
#include "monte_carlo.hpp"
#include "scanner_circular_binomial.hpp"
#include "scanner_circular_poisson.hpp"
#include "scanner_flexible_binomial.hpp"
#include "scanner_flexible_poisson.hpp"

using namespace flexscanx_core;

namespace flexscanx_pybind {

namespace {

// Cooperative interrupt poll installed into ScanSettings::interrupt_poll for the
// duration of one ScanSlice call (FXA-188 / D6).  Only the thread that entered
// the binding may run Python signal handlers (PyErr_CheckSignals is a no-op on
// other threads), so worker threads return immediately.  The owning thread
// reacquires the GIL at most every 50 ms; a pending KeyboardInterrupt becomes a
// C++ exception that unwinds through the schedulers' cancel/join/rethrow path
// and reaches Python as the original exception.
struct InterruptPoller {
  std::thread::id owner{std::this_thread::get_id()};
  std::chrono::steady_clock::time_point last{std::chrono::steady_clock::now()};
  static constexpr std::chrono::milliseconds cadence{50};

  void operator()() {
    if (std::this_thread::get_id() != owner)
      return;
    const auto now = std::chrono::steady_clock::now();
    if (now - last < cadence)
      return;
    last = now;
    py::gil_scoped_acquire gil;
    if (PyErr_CheckSignals() != 0)
      throw py::error_already_set();
  }
};

void to_upper(std::string &value) {
  for (auto &ch : value)
    ch = (char)std::toupper((unsigned char)ch);
}

ScanSettings parse_scan_settings(py::dict settings) {
  ScanSettings st;
  if (settings.contains("ralpha"))
    st.ralpha = settings[py::str("ralpha")].cast<double>();
  if (settings.contains("clustersize"))
    st.clustersize = settings[py::str("clustersize")].cast<int>();
  if (settings.contains("n_jobs_centers"))
    st.n_jobs_centers = settings[py::str("n_jobs_centers")].cast<int>();
  if (settings.contains("debug_prune"))
    st.debug_prune = settings[py::str("debug_prune")].cast<bool>();
  if (settings.contains("enable_center_ub_gating"))
    st.enable_center_ub_gating = settings[py::str("enable_center_ub_gating")].cast<bool>();
  if (settings.contains("cache_candidate_llr"))
    st.cache_candidate_llr = settings[py::str("cache_candidate_llr")].cast<bool>();
  if (settings.contains("enable_frame_ub_prune"))
    st.enable_frame_ub_prune = settings[py::str("enable_frame_ub_prune")].cast<bool>();
  if (settings.contains("scheduler_backend")) {
    std::string scheduler = settings[py::str("scheduler_backend")].cast<std::string>();
    to_upper(scheduler);
    if (scheduler == "WORK_STEAL")
      st.scheduler_backend = ScanSettings::SchedulerBackend::WORK_STEAL;
    else if (scheduler == "ATOMIC")
      st.scheduler_backend = ScanSettings::SchedulerBackend::ATOMIC;
    else
      throw py::value_error("scheduler_backend must be one of {'ATOMIC', 'WORK_STEAL'}");
  }
  return st;
}

std::vector<int> parse_priority_centers(py::dict settings) {
  std::vector<int> priority;
  if (!settings.contains("priority_centers"))
    return priority;
  auto values = settings[py::str("priority_centers")].cast<std::vector<int>>();
  priority.reserve(values.size());
  for (int value : values) {
    if (value > 0)
      priority.push_back(value - 1);
  }
  return priority;
}

void build_pre_gating_cache(ScanExecutionContext &ctx, const std::vector<int> &priority) {
  if (ctx.stattype != "RESTRICTED")
    return;
  if (!priority.empty()) {
    ctx.have_pre_gating = false;
    return;
  }
  const auto &observed = *ctx.observed;
  const auto &expected = *ctx.expected;
  if (ctx.model == "POISSON") {
    ctx.pre_gating.pv_hot.resize(observed.size());
    ctx.pre_gating.pv_cold.resize(observed.size());
    for (size_t i = 0; i < observed.size(); ++i) {
      ctx.pre_gating.pv_hot[i] = poisson_middle_p_upper(observed[i], expected[i]);
      ctx.pre_gating.pv_cold[i] = poisson_middle_p_lower(observed[i], expected[i]);
    }
    ctx.have_pre_gating = true;
    return;
  }
  if (ctx.model == "BINOMIAL") {
    double nG0 = 0.0;
    double mG0 = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
      nG0 += observed[i];
      mG0 += expected[i];
    }
    double p0m = (mG0 > 0.0) ? (nG0 / mG0) : 0.0;
    ctx.pre_gating.pv_hot.resize(observed.size());
    ctx.pre_gating.pv_cold.resize(observed.size());
    for (size_t i = 0; i < observed.size(); ++i) {
      int bn = (int)std::llround(expected[i]);
      ctx.pre_gating.pv_hot[i] = binomial_middle_p_upper(observed[i], bn, p0m);
      ctx.pre_gating.pv_cold[i] = binomial_middle_p_lower(observed[i], bn, p0m);
    }
    ctx.have_pre_gating = true;
  }
}

std::vector<int> build_center_order(const ScanExecutionContext &ctx,
                                    const std::vector<int> &priority) {
  std::vector<int> center_order;
  if (ctx.stattype == "RESTRICTED" && ctx.have_pre_gating) {
    double threshold = ctx.scan_settings.ralpha;
    center_order.reserve(ctx.handle->pre->N);
    for (int center = 0; center < ctx.handle->pre->N; ++center) {
      double p_hot =
          (center < (int)ctx.pre_gating.pv_hot.size()) ? ctx.pre_gating.pv_hot[center] : 1.0;
      double p_cold =
          (center < (int)ctx.pre_gating.pv_cold.size()) ? ctx.pre_gating.pv_cold[center] : 1.0;
      bool pass = false;
      if (ctx.clustertype == "HOT")
        pass = (p_hot < threshold);
      else if (ctx.clustertype == "COLD")
        pass = (p_cold < threshold);
      else
        pass = (p_hot < threshold) || (p_cold < threshold);
      if (pass)
        center_order.push_back(center);
    }
  }
  if (priority.empty())
    return center_order;
  std::vector<char> keep(ctx.handle->pre->N, 0);
  if (!center_order.empty()) {
    for (int value : center_order)
      keep[value] = 1;
  } else {
    std::fill(keep.begin(), keep.end(), 1);
    center_order.reserve(ctx.handle->pre->N);
    for (int center = 0; center < ctx.handle->pre->N; ++center)
      center_order.push_back(center);
  }
  std::vector<char> seen(ctx.handle->pre->N, 0);
  std::vector<int> merged;
  merged.reserve(center_order.size());
  for (int value : priority) {
    if (value >= 0 && value < ctx.handle->pre->N && keep[value] && !seen[value]) {
      merged.push_back(value);
      seen[value] = 1;
    }
  }
  for (int value : center_order) {
    if (value >= 0 && value < ctx.handle->pre->N && keep[value] && !seen[value]) {
      merged.push_back(value);
      seen[value] = 1;
    }
  }
  return merged;
}

void warn_on_sparse_radius(const ScanExecutionContext &ctx) {
  if (!std::isfinite(ctx.handle->pre->radius))
    return;
  try {
    int sparse = 0;
    for (int center = 0; center < ctx.handle->pre->N; ++center) {
      int within = 0;
      for (size_t k = 0; k < ctx.handle->pre->knn_idx[center].size(); ++k) {
        const int idx = ctx.handle->pre->knn_idx[center][k];
        if (euclid(ctx.handle->pre->x[center], ctx.handle->pre->y[center], ctx.handle->pre->x[idx],
                   ctx.handle->pre->y[idx]) <= ctx.handle->pre->radius) {
          within++;
        }
      }
      if (within <= 1)
        sparse++;
    }
    double fraction = (ctx.handle->pre->N > 0) ? (double)sparse / (double)ctx.handle->pre->N : 0.0;
    if (fraction >= 0.5) {
      py::module_::import("warnings")
          .attr("warn")("clusterradius excludes nearly all neighbors for many centers (>=50%). "
                        "Consider increasing radius, decreasing K, or checking CRS/units.");
    }
  } catch (...) {
  }
}

ScanExecutionContext prepare_scan_execution(PreHandle *handle, const std::vector<int> &observed,
                                            const std::vector<double> &expected,
                                            py::dict settings) {
  ScanExecutionContext ctx;
  ctx.handle = handle;
  ctx.observed = &observed;
  ctx.expected = &expected;
  ctx.scan_settings = parse_scan_settings(settings);
  ctx.model = settings.contains("model") ? settings[py::str("model")].cast<std::string>()
                                         : std::string("POISSON");
  ctx.scanmethod = settings.contains("scanmethod")
                       ? settings[py::str("scanmethod")].cast<std::string>()
                       : std::string("FLEXIBLE");
  ctx.stattype = settings.contains("stattype") ? settings[py::str("stattype")].cast<std::string>()
                                               : std::string("RESTRICTED");
  ctx.clustertype = settings.contains("clustertype")
                        ? settings[py::str("clustertype")].cast<std::string>()
                        : std::string("HOT");
  to_upper(ctx.model);
  to_upper(ctx.scanmethod);
  to_upper(ctx.stattype);
  to_upper(ctx.clustertype);
  if (ctx.model != "POISSON" && ctx.model != "BINOMIAL")
    throw py::value_error("model must be one of {'POISSON', 'BINOMIAL'}");
  if (ctx.scanmethod != "FLEXIBLE" && ctx.scanmethod != "CIRCULAR")
    throw py::value_error("scanmethod must be one of {'FLEXIBLE', 'CIRCULAR'}");
  if (ctx.stattype != "RESTRICTED" && ctx.stattype != "ORIGINAL")
    throw py::value_error("stattype must be one of {'RESTRICTED', 'ORIGINAL'}");
  if (ctx.clustertype != "HOT" && ctx.clustertype != "COLD")
    throw py::value_error("clustertype must be one of {'HOT', 'COLD'}");
  if (!std::isfinite(ctx.scan_settings.ralpha))
    throw py::value_error("ralpha must be finite");
  if (ctx.scan_settings.ralpha < 0.0 || ctx.scan_settings.ralpha > 1.0)
    throw py::value_error("ralpha must be in [0,1]");
  if (ctx.scan_settings.clustersize < 1)
    throw py::value_error("clustersize must be >= 1");
  if (ctx.scan_settings.n_jobs_centers < 1)
    throw py::value_error("n_jobs_centers must be >= 1");
  for (size_t i = 0; i < observed.size(); ++i) {
    if (observed[i] < 0)
      throw py::value_error("observed must be non-negative");
    if (!std::isfinite(expected[i]))
      throw py::value_error(ctx.model == "BINOMIAL" ? "population must be finite"
                                                    : "expected must be finite");
    if (ctx.model == "POISSON" && expected[i] < 0.0)
      throw py::value_error("expected must be non-negative");
    if (ctx.model == "BINOMIAL") {
      if (expected[i] <= 0.0 || std::floor(expected[i]) != expected[i])
        throw py::value_error("population must contain integers greater than zero");
      if (expected[i] > (double)std::numeric_limits<int>::max())
        throw py::value_error("population exceeds the supported int range");
      if ((double)observed[i] > expected[i])
        throw py::value_error("observed cannot exceed population");
    }
  }
  ctx.hot = (ctx.clustertype == "HOT");
  auto priority = parse_priority_centers(settings);
  build_pre_gating_cache(ctx, priority);
  ctx.center_order = build_center_order(ctx, priority);
  warn_on_sparse_radius(ctx);
  ctx.mc_progress_every = settings.contains("mc_progress_every")
                              ? settings[py::str("mc_progress_every")].cast<int>()
                              : 0;
  if (settings.contains("mc_progress_every") && ctx.mc_progress_every < 1)
    throw py::value_error("mc_progress_every must be >= 1");
  ctx.mc_progress =
      settings.contains("mc_progress") ? (py::object)settings[py::str("mc_progress")] : py::none();
  ctx.allow_release = ctx.mc_progress.is_none() && ctx.mc_progress_every <= 0;
  ctx.simcount = settings.contains("simcount") ? settings[py::str("simcount")].cast<int>() : 0;
  if (ctx.simcount < 0)
    throw py::value_error("simcount must be >= 0");
  ctx.rantype =
      settings.contains("rantype")
          ? settings[py::str("rantype")].cast<std::string>()
          : (ctx.model == "BINOMIAL" ? std::string("BINOMIAL") : std::string("MULTINOMIAL"));
  to_upper(ctx.rantype);
  if ((ctx.model == "POISSON" && ctx.rantype != "MULTINOMIAL" && ctx.rantype != "POISSON") ||
      (ctx.model == "BINOMIAL" && ctx.rantype != "BINOMIAL"))
    throw py::value_error("rantype is invalid for the selected model");
  if (ctx.rantype == "POISSON") {
    for (double value : expected) {
      if (!poisson_mean_supported(value))
        throw py::value_error("Poisson mean exceeds the supported int range");
    }
  }
  if (ctx.simcount > 0 && ctx.rantype == "MULTINOMIAL") {
    long long observed_total = 0;
    for (int value : observed) {
      observed_total += (long long)value;
      if (observed_total > (long long)std::numeric_limits<int>::max())
        throw py::value_error("multinomial total exceeds the supported int range");
    }
  }
  ctx.crn = settings.contains("crn") ? settings[py::str("crn")].cast<bool>() : false;
  bool has_seed = settings.contains("seed");
  ctx.seed_base = has_seed ? settings[py::str("seed")].cast<long long>() : 0xDEADBEEF;
  if (has_seed && (ctx.seed_base < 0 || ctx.seed_base > 2147483647LL))
    throw py::value_error("seed must be in [0, 2147483647]");
  ctx.time_index =
      settings.contains("time_index") ? settings[py::str("time_index")].cast<int>() : 0;
  if (ctx.time_index < 0)
    throw py::value_error("time_index must be >= 0");
  ctx.mc_early_stop =
      settings.contains("mc_early_stop") ? settings[py::str("mc_early_stop")].cast<bool>() : false;
  ctx.mc_alpha =
      settings.contains("mc_alpha") ? settings[py::str("mc_alpha")].cast<double>() : 0.05;
  if (!std::isfinite(ctx.mc_alpha) || ctx.mc_alpha <= 0.0 || ctx.mc_alpha >= 1.0)
    throw py::value_error("mc_alpha must be in (0,1)");
  ctx.mc_alias = settings.contains("mc_alias") ? settings[py::str("mc_alias")].cast<bool>() : false;
  ctx.secondary_null = settings.contains("secondary_null")
                           ? settings[py::str("secondary_null")].cast<std::string>()
                           : std::string("FULL");
  to_upper(ctx.secondary_null);
  if (ctx.secondary_null != "FULL" && ctx.secondary_null != "CONDITIONAL")
    throw py::value_error("secondary_null must be one of {'full', 'conditional'}");
  // Internal flag set only on the BOTH envelope's two one-sided sub-scans so the
  // per-direction full-region null-max array is returned to Python (MC-3).
  ctx.emit_sim_max =
      settings.contains("_emit_sim_max") ? settings[py::str("_emit_sim_max")].cast<bool>() : false;
  std::string ratio_smoothing = settings.contains("ratio_smoothing")
                                    ? settings[py::str("ratio_smoothing")].cast<std::string>()
                                    : std::string("NONE");
  to_upper(ratio_smoothing);
  if (ratio_smoothing != "NONE" && ratio_smoothing != "HALDANE" && ratio_smoothing != "EB") {
    throw py::value_error("ratio_smoothing must be one of {'NONE', 'HALDANE', 'EB'}");
  }
  ctx.use_haldane = (ratio_smoothing == "HALDANE");
  ctx.use_eb = (ratio_smoothing == "EB");
  if (!ctx.use_eb)
    return ctx;
  if (ctx.model == "POISSON") {
    std::vector<double> rates;
    rates.reserve(observed.size());
    double sum_inv_m = 0.0;
    int count = 0;
    double sum_m = 0.0;
    double sum_n = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
      double m0 = expected[i];
      int n0 = observed[i];
      if (!(m0 > 0.0) || !std::isfinite(m0))
        continue;
      double ri = (double)n0 / m0;
      if (!std::isfinite(ri))
        continue;
      rates.push_back(ri);
      sum_inv_m += 1.0 / m0;
      sum_m += m0;
      sum_n += n0;
      count++;
    }
    if (count >= 2 && sum_m > 0.0) {
      double rbar = sum_n / sum_m;
      double s2 = 0.0;
      for (double value : rates) {
        double diff = value - rbar;
        s2 += diff * diff;
      }
      s2 /= (double)count;
      double mean_inv_m = sum_inv_m / (double)count;
      double mu_var_hat = s2 - rbar * mean_inv_m;
      if (mu_var_hat > 0.0 && std::isfinite(mu_var_hat) && rbar > 0.0 && std::isfinite(rbar)) {
        ctx.eb_a = (rbar * rbar) / mu_var_hat;
        ctx.eb_b = rbar / mu_var_hat;
        if (std::isfinite(ctx.eb_a) && std::isfinite(ctx.eb_b) && ctx.eb_a > 0.0 &&
            ctx.eb_b > 0.0) {
          ctx.eb_ok = true;
        }
      }
    }
  } else if (ctx.model == "BINOMIAL") {
    std::vector<double> probs;
    probs.reserve(observed.size());
    double sum_inv_m = 0.0;
    int count = 0;
    double sum_m = 0.0;
    double sum_y = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
      double md = expected[i];
      if (!std::isfinite(md))
        continue;
      long long mi = (long long)std::llround(md);
      if (mi <= 0)
        continue;
      int yi = observed[i];
      if (yi < 0)
        yi = 0;
      if (yi > mi)
        yi = (int)mi;
      double pi = (double)yi / (double)mi;
      probs.push_back(pi);
      sum_inv_m += 1.0 / (double)mi;
      sum_m += (double)mi;
      sum_y += (double)yi;
      count++;
    }
    if (count >= 2 && sum_m > 0.0) {
      double pbar = sum_y / sum_m;
      double s2 = 0.0;
      for (double value : probs) {
        double diff = value - pbar;
        s2 += diff * diff;
      }
      s2 /= (double)count;
      double mean_noise = pbar * (1.0 - pbar) * (sum_inv_m / (double)count);
      double tau_hat = s2 - mean_noise;
      if (tau_hat > 0.0 && std::isfinite(tau_hat) && pbar > 0.0 && pbar < 1.0 &&
          std::isfinite(pbar)) {
        double t = pbar * (1.0 - pbar) / tau_hat - 1.0;
        if (std::isfinite(t) && t > 0.0) {
          ctx.eb_alpha = pbar * t;
          ctx.eb_beta = (1.0 - pbar) * t;
          if (std::isfinite(ctx.eb_alpha) && std::isfinite(ctx.eb_beta) && ctx.eb_alpha > 0.0 &&
              ctx.eb_beta > 0.0) {
            ctx.eb_ok = true;
          }
        }
      }
    }
  }
  return ctx;
}

ClusterSelection route_scan(const ScanExecutionContext &ctx, const std::vector<int> &observed,
                            const std::vector<char> *excluded, const std::vector<int> *center_order,
                            const GatingCache *gating_ext) {
  const GatingCache *gating_ptr =
      (ctx.stattype == "RESTRICTED" && gating_ext) ? gating_ext : nullptr;
  const auto &expected = *ctx.expected;
  if (ctx.model == "POISSON") {
    if (ctx.scanmethod == "FLEXIBLE") {
      if (ctx.stattype == "RESTRICTED") {
        return ctx.hot ? scan_flexible_hot_restricted_poisson(*ctx.handle->pre, observed, expected,
                                                              ctx.scan_settings, excluded,
                                                              center_order, gating_ptr)
                       : scan_flexible_cold_restricted_poisson(*ctx.handle->pre, observed, expected,
                                                               ctx.scan_settings, excluded,
                                                               center_order, gating_ptr);
      }
      if (ctx.stattype == "ORIGINAL") {
        return ctx.hot ? scan_flexible_hot_original_poisson(*ctx.handle->pre, observed, expected,
                                                            ctx.scan_settings, excluded,
                                                            center_order, gating_ptr)
                       : scan_flexible_cold_original_poisson(*ctx.handle->pre, observed, expected,
                                                             ctx.scan_settings, excluded,
                                                             center_order, gating_ptr);
      }
      throw std::runtime_error("Unknown stattype");
    }
    if (ctx.scanmethod == "CIRCULAR") {
      if (ctx.stattype == "RESTRICTED") {
        return scan_circular_poisson(*ctx.handle->pre, observed, expected, ctx.scan_settings,
                                     /*restricted=*/true, /*hot=*/ctx.hot, excluded, center_order,
                                     gating_ptr);
      }
      if (ctx.stattype == "ORIGINAL") {
        return scan_circular_poisson(*ctx.handle->pre, observed, expected, ctx.scan_settings,
                                     /*restricted=*/false, /*hot=*/ctx.hot, excluded, center_order,
                                     gating_ptr);
      }
      throw std::runtime_error("Unknown stattype");
    }
    throw std::runtime_error("Unknown scanmethod");
  }
  if (ctx.model == "BINOMIAL") {
    if (ctx.scanmethod == "FLEXIBLE") {
      if (ctx.stattype == "RESTRICTED") {
        return ctx.hot ? scan_flexible_hot_restricted_binomial(*ctx.handle->pre, observed, expected,
                                                               ctx.scan_settings, excluded,
                                                               center_order, gating_ptr)
                       : scan_flexible_cold_restricted_binomial(*ctx.handle->pre, observed,
                                                                expected, ctx.scan_settings,
                                                                excluded, center_order, gating_ptr);
      }
      if (ctx.stattype == "ORIGINAL") {
        return ctx.hot ? scan_flexible_hot_original_binomial(*ctx.handle->pre, observed, expected,
                                                             ctx.scan_settings, excluded,
                                                             center_order, gating_ptr)
                       : scan_flexible_cold_original_binomial(*ctx.handle->pre, observed, expected,
                                                              ctx.scan_settings, excluded,
                                                              center_order, gating_ptr);
      }
      throw std::runtime_error("Unknown stattype");
    }
    if (ctx.scanmethod == "CIRCULAR") {
      if (ctx.stattype == "RESTRICTED") {
        return scan_circular_binomial(*ctx.handle->pre, observed, expected, ctx.scan_settings,
                                      /*restricted=*/true, /*hot=*/ctx.hot, excluded, center_order,
                                      gating_ptr);
      }
      if (ctx.stattype == "ORIGINAL") {
        return scan_circular_binomial(*ctx.handle->pre, observed, expected, ctx.scan_settings,
                                      /*restricted=*/false, /*hot=*/ctx.hot, excluded, center_order,
                                      gating_ptr);
      }
      throw std::runtime_error("Unknown stattype");
    }
    throw std::runtime_error("Unknown scanmethod");
  }
  throw std::runtime_error("Unknown model");
}

std::vector<double> compute_scan_sim_max(ScanExecutionContext &ctx,
                                         const std::vector<char> *excluded_ptr,
                                         const ClusterSelection &observed_selection) {
  const auto &observed = *ctx.observed;
  const auto &expected = *ctx.expected;
  std::vector<double> sim_max_local;
  if (ctx.simcount <= 0)
    return sim_max_local;
  double nG = 0.0;
  double mG = 0.0;
  for (int value : observed)
    nG += value;
  for (double value : expected)
    mG += value;
  double p0 = (mG > 0.0) ? (nG / mG) : 0.0;
  if (!(p0 >= 0.0) || !(p0 <= 1.0) || !std::isfinite(p0))
    p0 = 0.0;
  sim_max_local.resize(std::max(0, ctx.simcount));
  // Wilson early stopping is evaluated at most every 25 replicates.  This deliberately
  // gives the documented/recommended simcount=199 a chance to stop before the
  // requested run completes, while keeping callback-free batches amortized.
  constexpr int kEarlyStopCadence = 25;
  int batch = std::min(ctx.mc_early_stop ? kEarlyStopCadence : 200, ctx.simcount);
  if (!ctx.allow_release && ctx.mc_progress_every > 0)
    batch = std::min(batch, ctx.mc_progress_every);
  int done = 0;
  auto ge_count_limited = [&](double threshold, const std::vector<double> &values, int limit) {
    int count = 0;
    int upper = std::max(0, std::min((int)values.size(), limit));
    for (int i = 0; i < upper; ++i) {
      if (values[i] >= threshold)
        ++count;
    }
    return count;
  };
  auto wilson_ci = [](double phat, int n) {
    const double z = 1.96;
    if (n <= 0)
      return std::pair<double, double>(0.0, 1.0);
    double denom = 1.0 + (z * z) / n;
    double center = (phat + (z * z) / (2.0 * n)) / denom;
    double half = (z / denom) * std::sqrt((phat * (1.0 - phat)) / n + (z * z) / (4.0 * n * n));
    double lo = std::max(0.0, center - half);
    double hi = std::min(1.0, center + half);
    return std::pair<double, double>(lo, hi);
  };
  bool debug_saved = ctx.scan_settings.debug_prune;
  int jobs_saved = ctx.scan_settings.n_jobs_centers;
  struct RestoreScanSettings {
    ScanSettings &settings;
    bool debug_prune;
    int n_jobs_centers;
    ~RestoreScanSettings() {
      settings.debug_prune = debug_prune;
      settings.n_jobs_centers = n_jobs_centers;
    }
  } restore_settings{ctx.scan_settings, debug_saved, jobs_saved};
  ctx.scan_settings.debug_prune = false;
  AliasTable alias_table;
  if (ctx.rantype == "MULTINOMIAL" && ctx.mc_alias)
    alias_table = build_alias_from_weights(expected);
  auto mc_loop_seq = [&](auto &&route_fn) {
    while (done < ctx.simcount) {
      int todo = std::min(batch, ctx.simcount - done);
      for (int s = 0; s < todo; ++s) {
        if (ctx.scan_settings.interrupt_poll)
          ctx.scan_settings.interrupt_poll();
        int rep_index = done + s;
        uint64_t seed_for_seq =
            ctx.crn ? (uint64_t)ctx.seed_base
                    : ((uint64_t)ctx.seed_base ^
                       (uint64_t)(0x94D049BB133111EBULL * (uint64_t)ctx.time_index));
        std::vector<int> sim_obs;
        if (ctx.rantype == "MULTINOMIAL") {
          if (ctx.mc_alias) {
            sim_obs = multinomial_counts_alias_crn(alias_table, (int)nG, seed_for_seq, rep_index);
          } else {
            sim_obs = multinomial_counts_crn(expected, (int)nG, seed_for_seq, rep_index);
          }
        } else if (ctx.rantype == "POISSON") {
          sim_obs = poisson_counts_crn(expected, seed_for_seq, rep_index);
        } else if (ctx.rantype == "BINOMIAL") {
          sim_obs = binomial_counts_crn(expected, p0, seed_for_seq, rep_index);
        } else {
          sim_obs = multinomial_counts_crn(expected, (int)nG, seed_for_seq, rep_index);
        }
        auto selection = route_fn(sim_obs, excluded_ptr);
        sim_max_local[rep_index] = selection.stats;
      }
      done += todo;
      if (!ctx.allow_release) {
        bool do_report = (ctx.mc_progress_every <= 0) || (done % ctx.mc_progress_every == 0) ||
                         (done >= ctx.simcount);
        if (do_report && (!ctx.mc_progress.is_none() || ctx.mc_progress_every > 0)) {
          int ge = ge_count_limited(observed_selection.stats, sim_max_local, done);
          double phat = double(ge + 1) / double(done + 1);
          auto ci = wilson_ci(phat, done + 1);
          if (!ctx.mc_progress.is_none()) {
            py::dict stat;
            stat["done"] = done;
            stat["simcount"] = ctx.simcount;
            stat["phat"] = phat;
            stat["ci"] = py::make_tuple(ci.first, ci.second);
            ctx.mc_progress(stat);
          } else if (ctx.mc_progress_every > 0) {
            // When mc_progress_every is set but mc_progress callback is None,
            // we emit a plain py::print() line.  This intentionally bypasses
            // Python's logging framework; users who need logging-level control
            // should provide an explicit mc_progress= callback instead.
            py::print("MC progress:", done, "/", ctx.simcount, "phat=", phat,
                      "CI=", py::make_tuple(ci.first, ci.second));
          }
        }
      }
      if (ctx.mc_early_stop) {
        int ge = ge_count_limited(observed_selection.stats, sim_max_local, done);
        double phat = double(ge + 1) / double(done + 1);
        auto ci = wilson_ci(phat, done + 1);
        if (ci.second < ctx.mc_alpha || ci.first > ctx.mc_alpha)
          break;
      }
    }
  };
  bool parallel_ok = ctx.allow_release && !ctx.mc_early_stop;
  if (parallel_ok) {
    py::gil_scoped_release nogil;
    ctx.scan_settings.n_jobs_centers = 1;
    int hardware = (int)std::thread::hardware_concurrency();
    if (hardware <= 0)
      hardware = 1;
    int workers = Scheduler::bounded_worker_count(jobs_saved, ctx.simcount, hardware);
    auto worker_fn = [&](int worker_index, const std::atomic<bool> &cancel) {
      auto route_fn = [&](const std::vector<int> &sim_obs, const std::vector<char> *ex_ptr) {
        return route_scan(ctx, sim_obs, ex_ptr, nullptr, nullptr);
      };
      for (int rep = worker_index; rep < ctx.simcount; rep += workers) {
        if (cancel.load(std::memory_order_acquire))
          return;
        if (worker_index == 0 && ctx.scan_settings.interrupt_poll)
          ctx.scan_settings.interrupt_poll();
        uint64_t seed_for_seq =
            ctx.crn ? (uint64_t)ctx.seed_base
                    : ((uint64_t)ctx.seed_base ^
                       (uint64_t)(0x94D049BB133111EBULL * (uint64_t)ctx.time_index));
        std::vector<int> sim_obs;
        if (ctx.rantype == "MULTINOMIAL") {
          if (ctx.mc_alias) {
            sim_obs = multinomial_counts_alias_crn(alias_table, (int)nG, seed_for_seq, rep);
          } else {
            sim_obs = multinomial_counts_crn(expected, (int)nG, seed_for_seq, rep);
          }
        } else if (ctx.rantype == "POISSON") {
          sim_obs = poisson_counts_crn(expected, seed_for_seq, rep);
        } else if (ctx.rantype == "BINOMIAL") {
          sim_obs = binomial_counts_crn(expected, p0, seed_for_seq, rep);
        } else {
          sim_obs = multinomial_counts_crn(expected, (int)nG, seed_for_seq, rep);
        }
        auto selection = route_fn(sim_obs, excluded_ptr);
        sim_max_local[rep] = selection.stats;
      }
    };
    Scheduler::for_each_worker(workers, worker_fn);
  } else {
    mc_loop_seq([&](const std::vector<int> &sim_obs, const std::vector<char> *ex_ptr) {
      return route_scan(ctx, sim_obs, ex_ptr, nullptr, nullptr);
    });
    if (ctx.mc_early_stop && done < ctx.simcount)
      sim_max_local.resize(std::max(0, done));
  }
  return sim_max_local;
}

} // namespace

py::dict execute_scan_slice(PreHandle *handle, const std::vector<int> &observed,
                            const std::vector<double> &expected, py::dict settings) {
  if (!handle || !handle->pre)
    throw std::runtime_error("invalid precompute handle");
  if ((int)observed.size() != handle->pre->N || (int)expected.size() != handle->pre->N) {
    throw std::runtime_error(
        "ScanSlice: observed and expected length must equal N=" + std::to_string(handle->pre->N) +
        " (got observed=" + std::to_string(observed.size()) +
        ", expected=" + std::to_string(expected.size()) + ")");
  }
  ScanExecutionContext ctx = prepare_scan_execution(handle, observed, expected, settings);
  InterruptPoller poller;
  ctx.scan_settings.interrupt_poll = [&poller]() { poller(); };
  ClusterSelection primary_selection;
  if (ctx.allow_release) {
    py::gil_scoped_release nogil;
    primary_selection =
        route_scan(ctx, observed, nullptr, ctx.center_order.empty() ? nullptr : &ctx.center_order,
                   ctx.have_pre_gating ? &ctx.pre_gating : nullptr);
  } else {
    primary_selection =
        route_scan(ctx, observed, nullptr, ctx.center_order.empty() ? nullptr : &ctx.center_order,
                   ctx.have_pre_gating ? &ctx.pre_gating : nullptr);
  }
  std::vector<double> primary_sim_max = compute_scan_sim_max(ctx, nullptr, primary_selection);
  py::dict primary = assemble_scan_result(ctx, primary_selection, primary_sim_max);
  // MC-3: only the PRIMARY's full-region null-max array feeds the BOTH combined
  // two-sided null; the per-secondary copies were emitted but always stripped in
  // _build_both_result. Gate emission off for the secondaries so we don't copy the
  // (potentially large) array across the pybind boundary once per secondary. The
  // secondaries are still ranked against primary_sim_max below -- only the unused
  // attachment is suppressed.
  ctx.emit_sim_max = false;
  int secondary = settings.contains("secondary") ? settings[py::str("secondary")].cast<int>() : 0;
  if (secondary < 0)
    throw py::value_error("secondary must be >= 0");
  double secondary_p_stop = settings.contains("secondary_p_stop")
                                ? settings[py::str("secondary_p_stop")].cast<double>()
                                : 1.0;
  if (!std::isfinite(secondary_p_stop) || secondary_p_stop < 0.0 || secondary_p_stop > 1.0)
    throw py::value_error("secondary_p_stop must be in [0,1]");
  double secondary_stats_stop = settings.contains("secondary_stats_stop")
                                    ? settings[py::str("secondary_stats_stop")].cast<double>()
                                    : -std::numeric_limits<double>::infinity();
  if (settings.contains("secondary_stats_stop") &&
      (!std::isfinite(secondary_stats_stop) || secondary_stats_stop < 0.0))
    throw py::value_error("secondary_stats_stop must be finite and non-negative");
  std::vector<char> excluded(ctx.handle->pre->N, 0);
  for (int value : primary_selection.area)
    excluded[value] = 1;
  py::list secondaries;
  for (int i = 0; i < secondary; ++i) {
    // Secondary WINDOW SEARCH always excludes already-detected areas (non-overlap
    // per DEF-STOP) -- this is correct regardless of secondary_null.
    auto selection = route_scan(ctx, observed, &excluded, nullptr,
                                ctx.have_pre_gating ? &ctx.pre_gating : nullptr);
    if (selection.area.empty())
      break;
    if (std::isfinite(secondary_stats_stop) && !(selection.stats > secondary_stats_stop))
      break;
    // Only the NULL DISTRIBUTION the secondary is ranked against depends on the
    // switch. "FULL" (paper-faithful, DEF-STOP) ranks each secondary against the
    // same unconditional full-region null already computed for the primary, so
    // its p-value is "as if it were the primary cluster". "CONDITIONAL" (prior
    // behaviour) recomputes the null over only the non-overlapping windows, whose
    // maxima are <= the full-region maxima -> anti-conservative secondary p-values.
    py::dict assembled;
    if (ctx.secondary_null == "CONDITIONAL") {
      std::vector<double> sim_max_secondary = compute_scan_sim_max(ctx, &excluded, selection);
      assembled = assemble_scan_result(ctx, selection, sim_max_secondary);
    } else {
      assembled = assemble_scan_result(ctx, selection, primary_sim_max);
    }
    if (std::isfinite(secondary_p_stop)) {
      try {
        if (assembled.contains("pval") && !assembled[py::str("pval")].is_none()) {
          double pval = assembled[py::str("pval")].cast<double>();
          if (!(pval <= secondary_p_stop))
            break;
        }
      } catch (...) {
      }
    }
    secondaries.append(assembled);
    for (int value : selection.area)
      excluded[value] = 1;
  }
  py::dict out = primary;
  out["secondaries"] = secondaries;
  return out;
}

} // namespace flexscanx_pybind
