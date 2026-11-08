#pragma once

#include <string>
#include <vector>

#include "bindings_support.hpp"
#include "scanner_flexible_common.hpp"

namespace flexscanx_pybind {

struct ScanExecutionContext {
  PreHandle *handle = nullptr;
  const std::vector<int> *observed = nullptr;
  const std::vector<double> *expected = nullptr;

  flexscanx_core::ScanSettings scan_settings;

  std::string model = "POISSON";
  std::string scanmethod = "FLEXIBLE";
  std::string stattype = "RESTRICTED";
  std::string clustertype = "HOT";
  bool hot = true;

  std::vector<int> center_order;
  flexscanx_core::GatingCache pre_gating;
  bool have_pre_gating = false;

  int mc_progress_every = 0;
  py::object mc_progress;
  bool allow_release = true;

  int simcount = 0;
  std::string rantype = "MULTINOMIAL";
  bool crn = false;
  long long seed_base = 0xDEADBEEF;
  int time_index = 0;
  bool mc_early_stop = false;
  double mc_alpha = 0.05;
  bool mc_alias = false;

  // Null distribution used to rank SECONDARY clusters (MC-6 / DEF-STOP).
  //   "FULL"        - reuse the unconditional full-region primary null (paper-
  //                   faithful default; secondaries ranked "as if primary").
  //   "CONDITIONAL" - recompute the null over windows that avoid detected areas
  //                   (prior, anti-conservative behaviour; opt-in back-compat).
  // The secondary WINDOW SEARCH always excludes detected areas regardless of this
  // setting; only the null DISTRIBUTION it is ranked against changes.
  std::string secondary_null = "FULL";

  // MC-3 (combined two-sided BOTH null). When true, assemble_scan_result also
  // attaches the per-direction full-region null-max array as out["sim_max"] so
  // Python can form the paper-correct two-sided null max(hot_sim_max, cold_sim_max)
  // for clustertype="BOTH". Set only on the two one-sided sub-scans of the BOTH
  // envelope (settings["_emit_sim_max"]); single-direction HOT/COLD scans leave it
  // false, so their result dicts stay byte-identical.
  bool emit_sim_max = false;

  bool use_haldane = false;
  bool use_eb = false;
  bool eb_ok = false;
  double eb_a = 0.0;
  double eb_b = 0.0;
  double eb_alpha = 0.0;
  double eb_beta = 0.0;
};

py::dict assemble_scan_result(const ScanExecutionContext &ctx,
                              const flexscanx_core::ClusterSelection &selection,
                              const std::vector<double> &simdist);

py::dict execute_scan_slice(PreHandle *handle, const std::vector<int> &observed,
                            const std::vector<double> &expected, py::dict settings);

} // namespace flexscanx_pybind
