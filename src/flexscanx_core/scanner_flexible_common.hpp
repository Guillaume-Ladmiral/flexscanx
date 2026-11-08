#pragma once
#include <algorithm>
#include <functional>
#include <vector>

namespace flexscanx_core {

struct ScanSettings {
  double ralpha{0.2};
  int clustersize{15};
  int n_jobs_centers{1};
  bool debug_prune{false};
  // Heuristics and scheduling.
  // Center-level UB gating default OFF (audit 2026-06-11, review of CC-1/CC-2):
  // it shares greedy_upper_bound's unproven early-stop with the frame prune, so
  // it may prune a center holding the true maximum-likelihood cluster. Opt-in
  // only; results may be suboptimal when enabled.
  bool enable_center_ub_gating{false};
  bool cache_candidate_llr{true};
  // Per-frame optimistic bound pruning inside flexible DFS.
  // Default OFF (audit 2026-06-11, finding CC-1): greedy_upper_bound is not a
  // valid upper bound (early stop + frontier-only candidates), so enabling it
  // can discard the true maximum-likelihood cluster. Opt-in only, results may
  // be suboptimal when enabled.
  bool enable_frame_ub_prune{false};
  enum class SchedulerBackend { ATOMIC = 0, WORK_STEAL = 1 };
  SchedulerBackend scheduler_backend{SchedulerBackend::ATOMIC};
  // Optional cooperative interrupt hook (FXA-188 / D6).  The schedulers call it
  // on the calling thread before each center; the binding installs a poller
  // that, on the thread that owns the Python call and at most every 50 ms,
  // reacquires the GIL and runs PyErr_CheckSignals(), throwing when a signal
  // is pending.  The throw unwinds through the schedulers' existing
  // record/cancel/join/rethrow path.  Empty by default: the native core stays
  // usable without Python.
  std::function<void()> interrupt_poll;
};

struct ClusterSelection {
  std::vector<int> area; // 0-based indices
  double nZ{0.0};
  double mZ{0.0};
  double stats{0.0};
};

// Deterministic tie-break for equal-LLR cluster selections (audit CC-2).
// Returns true when `cand_area` should replace `cur_area` given equal stats:
// prefer the smaller cluster; for equal sizes, the lexicographically smaller
// sorted area list. This is a strict total order on candidate clusters, so the
// selected winner is independent of thread arrival order (parallel scans match
// the sequential result bit-for-bit). Ties are rare; the sort cost is paid
// only on exact stats equality.
inline bool tie_break_prefers(const std::vector<int> &cand_area, const std::vector<int> &cur_area) {
  if (cand_area.size() != cur_area.size())
    return cand_area.size() < cur_area.size();
  std::vector<int> a(cand_area);
  std::vector<int> b(cur_area);
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a < b;
}

// Optional per-call gating cache to reuse p-values across scans
struct GatingCache {
  std::vector<double> pv_hot;  // upper-tail middle p per area
  std::vector<double> pv_cold; // lower-tail middle p per area
  // Note: Presence is indicated by non-empty vectors (!pv_hot.empty(), !pv_cold.empty())
};

} // namespace flexscanx_core
