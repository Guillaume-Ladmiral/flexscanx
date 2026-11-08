#pragma once
#include "flexscanx_core/algorithms/debug_stats.hpp"
#include "flexscanx_core/algorithms/gating.hpp"
#include "flexscanx_core/algorithms/scheduler.hpp"
#include "flexscanx_core/algorithms/universe.hpp"
#include "flexscanx_core/policies/model_binomial.hpp"
#include "flexscanx_core/policies/model_poisson.hpp"
#include "flexscanx_core/policies/stat_type.hpp"
#include "flexscanx_core/policies/tail.hpp"
#include "flexscanx_core/scanner_flexible_common.hpp" // brings ScanSettings, ClusterSelection, GatingCache
#include "flexscanx_core/static_precompute.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

namespace flexscanx_core {

// Templated circular scanner: iterates KNN prefixes around center c
// - Traits: PoissonTraits or BinomialTraits
// - Tail: HotTag or ColdTag
// - StatTag: RestrictedTag or OriginalTag
// observed: counts; denom: expected (Poisson) or population (Binomial)

template <class Traits, class Tail, class StatTag>
inline ClusterSelection
scan_circular_core(const Precompute &pre, const std::vector<int> &observed,
                   const std::vector<typename Traits::DenomT> &denom, const ScanSettings &st,
                   const std::vector<char> *excluded, const std::vector<int> *center_order,
                   const GatingCache *gating) {
  const int N = pre.N;
  double nG = 0.0, mG = 0.0;
  for (int i = 0; i < N; ++i) {
    nG += observed[i];
    mG += denom[i];
  }
  // Precompute baseline once — avoids recomputing nG*log(nG/mG) at every KNN prefix.
  const double baseline = Traits::compute_baseline(nG, mG);

  // Build gating vector only for Restricted
  std::vector<double> pv_scratch;
  const std::vector<double> *pv_ptr = nullptr;
  if constexpr (std::is_same_v<StatTag, RestrictedTag>) {
    if constexpr (std::is_same_v<Traits, BinomialTraits>) {
      typename BinomialTraits::PvContext ctx{(mG > 0.0) ? (nG / mG) : 0.0};
      pv_ptr = &GatingProvider<Traits, Tail>::get(N, observed, denom, nG, mG, st, gating,
                                                  pv_scratch, &ctx);
    } else {
      pv_ptr =
          &GatingProvider<Traits, Tail>::get(N, observed, denom, nG, mG, st, gating, pv_scratch);
    }
  }

  ClusterSelection best;
  best.stats = 0.0;
  best.nZ = 0.0;
  best.mZ = 0.0;
  best.area.clear();
  DebugStats dbg;
  std::mutex best_mtx;
  // Atomic shadow of best.stats for lock-free speculative reads across threads.
  // All writes to best.stats must also update best_stats_atomic (inside best_mtx).
  std::atomic<double> best_stats_atomic{0.0};

  auto try_center = [&](int c) {
    if (excluded && (*excluded)[c])
      return;
    if constexpr (std::is_same_v<StatTag, RestrictedTag>) {
      if (!((*pv_ptr)[c] < st.ralpha)) {
        dbg.centers_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    dbg.centers_scanned.fetch_add(1, std::memory_order_relaxed);

    double nZ = 0.0, mZ = 0.0;
    // Build ordered neighbor sequence, optionally refilling excluded entries
    static thread_local std::vector<int> order;
    static thread_local std::vector<double> order_dist2;
    order.clear();
    order_dist2.clear();
    order.reserve(pre.K);
    order_dist2.reserve(pre.K);
    build_knn_order(pre, c, excluded, order, order_dist2);

    for (int k = 0; k < (int)order.size(); ++k) {
      int idx = order[k];
      if constexpr (std::is_same_v<StatTag, RestrictedTag>) {
        if (!((*pv_ptr)[idx] < st.ralpha))
          // By design (following R flexscan convention): stop expanding this center the
          // moment the KNN-distance-sorted sequence hits a node that fails the restricted
          // gate. This is intentional — the circular scan uses strict distance ordering,
          // so any farther node is guaranteed to include this failing node in its cluster.
          // Change to `continue` only if you want a "skip-and-continue" strategy, which
          // would change statistical behavior and diverge from the R reference.
          break;
      }
      nZ += observed[idx];
      mZ += denom[idx];
      dbg.dfs_nodes_expanded.fetch_add(1, std::memory_order_relaxed);
      double stats = Traits::template llr_with_baseline<Tail>(nZ, mZ, nG, mG, baseline);
      // Speculative read admits exact positive ties so the deterministic
      // tie-break (audit CC-2) can decide a thread-order-independent winner.
      double bs_seen = best_stats_atomic.load(std::memory_order_relaxed);
      if (stats > bs_seen || (stats == bs_seen && stats > 0.0)) {
        std::lock_guard<std::mutex> lk(best_mtx);
        bool replace = stats > best.stats;
        if (!replace && stats == best.stats && stats > 0.0) {
          // Size-first: only materialise the candidate prefix when sizes match
          // (tie_break_prefers prefers the smaller cluster outright).
          size_t cand_size = (size_t)(k + 1);
          if (cand_size != best.area.size()) {
            replace = cand_size < best.area.size();
          } else {
            std::vector<int> cand(order.begin(), order.begin() + k + 1);
            replace = tie_break_prefers(cand, best.area);
          }
        }
        if (replace) {
          best.stats = stats;
          best_stats_atomic.store(stats, std::memory_order_relaxed);
          best.nZ = nZ;
          best.mZ = mZ;
          best.area.clear();
          for (int j = 0; j <= k; ++j)
            best.area.push_back(order[j]);
          dbg.best_updates.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if ((int)(k + 1) >= st.clustersize)
        break;
    }
  };

  // Build centers list from provided order or default 0..N-1
  std::vector<int> centers;
  if (center_order) {
    centers.reserve(center_order->size());
    for (int c : *center_order)
      if (c >= 0 && c < N)
        centers.push_back(c);
  } else {
    centers.resize(N);
    for (int c = 0; c < N; ++c)
      centers[c] = c;
  }
  if (st.scheduler_backend == ScanSettings::SchedulerBackend::ATOMIC) {
    Scheduler::for_each_center_atomic(centers, std::max(1, st.n_jobs_centers), try_center,
                                      st.interrupt_poll);
  } else {
    Scheduler::for_each_center_worksteal(centers, std::max(1, st.n_jobs_centers), try_center,
                                         st.interrupt_poll);
  }

  if (st.debug_prune) {
    print_stats_circular<Traits, Tail, StatTag>(dbg);
  }
  return best;
}

} // namespace flexscanx_core
