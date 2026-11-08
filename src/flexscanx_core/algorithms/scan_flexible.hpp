#pragma once
#include "flexscanx_core/algorithms/bounds.hpp"
#include "flexscanx_core/algorithms/centers.hpp"
#include "flexscanx_core/algorithms/debug_stats.hpp"
#include "flexscanx_core/algorithms/frontier.hpp"
#include "flexscanx_core/algorithms/gating.hpp"
#include "flexscanx_core/algorithms/scheduler.hpp"
#include "flexscanx_core/algorithms/universe.hpp"
#include "flexscanx_core/policies/model_binomial.hpp"
#include "flexscanx_core/policies/model_poisson.hpp"
#include "flexscanx_core/policies/stat_type.hpp"
#include "flexscanx_core/policies/tail.hpp"
#include "flexscanx_core/scanner_flexible_common.hpp" // ScanSettings, ClusterSelection, GatingCache
#include "flexscanx_core/static_precompute.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

namespace flexscanx_core {

namespace detail {

using FlexibleFrontier = std::vector<std::pair<int, double>>;

// One pool per calling thread, shared by every flexible-scan specialization.
// Every DFS frame, including the root frame, acquires a vector here and returns
// it on pop. This balanced ownership prevents process-lifetime growth per
// scanned centre while retaining allocations after warm-up.
inline thread_local std::vector<FlexibleFrontier> tl_flexible_frontier_pool;

inline std::vector<FlexibleFrontier> &flexible_frontier_pool() noexcept {
  return tl_flexible_frontier_pool;
}

} // namespace detail

// Flexible DFS-based scanner core (templated by model Traits, Tail, and StatTag)
// observed: counts; denom: expected (Poisson) or population (Binomial)

template <class Traits, class Tail, class StatTag>
inline ClusterSelection
scan_flexible_core(const Precompute &pre, const std::vector<int> &observed,
                   const std::vector<typename Traits::DenomT> &denom, const ScanSettings &st,
                   const std::vector<char> *excluded, const std::vector<int> *center_order,
                   const GatingCache *gating) {
  const int N = pre.N;
  double nG = 0.0, mG = 0.0;
  for (int i = 0; i < N; ++i) {
    nG += observed[i];
    mG += denom[i];
  }
  // Precompute once: the baseline term (nG*log(nG/mG) for Poisson; analogous for Binomial)
  // is constant for the entire scan and would otherwise be recomputed at every DFS node.
  const double baseline = Traits::compute_baseline(nG, mG);

  // Gating vector (Restricted only)
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

  // Center ordering: default by (pv asc if restricted, then single-node LLR desc)
  std::vector<double> center_llr1(N, 0.0);
  for (int i = 0; i < N; ++i)
    center_llr1[i] =
        Traits::template llr_with_baseline<Tail>(observed[i], denom[i], nG, mG, baseline);

  std::vector<int> centers;
  if constexpr (std::is_same_v<StatTag, RestrictedTag>) {
    if (center_order && !center_order->empty()) {
      centers = build_centers(N, center_order);
    } else {
      centers = build_centers(N, nullptr, [&](int c) {
        double pv = (pv_ptr && c < (int)pv_ptr->size()) ? (*pv_ptr)[c] : 1.0;
        return std::tuple<double, double>(pv, -center_llr1[c]);
      });
    }
  } else {
    centers = build_centers(N, center_order);
  }

  std::mutex best_mtx;
  ClusterSelection best;
  best.stats = 0.0;
  best.nZ = 0.0;
  best.mZ = 0.0;
  best.area.clear();
  // Atomic shadow of best.stats for lock-free speculative reads across threads.
  // All writes to best.stats must also update best_stats_atomic (inside best_mtx).
  std::atomic<double> best_stats_atomic{0.0};
  DebugStats dbg;

  auto try_center = [&](int c) {
    if (excluded && (*excluded)[c])
      return;
    if constexpr (std::is_same_v<StatTag, RestrictedTag>) {
      double pv = 1.0;
      if (pv_ptr && c < (int)pv_ptr->size()) {
        pv = (*pv_ptr)[c];
      } else {
        if constexpr (std::is_same_v<Traits, PoissonTraits>) {
          if constexpr (std::is_same_v<Tail, HotTag>)
            pv = PoissonTraits::pv_upper(observed[c], denom[c]);
          else
            pv = PoissonTraits::pv_lower(observed[c], denom[c]);
        } else {
          typename BinomialTraits::PvContext bctx{(mG > 0.0) ? (nG / mG) : 0.0};
          if constexpr (std::is_same_v<Tail, HotTag>)
            pv = BinomialTraits::pv_upper(observed[c], (int)std::llround(denom[c]), bctx);
          else
            pv = BinomialTraits::pv_lower(observed[c], (int)std::llround(denom[c]), bctx);
        }
      }
      if (!(pv < st.ralpha)) {
        dbg.centers_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
    dbg.centers_scanned.fetch_add(1, std::memory_order_relaxed);

    // Universe set for this center: thread-local to avoid per-center heap allocation.
    // build_universe handles resize + zero-fill internally.
    static thread_local std::vector<char> in_universe;
    build_universe(pre, c, excluded, in_universe);

    // Used mask: thread-local to avoid per-center heap allocation.
    static thread_local std::vector<char> used;
    if ((int)used.size() < N)
      used.assign(N, 0);
    else
      std::fill(used.begin(), used.end(), 0);
    std::vector<int> cluster;
    cluster.reserve(st.clustersize);

    // Initial frontier from center's neighbors
    used[c] = 1;
    cluster.push_back(c);
    static thread_local FrontierMarker fmark;
    fmark.ensure(N);
    std::vector<std::pair<int, double>> frontier;
    auto &frontier_pool = detail::flexible_frontier_pool();
    if (!frontier_pool.empty()) {
      frontier = std::move(frontier_pool.back());
      frontier_pool.pop_back();
      frontier.clear();
    } else {
      frontier.reserve(pre.K);
    }
    int rs0 = pre.csr.row_ptr[c], re0 = pre.csr.row_ptr[c + 1];
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    for (int p = rs0; p < re0; ++p) {
      int nb = pre.csr.col_idx[p];
      if (!in_universe[nb] || used[nb])
        continue;
      if (!fmark.is_marked(nb)) {
        frontier.emplace_back(nb, NEG_INF);
        fmark.mark(nb);
      }
    }

    // Evaluate single-node cluster
    {
      double nZ = observed[c], mZ = denom[c];
      double stats = Traits::template llr_with_baseline<Tail>(nZ, mZ, nG, mG, baseline);
      std::lock_guard<std::mutex> lk(best_mtx);
      if (stats > best.stats ||
          (stats == best.stats && stats > 0.0 && tie_break_prefers(cluster, best.area))) {
        best.stats = stats;
        best_stats_atomic.store(stats, std::memory_order_relaxed);
        best.nZ = nZ;
        best.mZ = mZ;
        best.area = cluster;
      }
    }

    // Center-level optimistic bound gating using full KNN-universe candidates (ignoring
    // connectivity). Use all nodes in the center's KNN universe within radius (excluding the center
    // and excluded/used).
    if (st.enable_center_ub_gating && st.clustersize > 1) {
      int remaining_slots = st.clustersize - (int)cluster.size();
      std::vector<int> cand;
      const std::vector<int> &knn_source = (pre.refill_enabled && !pre.knn_idx_refill.empty())
                                               ? pre.knn_idx_refill[c]
                                               : pre.knn_idx[c];
      cand.reserve(knn_source.size());
      // Keep center-level UB aligned with the effective KNN universe, including refill entries
      // admitted after exclusions. Filtering through in_universe preserves the exact K-limited set.
      for (int idx : knn_source) {
        if (in_universe[idx] && !used[idx] && idx != c)
          cand.push_back(idx);
      }
      if (!cand.empty()) {
        double ub0 =
            greedy_upper_bound<Traits, Tail>(cand, 0, remaining_slots, (double)observed[c],
                                             (double)denom[c], nG, mG, observed, denom, baseline);
        double bs = best_stats_atomic.load(std::memory_order_relaxed);
        // Strict <: a center whose bound exactly ties the incumbent may still
        // hold the tie-break-preferred cluster and must not be pruned.
        if (ub0 < bs) {
          dbg.center_ub_prunes.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
    }

    struct Frame {
      int next_i;
      std::vector<std::pair<int, double>> frontier_snapshot;
      double nZ, mZ;
    };
    std::vector<Frame> stack;
    stack.push_back(Frame{0, std::move(frontier), (double)observed[c], (double)denom[c]});

    while (!stack.empty()) {
      auto &fr = stack.back();
      if ((int)cluster.size() >= st.clustersize || fr.next_i >= (int)fr.frontier_snapshot.size()) {
        // Every frame acquired its frontier from the same pool, including the
        // root frame, so returning it here preserves balanced ownership.
        frontier_pool.push_back(std::move(fr.frontier_snapshot));
        if ((int)cluster.size() <= 1) {
          stack.pop_back();
          break;
        }
        int removed = cluster.back();
        cluster.pop_back();
        used[removed] = 0;
        stack.pop_back();
        continue;
      }
      // Per-frame optimistic bound (optional)
      if (st.enable_frame_ub_prune && !fr.frontier_snapshot.empty()) {
        int remaining_slots = st.clustersize - (int)cluster.size();
        std::vector<int> cand2;
        cand2.reserve(fr.frontier_snapshot.size());
        for (auto &it : fr.frontier_snapshot)
          cand2.push_back(it.first);
        double ub = greedy_upper_bound<Traits, Tail>(cand2, fr.next_i, remaining_slots, fr.nZ,
                                                     fr.mZ, nG, mG, observed, denom, baseline);
        double bs = best_stats_atomic.load(std::memory_order_relaxed);
        // Strict <: keep subtrees whose bound ties the incumbent so the
        // deterministic tie-break can evaluate their candidates.
        if (ub < bs) {
          dbg.bound_prunes.fetch_add(1, std::memory_order_relaxed);
          fr.next_i = (int)fr.frontier_snapshot.size();
          continue;
        }
      }
      // Best next candidate (with cache)
      if (fr.next_i < (int)fr.frontier_snapshot.size()) {
        int best_pos;
        if (st.cache_candidate_llr) {
          best_pos = select_best_candidate<Traits, Tail, true>(
              fr.frontier_snapshot, fr.next_i, fr.nZ, fr.mZ, nG, mG, observed, denom, baseline);
        } else {
          best_pos = select_best_candidate<Traits, Tail, false>(
              fr.frontier_snapshot, fr.next_i, fr.nZ, fr.mZ, nG, mG, observed, denom, baseline);
        }
        std::swap(fr.frontier_snapshot[fr.next_i], fr.frontier_snapshot[best_pos]);
      }
      int r = fr.frontier_snapshot[fr.next_i++].first;
      if (used[r])
        continue;
      if (!in_universe[r])
        continue;
      if constexpr (std::is_same_v<StatTag, RestrictedTag>) {
        double pv = 1.0;
        if (pv_ptr && r < (int)pv_ptr->size()) {
          pv = (*pv_ptr)[r];
        } else {
          if constexpr (std::is_same_v<Traits, PoissonTraits>) {
            if constexpr (std::is_same_v<Tail, HotTag>)
              pv = PoissonTraits::pv_upper(observed[r], denom[r]);
            else
              pv = PoissonTraits::pv_lower(observed[r], denom[r]);
          } else {
            typename BinomialTraits::PvContext bctx{(mG > 0.0) ? (nG / mG) : 0.0};
            if constexpr (std::is_same_v<Tail, HotTag>)
              pv = BinomialTraits::pv_upper(observed[r], (int)std::llround(denom[r]), bctx);
            else
              pv = BinomialTraits::pv_lower(observed[r], (int)std::llround(denom[r]), bctx);
          }
        }
        if (!(pv < st.ralpha))
          continue; // node gating (skip)
      }

      // Choose r: update cluster and frontier
      used[r] = 1;
      cluster.push_back(r);
      dbg.dfs_nodes_expanded.fetch_add(1, std::memory_order_relaxed);

      // Build child frontier in one pass, marking nf for dedup. Reuse a pooled
      // vector to avoid per-node heap allocation. Cache values are reset to
      // NEG_INF — the parent's cached LLRs were computed at the old (nZ, mZ)
      // state and are stale after adding node r.
      //
      // Enumeration semantics (audit ENUM-2 / "F-7b"): the child inherits only
      // the parent's NOT-YET-TRIED candidates (positions >= fr.next_i; r sits at
      // fr.next_i - 1 after the increment). This forbids an earlier-tried sibling
      // from the *immediate* child, de-duplicating acyclic (chain) insertion
      // orders — the factorial blow-up that made unpruned scans intractable. It
      // is NOT a persistent per-frame forbid: used[] is cleared on backtrack
      // (see :220), so an earlier sibling becomes addable again and re-enters a
      // *later* node's frontier as a fresh neighbour. A connected subset whose
      // induced subgraph contains a cycle is therefore regenerated once per
      // cyclic insertion order (re-visit factor up to ~107x on K8; far smaller on
      // sparse planar adjacency). COMPLETENESS holds — every connected subset is
      // generated >= once (verified == brute-force connected-subset enumeration)
      // — but visit-ONCE does NOT. The duplicates are inference-neutral: the scan
      // keeps only the max LLR under the strict total-order tie-break
      // (tie_break_prefers), and re-evaluating an identical window is idempotent,
      // so the MLC and every per-replicate null max — hence all p-values — are
      // unchanged (verified: MLC/stats/p == brute-force max, Poisson & Binomial).
      // A true per-frame "finished" forbid (à la rflexscan's masksw) that would
      // restore visit-once is a deferred perf optimisation, not a correctness fix.
      std::vector<std::pair<int, double>> new_frontier;
      if (!frontier_pool.empty()) {
        new_frontier = std::move(frontier_pool.back());
        frontier_pool.pop_back();
        new_frontier.clear(); // O(1): retains heap capacity for reuse
      } else {
        new_frontier.reserve(fr.frontier_snapshot.size());
      }
      static thread_local FrontierMarker nf;
      nf.ensure(N);
      for (int q = fr.next_i; q < (int)fr.frontier_snapshot.size(); ++q) {
        int cand_node = fr.frontier_snapshot[q].first;
        new_frontier.emplace_back(cand_node, NEG_INF); // reset cache: stale at new (nZ,mZ)
        if (cand_node >= 0 && cand_node < N)
          nf.mark(cand_node);
      }
      int rs2 = pre.csr.row_ptr[r], re2 = pre.csr.row_ptr[r + 1];
      for (int p = rs2; p < re2; ++p) {
        int nb = pre.csr.col_idx[p];
        if (!in_universe[nb] || used[nb])
          continue;
        if (!nf.is_marked(nb)) {
          new_frontier.emplace_back(nb, NEG_INF);
          nf.mark(nb);
        }
      }

      double nZp = fr.nZ + observed[r], mZp = fr.mZ + denom[r];
      double stats = Traits::template llr_with_baseline<Tail>(nZp, mZp, nG, mG, baseline);
      // Speculative read admits exact ties (stats == seen, positive only) so the
      // deterministic tie-break below can run; zero-stat evaluations never tie.
      double bs_seen = best_stats_atomic.load(std::memory_order_relaxed);
      if (stats > bs_seen || (stats == bs_seen && stats > 0.0)) {
        std::lock_guard<std::mutex> lk(best_mtx);
        if (stats > best.stats ||
            (stats == best.stats && stats > 0.0 && tie_break_prefers(cluster, best.area))) {
          best.stats = stats;
          best_stats_atomic.store(stats, std::memory_order_relaxed);
          best.nZ = nZp;
          best.mZ = mZp;
          best.area = cluster;
          dbg.best_updates.fetch_add(1, std::memory_order_relaxed);
        }
      }
      stack.push_back(Frame{0, std::move(new_frontier), nZp, mZp});
    }
  };

  if (st.scheduler_backend == ScanSettings::SchedulerBackend::ATOMIC) {
    Scheduler::for_each_center_atomic(centers, std::max(1, st.n_jobs_centers), try_center,
                                      st.interrupt_poll);
  } else {
    Scheduler::for_each_center_worksteal(centers, std::max(1, st.n_jobs_centers), try_center,
                                         st.interrupt_poll);
  }
  if (st.debug_prune) {
    print_stats_flexible<Traits, Tail, StatTag>(dbg);
  }
  return best;
}

} // namespace flexscanx_core
