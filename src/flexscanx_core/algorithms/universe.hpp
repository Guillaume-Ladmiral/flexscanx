#pragma once
#include "flexscanx_core/static_precompute.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace flexscanx_core {

inline long double coordinate_distance(const Precompute &pre, int c, int idx) {
  const long double dx = static_cast<long double>(pre.x[c]) - static_cast<long double>(pre.x[idx]);
  const long double dy = static_cast<long double>(pre.y[c]) - static_cast<long double>(pre.y[idx]);
  return std::hypot(dx, dy);
}

// Build a per-center KNN "universe" bitset that respects radius and excluded mask.
// Refills up to K if pre.refill_enabled, otherwise uses top-K knn_idx.
inline void build_universe(const Precompute &pre, int c, const std::vector<char> *excluded,
                           std::vector<char> &in_universe) {
  const int N = pre.N;
  if ((int)in_universe.size() < N)
    in_universe.assign(N, 0);
  std::fill(in_universe.begin(), in_universe.end(), 0);

  auto within_radius = [&](int idx) {
    if (pre.radius < std::numeric_limits<double>::infinity()) {
      return coordinate_distance(pre, c, idx) <= static_cast<long double>(pre.radius);
    }
    return true;
  };

  int added = 0;
  if (pre.refill_enabled && !pre.knn_idx_refill.empty()) {
    for (int idx : pre.knn_idx_refill[c]) {
      if (!within_radius(idx))
        continue;
      if (excluded && (*excluded)[idx])
        continue;
      if (!in_universe[idx]) {
        in_universe[idx] = 1;
        if (++added >= pre.K)
          break;
      }
    }
  }
  if (added < pre.K) {
    const bool finite_radius = (pre.radius < std::numeric_limits<double>::infinity());
    for (int k = 0; k < (int)pre.knn_idx[c].size(); ++k) {
      int idx = pre.knn_idx[c][k];
      // When a finite radius is active, recompute distance in extended precision to match
      // the refill path and avoid cached-float boundary discrepancies at the radius edge.
      if (finite_radius) {
        const long double distance_value = coordinate_distance(pre, c, idx);
        if (distance_value > static_cast<long double>(pre.radius))
          break; // list is distance-sorted; no farther entry can be within radius
      }
      // No radius check needed when radius == inf (all entries are within range).
      if (excluded && (*excluded)[idx])
        continue;
      if (!in_universe[idx]) {
        in_universe[idx] = 1;
        if (++added >= pre.K)
          break;
      }
    }
  }
}

// Build an ordered KNN list for circular scanning. Refills first (if enabled), then top-K list.
// Stops on radius overflow (lists are distance-sorted). Skips excluded indices. Avoids duplicates.
// O(K) duplicate detection via thread-local bool vector (vs. O(K²) linear scan).
inline void build_knn_order(const Precompute &pre, int c, const std::vector<char> *excluded,
                            std::vector<int> &order, std::vector<double> &order_dist2) {
  order.clear();
  order_dist2.clear();
  order.reserve(pre.K);
  order_dist2.reserve(pre.K);

  // Thread-local membership bitvector; resized lazily, reset only for entries we wrote.
  static thread_local std::vector<bool> in_order_tl;
  if ((int)in_order_tl.size() < pre.N)
    in_order_tl.assign(pre.N, false);

  if (pre.refill_enabled && !pre.knn_idx_refill.empty()) {
    for (int idx : pre.knn_idx_refill[c]) {
      const long double distance_value = coordinate_distance(pre, c, idx);
      if (pre.radius < std::numeric_limits<double>::infinity() &&
          distance_value > static_cast<long double>(pre.radius))
        break;
      if (excluded && (*excluded)[idx])
        continue;
      if (in_order_tl[idx])
        continue;
      in_order_tl[idx] = true;
      order.push_back(idx);
      order_dist2.push_back(static_cast<double>(distance_value * distance_value));
      if ((int)order.size() >= pre.K)
        break;
    }
  }
  if ((int)order.size() < pre.K) {
    const bool finite_radius_ord = (pre.radius < std::numeric_limits<double>::infinity());
    for (int k = 0; k < (int)pre.knn_idx[c].size() && (int)order.size() < pre.K; ++k) {
      int idx = pre.knn_idx[c][k];
      // Compute squared distance: extended precision when radius filtering is active
      // (matches refill path); float cast acceptable otherwise (no radius check performed).
      double d2;
      if (finite_radius_ord) {
        const long double distance_value = coordinate_distance(pre, c, idx);
        if (distance_value > static_cast<long double>(pre.radius))
          break; // list is distance-sorted; no farther entry can be within radius
        d2 = static_cast<double>(distance_value * distance_value);
      } else {
        d2 = (double)pre.knn_dist[c][k];
      }
      if (excluded && (*excluded)[idx])
        continue;
      if (in_order_tl[idx])
        continue;
      in_order_tl[idx] = true;
      order.push_back(idx);
      order_dist2.push_back(d2);
    }
  }

  // Reset only the entries we marked — O(K), not O(N).
  for (int v : order)
    in_order_tl[v] = false;
}

} // namespace flexscanx_core
