#pragma once
#include "flexscanx_core/policies/tail.hpp"
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace flexscanx_core {

// Greedy optimistic upper bound: allow selecting up to `slots` best candidates by LLR.
// candidates: indices of remaining nodes; start_idx: starting position into candidates to consider.
// observed and denom are the input vectors; Traits::llr_with_baseline<Tail> computes the LLR
// using a precomputed baseline to avoid recomputing nG*log(nG/mG) on every call.

template <class Traits, class Tail>
inline double
greedy_upper_bound(const std::vector<int> &candidates, int start_idx, int slots, double nZ,
                   double mZ, double nG, double mG, const std::vector<int> &observed,
                   const std::vector<typename Traits::DenomT> &denom, double baseline) {
  double best = Traits::template llr_with_baseline<Tail>(nZ, mZ, nG, mG, baseline);
  if (slots <= 0)
    return best;
  // Use a thread-local scratch buffer to avoid per-call heap allocation in the DFS hot loop.
  static thread_local std::vector<char> taken_buf;
  if (taken_buf.size() < candidates.size())
    taken_buf.resize(candidates.size(), 0);
  std::fill(taken_buf.begin(), taken_buf.begin() + (int)candidates.size(), 0);
  auto &taken = taken_buf;
  for (int s = 0; s < slots; ++s) {
    double local_best = best;
    int best_j = -1;
    for (int j = start_idx; j < (int)candidates.size(); ++j) {
      if (taken[j])
        continue;
      int idx = candidates[j];
      double val = Traits::template llr_with_baseline<Tail>(nZ + observed[idx], mZ + denom[idx], nG,
                                                            mG, baseline);
      if (val > local_best) {
        local_best = val;
        best_j = j;
      }
    }
    if (best_j < 0)
      break;
    int idx = candidates[best_j];
    taken[best_j] = 1;
    nZ += observed[idx];
    mZ += denom[idx];
    if (local_best > best)
      best = local_best;
  }
  return best;
}

// Best-candidate single-pass chooser (no global sort). If CacheLLR=true, uses per-item cache.
// baseline: precomputed nG*log(nG/mG) (or Binomial equivalent) — constant for the simulation.

template <class Traits, class Tail, bool CacheLLR>
inline int
select_best_candidate(std::vector<std::pair<int, double>> &frontier, int start, double nZ,
                      double mZ, double nG, double mG, const std::vector<int> &observed,
                      const std::vector<typename Traits::DenomT> &denom, double baseline) {
  int best_pos = start;
  double best_val = -std::numeric_limits<double>::infinity();
  for (int j = start; j < (int)frontier.size(); ++j) {
    int a = frontier[j].first;
    double &cached = frontier[j].second;
    double va;
    if constexpr (CacheLLR) {
      if (cached == -std::numeric_limits<double>::infinity()) {
        cached = Traits::template llr_with_baseline<Tail>(nZ + observed[a], mZ + denom[a], nG, mG,
                                                          baseline);
      }
      va = cached;
    } else {
      va = Traits::template llr_with_baseline<Tail>(nZ + observed[a], mZ + denom[a], nG, mG,
                                                    baseline);
    }
    if (va > best_val) {
      best_val = va;
      best_pos = j;
    }
  }
  return best_pos;
}

} // namespace flexscanx_core
