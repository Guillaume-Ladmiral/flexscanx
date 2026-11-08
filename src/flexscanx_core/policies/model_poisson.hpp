#pragma once
#include "flexscanx_core/llr_poisson.hpp"
#include "flexscanx_core/middle_p_poisson.hpp"
#include "flexscanx_core/policies/tail.hpp"
#include <cmath>
#include <vector>

namespace flexscanx_core {

struct PoissonTraits {
  using DenomT = double; // expected

  // Precompute the constant baseline term once per simulation; pass to llr_with_baseline<Tail>
  // to skip the nG*log(nG/mG) recomputation at every DFS node.
  static inline double compute_baseline(double nG, double mG) {
    return poisson_baseline(nG, mG);
  }

  // Standard 4-parameter llr (computes baseline internally — backward compatible).
  template <class Tail> static inline double llr(double nZ, double mZ, double nG, double mG);

  // Baseline-aware 5-parameter llr (hot path — skips baseline recomputation).
  template <class Tail>
  static inline double llr_with_baseline(double nZ, double mZ, double nG, double mG, double b);

  // Gating p-values (restricted)
  static inline double pv_upper(int ax, double mean) {
    return poisson_middle_p_upper(ax, mean);
  }
  static inline double pv_lower(int ax, double mean) {
    return poisson_middle_p_lower(ax, mean);
  }
};

// Standard llr<Tail> — keeps backward compatibility for any caller without a precomputed baseline.
template <>
inline double PoissonTraits::llr<struct HotTag>(double nZ, double mZ, double nG, double mG) {
  return llr_poisson_original_hot(nZ, mZ, nG, mG);
}
template <>
inline double PoissonTraits::llr<struct ColdTag>(double nZ, double mZ, double nG, double mG) {
  return llr_poisson_original_cold(nZ, mZ, nG, mG);
}

// llr_with_baseline<Tail> — hot path for scan cores that precompute the baseline.
template <>
inline double PoissonTraits::llr_with_baseline<struct HotTag>(double nZ, double mZ, double nG,
                                                              double mG, double b) {
  return llr_poisson_original_hot_b(nZ, mZ, nG, mG, b);
}
template <>
inline double PoissonTraits::llr_with_baseline<struct ColdTag>(double nZ, double mZ, double nG,
                                                               double mG, double b) {
  return llr_poisson_original_cold_b(nZ, mZ, nG, mG, b);
}

} // namespace flexscanx_core
