#pragma once
#include "flexscanx_core/llr_binomial.hpp"
#include "flexscanx_core/middle_p_binomial.hpp"
#include "flexscanx_core/policies/tail.hpp"
#include <cmath>
#include <vector>

namespace flexscanx_core {

struct BinomialTraits {
  using DenomT = double; // population
  struct PvContext {
    double pTot{0.0};
  };

  // Precompute the constant baseline term once per simulation; pass to llr_with_baseline<Tail>
  // to skip the nG*log(pTot)+(mG-nG)*log(1-pTot) recomputation at every DFS node.
  static inline double compute_baseline(double nG, double mG) {
    return binomial_baseline(nG, mG);
  }

  // Standard 4-parameter llr (computes baseline internally — backward compatible).
  template <class Tail> static inline double llr(double nZ, double mZ, double nG, double mG);

  // Baseline-aware 5-parameter llr (hot path — skips baseline recomputation).
  template <class Tail>
  static inline double llr_with_baseline(double nZ, double mZ, double nG, double mG, double b);

  static inline double pv_upper(int ax, int n, const PvContext &ctx) {
    return binomial_middle_p_upper(ax, n, ctx.pTot);
  }
  static inline double pv_lower(int ax, int n, const PvContext &ctx) {
    return binomial_middle_p_lower(ax, n, ctx.pTot);
  }
};

// Standard llr<Tail>
template <>
inline double BinomialTraits::llr<struct HotTag>(double nZ, double mZ, double nG, double mG) {
  return llr_binomial_original_hot(nZ, mZ, nG, mG);
}
template <>
inline double BinomialTraits::llr<struct ColdTag>(double nZ, double mZ, double nG, double mG) {
  return llr_binomial_original_cold(nZ, mZ, nG, mG);
}

// llr_with_baseline<Tail>
template <>
inline double BinomialTraits::llr_with_baseline<struct HotTag>(double nZ, double mZ, double nG,
                                                               double mG, double b) {
  return llr_binomial_original_hot_b(nZ, mZ, nG, mG, b);
}
template <>
inline double BinomialTraits::llr_with_baseline<struct ColdTag>(double nZ, double mZ, double nG,
                                                                double mG, double b) {
  return llr_binomial_original_cold_b(nZ, mZ, nG, mG, b);
}

} // namespace flexscanx_core
