#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace flexscanx_core {

// One-time baseline: nG*log(nG/mG), guarded for degenerate inputs.
// Identical to the c3 term previously computed inside every llr call.
inline double poisson_baseline(double nG, double mG) {
  return (nG > 0.0 && mG > 0.0) ? nG * std::log(nG / mG) : 0.0;
}

// Template-based shared implementation to eliminate code duplication.
// IsHot: true for HOT (rIn > rOut), false for COLD (rIn < rOut).
// c3_precomp: precomputed baseline — pass poisson_baseline(nG, mG) to skip the inner log().
template <bool IsHot>
inline double llr_poisson_original_impl(double nZ, double mZ, double nG, double mG,
                                        double c3_precomp) {
  (void)c3_precomp; // Retained for the baseline-aware public wrapper ABI.
  double nOut = nG - nZ;
  double mOut = mG - mZ;
  if (mZ <= 0.0 || mOut <= 0.0 || nG <= 0.0 || mG <= 0.0)
    return 0.0;

  double rIn = nZ / mZ;
  double rOut = nOut / mOut;

  // Direction-specific check
  if constexpr (IsHot) {
    if (!(rIn > rOut))
      return 0.0;
  } else {
    if (!(rIn < rOut))
      return 0.0;
  }

  // Express the statistic relative to the global rate.  This is algebraically
  // identical to c1+c2-c3, but avoids subtracting O(nG) log-likelihoods to
  // recover an arbitrarily small positive signal.
  const long double n_global = static_cast<long double>(nG);
  const long double m_global = static_cast<long double>(mG);
  auto contribution = [&](double count, double exposure) {
    if (!(count > 0.0))
      return 0.0L;
    const long double count_ld = static_cast<long double>(count);
    const long double exposure_ld = static_cast<long double>(exposure);
    // Form (count * mG - nG * exposure) before division.  Computing
    // (count/nG)*(mG/exposure)-1 loses the small relative shift before the
    // count multiplier restores its scale, even when the product uses fma.
    const long double relative_minus_one =
        (count_ld * m_global - n_global * exposure_ld) / (n_global * exposure_ld);
    return count_ld * std::log1p(relative_minus_one);
  };
  const long double value = contribution(nZ, mZ) + contribution(nOut, mOut);
  return static_cast<double>(std::max(0.0L, value));
}

// Backward-compatible overload: computes baseline internally.
// Used by any caller that does not precompute the baseline.
template <bool IsHot>
inline double llr_poisson_original_impl(double nZ, double mZ, double nG, double mG) {
  return llr_poisson_original_impl<IsHot>(nZ, mZ, nG, mG, poisson_baseline(nG, mG));
}

// Named wrappers used by model policy dispatch.
inline double llr_poisson_original_hot(double nZ, double mZ, double nG, double mG) {
  return llr_poisson_original_impl<true>(nZ, mZ, nG, mG);
}
inline double llr_poisson_original_cold(double nZ, double mZ, double nG, double mG) {
  return llr_poisson_original_impl<false>(nZ, mZ, nG, mG);
}

// Baseline-aware named wrappers (hot path — avoids recomputing nG*log(nG/mG)).
inline double llr_poisson_original_hot_b(double nZ, double mZ, double nG, double mG, double b) {
  return llr_poisson_original_impl<true>(nZ, mZ, nG, mG, b);
}
inline double llr_poisson_original_cold_b(double nZ, double mZ, double nG, double mG, double b) {
  return llr_poisson_original_impl<false>(nZ, mZ, nG, mG, b);
}

} // namespace flexscanx_core
