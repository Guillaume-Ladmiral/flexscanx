#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace flexscanx_core {

// One-time baseline: nG*log(pTot) + (mG-nG)*log(1-pTot), guarded for degenerate inputs.
// Identical to the c0 term previously computed inside every llr call.
inline double binomial_baseline(double nG, double mG) {
  double pTot = (mG > 0.0) ? (nG / mG) : 0.0;
  if (pTot <= 0.0 || pTot >= 1.0)
    return 0.0;
  return nG * std::log(pTot) + (mG - nG) * std::log(1.0 - pTot);
}

// Template-based shared implementation to eliminate code duplication.
// IsHot: true for HOT (pIn > pOut), false for COLD (pIn < pOut).
// c0_precomp: precomputed baseline — pass binomial_baseline(nG, mG) to skip the inner logs.
template <bool IsHot>
inline double llr_binomial_original_impl(double nZ, double mZ, double nG, double mG,
                                         double c0_precomp) {
  (void)c0_precomp; // Retained for the baseline-aware public wrapper ABI.
  double nOut = nG - nZ;
  double mOut = mG - mZ;
  if (mZ <= 0.0 || mOut <= 0.0)
    return 0.0;

  double pIn = nZ / mZ;
  double pOut = nOut / mOut;

  // Direction-specific check
  if constexpr (IsHot) {
    if (!(pIn > pOut))
      return 0.0;
  } else {
    if (!(pIn < pOut))
      return 0.0;
  }

  const long double p_total = static_cast<long double>(nG) / static_cast<long double>(mG);
  if (!(p_total > 0.0L && p_total < 1.0L))
    return 0.0;
  auto bernoulli_kl = [&](double successes, double trials) {
    const long double probability =
        static_cast<long double>(successes) / static_cast<long double>(trials);
    if (probability <= 0.0L)
      return -std::log1p(-p_total);
    if (probability >= 1.0L)
      return -std::log(p_total);
    const long double delta = probability - p_total;
    return probability * std::log1p(delta / p_total) +
           (1.0L - probability) * std::log1p(-delta / (1.0L - p_total));
  };
  const long double value = static_cast<long double>(mZ) * bernoulli_kl(nZ, mZ) +
                            static_cast<long double>(mOut) * bernoulli_kl(nOut, mOut);
  return static_cast<double>(std::max(0.0L, value));
}

// Backward-compatible overload: computes baseline internally.
template <bool IsHot>
inline double llr_binomial_original_impl(double nZ, double mZ, double nG, double mG) {
  return llr_binomial_original_impl<IsHot>(nZ, mZ, nG, mG, binomial_baseline(nG, mG));
}

// Named wrappers used by model policy dispatch.
inline double llr_binomial_original_hot(double nZ, double mZ, double nG, double mG) {
  return llr_binomial_original_impl<true>(nZ, mZ, nG, mG);
}
inline double llr_binomial_original_cold(double nZ, double mZ, double nG, double mG) {
  return llr_binomial_original_impl<false>(nZ, mZ, nG, mG);
}

// Baseline-aware named wrappers (hot path — avoids recomputing the baseline logs).
inline double llr_binomial_original_hot_b(double nZ, double mZ, double nG, double mG, double b) {
  return llr_binomial_original_impl<true>(nZ, mZ, nG, mG, b);
}
inline double llr_binomial_original_cold_b(double nZ, double mZ, double nG, double mG, double b) {
  return llr_binomial_original_impl<false>(nZ, mZ, nG, mG, b);
}

} // namespace flexscanx_core
