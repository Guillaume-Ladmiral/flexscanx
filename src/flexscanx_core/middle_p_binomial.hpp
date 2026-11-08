#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace flexscanx_core {

namespace detail {

constexpr int kBetaMaxIterations = 100000;
constexpr long double kBetaEpsilon = 8.0L * std::numeric_limits<long double>::epsilon();
constexpr long double kBetaFpMin = std::numeric_limits<long double>::min() / kBetaEpsilon;

// Modified Lentz evaluation of the incomplete-beta continued fraction.  It
// avoids the O(n) loop required to sum individual binomial masses.
inline long double beta_continued_fraction(long double a, long double b, long double x) {
  const long double qab = a + b;
  const long double qap = a + 1.0L;
  const long double qam = a - 1.0L;
  long double c = 1.0L;
  long double d = 1.0L - qab * x / qap;
  if (std::fabs(d) < kBetaFpMin)
    d = kBetaFpMin;
  d = 1.0L / d;
  long double h = d;
  for (int m = 1; m <= kBetaMaxIterations; ++m) {
    const int m2 = 2 * m;
    long double aa = static_cast<long double>(m) * (b - static_cast<long double>(m)) * x /
                     ((qam + static_cast<long double>(m2)) * (a + static_cast<long double>(m2)));
    d = 1.0L + aa * d;
    if (std::fabs(d) < kBetaFpMin)
      d = kBetaFpMin;
    c = 1.0L + aa / c;
    if (std::fabs(c) < kBetaFpMin)
      c = kBetaFpMin;
    d = 1.0L / d;
    h *= d * c;

    aa = -(a + static_cast<long double>(m)) * (qab + static_cast<long double>(m)) * x /
         ((a + static_cast<long double>(m2)) * (qap + static_cast<long double>(m2)));
    d = 1.0L + aa * d;
    if (std::fabs(d) < kBetaFpMin)
      d = kBetaFpMin;
    c = 1.0L + aa / c;
    if (std::fabs(c) < kBetaFpMin)
      c = kBetaFpMin;
    d = 1.0L / d;
    const long double delta = d * c;
    h *= delta;
    if (std::fabs(delta - 1.0L) <= kBetaEpsilon)
      break;
  }
  return h;
}

inline double regularized_beta_lower(double a, double b, double x) {
  if (x <= 0.0)
    return 0.0;
  if (x >= 1.0)
    return 1.0;
  const long double a_ld = static_cast<long double>(a);
  const long double b_ld = static_cast<long double>(b);
  const long double x_ld = static_cast<long double>(x);
  const long double front =
      std::exp(std::lgamma(a_ld + b_ld) - std::lgamma(a_ld) - std::lgamma(b_ld) +
               a_ld * std::log(x_ld) + b_ld * std::log1p(-x_ld));
  long double result;
  if (x_ld < (a_ld + 1.0L) / (a_ld + b_ld + 2.0L)) {
    result = front * beta_continued_fraction(a_ld, b_ld, x_ld) / a_ld;
  } else {
    result = 1.0L - front * beta_continued_fraction(b_ld, a_ld, 1.0L - x_ld) / b_ld;
  }
  return static_cast<double>(std::min(1.0L, std::max(0.0L, result)));
}

} // namespace detail

inline double binom_pmf(int k, int n, double p) {
  if (k < 0 || k > n)
    return 0.0;
  if (n == 0)
    return (k == 0) ? 1.0 : 0.0;
  if (p <= 0.0)
    return (k == 0) ? 1.0 : 0.0;
  if (p >= 1.0)
    return (k == n) ? 1.0 : 0.0;
  const long double logpmf =
      std::lgamma(static_cast<long double>(n) + 1.0L) -
      std::lgamma(static_cast<long double>(k) + 1.0L) -
      std::lgamma(static_cast<long double>(n - k) + 1.0L) +
      static_cast<long double>(k) * std::log(static_cast<long double>(p)) +
      static_cast<long double>(n - k) * std::log1p(-static_cast<long double>(p));
  const double v = static_cast<double>(std::exp(logpmf));
  if (!std::isfinite(v))
    return 0.0;
  return v;
}

inline double binom_cdf(int k, int n, double p) {
  if (k < 0)
    return 0.0;
  if (k >= n)
    return 1.0;
  if (n == 0)
    return 1.0;
  if (p <= 0.0)
    return 1.0; // all mass at 0
  if (p >= 1.0)
    return (k >= n) ? 1.0 : 0.0;
  // For p > 0.5 use the symmetric complement: P(X<=k | n,p) = 1 - P(X<=n-k-1 | n,1-p).
  // This avoids the ratio p/(1-p) >> 1 that causes the recurrence to overflow for p near 1.
  if (p > 0.5)
    return 1.0 - binom_cdf(n - k - 1, n, 1.0 - p);
  double log1mp = std::log(1.0 - p);
  double log_pmf0 = n * log1mp;
  if (log_pmf0 > -700.0) {
    // Fast path: pmf(0) = (1-p)^n is representable, use the linear recurrence
    // pmf(i) = pmf(i-1)*(n-i+1)/i * p/(1-p). The ratio p/(1-p) <= 1 here, so
    // no overflow; terms only grow toward the mode, so no mid-sum underflow.
    double pmf0 = std::exp(log_pmf0);
    double cdf = pmf0;
    double pmf = pmf0;
    for (int i = 1; i <= k; ++i) {
      pmf *= ((double)(n - i + 1) / (double)i) * (p / (1.0 - p));
      cdf += pmf;
      if (cdf >= 1.0) {
        cdf = 1.0;
        break;
      }
    }
    if (cdf < 0.0)
      cdf = 0.0;
    if (cdf > 1.0)
      cdf = 1.0;
    return cdf;
  }
  // Underflow regime (n*|log(1-p)| > ~708, e.g. n >~ 1021 at p = 0.5): the
  // base term underflows to 0 in double, which previously made the whole CDF
  // collapse to 0 (audit 2026-06-11, finding CC-3) and corrupted RESTRICTED
  // BINOMIAL gating at large populations. Accumulate the sum in log space via
  // a streaming logaddexp over the same recurrence instead.
  double log_ratio = std::log(p) - log1mp;
  double log_pmf = log_pmf0;
  double log_cdf = log_pmf0;
  for (int i = 1; i <= k; ++i) {
    log_pmf += std::log((double)(n - i + 1) / (double)i) + log_ratio;
    double hi = (log_cdf > log_pmf) ? log_cdf : log_pmf;
    double lo = (log_cdf > log_pmf) ? log_pmf : log_cdf;
    log_cdf = hi + std::log1p(std::exp(lo - hi));
    if (log_cdf >= 0.0)
      return 1.0;
  }
  double cdf = std::exp(log_cdf);
  if (cdf < 0.0)
    cdf = 0.0;
  if (cdf > 1.0)
    cdf = 1.0;
  return cdf;
}

// Upper-tail middle p (HOT): midp_upper(ax) = Pr{X > ax} + 0.5*Pr{X=ax}.
// For ax < 0, return 1.0 by convention so that all middle p-values lie in [0,1].
inline double binomial_middle_p_upper(int ax, int n, double p) {
  if (ax < 0)
    return 1.0; // by convention for middle p
  if (ax > n)
    return 0.0;
  if (p == 0.5 && ax == n - ax)
    return 0.5;
  const double pmf = binom_pmf(ax, n, p);
  const double upper_strict = (ax < n)
                                  ? detail::regularized_beta_lower(static_cast<double>(ax) + 1.0,
                                                                   static_cast<double>(n - ax), p)
                                  : 0.0;
  double mid = upper_strict + 0.5 * pmf;
  if (mid > 0.5) {
    const double lower_strict =
        (ax > 0) ? detail::regularized_beta_lower(static_cast<double>(n - ax) + 1.0,
                                                  static_cast<double>(ax), 1.0 - p)
                 : 0.0;
    mid = 1.0 - (lower_strict + 0.5 * pmf);
  }
  if (mid < 0.0)
    mid = 0.0;
  if (mid > 1.0)
    mid = 1.0;
  return mid;
}

// Lower-tail middle p (COLD): F(ax) - 0.5*Pr{X=ax}.
// Identity: for all ax >= 0, midp_lower(ax) = 1 - midp_upper(ax).
// For ax < 0, return 0.0 by convention (gives midp_lower(0) = 0.5*Pr{X=0}).
inline double binomial_middle_p_lower(int ax, int n, double p) {
  if (ax < 0)
    return 0.0; // lower tail at x<0
  if (ax > n)
    return 1.0;
  if (p == 0.5 && ax == n - ax)
    return 0.5;
  const double pmf = binom_pmf(ax, n, p);
  const double lower_strict =
      (ax > 0) ? detail::regularized_beta_lower(static_cast<double>(n - ax) + 1.0,
                                                static_cast<double>(ax), 1.0 - p)
               : 0.0;
  double low = lower_strict + 0.5 * pmf;
  if (low > 0.5) {
    const double upper_strict = (ax < n)
                                    ? detail::regularized_beta_lower(static_cast<double>(ax) + 1.0,
                                                                     static_cast<double>(n - ax), p)
                                    : 0.0;
    low = 1.0 - (upper_strict + 0.5 * pmf);
  }
  if (low < 0.0)
    low = 0.0;
  if (low > 1.0)
    low = 1.0;
  return low;
}

} // namespace flexscanx_core
