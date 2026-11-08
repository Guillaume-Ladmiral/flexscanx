#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace flexscanx_core {

namespace detail {

constexpr int kGammaItMax = 100000;
constexpr double kGammaEps = 3e-14;
constexpr double kGammaFpMin = std::numeric_limits<double>::min() / kGammaEps;

inline double gamma_prefactor(double a, double x) {
  return std::exp(a * std::log(x) - x - std::lgamma(a));
}

inline double regularized_gamma_p_series(double a, double x) {
  double sum = 1.0 / a;
  double del = sum;
  double ap = a;
  for (int n = 1; n <= kGammaItMax; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::fabs(del) <= std::fabs(sum) * kGammaEps)
      break;
  }
  return sum * gamma_prefactor(a, x);
}

inline double regularized_gamma_q_cf(double a, double x) {
  double b = x + 1.0 - a;
  double c = 1.0 / kGammaFpMin;
  if (std::fabs(b) < kGammaFpMin)
    b = (b < 0.0) ? -kGammaFpMin : kGammaFpMin;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= kGammaItMax; ++i) {
    double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < kGammaFpMin)
      d = kGammaFpMin;
    c = b + an / c;
    if (std::fabs(c) < kGammaFpMin)
      c = kGammaFpMin;
    d = 1.0 / d;
    double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) <= kGammaEps)
      break;
  }
  return gamma_prefactor(a, x) * h;
}

inline double regularized_gamma_p(double a, double x) {
  if (!(a > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  if (x <= 0.0)
    return 0.0;
  if (x < a + 1.0)
    return std::min(1.0, std::max(0.0, regularized_gamma_p_series(a, x)));
  double q = regularized_gamma_q_cf(a, x);
  return std::min(1.0, std::max(0.0, 1.0 - q));
}

inline double regularized_gamma_q(double a, double x) {
  if (!(a > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  if (x <= 0.0)
    return 1.0;
  if (x < a + 1.0) {
    double p = regularized_gamma_p_series(a, x);
    return std::min(1.0, std::max(0.0, 1.0 - p));
  }
  return std::min(1.0, std::max(0.0, regularized_gamma_q_cf(a, x)));
}

// The mid-p boundary removes the leading lattice correction.  The two-term
// Edgeworth expression is therefore accurate in a *narrow* central window;
// 80-decimal fixtures bound its absolute error below 1e-9 at the selected
// handoff.  Blend that window in over [1e6, 2e6].  The independently pinned
// z<=6 high-mean regime is blended separately over [5e7, 1e8], replacing the
// legacy discontinuous 1e8 switch without applying the expansion prematurely
// in the off-centre intermediate regime.
inline double poisson_midp_asymptotic_weight(int k, double mean) {
  constexpr double kCentralTransitionStart = 1.0e6;
  constexpr double kCentralTransitionEnd = 2.0e6;
  constexpr double kCentralZLimit = 0.05;
  constexpr double kBroadTransitionStart = 5.0e7;
  constexpr double kBroadTransitionEnd = 1.0e8;
  constexpr double kBroadZLimit = 6.0;
  if (!(mean >= kCentralTransitionStart))
    return 0.0;
  const double z = (static_cast<double>(k) - mean) / std::sqrt(mean);
  const double abs_z = std::fabs(z);
  if (!(abs_z <= kBroadZLimit))
    return 0.0;
  if (abs_z <= kCentralZLimit) {
    if (mean >= kCentralTransitionEnd)
      return 1.0;
    const double t =
        (mean - kCentralTransitionStart) / (kCentralTransitionEnd - kCentralTransitionStart);
    return t * t * (3.0 - 2.0 * t);
  }
  if (!(mean > kBroadTransitionStart))
    return 0.0;
  if (mean >= kBroadTransitionEnd)
    return 1.0;
  const double t = (mean - kBroadTransitionStart) / (kBroadTransitionEnd - kBroadTransitionStart);
  return t * t * (3.0 - 2.0 * t);
}

inline double poisson_midp_asymptotic(int k, double mean, bool upper) {
  constexpr double kInvSqrtTwo = 0.707106781186547524400844362104849039;
  constexpr double kInvSqrtTwoPi = 0.398942280401432677939946059934381868;
  const double root_mean = std::sqrt(mean);
  const double z = (static_cast<double>(k) - mean) / root_mean;
  const double z2 = z * z;
  const double z3 = z2 * z;
  const double z5 = z3 * z2;
  const double correction_first = (1.0 - z2) / (6.0 * root_mean);
  const double correction_second =
      -((z3 - 3.0 * z) / 24.0 + (z5 - 10.0 * z3 + 15.0 * z) / 72.0) / mean;
  const double correction =
      kInvSqrtTwoPi * std::exp(-0.5 * z2) * (correction_first + correction_second);
  const double result = upper ? 0.5 * std::erfc(z * kInvSqrtTwo) - correction
                              : 0.5 * std::erfc(-z * kInvSqrtTwo) + correction;
  return std::min(1.0, std::max(0.0, result));
}

} // namespace detail

// Poisson PMF via log-gamma; remains stable for large counts/means.
inline double poisson_pmf(int k, double mean) {
  if (k < 0)
    return 0.0;
  if (!(mean >= 0.0) || !std::isfinite(mean))
    return 0.0;
  if (mean == 0.0)
    return (k == 0) ? 1.0 : 0.0;
  double logp = static_cast<double>(k) * std::log(mean) - mean - std::lgamma(k + 1.0);
  return std::exp(logp);
}

// Poisson CDF via the regularized upper incomplete gamma:
//   Pr{X <= k} = Q(k + 1, mean)
inline double poisson_cdf(int k, double mean) {
  if (k < 0)
    return 0.0;
  if (!(mean >= 0.0) || !std::isfinite(mean))
    return 0.0;
  if (mean == 0.0)
    return 1.0;
  double cdf = detail::regularized_gamma_q(static_cast<double>(k) + 1.0, mean);
  return std::min(std::max(cdf, 0.0), 1.0);
}

// Upper-tail middle p: midp_upper(ax) = Pr{X > ax} + 0.5*Pr{X=ax}
inline double poisson_middle_p_upper(int ax, double mean) {
  if (ax < 0)
    return 1.0; // x<0 corner case per spec
  const double asymptotic_weight = detail::poisson_midp_asymptotic_weight(ax, mean);
  if (asymptotic_weight == 1.0)
    return detail::poisson_midp_asymptotic(ax, mean, true);
  double pmf = poisson_pmf(ax, mean);
  if (mean == 0.0)
    return (ax == 0) ? 0.5 : 0.0;
  double upper_strict =
      detail::regularized_gamma_p(static_cast<double>(ax) + 1.0, mean); // Pr{X > ax}
  double mid = upper_strict + 0.5 * pmf;
  if (mid < 0.0)
    mid = 0.0;
  if (mid > 1.0)
    mid = 1.0;
  if (asymptotic_weight > 0.0) {
    const double asymptotic = detail::poisson_midp_asymptotic(ax, mean, true);
    mid += asymptotic_weight * (asymptotic - mid);
  }
  return mid;
}

// Lower-tail middle p (for COLD): F(ax) - 0.5*Pr{X=ax}
// Identity: for all ax >= 0, midp_lower(ax) = 1 - midp_upper(ax).
// Corner case ax < 0 returns 0.0 by convention, giving midp_lower(0) = 0.5 * Pr{X=0}.
// Some older notes may show 1 - Ppfm(ax-1); that is incorrect for mid-p and was a documentation
// typo.
inline double poisson_middle_p_lower(int ax, double mean) {
  if (ax < 0)
    return 0.0; // by convention for lower tail at x<0
  const double asymptotic_weight = detail::poisson_midp_asymptotic_weight(ax, mean);
  if (asymptotic_weight == 1.0)
    return detail::poisson_midp_asymptotic(ax, mean, false);
  double Fx = poisson_cdf(ax, mean);
  double pmf = poisson_pmf(ax, mean);
  double low = Fx - 0.5 * pmf;
  if (low < 0.0)
    low = 0.0;
  if (low > 1.0)
    low = 1.0;
  if (asymptotic_weight > 0.0) {
    const double asymptotic = detail::poisson_midp_asymptotic(ax, mean, false);
    low += asymptotic_weight * (asymptotic - low);
  }
  return low;
}

} // namespace flexscanx_core
