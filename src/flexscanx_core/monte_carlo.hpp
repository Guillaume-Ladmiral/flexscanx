#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace flexscanx_core {

// std::poisson_distribution<int> cannot represent a mean beyond its result
// type's positive range.  Reject it explicitly instead of entering a
// libstdc++ sampling path that can fail to terminate for oversized means.
inline bool poisson_mean_supported(double mean) noexcept {
  return std::isfinite(mean) && mean >= 0.0 &&
         mean <= static_cast<double>(std::numeric_limits<int>::max());
}

inline void require_supported_poisson_mean(double mean) {
  if (std::isfinite(mean) && mean > static_cast<double>(std::numeric_limits<int>::max()))
    throw std::domain_error("Poisson mean exceeds the supported int range");
}

// -----------------------------------------------------------------------------
// Xoshiro256** RNG (fast URBG for distributions)
// -----------------------------------------------------------------------------
namespace detail_crn {
// SplitMix64 hash to scramble inputs into 64-bit values
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct Xoshiro256ss {
  using result_type = uint64_t;

  uint64_t s[4];
  Xoshiro256ss() {
    s[0] = s[1] = s[2] = s[3] = 0;
  }
  void seed(uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < 4; ++i) {
      s[i] = splitmix64(x);
      x ^= 0xD1B54A32D192ED03ULL + (uint64_t)i;
    }
    // Ensure non-zero state
    if (s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0) {
      s[0] = 1ULL;
    }
  }
  static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  uint64_t operator()() {
    const uint64_t result = rotl(s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }
  static constexpr uint64_t min() {
    return 0ULL;
  }
  static constexpr uint64_t max() {
    return UINT64_MAX;
  }
};

inline uint64_t make_seed(uint64_t seed_base, uint64_t rep, uint64_t area, uint64_t stream = 0) {
  uint64_t x = seed_base;
  x ^= 0xD1B54A32D192ED03ULL * (rep + 0x9E3779B97F4A7C15ULL);
  x ^= 0xA24BAED4963EE407ULL * (area + 0xBF58476D1CE4E5B9ULL);
  x ^= 0x9E3779B185EBCA87ULL * (stream + 0x94D049BB133111EBULL);
  return splitmix64(x);
}
} // namespace detail_crn

// -----------------------------------------------------------------------------
// Standard RNG-based generators (templated URBG)
// -----------------------------------------------------------------------------
template <class URBG>
inline std::vector<int> multinomial_counts(const std::vector<double> &expected, int total,
                                           URBG &rng) {
  std::vector<int> out(expected.size(), 0);
  if (total <= 0 || expected.empty())
    return out;
  std::vector<double> w;
  w.reserve(expected.size());
  double mass = 0.0;
  for (double e : expected) {
    double v = (e > 0.0 && std::isfinite(e)) ? e : 0.0;
    w.push_back(v);
    mass += v;
  }
  if (mass <= 0.0)
    return out;
  std::discrete_distribution<int> dist(w.begin(), w.end());
  for (int t = 0; t < total; ++t) {
    int k = dist(rng);
    ++out[k];
  }
  return out;
}

template <class URBG>
inline std::vector<int> poisson_counts(const std::vector<double> &mean, URBG &rng) {
  std::vector<int> out(mean.size(), 0);
  for (size_t i = 0; i < mean.size(); ++i) {
    double mu = mean[i];
    require_supported_poisson_mean(mu);
    if (!(mu > 0.0) || !std::isfinite(mu)) {
      out[i] = 0;
      continue;
    }
    std::poisson_distribution<int> pd(mu);
    out[i] = pd(rng);
  }
  return out;
}

template <class URBG>
inline std::vector<int> binomial_counts(const std::vector<double> &population, double p,
                                        URBG &rng) {
  std::vector<int> out(population.size(), 0);
  if (!(p >= 0.0) || !(p <= 1.0) || !std::isfinite(p))
    p = 0.0;
  for (size_t i = 0; i < population.size(); ++i) {
    double pop = population[i];
    if (!(pop > 0.0) || !std::isfinite(pop)) {
      out[i] = 0;
      continue;
    }
    int n = (int)std::llround(pop);
    if (n <= 0) {
      out[i] = 0;
      continue;
    }
    std::binomial_distribution<int> bd(n, p);
    out[i] = bd(rng);
  }
  return out;
}

// -----------------------------------------------------------------------------
// Full CRN generators: stateless, counter-based seeding with Xoshiro256**
// -----------------------------------------------------------------------------
inline std::vector<int> poisson_counts_crn(const std::vector<double> &mean, uint64_t seed_base,
                                           int replication) {
  std::vector<int> out(mean.size(), 0);
  for (size_t i = 0; i < mean.size(); ++i) {
    double mu = mean[i];
    require_supported_poisson_mean(mu);
    if (!(mu > 0.0) || !std::isfinite(mu)) {
      out[i] = 0;
      continue;
    }
    detail_crn::Xoshiro256ss rng;
    rng.seed(detail_crn::make_seed((uint64_t)seed_base, (uint64_t)replication, (uint64_t)i,
                                   /*stream*/ 0));
    std::poisson_distribution<int> pd(mu);
    out[i] = pd(rng);
  }
  return out;
}

inline std::vector<int> binomial_counts_crn(const std::vector<double> &population, double p,
                                            uint64_t seed_base, int replication) {
  std::vector<int> out(population.size(), 0);
  if (!(p >= 0.0) || !(p <= 1.0) || !std::isfinite(p))
    p = 0.0;
  for (size_t i = 0; i < population.size(); ++i) {
    double pop = population[i];
    if (!(pop > 0.0) || !std::isfinite(pop)) {
      out[i] = 0;
      continue;
    }
    int n = (int)std::llround(pop);
    if (n <= 0) {
      out[i] = 0;
      continue;
    }
    detail_crn::Xoshiro256ss rng;
    rng.seed(detail_crn::make_seed((uint64_t)seed_base, (uint64_t)replication, (uint64_t)i,
                                   /*stream*/ 1));
    std::binomial_distribution<int> bd(n, p);
    out[i] = bd(rng);
  }
  return out;
}

inline std::vector<int> multinomial_counts_crn(const std::vector<double> &expected, int total,
                                               uint64_t seed_base, int replication) {
  std::vector<int> out(expected.size(), 0);
  if (total <= 0 || expected.empty())
    return out;
  std::vector<double> w;
  w.reserve(expected.size());
  double mass = 0.0;
  for (double e : expected) {
    double v = (e > 0.0 && std::isfinite(e)) ? e : 0.0;
    w.push_back(v);
    mass += v;
  }
  if (mass <= 0.0)
    return out;
  detail_crn::Xoshiro256ss rng;
  rng.seed(detail_crn::make_seed((uint64_t)seed_base, (uint64_t)replication,
                                 /*area*/ 0ULL, /*stream*/ 2));
  std::discrete_distribution<int> dist(w.begin(), w.end());
  for (int t = 0; t < total; ++t) {
    int k = dist(rng);
    ++out[k];
  }
  return out;
}

// -----------------------------------------------------------------------------
// Alias table (Vose) for multinomial draws
// -----------------------------------------------------------------------------
struct AliasTable {
  std::vector<double> prob; // acceptance probabilities in [0,1]
  std::vector<int> alias;   // alias indices
  int K = 0;                // number of categories
};

inline AliasTable build_alias_from_weights(const std::vector<double> &expected) {
  AliasTable A;
  const int K = (int)expected.size();
  A.K = K;
  if (K <= 0) {
    return A;
  }
  std::vector<double> w;
  w.reserve(K);
  double mass = 0.0;
  for (double e : expected) {
    double v = (e > 0.0 && std::isfinite(e)) ? e : 0.0;
    w.push_back(v);
    mass += v;
  }
  A.prob.assign(K, 0.0);
  A.alias.assign(K, 0);
  if (mass <= 0.0) {
    // Degenerate: keep prob=1 and alias=i so sampling falls back to identity if used
    for (int i = 0; i < K; ++i) {
      A.prob[i] = 1.0;
      A.alias[i] = i;
    }
    return A;
  }
  // Normalize and scale by K
  std::vector<double> q(K, 0.0);
  for (int i = 0; i < K; ++i) {
    q[i] = (w[i] / mass) * (double)K;
  }
  std::vector<int> small;
  std::vector<int> large;
  small.reserve(K);
  large.reserve(K);
  for (int i = 0; i < K; ++i) {
    if (q[i] < 1.0)
      small.push_back(i);
    else
      large.push_back(i);
  }
  while (!small.empty() && !large.empty()) {
    int s = small.back();
    small.pop_back();
    int l = large.back();
    large.pop_back();
    A.prob[s] = q[s];
    A.alias[s] = l;
    q[l] = (q[l] + q[s]) - 1.0; // reduce l's overfull mass by the amount used to top up s
    if (q[l] < 1.0)
      small.push_back(l);
    else
      large.push_back(l);
  }
  // Whatever remains has prob=1
  while (!large.empty()) {
    int l = large.back();
    large.pop_back();
    A.prob[l] = 1.0;
    A.alias[l] = l;
  }
  while (!small.empty()) {
    int s = small.back();
    small.pop_back();
    A.prob[s] = 1.0;
    A.alias[s] = s;
  }
  return A;
}

inline std::vector<int> multinomial_counts_alias_crn(const AliasTable &A, int total,
                                                     uint64_t seed_base, int replication) {
  std::vector<int> out((size_t)A.K, 0);
  if (total <= 0 || A.K <= 0)
    return out;
  detail_crn::Xoshiro256ss rng;
  // Reuse stream id 2 for a stable alias-specific stream. The alias and
  // std::discrete_distribution paths consume RNG outputs differently, so they
  // are not draw-for-draw CRN-aligned for the same public seed.
  rng.seed(detail_crn::make_seed((uint64_t)seed_base, (uint64_t)replication, /*area*/ 0ULL,
                                 /*stream*/ 2));
  std::uniform_int_distribution<int> ui(0, A.K - 1);
  std::uniform_real_distribution<double> ur(0.0, 1.0);
  for (int t = 0; t < total; ++t) {
    int i = ui(rng);
    double u = ur(rng);
    int k = (u < A.prob[i]) ? i : A.alias[i];
    ++out[k];
  }
  return out;
}

} // namespace flexscanx_core
