#pragma once
#include "flexscanx_core/policies/model_binomial.hpp"
#include "flexscanx_core/policies/model_poisson.hpp"
#include "flexscanx_core/policies/stat_type.hpp"
#include "flexscanx_core/policies/tail.hpp"
#include "flexscanx_core/scanner_flexible_common.hpp" // brings ScanSettings, GatingCache
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

namespace flexscanx_core {

// Restricted gating provider: builds/returns per-node middle p vectors.
// For Poisson, p-values use mean = expected[i].
// For Binomial, requires PvContext with pTot = nG/mG.

template <class Traits, class Tail> struct GatingProvider {
  // Base template: fires a compile-time error if instantiated without a specialization.
  // When adding a new Traits type, add a full specialization of GatingProvider below.
  template <class DenomVec, class PvContextT = void>
  static inline const std::vector<double> &
  get(int, const std::vector<int> &, const DenomVec &, double, double, const ScanSettings &,
      const GatingCache *, std::vector<double> &, const PvContextT * = nullptr) {
    static_assert(std::is_same_v<Traits, void>, // always false for any real Traits type
                  "GatingProvider: no specialization exists for this Traits type. "
                  "Add one in gating.hpp (see PoissonTraits and BinomialTraits specializations).");
    // unreachable:
    static std::vector<double> _empty;
    return _empty;
  }
};

// Poisson specialization

template <class Tail> struct GatingProvider<PoissonTraits, Tail> {
  static inline const std::vector<double> &
  get(int N, const std::vector<int> &observed, const std::vector<double> &expected, double /*nG*/,
      double /*mG*/, const ScanSettings & /*st*/, const GatingCache *cache,
      std::vector<double> &scratch, const void * = nullptr) {
    if constexpr (std::is_same_v<Tail, HotTag>) {
      if (cache && !cache->pv_hot.empty() && (int)cache->pv_hot.size() == N)
        return cache->pv_hot;
    } else { // ColdTag
      if (cache && !cache->pv_cold.empty() && (int)cache->pv_cold.size() == N)
        return cache->pv_cold;
    }
    scratch.resize(N);
    for (int i = 0; i < N; ++i) {
      if constexpr (std::is_same_v<Tail, HotTag>) {
        scratch[i] = PoissonTraits::pv_upper(observed[i], expected[i]);
      } else {
        scratch[i] = PoissonTraits::pv_lower(observed[i], expected[i]);
      }
    }
    return scratch;
  }
};

// Binomial specialization

template <class Tail> struct GatingProvider<BinomialTraits, Tail> {
  static inline const std::vector<double> &
  get(int N, const std::vector<int> &observed, const std::vector<double> &population, double nG,
      double mG, const ScanSettings & /*st*/, const GatingCache *cache,
      std::vector<double> &scratch, const typename BinomialTraits::PvContext *ctx_in = nullptr) {
    BinomialTraits::PvContext ctx;
    if (ctx_in)
      ctx = *ctx_in;
    else
      ctx.pTot = (mG > 0.0) ? (nG / mG) : 0.0;
    if constexpr (std::is_same_v<Tail, HotTag>) {
      if (cache && !cache->pv_hot.empty() && (int)cache->pv_hot.size() == N)
        return cache->pv_hot;
    } else {
      if (cache && !cache->pv_cold.empty() && (int)cache->pv_cold.size() == N)
        return cache->pv_cold;
    }
    scratch.resize(N);
    for (int i = 0; i < N; ++i) {
      int bn = (int)std::llround(population[i]);
      if constexpr (std::is_same_v<Tail, HotTag>) {
        scratch[i] = BinomialTraits::pv_upper(observed[i], bn, ctx);
      } else {
        scratch[i] = BinomialTraits::pv_lower(observed[i], bn, ctx);
      }
    }
    return scratch;
  }
};

} // namespace flexscanx_core
