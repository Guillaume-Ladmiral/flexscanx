#include "scanner_circular_binomial.hpp"
#include "algorithms/scan_circular.hpp"
#include "policies/model_binomial.hpp"
#include "policies/stat_type.hpp"
#include "policies/tail.hpp"

namespace flexscanx_core {

ClusterSelection scan_circular_binomial(const Precompute &pre, const std::vector<int> &observed,
                                        const std::vector<double> &population,
                                        const ScanSettings &st, bool restricted, bool hot,
                                        const std::vector<char> *excluded,
                                        const std::vector<int> *center_order,
                                        const GatingCache *gating) {
  if (restricted)
    return hot ? scan_circular_core<BinomialTraits, HotTag, RestrictedTag>(
                     pre, observed, population, st, excluded, center_order, gating)
               : scan_circular_core<BinomialTraits, ColdTag, RestrictedTag>(
                     pre, observed, population, st, excluded, center_order, gating);
  return hot ? scan_circular_core<BinomialTraits, HotTag, OriginalTag>(
                   pre, observed, population, st, excluded, center_order, gating)
             : scan_circular_core<BinomialTraits, ColdTag, OriginalTag>(
                   pre, observed, population, st, excluded, center_order, gating);
}

} // namespace flexscanx_core
