#include "scanner_circular_poisson.hpp"
#include "algorithms/scan_circular.hpp"
#include "policies/model_poisson.hpp"
#include "policies/stat_type.hpp"
#include "policies/tail.hpp"

namespace flexscanx_core {

ClusterSelection scan_circular_poisson(const Precompute &pre, const std::vector<int> &observed,
                                       const std::vector<double> &expected, const ScanSettings &st,
                                       bool restricted, bool hot, const std::vector<char> *excluded,
                                       const std::vector<int> *center_order,
                                       const GatingCache *gating) {
  if (restricted)
    return hot ? scan_circular_core<PoissonTraits, HotTag, RestrictedTag>(
                     pre, observed, expected, st, excluded, center_order, gating)
               : scan_circular_core<PoissonTraits, ColdTag, RestrictedTag>(
                     pre, observed, expected, st, excluded, center_order, gating);
  return hot ? scan_circular_core<PoissonTraits, HotTag, OriginalTag>(
                   pre, observed, expected, st, excluded, center_order, gating)
             : scan_circular_core<PoissonTraits, ColdTag, OriginalTag>(
                   pre, observed, expected, st, excluded, center_order, gating);
}

} // namespace flexscanx_core
