#include "flexscanx_core/scanner_flexible_poisson.hpp"
#include "flexscanx_core/algorithms/scan_flexible.hpp"
#include "flexscanx_core/policies/model_poisson.hpp"
#include "flexscanx_core/policies/stat_type.hpp"
#include "flexscanx_core/policies/tail.hpp"

namespace flexscanx_core {

ClusterSelection scan_flexible_hot_restricted_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<PoissonTraits, HotTag, RestrictedTag>(pre, observed, expected, st,
                                                                  excluded, center_order, gating);
}

ClusterSelection scan_flexible_cold_restricted_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<PoissonTraits, ColdTag, RestrictedTag>(pre, observed, expected, st,
                                                                   excluded, center_order, gating);
}

ClusterSelection scan_flexible_hot_original_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<PoissonTraits, HotTag, OriginalTag>(pre, observed, expected, st,
                                                                excluded, center_order, gating);
}

ClusterSelection scan_flexible_cold_original_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<PoissonTraits, ColdTag, OriginalTag>(pre, observed, expected, st,
                                                                 excluded, center_order, gating);
}

} // namespace flexscanx_core
