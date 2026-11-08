#include "flexscanx_core/algorithms/scan_flexible.hpp"
#include "flexscanx_core/policies/model_binomial.hpp"
#include "flexscanx_core/policies/stat_type.hpp"
#include "flexscanx_core/policies/tail.hpp"
#include "flexscanx_core/scanner_flexible_binomial_restricted.hpp"

namespace flexscanx_core {

ClusterSelection scan_flexible_hot_restricted_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<BinomialTraits, HotTag, RestrictedTag>(pre, observed, population, st,
                                                                   excluded, center_order, gating);
}

ClusterSelection scan_flexible_cold_restricted_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<BinomialTraits, ColdTag, RestrictedTag>(pre, observed, population, st,
                                                                    excluded, center_order, gating);
}

ClusterSelection scan_flexible_hot_original_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<BinomialTraits, HotTag, OriginalTag>(pre, observed, population, st,
                                                                 excluded, center_order, gating);
}

ClusterSelection scan_flexible_cold_original_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded, const std::vector<int> *center_order,
    const GatingCache *gating) {
  return scan_flexible_core<BinomialTraits, ColdTag, OriginalTag>(pre, observed, population, st,
                                                                  excluded, center_order, gating);
}

} // namespace flexscanx_core
