#pragma once
// NOTE: Prefer including the aggregator header "flexscanx_core/scanner_flexible_binomial.hpp"
// rather than this per-variant header when consuming from other translation units.
#include "flexscanx_core/scanner_flexible_common.hpp"
#include "flexscanx_core/static_precompute.hpp"
#include <vector>

namespace flexscanx_core {

ClusterSelection scan_flexible_hot_restricted_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

ClusterSelection scan_flexible_cold_restricted_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

} // namespace flexscanx_core
