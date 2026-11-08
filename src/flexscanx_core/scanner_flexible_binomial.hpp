#pragma once
// Aggregator header for Binomial flexible scanners.

#include "flexscanx_core/scanner_flexible_binomial_restricted.hpp"

namespace flexscanx_core {

ClusterSelection scan_flexible_hot_original_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

ClusterSelection scan_flexible_cold_original_binomial(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &population,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

} // namespace flexscanx_core
