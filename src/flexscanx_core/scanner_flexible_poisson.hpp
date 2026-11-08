#pragma once
// Aggregator header to provide a single include point for Poisson flexible scanners.

#include "flexscanx_core/scanner_flexible_poisson_restricted.hpp"

namespace flexscanx_core {

ClusterSelection scan_flexible_hot_original_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

ClusterSelection scan_flexible_cold_original_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

} // namespace flexscanx_core
