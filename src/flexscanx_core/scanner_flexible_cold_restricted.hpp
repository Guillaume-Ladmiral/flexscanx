#pragma once
#include "flexscanx_core/scanner_flexible_common.hpp"
#include "static_precompute.hpp"
#include <vector>

namespace flexscanx_core {

ClusterSelection scan_flexible_cold_restricted_poisson(
    const Precompute &pre, const std::vector<int> &observed, const std::vector<double> &expected,
    const ScanSettings &st, const std::vector<char> *excluded = nullptr,
    const std::vector<int> *center_order = nullptr, const GatingCache *gating = nullptr);

} // namespace flexscanx_core
