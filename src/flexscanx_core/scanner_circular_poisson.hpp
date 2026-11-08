#pragma once
#include "flexscanx_core/scanner_flexible_common.hpp" // brings ScanSettings, ClusterSelection
#include "static_precompute.hpp"

namespace flexscanx_core {

ClusterSelection scan_circular_poisson(const Precompute &pre, const std::vector<int> &observed,
                                       const std::vector<double> &expected, const ScanSettings &st,
                                       bool restricted, bool hot,
                                       const std::vector<char> *excluded = nullptr,
                                       const std::vector<int> *center_order = nullptr,
                                       const GatingCache *gating = nullptr);

} // namespace flexscanx_core
