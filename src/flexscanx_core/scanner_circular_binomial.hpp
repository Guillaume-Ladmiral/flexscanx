#pragma once
#include "flexscanx_core/scanner_flexible_common.hpp" // brings ScanSettings, ClusterSelection, GatingCache
#include "static_precompute.hpp"

namespace flexscanx_core {

ClusterSelection scan_circular_binomial(const Precompute &pre, const std::vector<int> &observed,
                                        const std::vector<double> &population,
                                        const ScanSettings &st, bool restricted, bool hot,
                                        const std::vector<char> *excluded = nullptr,
                                        const std::vector<int> *center_order = nullptr,
                                        const GatingCache *gating = nullptr);

}
