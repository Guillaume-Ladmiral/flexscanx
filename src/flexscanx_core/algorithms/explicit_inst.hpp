#pragma once
// Explicit template instantiations to reduce compile times for common combinations.
// If you change the set of exposed combinations, update this file accordingly.

#include "scan_circular.hpp"
#include "scan_flexible.hpp"

namespace flexscanx_core {

// Flexible: Poisson
extern template ClusterSelection scan_flexible_core<PoissonTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_flexible_core<PoissonTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_flexible_core<PoissonTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_flexible_core<PoissonTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Flexible: Binomial
extern template ClusterSelection scan_flexible_core<BinomialTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_flexible_core<BinomialTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_flexible_core<BinomialTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_flexible_core<BinomialTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Circular: Poisson
extern template ClusterSelection scan_circular_core<PoissonTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_circular_core<PoissonTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_circular_core<PoissonTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_circular_core<PoissonTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Circular: Binomial
extern template ClusterSelection scan_circular_core<BinomialTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_circular_core<BinomialTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_circular_core<BinomialTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
extern template ClusterSelection scan_circular_core<BinomialTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

} // namespace flexscanx_core
