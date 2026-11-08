#include "explicit_inst.hpp"

namespace flexscanx_core {

// Flexible: Poisson
template ClusterSelection scan_flexible_core<PoissonTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_flexible_core<PoissonTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_flexible_core<PoissonTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_flexible_core<PoissonTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Flexible: Binomial
template ClusterSelection scan_flexible_core<BinomialTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_flexible_core<BinomialTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_flexible_core<BinomialTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_flexible_core<BinomialTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Circular: Poisson
template ClusterSelection scan_circular_core<PoissonTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_circular_core<PoissonTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_circular_core<PoissonTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_circular_core<PoissonTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Circular: Binomial
template ClusterSelection scan_circular_core<BinomialTraits, HotTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_circular_core<BinomialTraits, ColdTag, RestrictedTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_circular_core<BinomialTraits, HotTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);
template ClusterSelection scan_circular_core<BinomialTraits, ColdTag, OriginalTag>(
    const Precompute &, const std::vector<int> &, const std::vector<double> &, const ScanSettings &,
    const std::vector<char> *, const std::vector<int> *, const GatingCache *);

// Force instantiation by referencing the symbols from this TU.
// Some compilers/linkers require explicit use; here we rely on ODR-use via the wrappers.

} // namespace flexscanx_core
