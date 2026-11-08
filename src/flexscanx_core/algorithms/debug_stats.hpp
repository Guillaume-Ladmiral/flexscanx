#pragma once
#include <atomic>
#include <iostream>
#include <string>
#include <type_traits>

#include "flexscanx_core/policies/model_binomial.hpp"
#include "flexscanx_core/policies/model_poisson.hpp"
#include "flexscanx_core/policies/stat_type.hpp"
#include "flexscanx_core/policies/tail.hpp"

namespace flexscanx_core {

struct DebugStats {
  std::atomic<long long> centers_scanned{0};
  std::atomic<long long> centers_skipped{0};
  std::atomic<long long> dfs_nodes_expanded{0};
  std::atomic<long long> bound_prunes{0};
  std::atomic<long long> center_ub_prunes{0};
  std::atomic<long long> best_updates{0};
};

// Compile-time names

template <class Traits> inline const char *model_name() {
  if constexpr (std::is_same_v<Traits, PoissonTraits>)
    return "poisson";
  else
    return "binomial";
}

template <class Tail> inline const char *tail_name() {
  if constexpr (std::is_same_v<Tail, HotTag>)
    return "hot";
  else
    return "cold";
}

template <class StatTag> inline const char *stat_name() {
  if constexpr (std::is_same_v<StatTag, RestrictedTag>)
    return "restricted";
  else
    return "original";
}

inline long long load_stat(const std::atomic<long long> &value) {
  return value.load(std::memory_order_relaxed);
}

template <class Traits, class Tail, class StatTag>
inline void print_stats_flexible(const DebugStats &ds) {
  std::cerr << "[flexscanx][debug_prune] scanner=flexible"
            << " model=" << model_name<Traits>() << " tail=" << tail_name<Tail>()
            << " stat=" << stat_name<StatTag>()
            << " centers_scanned=" << load_stat(ds.centers_scanned)
            << " centers_skipped=" << load_stat(ds.centers_skipped)
            << " dfs_nodes_expanded=" << load_stat(ds.dfs_nodes_expanded)
            << " bound_prunes=" << load_stat(ds.bound_prunes)
            << " center_ub_prunes=" << load_stat(ds.center_ub_prunes)
            << " best_updates=" << load_stat(ds.best_updates) << '\n';
}

template <class Traits, class Tail, class StatTag>
inline void print_stats_circular(const DebugStats &ds) {
  std::cerr << "[flexscanx][debug_prune] scanner=circular"
            << " model=" << model_name<Traits>() << " tail=" << tail_name<Tail>()
            << " stat=" << stat_name<StatTag>()
            << " centers_scanned=" << load_stat(ds.centers_scanned)
            << " centers_skipped=" << load_stat(ds.centers_skipped)
            << " dfs_nodes_expanded=" << load_stat(ds.dfs_nodes_expanded)
            << " best_updates=" << load_stat(ds.best_updates) << '\n';
}

} // namespace flexscanx_core
