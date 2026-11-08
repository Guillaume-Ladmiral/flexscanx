# flexscanx_core Developer Guide

This document explains the policy-based architecture used by the C++ core and how to add or modify scanners.

1. Overview
- We support two models (Poisson, Binomial), two tails (HOT, COLD), and two stat types (Restricted, Original) across two scan methods (Flexible, Circular).
- Shared template implementations eliminate duplication:
  - algorithms/scan_circular.hpp: scan_circular_core<Traits, Tail, StatTag>
  - algorithms/scan_flexible.hpp: scan_flexible_core<Traits, Tail, StatTag>
- Thin wrappers provide stable C APIs used by pybind11 bindings.

2. Key components
- policies/
  - model_poisson.hpp: PoissonTraits with llr<Tail>(), llr_with_baseline<Tail>(),
    compute_baseline(), pv_upper(), pv_lower()
  - model_binomial.hpp: BinomialTraits with llr<Tail>(), llr_with_baseline<Tail>(),
    compute_baseline(), PvContext{pTot}, pv_upper()/pv_lower()
  - tail.hpp: HotTag, ColdTag
  - stat_type.hpp: RestrictedTag, OriginalTag
- llr_poisson.hpp / llr_binomial.hpp: LLR formulas with precomputed-baseline overloads.
  poisson_baseline(nG,mG) / binomial_baseline(nG,mG) compute the constant null term once;
  llr_with_baseline<Tail> receives it to skip the log() recomputation in tight loops.
- algorithms/
  - universe.hpp: build_universe() (flexible) and build_knn_order() (circular).
    Both recompute distances at full double precision when a finite radius is active,
    to avoid float/double boundary inconsistencies from stored knn_dist (float32).
  - gating.hpp: GatingProvider<Traits, Tail>::get() builds/reuses per-node middle p vectors (Restricted only)
  - bounds.hpp: greedy_upper_bound<Traits, Tail>(..., double baseline) and
    select_best_candidate<Traits, Tail, CacheLLR>(..., double baseline) — both receive
    the precomputed baseline to avoid redundant log() calls.
  - frontier.hpp: FrontierMarker with generation-marking for O(1) clears.
  - centers.hpp: build_centers() with optional sorting
  - scheduler.hpp: for_each_center_atomic() (default, simple atomic index) and
    for_each_center_worksteal() (LIFO own queue, FIFO steal — standard work-stealing pattern).
  - scan_circular.hpp / scan_flexible.hpp: templated scanner implementations

3. Adding a new combination
- Choose Traits: PoissonTraits or BinomialTraits
- Choose Tail: HotTag or ColdTag
- Choose StatTag: RestrictedTag or OriginalTag
- Call the appropriate core function from a thin wrapper:
  - Flexible: scan_flexible_core<Traits, Tail, StatTag>(...)
  - Circular: scan_circular_core<Traits, Tail, StatTag>(...)
- Keep function signatures stable for bindings. The wrappers live in scanner_*.cpp files.

4. Gating (Restricted only)
- GatingProvider computes per-area middle p (upper for HOT, lower for COLD).
- Flexible: nodes failing gating are skipped.
- Circular: prefixes stop at first failure to maintain radial inclusion order.

5. Bounds and candidate ordering (Flexible)
- greedy_upper_bound provides optional optimistic pruning per-frame and at center
  level. **2026-06 audit: it is NOT a valid upper bound** (early stop on no
  single-step improvement; the frame-level candidate set omits deeper-reachable
  nodes), so both prunes are OFF by default (`enable_frame_ub_prune`,
  `enable_center_ub_gating` in ScanSettings) and documented as opt-in heuristics
  that may return suboptimal clusters. Prune comparisons are strict (`<`) so an
  enabled heuristic cannot drop exact-tie candidates.
- select_best_candidate implements single-pass best-next selection. CacheLLR=true caches per-item LLR.

6. Frontier and memory reuse
- FrontierMarker uses generation counters to avoid O(N) clears.
- scan_flexible.hpp uses thread_local FrontierMarker instances (fmark, nf) to reduce per-center allocations.
- scan_flexible.hpp also maintains a thread_local frontier pool (tl_frontier_pool) that recycles
  std::vector<pair<int,double>> objects across DFS node expansions within the same thread.
  The pool grows to at most clustersize depth and then stays constant — all child-frame
  allocations amortise to O(1) clear() calls after warmup.
- Child frontier entries are always reset to NEG_INF (not copied from parent) to ensure the
  LLR candidate cache reflects the current cluster state (nZ, mZ) rather than the parent’s.
- **Complete enumeration with cyclic revisitation (2026-07 audit):** a child frame
  inherits only the parent's NOT-yet-tried candidates (positions >= next_i) plus
  the chosen node's new neighbours. This de-duplicates acyclic insertion orders,
  but the forbid is not persistent across backtracking: connected subsets whose
  induced graph contains a cycle can be revisited through multiple insertion
  orders. Re-evaluation is inference-neutral because identical windows are
  idempotent under the strict total-order tie-break, but it can add work on dense
  cyclic graphs. Completeness and result equivalence are locked in by
  `tests/test_enum2_cyclic_invariance.py`, which compares MLCs and Monte Carlo
  ranks against an independent brute-force connected-subset enumerator.

7. Parallelization
- Scheduler::for_each_center_atomic is the default across all flexible and circular scanners.
  Simple, robust, and optimal for variable-cost centers (dynamic load balancing via atomic counter).
- Scheduler::for_each_center_worksteal is available via scheduler_backend="WORK_STEAL" setting.
  Uses LIFO own-queue and FIFO steal (standard Cilk/TBB pattern) for cache locality.
- Best-cluster updates are deterministic under exact LLR ties: tie_break_prefers()
  (scanner_flexible_common.hpp) imposes a strict total order (smaller cluster, then
  lexicographically smaller sorted area), so parallel results equal the sequential scan
  bit-for-bit regardless of thread timing (2026-06 audit, CC-2).

8. Settings and logging
- ScanSettings is defined in scanner_flexible_common.hpp and is shared across scanners.
- Consider adding flags to ScanSettings to toggle pruning/scheduling heuristics; the template cores already isolate the necessary decisions.

9. Testing
- Ensure any change passes:
  - Unit tests for Poisson/Binomial, HOT/COLD, Restricted/Original
  - Property tests (K=1 behavior, monotonicity, invariance to renumbering)
  - Integration tests (radius filter, adjacency correctness)
  - The C++ unit suite: configure with -DFLEXSCANX_BUILD_PYEXT=OFF
    -DFLEXSCANX_BUILD_CPP_TESTS=ON, build target flexscanx_core_tests, run the
    binary (CI does this on the base/3.12 matrix cells)
  - The audit regression suite tests/test_audit_v001c.py (brute-force
    optimality, parallel tie determinism, large-n binomial gating)

10. Performance tips
- Keep Traits methods inline to enable optimization in tight loops.
- Avoid unnecessary vector reallocations; reuse thread_local buffers where appropriate.
- Prefer accumulating doubles for nZ/mZ and pass integers only where required for pmfs.
