# Advanced Usage

<!-- BEGIN GENERATED: maintained_last_updated -->
**Last Updated:** 2026-11-08
<!-- END GENERATED: maintained_last_updated -->

Practical guidance for the parts of `flexscanx` that are useful after the first successful run:
Monte Carlo controls, time-series workflows, sequence analysis, geospatial helpers,
plotting/export, and performance/diagnostic switches.

## Monte Carlo Controls

- `rantype=None` is model-aware by default: Poisson resolves to `MULTINOMIAL`, Binomial resolves to `BINOMIAL`.
- `mc_early_stop=True` lets one-sided HOT/COLD Monte Carlo stop when the Wilson
  interval is already clearly above or below `mc_alpha`. BOTH scans deliberately
  exhaust `simcount` so their aligned two-sided null is not stopped by an invalid
  one-sided criterion. The one-sided rule is evaluated after at most 25 completed
  replicates, including recommended runs with `simcount=199`.
- `n_jobs_centers` is a total native worker budget: it caps centre workers for
  observed scans and replicate workers for parallel Monte Carlo; hardware and
  replication count can only lower that limit.
- `mc_use_alias=True` (or `FLEXSCANX_MC_ALIAS=1`) is available when multinomial sampling becomes the bottleneck in large simulation runs. It consumes the random stream differently from the default sampler, so the same `seed` yields a different null sample and p-values agree only within Monte Carlo error; the provenance keys `mc_use_alias` and `mc_alias_env_override` record which path ran.
- For quick iteration, start around `simcount=199`; use `999` or higher when you need stable p-values.
- `calibrate_llr_threshold` requires `n_sims >= 1`; passing 0 raises `ValueError`.

## Time-Series And Sequence Workflows

- Use `scan_series()` for repeated scans over `T x N` data.
- Provide `times=` when time labels matter downstream; summaries and CSV export preserve them.
- Keep `crn=True` when comparing slices over time. It reduces Monte Carlo noise in longitudinal comparisons.
  - **Seed note**: When `crn=True` and `seed=None`, an internal fixed seed (`_CRN_DEFAULT_SEED`) anchors the CRN stream. Pass an explicit `seed=` for full cross-run reproducibility.
- For temporal membership analysis, start with [sequence_analysis_workflow.qmd](../vignettes/workflows/sequence_analysis_workflow.qmd).

## Sequence Analysis — Distance Metrics

- `seq_dist_hamming(sequences, norm=True)` returns a full N×N symmetric distance matrix.
- For **large N** (> ~7 900 areas), the matrix exceeds 500 MB and a `UserWarning` is emitted. Use `condensed=True` to return a 1-D upper-triangle array (~half the memory) compatible with `scipy.spatial.distance.squareform`:
  ```python
  cdist = seq_dist_hamming(sequences, condensed=True)
  # Expand when needed: from scipy.spatial.distance import squareform; full = squareform(cdist)
  ```
- `seq_dist_gower` was removed in v0.0.1b — use `seq_dist_hamming` directly for categorical sequences (the two are equivalent for discrete states).

## Enumerating And Exporting Clusters

- Use `scan(..., secondary="ALL")` or `scan_series(..., secondary="ALL")` to enumerate the full non-overlapping cluster set instead of only the primary cluster.
- Use `flexscanx.summary.flatten_clusters()` when you need row-oriented output.
- Use `flexscanx.summary.write_clusters_csv()` for CSV export.
- Use `flexscanx.summarize_result()` when you want a quick DataFrame view for either a single scan or a series result.

## Geospatial Input Patterns

- `scan_from_geodataframe()` is the shortest path when your cluster IDs should remain in `id_col` space.
- `queen_adjacency()` and `rook_adjacency()` are the adjacency-first path for polygon workflows. Both evaluate their documented geometric predicate with shapely (queen: touch at any point; rook: shared boundary of non-zero length), so the graph does not change with the presence of libpysal; `csr_from_pysal()` remains available for a libpysal `W` built with libpysal's vertex-based rules.
- `csr_from_dense()` validates that the matrix is square, contains only `0`/`1` entries, and is symmetric; weighted, negative, `NaN` or string entries and asymmetric input raise `ValueError` naming the offending positions (weights used to be silently binarised, and asymmetric input used to produce silently wrong results).
- If you are starting from raw polygons and want a worked example, use [Manhattan_Breast_Cancer.qmd](../vignettes/workflows/Manhattan_Breast_Cancer.qmd).
- If you want to compare adjacency choices, use [contiguity_queen_vs_rook.qmd](../vignettes/workflows/contiguity_queen_vs_rook.qmd).

## Calibration: POISSON vs BINOMIAL

`calibrate_llr_threshold` follows the same parameter rules as `scan()`:

- **POISSON**: pass `expected=` (positive floats). Do not pass `population=`.
- **BINOMIAL**: pass `population=` (positive integer trial counts). Do not pass `expected=`.

Passing `expected=` for BINOMIAL (the old behaviour) now raises a clear `ValueError`.
The calibration cache key is keyed on the effective `rantype`, so `rantype=None` and
`rantype="MULTINOMIAL"` share the same cache entry for POISSON — no duplicate simulations.

## Visualization And Reports

- The plotting helpers in `flexscanx.plotting` cover primary-cluster maps, BOTH-mode side-by-side views, all-cluster grids, and overlay plots.
- **`flexscanx.export_cluster_report()`** is now available at the root namespace (previously accessible only via `flexscanx.report.export_cluster_report`). It builds a one-page map-plus-summary report in HTML, PDF, or PNG format.
  - Requires `matplotlib` — install with `pip install flexscanx[plot]`.
  - Requires `geopandas` — install with `pip install flexscanx[geo]`.
  - A clear `ImportError` is raised if either is missing, not a cryptic `AttributeError`.
- **`flexscanx.plot_consensus_frequency()`** maps the per-area inclusion frequencies from `consensus_sweep()` as a choropleth.
  - Also requires `matplotlib` (`pip install flexscanx[plot]`) and `geopandas`.
  - Pass `discrete_bins=[0, 0.25, 0.5, 0.75, 1.0]` for a four-tier colour scheme.
- If you only need a compact tabular export, prefer the summary/export helpers instead of full report generation.

## Performance And Diagnostics

- `K` controls the candidate horizon; `clustersize` controls the maximum admissible cluster size. They are not interchangeable.
- `knn_mode="K_REFILL"` is the default graph search mode; `K_FIXED` is useful when you want deterministic top-K without refill.
- `debug_prune=True` emits one concise per-scan diagnostic line from the core.
- When numba is installed (`pip install flexscanx[jit]`), `silhouette_graph` and
  `dunn_graph` use a JIT-compiled BFS, and `boundary_contrast` uses a
  JIT-compiled inner loop. The other graph metrics do not have numba-specific
  paths. Without numba, importing `flexscanx` remains silent; the first metric
  call that actually needs one of these pure-Python fallbacks emits one
  `UserWarning` per process pointing to the `[jit]` extra. Setting the
  environment variable `FLEXSCANX_DISABLE_JIT=1` before import forces the
  pure-Python paths even when numba is installed (no fallback warning is
  emitted in that case); CI uses it to exercise both paths.
- `csr_knn_from_coords(backend="balltree")` and `metric="haversine"` need
  scikit-learn, installed by `pip install flexscanx[knn]`; the default KD-tree
  path needs only SciPy.
- For script-driven benchmarking, use the supported tools documented in [scripts/README.md](../scripts/README.md).

## Where To Go Next

- Public contract: [API_REFERENCE.md](../API_REFERENCE.md)
- Workflow examples: [vignettes/README.md](../vignettes/README.md)
- Developer/build notes: [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md)
- Canonical doc map: [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)
