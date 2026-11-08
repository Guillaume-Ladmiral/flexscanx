# flexscanx API Reference

<!-- BEGIN GENERATED: api_reference_release_metadata -->
**Version**: 0.3.3<br>
**Release**: 2026-11-08<br>
**Status**: Release v0.3.3 — 2026-11-08; licensing decision D8 open
<!-- END GENERATED: api_reference_release_metadata -->

---

## Table of Contents

1. [Core Scanning Functions](#core-scanning-functions)
2. [Time-Series Analysis](#time-series-analysis)
3. [Sequence Analysis (Temporal Trajectories)](#sequence-analysis-temporal-trajectories)
4. [Graph and Adjacency](#graph-and-adjacency)
5. [Validation and Metrics](#validation-and-metrics)
   - [`calibrate_llr_threshold()`](#calibrate_llr_threshold)
   - [`stability_across_K()`](#stability_across_k)
   - [`find_stable_params_all()`](#find_stable_params_all)
   - [`find_stable_params_all_series()`](#find_stable_params_all_series)
   - [`grid_search_stable()`](#grid_search_stable)
   - [`screen_stable_significant()`](#screen_stable_significant)
   - [`conductance()` / `boundary_contrast()`](#conductance--boundary_contrast)
6. [Visualization](#visualization)
   - [`export_cluster_report()`](#export_cluster_report)
   - [`plot_consensus_frequency()`](#plot_consensus_frequency)
7. [Utilities and Helpers](#utilities-and-helpers)
8. [Data Structures](#data-structures)

---

## Core Scanning Functions

### `scan()`

Primary function for detecting spatial clusters in a single time slice.

```python
flexscanx.scan(
    *,
    coords=None,
    adjacency=None,
    observed,
    expected=None,
    population=None,
    model="POISSON",
    scanmethod="FLEXIBLE",
    stattype="RESTRICTED",
    clustertype="HOT",
    clustersize=None,
    clusterradius=None,
    ralpha=0.2,
    K=None,
    names=None,
    handle=None,
    secondary=None,
    seed=None,
    n_jobs_centers=None,
    simcount=999,
    rantype=None,
    mc_early_stop=False,
    mc_alpha=0.05,
    mc_progress=None,
    mc_progress_every=None,
    debug_prune=False,
    enable_center_ub_gating=None,
    cache_candidate_llr=None,
    enable_frame_ub_prune=None,
    scheduler_backend=None,
    knn_method=None,
    knn_mode=None,
    knn_refill_force=None,
    both_parallel=False,
    mc_use_alias=False,
    ratio_smoothing="NONE",
    secondary_p_stop=None,
    secondary_stats_stop=None,
    secondary_null="full"
) -> dict
```

#### Parameters

**Spatial Configuration**:
- **`coords`** *(array-like of shape (N, 2))*: Projected planar coordinates for each area. Required unless a precomputed `handle` is supplied, and must be omitted when one is: a handle owns its geometry, so passing `coords` beside `handle` raises `ValueError` instead of being silently ignored.
- **`adjacency`** *(tuple)*: Spatial adjacency in CSR format `(row_ptr, col_idx)`.
  - Use `flexscanx.graph.csr_from_pysal(...)` if you start from a `libpysal.weights.W` object.
  - In `knn_method="graph"` mode, CSR is validated (size, bounds, no self-loops/duplicates, symmetry)
  - In `knn_method="graph"` mode, CSR adjacency is required unless a precomputed `handle` is supplied; passing it beside `handle` raises `ValueError`
  - If `knn_method="geometry"`, adjacency is built from coords internally

**Data**:
- **`observed`** *(array-like of shape (N,))*: Observed counts per area. Required.
- **`expected`** *(array-like of shape (N,), optional)*: Expected counts under null. Required for Poisson model only.
- **`population`** *(array-like of shape (N,), optional)*: Population denominators. Required for Binomial model.
- **`names`** *(array-like of shape (N,), optional)*: Area labels attached as a `name` list beside each cluster's `area`. Default `None`: no `name` key is emitted, because the 1-based area identifiers are already carried by `area`. (`scan_from_geodataframe()` is the exception: it defaults `names` to the string form of `id_col`.)

**Model Configuration**:
- **`model`** *{"POISSON", "BINOMIAL"}*: Statistical model (default: "POISSON").
  - `"POISSON"`: Count data with expected rates
  - `"BINOMIAL"`: Proportion data with population denominators (`population` required)
- **`scanmethod`** *{"FLEXIBLE", "CIRCULAR"}*: Cluster shape constraint (default: "FLEXIBLE").
  - `"FLEXIBLE"`: Irregularly shaped clusters respecting adjacency
  - `"CIRCULAR"`: Circular/elliptical clusters based on distance
- **`stattype`** *{"RESTRICTED", "ORIGINAL"}*: Likelihood ratio type (default: "RESTRICTED").
  - `"RESTRICTED"`: More conservative, reduces false positives (recommended)
  - `"ORIGINAL"`: Classic formulation

**Cluster Configuration**:
- **`clustertype`** *{"HOT", "COLD", "BOTH"}*: Type of clusters to detect (default: "HOT").
  - `"HOT"`: High-rate clusters (excess cases)
  - `"COLD"`: Low-rate clusters (deficit)
  - `"BOTH"`: Both types detected separately

For `model="POISSON"`, every `expected` value must be finite and strictly
positive. Zero-exposure rows are rejected before scan inference.
- **`K`** *(int, optional)*: KNN search horizon (candidate universe size per center). Default: `min(15, N)`.
  - **Equidistant neighbours.** A centre's universe is its `K` nearest areas (itself included) ordered by Euclidean distance, ties broken by area index (row order). When the `K`-th and `(K+1)`-th nearest neighbours are exactly equidistant, which is routine on regular lattices and coarsely rounded coordinates, the universe boundary falls inside a tied group and which neighbour is admitted depends on the input row order; the reported cluster, statistic and p-value can therefore change under a relabelling (measured on square grids: 2.9 % of relabellings). `build_static()`, `scan()`, `scan_series()` and `calibrate_llr_threshold()` emit a `UserWarning` naming the number of affected centres (`describe_static(handle)["knn_tie_boundary_centers"]`). Remedies: choose a `K` that does not split a tied distance group, or break the ties with a negligible coordinate jitter. Results are always reproducible for a fixed input order and seed. Without boundary ties the reported cluster and statistic are independent of the row order (measured over 24 model/method/statistic/tail cells with three random relabellings each); the Monte Carlo rank and p-value are not, because the counter-based null stream is keyed by area index, so a relabelled input draws a different, equally valid null sample and its p-value moves within Monte Carlo error.
- **`clustersize`** *(int, optional)*: Maximum cluster cardinality. Default `None` resolves to 15 internally. If `clustersize > K`, effective clustersize is capped to `K`; a warning is emitted only when `clustersize` was passed explicitly (the default is capped silently, since the caller never chose it).
- **`clusterradius`** *(float)*: Maximum centre-radius (default: inf = no
  limit). This is the largest distance from the candidate's centre to any member;
  it is not the cluster diameter, so a reported cluster `max_dist` can be up to
  twice this bound.
- **`ralpha`** *(float in [0, 1])*: Restricted LLR threshold for admissibility (default: 0.2).
  - Lower values = more conservative (fewer false positives)
  - Typical range: [0.2, 0.3]

**Monte Carlo Configuration**:
- **`simcount`** *(int)*: Number of Monte Carlo simulations for p-value (default: 999).
  - 0 = skip Monte Carlo (deterministic scan only)
  - 199 = fast, less precise p-values
  - 999 = recommended balance
  - 9999 = high precision
  - **Interrupting a scan.** A running scan (the observed enumeration and the Monte Carlo replicates alike) checks for a pending signal about every 50 ms on the thread that called it, so `Ctrl-C` raises `KeyboardInterrupt` promptly and the worker threads are cancelled and joined. Two limits: a single centre's DFS is not interruptible mid-way (the check runs between centres and between replicates), and with `both_parallel=True` the two directional scans run on helper threads, so the interrupt is honoured only when they return.
<!-- BEGIN GENERATED: scan_mc_rantype_policy -->
- **`rantype`** *(str, optional)*: Monte Carlo null distribution. Default `None` resolves to `"MULTINOMIAL"` for `model="POISSON"` and `"BINOMIAL"` for `model="BINOMIAL"`. Valid explicit values are `{"MULTINOMIAL", "POISSON"}` for Poisson and `{"BINOMIAL"}` for Binomial.
<!-- END GENERATED: scan_mc_rantype_policy -->
- **`mc_early_stop`** *(bool)*: Enable one-sided HOT/COLD Monte Carlo early
  stopping (default: False). BOTH scans exhaust `simcount` so their aligned
  per-replicate two-sided maximum has a valid stopping basis.
- **`mc_alpha`** *(float)*: Significance threshold for early stopping (default: 0.05).

**Secondary Clusters**:
- **`secondary`** *(int or "ALL", optional)*: Number of secondary clusters to enumerate.
  - `None`: Primary cluster only
  - `int`: Enumerate N secondary clusters
  - `"ALL"`: Enumerate all non-overlapping clusters

**Performance**:
- **`n_jobs_centers`** *(int, optional)*: Number of threads for parallel center scanning.
  - `None`: Auto-detect (smart default based on N)
  - `1`: Single-threaded
  - `> 1`: Use specified threads
- **`handle`** *(optional)*: Precomputed static handle from `build_static()`.
- **`knn_method`** *{"graph", "geometry"}*: Method for building KNN structure.
- **`knn_mode`** *{"K_REFILL", "K_FIXED"}*: KNN candidate selection mode.
- **`knn_refill_force`** *(bool)*: Force KNN refill on large `N`.
- **`both_parallel`** *(bool)*: Parallelize HOT and COLD branches when `clustertype="BOTH"` by running the two directional scans on two Python threads. It forces `n_jobs_centers = 1` per direction (recorded in `scan_settings`), so the run uses two threads in total however many were requested; with `n_jobs_centers > 1` the default sequential path, whose worker family serves each direction in turn, is faster (measured 2026-09-07 on a 49-area exact FLEXIBLE scan with 999 replicates: 0.73 s at 8 threads sequentially against 3.3 s with `both_parallel=True`). Leave it `False` unless you deliberately want single-threaded directional scans overlapped.

**Reproducibility**:
- **`seed`** *(int, optional)*: Random seed for Monte Carlo simulations. Valid range: `[0, 2147483647]`.

**Debugging**:
- **`debug_prune`** *(bool)*: Emit one concise per-scan diagnostic summary to stderr with scan method/model/tail and pruning counters (default: False).

**Advanced**:
- **`mc_progress`** / **`mc_progress_every`**: Monte Carlo progress callback and callback cadence.
- **`enable_center_ub_gating`** *(bool, optional)*: Opt in to center-level optimistic bound pruning. **Off by default since the 2026-06 audit** — the greedy bound is heuristic and enabling it may return suboptimal clusters. Enabling either heuristic on a FLEXIBLE scan emits a `UserWarning` at the call site (the 2026-09-02 audit measured `enable_frame_ub_prune` losing the true maximum in 144 of 7,056 scans).
- **`cache_candidate_llr`** *(bool, optional)*: Cache best-next candidate LLR evaluations.
- **`enable_frame_ub_prune`** *(bool, optional)*: Opt in to per-frame optimistic bound pruning. **Off by default since the 2026-06 audit** — the greedy bound is not a valid upper bound and enabling it may return suboptimal clusters.
- **`scheduler_backend`** *{"ATOMIC", "WORK_STEAL"}*: Center scheduler backend.
- **`mc_use_alias`** *(bool)*: Use Vose alias sampling for the MULTINOMIAL Monte Carlo null (a speed option for large `simcount`). The alias sampler consumes the random stream differently from the default `std::discrete_distribution` sampler, so for the same `seed` it draws a *different* null sample: p-values agree within Monte Carlo error but not bit-for-bit, and `scan_settings["mc_use_alias"]` records which path ran. `calibrate_llr_threshold()` has its own Python alias implementation and draws from numpy's generator, so its threshold is likewise not draw-for-draw aligned with a scan's null.
- **`ratio_smoothing`** *{"NONE", "HALDANE", "EB"}*: Output-only smoothing for reported ratios. `scan_settings["ratio_smoothing"]` records the requested mode. Each cluster also records `ratio_smoothing_applied` as one of `"NONE"`, `"HALDANE"`, or `"EB"`; if empirical-Bayes smoothing was requested but is unavailable for that cluster, the applied value is `"NONE"`.
- **`secondary_p_stop`** / **`secondary_stats_stop`** *(float, optional)*: Early-stop thresholds for secondary enumeration.
  - Signature default is `None`
  - If `secondary` is requested and `secondary_p_stop` is omitted, flexscanx forwards an effective stop of `1.0` (no practical p-value pruning)
- **`secondary_null`** *{"full", "conditional"}*: Monte-Carlo null distribution each secondary cluster is ranked against (no effect on a one-direction MLC/primary). For a `clustertype="BOTH"` result, the BOTH envelope primary is always ranked against the combined full-region two-sided null; this setting applies only to directional secondaries.
  - `"full"` (default, paper-faithful): rank every secondary against the same unconditional full-region null computed for the primary — "as if it were the primary cluster" (DEF-STOP; Otani & Takahashi 2021 §2.4). The secondary window search still excludes already-detected areas (non-overlap); only the reference null is the full-region one.
  - `"conditional"`: recompute the null over only the non-overlapping windows. Its maxima are ≤ the full-region maxima, so secondary p-values are anti-conservative. This was flexscanx's behaviour through 0.0.2; pass it to reproduce the old secondary p-values.

#### Returns

**`result`** *(dict)*: Scan result.
- For single direction (`HOT` or `COLD`): keys include `area`, `n_case`, `expected`, `stats`, `rank`, optional `pval`, model-specific `Rate_Ratio`/`Risk_Ratio`, optional `Rate_Ratio_CI`/`Risk_Ratio_CI`, optional `secondaries`, and required top-level `scan_settings`.
  - `rank` is the Monte Carlo rank among simulated maxima when `simcount > 0`; it is `1` when Monte Carlo is disabled.
  - `scan_settings["simcount_requested"]` is the requested Monte Carlo budget and `scan_settings["simcount_actual"]` is the number actually completed for the primary cluster. The latter can be smaller when one-sided early stopping fires.
  - Each entry of `secondaries` carries its own `simcount_actual`, the completed replicates behind that cluster's `pval` (`pval == rank / (simcount_actual + 1)`). It equals the primary's value except under `secondary_null="conditional"` combined with `mc_early_stop=True`, where every secondary's conditional null stops independently and the primary's count would understate the denominator.
  - `Rate_Ratio` / `Risk_Ratio` may be `None` for degenerate or empty outputs.
- For `BOTH`: keys are `type="BOTH"`, `primary_type`, `primary`, `hot`, `cold`, and required top-level `scan_settings`. `primary["pval"]` is the inferential two-sided p-value; `primary["pval_hot"]` and `primary["pval_cold"]` retain the diagnostic one-sided branch values. Its `simcount_actual` equals the requested budget because aligned two-sided null maxima require exhausting both directions.

#### Examples

**Basic Poisson HOT Scan**:
```python
import flexscanx as flx
import numpy as np

# Data: 100 areas on a grid
coords = np.array([[i, j] for i in range(10) for j in range(10)])
observed = np.random.poisson(10, size=100)
expected = np.ones(100) * 10

# Build adjacency
row_ptr, col_idx = flx.csr_knn_from_coords(coords, k=8)

# Run scan
result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    model="POISSON",
    clustertype="HOT",
    K=20,
    simcount=999,
    seed=42
)

print(f"Detected cluster: {result['area']}")
print(f"LLR = {result['stats']:.2f}, p-value = {result['pval']:.4f}")
rr = result.get("Rate_Ratio")
if rr is not None:
    print(f"Relative Risk = {rr:.2f}")
```

**BOTH Scan (HOT + COLD)**:
```python
result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    clustertype="BOTH",  # Detect both types
    K=20,
    simcount=999
)

# Result contains HOT and COLD info
print(f"HOT cluster: {result['hot'].get('area', [])}")
print(f"COLD cluster: {result['cold'].get('area', [])}")
print(f"Primary direction: {result['primary_type']}")
```

**Enumerate All Clusters**:
```python
result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    K=20,
    secondary="ALL"  # Enumerate all non-overlapping
)

# Flatten and export
from flexscanx.summary import flatten_clusters, write_clusters_csv
clusters = flatten_clusters(result)
write_clusters_csv(result, "all_clusters.csv")
```

**Binomial Model**:
```python
# Proportion data (e.g., disease prevalence)
cases = np.array([15, 8, 20, 12, ...])  # observed cases
population = np.array([100, 80, 150, 95, ...])  # population sizes

result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=cases,
    population=population,  # Use population instead of expected
    model="BINOMIAL",
    clustertype="HOT",
    K=15
)
```

**Fast Scan (No Monte Carlo)**:
```python
# Deterministic scan only (no p-value)
result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    K=30,
    simcount=0  # Skip Monte Carlo for speed
)

# Use calibrated threshold instead
from flexscanx import calibrate_llr_threshold
cal = calibrate_llr_threshold(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    K=30,
    n_sims=999,
    alpha=0.05
)
tau = cal['tau']
is_significant = result['stats'] >= tau
```

---

<!-- BEGIN GENERATED: enumerating_all_clusters -->
### Enumerating All Clusters

Use `scan(..., secondary="ALL")` to enumerate the primary cluster plus all
non-overlapping secondary clusters. The v0.0.1 root API does not expose a
separate enumerator wrapper for this mode.
<!-- END GENERATED: enumerating_all_clusters -->

---

## Time-Series Analysis

### `scan_series()`

Detect clusters across multiple time points with efficient reuse and common random numbers.

```python
flexscanx.scan_series(
    *,
    observed,
    expected=None,
    population=None,
    coords=None,
    adjacency=None,
    names=None,
    model="POISSON",
    scanmethod="FLEXIBLE",
    stattype="RESTRICTED",
    clustertype="HOT",
    clustersize=None,
    clusterradius=None,
    ralpha=0.2,
    K=None,
    simcount=999,
    rantype=None,
    secondary=None,
    crn=True,
    seed=None,
    n_jobs_centers=None,
    handle=None,
    mc_early_stop=False,
    mc_alpha=0.05,
    mc_progress=None,
    mc_progress_every=None,
    enable_center_ub_gating=None,
    cache_candidate_llr=None,
    scheduler_backend=None,
    knn_method=None,
    knn_mode=None,
    knn_refill_force=None,
    both_parallel=False,
    enable_frame_ub_prune=None,
    n_jobs_time=None,
    warm_start=False,
    times=None,
    debug_prune=False,
    mc_use_alias=False,
    ratio_smoothing="NONE",
    secondary_p_stop=None,
    secondary_stats_stop=None,
    secondary_null="full"
) -> dict
```

#### Parameters

**Data** *(time-varying)*:
- **`observed`** *(array-like of shape (T, N))*: Observed counts at each time point.
- **`expected`** *(array-like of shape (T, N), optional)*: Expected counts (Poisson only).
- **`population`** *(array-like of shape (T, N), optional)*: Populations (Binomial; required for BINOMIAL model).

All other spatial/model parameters same as `scan()`.

**Time-Series Specific**:
- **`n_jobs_time`** *(int, optional)*: Parallel worker threads across time points.
- **`crn`** *(bool)*: Use Common Random Numbers across time (default: True).
  - Reduces variance when comparing clusters between time points
  - Recommended for stability analysis
  - **Seed note**: When `crn=True` and `seed=None`, an internal fixed seed (`_CRN_DEFAULT_SEED`) anchors the CRN stream for reproducibility. For full cross-run reproducibility pass an explicit `seed=`.
- **`times`** *(sequence, optional)*: Labels for time points in output; defaults to `range(T)`. Labels are stored as strict-JSON scalars: numpy integers and floats become Python numbers and datetime-like labels (`datetime`, `pandas.Timestamp`, `numpy.datetime64`) become ISO 8601 strings, so `json.dumps(result, allow_nan=False)` always succeeds.
- **`warm_start`** *(bool)*: Reuse priority centers across sequential time slices.
- **`debug_prune`** *(bool)*: Emit one concise per-scan core diagnostic summary to stderr.
- **`mc_use_alias`** *(bool)*: Use Vose alias sampling for the MULTINOMIAL Monte Carlo null; see `scan()` — the same seed draws a different null sample than the default sampler, so p-values agree within Monte Carlo error only.
- **`ratio_smoothing`** *{"NONE", "HALDANE", "EB"}*: Reporting-only smoothing for rate estimates (does not change scan statistic or p-values). The requested value remains in `scan_settings["ratio_smoothing"]`; each cluster's `ratio_smoothing_applied` is one of `"NONE"`, `"HALDANE"`, or `"EB"`, with an unavailable requested EB fallback reported as applied `"NONE"`.

<!-- BEGIN GENERATED: scan_series_returns -->
#### Returns

**`result`** *(dict)*: Series result with keys:
- **`per_time`** *(list of dict)*: List of T scan results (one per time point;
  same directional/BOTH payloads as `scan()`, but per-time entries omit `scan_settings`).
- **`times`** *(list)*: Time labels (provided via `times` or default `range(T)`).
- **`scan_settings`** *(dict)*: Shared provenance/configuration metadata for the series run.
<!-- END GENERATED: scan_series_returns -->

#### Examples

**Basic Time-Series Scan**:
```python
# Data: 12 months × 100 areas
T, N = 12, 100
observed_TN = np.random.poisson(10, size=(T, N))
expected_TN = np.ones((T, N)) * 10

result = flx.scan_series(
    observed=observed_TN,
    expected=expected_TN,
    coords=coords,
    adjacency=(row_ptr, col_idx),
    model="POISSON",
    clustertype="HOT",
    K=20,
    simcount=199,  # Faster for time-series
    crn=True,  # Stabilize temporal comparisons
    seed=42
)

# Inspect per-time results
for t, t_result in enumerate(result['per_time']):
    if t_result['area']:
        print(f"Month {t}: Cluster with {len(t_result['area'])} areas, "
              f"LLR={t_result['stats']:.2f}")
```

**Both HOT and COLD Over Time**:
```python
result = flx.scan_series(
    observed=observed_TN,
    expected=expected_TN,
    coords=coords,
    adjacency=(row_ptr, col_idx),
    clustertype="BOTH",
    K=20,
    simcount=199,
    crn=True
)

# Each time point has HOT and COLD
for t, t_result in enumerate(result['per_time']):
    hot_areas = t_result.get('hot', {}).get('area', [])
    cold_areas = t_result.get('cold', {}).get('area', [])
    print(f"t={t}: {len(hot_areas)} HOT, {len(cold_areas)} COLD")
```

---

## `scan_from_geodataframe()`

Convenience wrapper for polygon GeoDataFrames. It builds contiguity adjacency, extracts centroid
coordinates, runs `scan()`, and remaps returned cluster IDs into the GeoDataFrame `id_col` space.

```python
flexscanx.scan_from_geodataframe(
    gdf,
    *,
    id_col,
    contiguity="queen",
    K=None,
    clusterradius=float('inf'),
    knn_mode="K_REFILL",
    knn_refill_force=False,
    names=None,
    **kwargs
) -> dict
```

**Contract**:
- `gdf.crs` must be projected with linear units. Geographic or missing CRS inputs raise `ValueError`.
- `id_col` is required and must contain unique positive integer area IDs.
- Returned `area`, `from_area`, and `to_area` values are remapped into the `id_col` value space.
- If `names` is omitted, it defaults to the string form of `id_col`.

**Example**:
```python
import geopandas as gpd
import flexscanx as flx

gdf = gpd.read_file("regions.geojson").to_crs("EPSG:3857")
gdf["observed"] = [...]
gdf["expected"] = [...]

result = flx.scan_from_geodataframe(
    gdf,
    id_col="region_id",
    contiguity="queen",
    observed=gdf["observed"],
    expected=gdf["expected"],
    K=20,
)

print(result["area"])  # region_id values, not 1..N row positions
```

---

## Sequence Analysis (Temporal Trajectories)

Analyze cluster membership patterns over time.

### `extract_sequences()`

Convert `scan_series()` results to sequence matrix for trajectory analysis.

```python
from flexscanx.sequence import extract_sequences

extract_sequences(
    result,
    area_ids=None,
    include_type="both"
) -> Tuple[ndarray, List[int]]
```

#### Parameters

- **`result`** *(dict)*: Output from `scan_series()`.
- **`area_ids`** *(list of int, optional)*: Specific area IDs to extract.
  - If None, extracts all areas that appeared in at least one cluster
- **`include_type`** *{"hot", "cold", "both"}*: Which cluster types to encode.
  - `"hot"`: Code HOT=1, others=0
  - `"cold"`: Code COLD=-1, others=0
  - `"both"`: Code HOT=1, COLD=-1, none=0

If an area appears in both HOT and COLD clusters at one time point, `"both"`
encodes it as neutral `0`. This conflict rule is invariant to list-form cluster
ordering.

#### Returns

- **`sequences`** *(ndarray of shape (N_areas, T_timepoints))*: Membership matrix.
  - Values: 1 (HOT), -1 (COLD), 0 (none)
- **`area_ids`** *(list of int)*: Area IDs corresponding to rows (sorted).

#### Example

```python
from flexscanx.sequence import extract_sequences

result = flx.scan_series(...)  # Time-series scan

# Extract sequences for trajectory analysis
sequences, area_ids = extract_sequences(result, include_type='both')

print(f"Tracking {len(area_ids)} areas across {sequences.shape[1]} time points")
print(f"Area {area_ids[0]} trajectory: {sequences[0]}")
# Example output: [0, 1, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0]
# Interpretation: HOT at t=1,2,7; COLD at t=4; none otherwise
```

---

### Distance Metrics

#### `seq_dist_hamming()`

Compute pairwise Hamming distance between sequences.

```python
from flexscanx.sequence import seq_dist_hamming

seq_dist_hamming(
    sequences,
    weights=None,
    norm=True,
    condensed=False
) -> ndarray
```

**Parameters**:
- **`sequences`** *(ndarray of shape (N, T))*: Sequence matrix.
- **`weights`** *(ndarray of shape (T,), optional)*: Per-position weights. Must be non-negative and sum to a positive value.
- **`norm`** *(bool)*: Normalize by sequence length (default: True).
  - True: distances in [0, 1] (proportion of mismatches)
  - False: raw mismatch counts
- **`condensed`** *(bool)*: When True, return a 1-D condensed upper-triangle array compatible with `scipy.spatial.distance.squareform` instead of the full N×N matrix (default: False). Halves memory usage for large N.

**Returns**: 
- Shape `(N, N)` symmetric distance matrix (when `condensed=False`).
- Shape `(N*(N-1)//2,)` condensed upper-triangle vector (when `condensed=True`).

**Memory warning**: For N > ~7 900 areas the full N×N matrix exceeds 500 MB and a `UserWarning` is emitted. Use `condensed=True` to halve memory usage, or subsample `sequences` first.

**Example**:
```python
# Compute trajectory similarity (full matrix)
distances = seq_dist_hamming(sequences, norm=True)

# Find areas most similar to area 0
similar_idx = np.argsort(distances[0])[1:6]  # Top 5 (excluding self)
similar_areas = [area_ids[i] for i in similar_idx]
print(f"Areas most similar to {area_ids[0]}: {similar_areas}")

# Memory-efficient condensed form (for large N)
cdist = seq_dist_hamming(sequences, condensed=True)
# Expand when needed:
# from scipy.spatial.distance import squareform
# full = squareform(cdist)
```

---

#### `seq_dist_matrix_info()`

Summary statistics for a distance matrix.

```python
from flexscanx.sequence import seq_dist_matrix_info

seq_dist_matrix_info(D) -> dict
```

**Returns**: Dict with `'n'`, `'mean'`, `'std'`, `'min'`, `'max'`, `'median'`.

---

### Pattern Analysis

#### `seq_freq()`

Frequency table of unique sequence patterns.

```python
from flexscanx.sequence import seq_freq

seq_freq(
    sequences,
    top_n=10,
    normalize=True
) -> pd.DataFrame
```

**Parameters**:
- **`sequences`** *(ndarray)*: Sequence matrix.
- **`top_n`** *(int)*: Number of most frequent patterns to return.
- **`normalize`** *(bool)*: Include proportions and cumulative.

**Returns**: DataFrame with columns `['sequence', 'count', 'proportion', 'cumulative']`.

**Example**:
```python
freq_table = seq_freq(sequences, top_n=10)
print("Most common membership patterns:")
print(freq_table)

# Output:
#   sequence          count  proportion  cumulative
# 0 (0,0,0,0,...)      45      0.45        0.45
# 1 (1,1,1,0,...)      12      0.12        0.57
# ...
```

---

#### `seq_trans()`

State transition probability matrix.

```python
from flexscanx.sequence import seq_trans

seq_trans(
    sequences,
    lag=1,
    normalize=True
) -> pd.DataFrame
```

**Parameters**:
- **`lag`** *(int)*: Time lag for transitions (default: 1 = next time point).
- **`normalize`** *(bool)*: Return probabilities (True) or counts (False).

**Returns**: DataFrame with rows=from states, columns=to states.

**Example**:
```python
trans = seq_trans(sequences, lag=1, normalize=True)
print("\nTransition probabilities:")
print(trans)

# Interpret specific transitions
print(f"P(HOT → HOT) = {trans.loc[1, 1]:.3f}  (persistence)")
print(f"P(HOT → none) = {trans.loc[1, 0]:.3f}  (fade out)")
print(f"P(none → HOT) = {trans.loc[0, 1]:.3f}  (emergence)")
```

---

#### `seq_entropy()`

Shannon entropy of sequence distribution.

```python
from flexscanx.sequence import seq_entropy

seq_entropy(
    sequences,
    by_time=False,
    base=2.0,
    normalize=True
) -> Union[float, ndarray]
```

**Parameters**:
- **`by_time`** *(bool)*: 
  - False: overall sequence diversity (scalar)
  - True: per-position state diversity (array of length T)
- **`base`** *(float)*: Logarithm base (2.0=bits, np.e=nats).
- **`normalize`** *(bool)*: With `by_time=True`, divide by the alphabet's
  maximum entropy so values lie in [0, 1] (default: True). The overall
  sequence-pattern entropy is not normalized by this flag.

**Returns**: 
- Scalar entropy if `by_time=False`
- Array of length T if `by_time=True`

**Example**:
```python
# Overall pattern diversity
ent_overall = seq_entropy(sequences, by_time=False)
print(f"Pattern entropy: {ent_overall:.3f} bits")

# Per-time-point diversity
ent_by_time = seq_entropy(sequences, by_time=True)

import matplotlib.pyplot as plt
plt.plot(ent_by_time, 'o-')
plt.xlabel('Time Point')
plt.ylabel('Normalized entropy')
plt.title('Temporal Evolution of State Diversity')
plt.show()
```

---

### Sequence Visualization

#### `seq_iplot()`

Index plot: individual trajectories as colored horizontal bars.

```python
from flexscanx.sequence import seq_iplot

seq_iplot(
    sequences,
    area_ids=None,
    time_labels=None,
    colors=None,
    figsize=(12, 8),
    title="Cluster Membership Trajectories",
    show_grid=True,
    max_display=50
) -> Tuple[Figure, Axes]
```

**Parameters**:
- **`sequences`** *(ndarray of shape (N, T))*: Sequence matrix.
- **`area_ids`** *(list, optional)*: Area labels for y-axis.
- **`time_labels`** *(list, optional)*: Time labels for x-axis.
- **`colors`** *(dict, optional)*: Custom colors for states `{-1: 'blue', 0: 'white', 1: 'red'}`.
- **`max_display`** *(int)*: Limit to first N sequences (default: 50).

**Returns**: `(fig, ax)` tuple.

**Example**:
```python
fig, ax = seq_iplot(
    sequences[:30],  # First 30 areas
    area_ids=area_ids[:30],
    time_labels=['Jan', 'Feb', 'Mar', ..., 'Dec'],
    title="HOT/COLD Cluster Membership Over Time"
)
plt.savefig('trajectories.png', dpi=150)
```

---

#### `seq_dplot()`

Distribution plot: stacked bars showing proportion in each state over time.

```python
from flexscanx.sequence import seq_dplot

seq_dplot(
    sequences,
    time_labels=None,
    colors=None,
    figsize=(10, 6),
    title="Cluster Membership Distribution Over Time",
    show_entropy=False
) -> Tuple[Figure, Axes]
```

**Parameters**:
- **`show_entropy`** *(bool)*: Overlay entropy line on secondary y-axis.

**Example**:
```python
fig, ax = seq_dplot(
    sequences,
    time_labels=month_labels,
    show_entropy=True  # Show diversity measure
)
```

---

#### `seq_fplot()`

Frequency plot: horizontal bars of most common patterns.

```python
from flexscanx.sequence import seq_fplot

seq_fplot(
    sequences,
    top_n=10,
    figsize=(12, 6),
    title="Most Frequent Sequence Patterns"
) -> Tuple[Figure, Axes]
```

**Example**:
```python
fig, ax = seq_fplot(sequences, top_n=15)
# Shows top 15 patterns with notation: H=HOT, C=COLD, ·=none
```

---

## Graph and Adjacency

### `csr_from_dense()`

Build a CSR adjacency from a 0/1 dense matrix.

```python
import flexscanx as flx  # or: from flexscanx.graph import csr_from_dense

flx.csr_from_dense(adj_mat) -> tuple[list[int], list[int]]
```

**Parameters**:
- **`adj_mat`** *(iterable of iterables of int)*: Square 0/1 adjacency matrix. Every entry must be exactly `0` or `1` (integers, booleans, or the floats `0.0`/`1.0`). Diagonal entries never become edges.

**Raises**: `ValueError` if the matrix is not square, if any entry is outside `{0, 1}` (weights, negative values, `NaN`, strings; the message lists the offending `(row, column)` positions), or if it is not symmetric. The C++ core treats adjacency as a pure relation and requires an undirected (symmetric) graph; weighted or asymmetric input previously produced silently wrong or silently binarised results — both are now rejected explicitly. Binarise a weighted matrix yourself (for example `(W > 0).astype(int)`) if that is the intended graph.

**Returns**: `(row_ptr, col_idx)` CSR tuple.

---

### `csr_knn_from_coords()`

Build CSR adjacency from coordinates using KNN.
Also available as `flexscanx.csr_knn_from_coords` (top-level namespace since v0.0.1).

```python
import flexscanx as flx  # or: from flexscanx.graph import csr_knn_from_coords

flx.csr_knn_from_coords(
    coords,
    k,                      # required, no default
    *,
    radius=float('inf'),    # no distance cutoff by default
    include_self=False,
    backend='auto',
    metric='euclidean'
) -> tuple[list[int], list[int]]
```

**Parameters**:
- **`coords`** *(array-like of shape (N, 2))*: Coordinates.
- **`k`** *(int)*: Number of nearest neighbors (required).
- **`radius`** *(float)*: Maximum distance cutoff (default: `float('inf')` = no limit).
  For `metric='haversine'`, pass radius in **radians** (BallTree convention — use
  `numpy.radians()` to convert from degrees).
- **`include_self`** *(bool)*: Include self-loops in output (default: False).
- **`backend`** *{'auto', 'kdtree', 'balltree'}*: Search algorithm. `'auto'` and `'kdtree'` use SciPy's `cKDTree` (a runtime dependency); `'balltree'` uses scikit-learn's `BallTree`, which is the optional extra `pip install flexscanx[knn]`.
- **`metric`** *{'euclidean', 'haversine'}*: Distance metric.
  - `'euclidean'`: Planar coordinates.
  - `'haversine'`: Longitude/latitude coords in degrees. Requires scikit-learn (`pip install flexscanx[knn]`).

`csr_knn_from_coords(..., metric="haversine")` uses longitude/latitude only to
build the graph. In particular, do not pass those longitude/latitude coordinates to `scan(coords=...)`:
the scan core treats `coords` as Euclidean. Project the
coordinates first (or supply an appropriate planar coordinate array to `scan`)
while using the geographic graph if that is the desired adjacency.

**Returns**: `(row_ptr, col_idx)` CSR tuple.

**Example**:
```python
import flexscanx as flx

# Planar KNN
row_ptr, col_idx = flx.csr_knn_from_coords(coords, k=8)

# Geographic (haversine)
coords_lonlat = np.array([[lon, lat], ...])  # degrees
row_ptr, col_idx = flx.csr_knn_from_coords(
    coords_lonlat,
    k=50,
    metric='haversine',
    radius=np.radians(500 / 6371)  # ~500 km converted to radians
)
```

---

### `csr_from_geodataframe_contiguity()`

Build CSR from polygon contiguity (Queen or Rook).

```python
from flexscanx.graph import csr_from_geodataframe_contiguity

csr_from_geodataframe_contiguity(
    gdf,
    *,
    method="queen",
    queen_include_intersects=False
) -> Tuple[list[int], list[int]]
```

**Parameters**:
- **`gdf`** *(GeoDataFrame)*: Polygons with geometry column.
- **`method`** *{'queen', 'rook'}*: Contiguity definition, evaluated geometrically with shapely.
  - `'queen'`: the polygons touch at any point (a shared corner, or a vertex of one lying anywhere on an edge of the other), with disjoint interiors.
  - `'rook'`: the polygons share a boundary segment of non-zero length.
- **`queen_include_intersects`** *(bool)*: Also treat overlapping/intersecting
  polygons as Queen neighbors (default: False).

Both predicates are purely geometric and do not depend on shared vertices, so the
graph is the same whether or not libpysal is installed. (libpysal's own
`Queen`/`Rook` builders are vertex-based and disagree with these predicates on
meshes that are not vertex-congruent; use `csr_from_pysal` if you specifically want
libpysal's definition.)

**Returns**: `(row_ptr, col_idx)` CSR tuple of built-in Python integers, suitable
for JSON serialization.

**Example**:
```python
import geopandas as gpd
import numpy as np
from flexscanx.graph import csr_from_geodataframe_contiguity

gdf = gpd.read_file("regions.geojson").to_crs("EPSG:3857")
centroids = gdf.geometry.centroid
coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
row_ptr, col_idx = csr_from_geodataframe_contiguity(gdf, method='queen')

result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=gdf['cases'],
    expected=gdf['expected']
)
```

---

## Validation and Metrics

### `calibrate_llr_threshold()`

Calibrate LLR threshold for a target false alarm rate.

Under the default multinomial null, calibration fixes the simulated total at `sum(observed)` and redistributes those cases in proportion to `expected`. Seeded results are cached on disk keyed by all calibration parameters (including the effective `rantype`), so subsequent calls with identical settings are instant. The cache directory (`$XDG_CACHE_HOME/flexscanx` or `~/.cache/flexscanx`) is created private (`0700`) and its entries are HMAC-protected with a key stored beside them; a directory found group- or world-writable is tightened and its entries discarded, and a directory owned by another user disables the cache with a warning, because the key would otherwise be as writable as the data it protects.

```python
from flexscanx import calibrate_llr_threshold

calibrate_llr_threshold(
    *,
    coords=None,
    adjacency=None,
    handle=None,
    observed,
    expected=None,        # Required for POISSON; not accepted for BINOMIAL
    population=None,      # Required for BINOMIAL; not accepted for POISSON
    model="POISSON",
    scanmethod="FLEXIBLE",
    stattype="RESTRICTED",
    clustertype="HOT",
    K,
    clustersize,
    clusterradius=float('inf'),
    n_sims=999,
    alpha=0.05,
    seed=None,
    knn_mode="K_REFILL",
    knn_refill_force=None,
    rantype=None,
    ralpha=0.2
) -> dict
```

**Model parameter rules** (mirrors `scan()`):

- **POISSON**: pass `expected=` (positive floats). Do not pass `population=`.
- **BINOMIAL**: pass `population=` (positive integer trial counts). Do not pass `expected=`.

**Constraints**: `n_sims` must be ≥ 1; passing `n_sims=0` raises `ValueError`.

**Returns**: Dict with keys `'tau'` (threshold LLR), `'mu'`, `'sigma'`, `'alpha'`, `'n_sims'`.

**Poisson example**:
```python
cal = calibrate_llr_threshold(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,   # POISSON: pass expected=
    model="POISSON",
    K=30, clustersize=15,
    n_sims=999,
    alpha=0.05,
    seed=42,
)
tau = cal["tau"]
print(f"LLR threshold at alpha=0.05: {tau:.2f}")

result = flx.scan(..., simcount=0)
is_significant = result["stats"] >= tau
```

**Binomial example**:
```python
cal = calibrate_llr_threshold(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=cases,
    population=population,  # BINOMIAL: pass population=, not expected=
    model="BINOMIAL",
    K=30, clustersize=15,
    n_sims=999,
    alpha=0.05,
    seed=42,
)
```

---

### `stability_across_K()`

Evaluate cluster stability as the KNN horizon K varies across adjacent values.
Uses `secondary="ALL"` (no MC) to build a full cluster pool per K, then computes
pairwise Jaccard-matching metrics between adjacent K settings.

```python
from flexscanx import stability_across_K

stability_across_K(
    *,
    coords,
    adjacency,
    observed,
    expected=None,          # Required for POISSON; not accepted for BINOMIAL
    population=None,        # Required for BINOMIAL; not accepted for POISSON
    K_values,               # e.g. [20, 30, 40, 50]  — at least one value required
    clustersize,
    clusterradius=float('inf'),
    model="POISSON",
    scanmethod="FLEXIBLE",
    stattype="RESTRICTED",
    clustertype="HOT",
    knn_mode="K_FIXED",
    knn_refill_force=False,
    seed=None
) -> dict
```

**Returns**: Dict with keys:
- `'K_values'` *(list of int)*: The K values tested.
- `'runs'` *(list of dict)*: Per-K metrics — `'K'`, `'pool_size'`, `'stats'`, `'pval'`.
- `'pairwise'` *(list of dict)*: Jaccard-matching metrics between each adjacent pair of K values: `'mean_jaccard'`, `'min_jaccard'`, `'coverage'`, `'count_diff'`, `'nA'`, `'nB'`, `'K1'`, `'K2'`.

---

---

### `find_stable_params_all()`

Automatically select the smallest (K, S) pair that produces stable cluster sets
when K and S are decreased from their starting values. Uses coordinate descent with
optional binary refinement.

```python
from flexscanx import find_stable_params_all

find_stable_params_all(
    *,
    coords,
    adjacency,
    observed,
    expected=None,           # POISSON; not accepted for BINOMIAL
    population=None,         # BINOMIAL; not accepted for POISSON
    K_start=None,            # Default: min(max(15, round(25% of N)), 500), clipped to N
    S_start=None,            # Default: 12% of N, with a two-area start for N >= 4 when feasible
    K_min=None,              # Default: max(2, min(8, round(25% of N)))
    S_min=None,              # Default: max(1, min(5, round(6% of N)))
    stepK=0.8,               # Geometric step factor for K descent
    stepS=0.8,               # Geometric step factor for S descent
    require_K_gt_S=True,     # Enforce K > S (recommended)
    model="POISSON",
    scanmethod="FLEXIBLE",
    stattype="RESTRICTED",
    clustertype="BOTH",
    ralpha=0.2,
    clusterradius=float("inf"),
    thresholds=None,         # StabilityThresholds dataclass (default: mean_jaccard>=0.90, etc.)
    binary_refine=True,
    seed=None,
    n_jobs_centers=None,
    secondary_stats_stop=None
) -> dict
```

**Returns**: Dict with keys:
- `'best'` *(dict)*: `{'K': int, 'S': int}` — the smallest stable pair found.
- `'trace'` *(list of dict)*: Per-step stability checks with `'phase'`, `'from'`, `'to'`, `'metrics'`.
- `'metrics'` *(dict)*: Pool sizes for each `(K, S)` pair evaluated.

**Example**:
```python
from flexscanx import find_stable_params_all

sel = find_stable_params_all(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    model="POISSON",
    clustertype="BOTH",
)
print(f"Stable parameters: K={sel['best']['K']}, S={sel['best']['S']}")

# Use selected parameters for the main scan
result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    K=sel["best"]["K"],
    clustersize=sel["best"]["S"],
    simcount=999,
    seed=42,
)
```

---

### `find_stable_params_all_series()`

Run `find_stable_params_all` independently per time slice and aggregate by median.

```python
from flexscanx import find_stable_params_all_series

find_stable_params_all_series(
    *,
    observed,        # shape (T, N)
    expected=None,   # shape (T, N); POISSON only
    population=None, # shape (T, N); BINOMIAL only
    coords,
    adjacency,
    **kwargs         # Forwarded to find_stable_params_all
) -> dict
```

**Returns**: Dict with keys `'best'` (median K/S), `'K_values'`, `'S_values'`, `'per_time'`.

---

### `grid_search_stable()`

Evaluate a full (K, S) grid and compute Jaccard-stability metrics vs the highest anchor.

```python
from flexscanx import grid_search_stable

grid_search_stable(
    *,
    coords,
    adjacency,
    observed,
    expected=None,
    population=None,
    K_grid,          # e.g. [20, 30, 40]
    S_grid,          # e.g. [10, 15, 20]
    model="POISSON",
    scanmethod="FLEXIBLE",
    stattype="RESTRICTED",
    clustertype="BOTH",
    ralpha=0.2,
    clusterradius=float('inf'),
    simcount=0,
    seed=None,
    n_jobs_centers=None,
    secondary_stats_stop=None
) -> pd.DataFrame
```

**Returns**: DataFrame with columns `K, S, pool_size, stats, pval, mean_jaccard, min_jaccard, coverage, count_diff`.
All grids must contain at least one feasible `K > S` cell; a wholly infeasible
grid raises `ValueError` instead of returning an ambiguous empty frame.

---

### `screen_stable_significant()`

Filter `grid_search_stable` output to rows that pass both stability thresholds and significance.

```python
from flexscanx import screen_stable_significant

screen_stable_significant(
    *,
    coords,
    adjacency,
    observed,
    expected=None,
    population=None,
    K_grid,
    S_grid,
    alpha=0.05,
    simcount=999,
    thresholds=None,  # StabilityThresholds dataclass
    **kwargs          # Forwarded to grid_search_stable
) -> pd.DataFrame
```

---

### `conductance()` / `boundary_contrast()`

Graph quality metrics for clusters.

```python
from flexscanx import conductance, boundary_contrast

phi = conductance(row_ptr, col_idx, membership)
bc = boundary_contrast(observed, expected, row_ptr, col_idx, membership)
```

**Parameters**:
- **`membership`** *(array-like of shape (N,))*: Binary membership (1=in cluster).

For `silhouette_graph`, an unreachable complement gives silhouette `+1` for a
connected cluster (the finite-distance limit), while a split cluster gives `-1`.

---

## Visualization

### `plot_cluster_membership()`

Choropleth map of cluster membership.

Use an `id_col` whose integer values match the result's area IDs. For plain `scan()` and
`scan_series()` results those IDs are 1-based row-order IDs; `scan_from_geodataframe()` remaps
them into the `id_col` space you supplied there. `plot_cluster_membership()`,
`plot_all_clusters()` and `plot_all_both_clusters()` convert the column like
`pool_to_geodataframe()`: integers and their decimal string form are accepted, any other value
raises `ValueError` naming the column, row and value, and a result whose cluster areas are all
absent from the column raises `ValueError` ("No cluster IDs matched ...") instead of drawing an
empty map. A cluster that is only partly present is drawn partly.

```python
from flexscanx import plotting

plotting.plot_cluster_membership(
    geodata,
    result,
    id_col,
    *,
    which='primary',
    ax=None,
    cmap='Reds',
    edgecolor='white',
    linewidth=0.5,
    missing_color='#f0f0f0',
    legend=False,
    title=None,
    alpha=1.0,
    save_path=None,
    dpi=300,
    return_fig=False
) -> Union[Axes, Tuple[Figure, Axes]]
```

**Example**:
```python
from flexscanx import plotting

fig, ax = plotting.plot_cluster_membership(
    gdf,
    result,
    id_col='id',
    title='Detected HOT Cluster',
    save_path='cluster_map.png',
    return_fig=True
)
```

---

### `plot_both_membership()`

Side-by-side maps for BOTH scans.

```python
plotting.plot_both_membership(
    geodata,
    result_both,
    id_col,
    *,
    titles=('HOT', 'COLD'),
    cmaps=('Reds', 'Blues'),
    edgecolor='white',
    linewidth=0.5,
    alpha=1.0,
    save_path=None,
    dpi=300,
    return_fig=False
)
```

---

### `plot_all_clusters()`

Grid of all detected clusters (primary + secondaries).

```python
plotting.plot_all_clusters(
    gdf,
    result,
    id_col='id',
    overlay=False,
    cols=3,
    label_stats=True,
    save_path=None,
    return_fig=False
)
```

**Parameters**:
- **`overlay`** *(bool)*: Overlay all on single map (True) vs grid (False).
- **`cols`** *(int)*: Number of columns in grid.
- **`label_stats`** *(bool)*: Add LLR/p-value to titles.

`result` must be one single-direction scan result with an `area` list and
list-valued secondary areas. `scan_series` results are not displayable here;
select one time slice. For a `BOTH` result use `plot_all_both_clusters()` so
HOT and COLD retain their separate visual semantics. With a non-`None`
`significance_alpha`, clusters whose p-value is missing (as with `simcount=0`)
are excluded; set it to `None` to show untested clusters.

When `return_fig=False`, plotting functions close only the figure they created;
an axis supplied by the caller remains open and owned by the caller.

---

### `choropleth()`

General-purpose choropleth for arbitrary values.

```python
plotting.choropleth(
    geodata,
    values_by_id,
    id_col,
    *,
    ax=None,
    cmap="OrRd",
    edgecolor="white",
    linewidth=0.5,
    missing_color="#f0f0f0",
    legend=True,
    legend_kwds=None,
    title=None,
    scheme=None,
    k=5,
    classification_kwds=None,
    vmin=None,
    vmax=None,
    save_path=None,
    dpi=300,
    return_fig=False
) -> Any
```

**Parameters**:
- **`values_by_id`** *(Series, dict, or sequence)*: Values keyed by area ID.
  Plain sequences and default-index Series are interpreted in 1-based area-id order.
- **`scheme`** *(str, optional)*: Classification scheme ('quantiles', 'equal_interval', etc.).
- **`vmin`/`vmax`** *(float, optional)*: Color scale limits.

---

### `export_cluster_report()`

Export one-page HTML/PDF/PNG report with cluster maps and a summary table.
Available at the root namespace since v0.0.1b.

```python
import flexscanx as flx                          # root namespace (recommended)
# or: from flexscanx.report import export_cluster_report

flx.export_cluster_report(
    geodata,                    # path or GeoDataFrame with integer id_col
    result,                     # dict from scan() — single direction or BOTH
    id_col,                     # name of the integer-ID column
    path,                       # output path (.html, .pdf, or .png)
    *,
    cols=3,                     # map grid columns
    max_clusters=None,          # cap clusters shown per direction
    title=None,
    figsize=None,
    hot_cmap="Reds",
    cold_cmap="Blues",
    table_max_rows=15,
    consensus_areas=None,       # optional consensus overlay (single-direction only)
    consensus_title=None,
    consensus_cmap="Greens"
)
```

**Supported formats** (by file extension):
- `.html`: embedded base64 PNG inside a self-contained HTML page
- `.pdf`: vector PDF
- `.png`: raster PNG at 300 DPI

**Requires**: `matplotlib` (`pip install flexscanx[plot]`) and `geopandas` (`pip install flexscanx[geo]`).

`scan_series` results are not supported directly: select one time slice before
exporting. Set `table_max_rows=0` to suppress the table panel deliberately.

For a `BOTH` result, the report shows the envelope primary table first. Its `P`
column is the combined-null two-sided inferential value and `P_hot`/`P_cold`
remain visible as directional diagnostics. Separate HOT and COLD one-sided
diagnostic tables retain every branch cluster, including clusters from the
non-winning direction.

**Example**:
```python
import flexscanx as flx

result = flx.scan(
    coords=coords, adjacency=(row_ptr, col_idx),
    observed=observed, expected=expected, K=20, simcount=999,
)
flx.export_cluster_report(gdf, result, id_col="region_id", path="report.html")
```

---

### `plot_consensus_frequency()`

Choropleth map of consensus sweep area-inclusion frequencies.

Use after `consensus_sweep()` to visualize which areas were most consistently included
across the parameter grid.

**Requires**: `matplotlib` (`pip install flexscanx[plot]`) and `geopandas` (`pip install flexscanx[geo]`). Both are optional dependencies; a clear `ImportError` is raised if either is missing.

```python
import flexscanx as flx

flx.plot_consensus_frequency(
    geodata,
    freq,
    id_col,
    *,
    ax=None,
    cmap="YlOrRd",
    legend=True,
    title=None,
    vmin=0.0,
    vmax=1.0,
    discrete_bins=None,
    edgecolor="white",
    linewidth=0.5,
    missing_color="#f0f0f0",
    save_path=None,
    dpi=300,
    return_fig=False
) -> Axes | tuple[Figure, Axes]
```

**Parameters**:
- **`geodata`** *(str or GeoDataFrame)*: Path or GeoDataFrame whose `id_col` values match area IDs.
- **`freq`** *(list of float)*: Per-area consensus frequencies in 1..N order, as returned by `consensus_sweep()["freq"]`.
- **`id_col`** *(str)*: Column in `geodata` containing the 1-based integer area IDs (integers or their decimal string form, as for `pool_to_geodataframe()`). Any other value raises `ValueError` naming the column, row and value.
- **`discrete_bins`** *(list of float, optional)*: Bin edges for discrete colour bands, e.g. `[0, 0.25, 0.5, 0.75, 1.0]`.
- **`return_fig`** *(bool)*: When `True`, return `(fig, ax)`; otherwise return `ax` only.

`freq` must have exactly one value for every geodata row; a short or oversized
vector is rejected rather than silently rendering an incomplete map. A column in
which no value matches an area ID in `1..N` is likewise rejected (`ValueError`)
instead of rendering every area in `missing_color`; rows whose ID falls outside
`1..N` (for example a 0-based column) keep the missing colour.

**Example**:
```python
# Run consensus sweep over a K/ralpha grid
sweep = flx.consensus_sweep(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    clustersize=15,
    grid={"K": [20, 30, 40], "ralpha": [0.1, 0.2, 0.3]},
)

# Plot consensus frequency map
flx.plot_consensus_frequency(
    gdf,
    sweep["freq"],
    id_col="region_id",
    cmap="YlOrRd",
    title="Consensus inclusion frequency",
    discrete_bins=[0, 0.25, 0.5, 0.75, 1.0],
    save_path="consensus_map.png",
)
```

---

## Utilities and Helpers

### `flatten_clusters()`

Convert nested cluster structure to flat list.

```python
from flexscanx.summary import flatten_clusters

clusters = flatten_clusters(result)
# Each row includes max_dist when available.
```

For a `BOTH` result, rows represent the combined-null envelope primary (and its
secondaries): `pval` is the two-sided inferential value. The primary row also
has `pval_hot` and `pval_cold` for the directional one-sided diagnostics.

<!-- BEGIN GENERATED: series_time_labels_note -->
For `scan_series()` results, the flattened `time` field preserves `result["times"]` when provided.
<!-- END GENERATED: series_time_labels_note -->

---

### `write_clusters_csv()`

Export clusters to CSV.

```python
from flexscanx.summary import write_clusters_csv

write_clusters_csv(result, 'clusters.csv')
```

---

### `cluster_pool()` / `filter_significant_clusters()`

Extract a cluster pool from a scan result and filter it by significance, size, or radius.

```python
from flexscanx.helpers import cluster_pool, filter_significant_clusters

pooled = cluster_pool(result)
significant = filter_significant_clusters(result, alpha=0.05, max_radius=10.0)
```

`max_radius` uses each cluster's `max_dist` metadata. It is only available when you pass a scan result dict; a bare pool of `frozenset`s does not carry radius information.

For `BOTH` results, significance filtering uses the envelope primary's two-sided
`pval`, not either directional one-sided diagnostic.

With the default finite `alpha`, a missing `pval` means the cluster was not
Monte-Carlo tested (for example `simcount=0`) and is excluded. Pass `alpha=None`
to opt out of p-value filtering and retain those clusters.

---

## Data Structures

### ScanResult

Result from `scan()`:

```python
{
    'area': List[int],           # Result area IDs (1-based for scan()/scan_series(); remapped by scan_from_geodataframe())
    'n_case': float,             # Cluster count
    'expected': float,           # Expected count (or population in BINOMIAL)
    'type': str,                 # 'HOT' or 'COLD'
    'stats': float,              # LLR statistic
    'pval': float | None,        # Monte Carlo p-value when requested
    'rank': int,                 # Monte Carlo rank among simulated maxima; 1 when simcount == 0
    'Rate_Ratio': float | None,  # Relative risk (Poisson only; None when degenerate or non-finite)
    'Rate_Ratio_CI': tuple | None, # Reporting CI for Rate_Ratio
    'Risk_Ratio': float | None,  # Relative risk (Binomial only; None when degenerate or non-finite)
    'Risk_Ratio_CI': tuple | None, # Reporting CI for Risk_Ratio
    'max_dist': float,           # Max distance within cluster
    'from_area': int,            # One endpoint of max_dist
    'to_area': int,              # Other endpoint of max_dist
    'secondaries': List[dict],   # Secondary clusters (if requested); each also carries
                                 #   'simcount_actual': int, the completed replicates behind
                                 #   its own 'pval' (pval == rank / (simcount_actual + 1))
    'scan_settings': dict,       # Provenance; scalar simcount_requested/simcount_actual
    'ratio_smoothing_applied': str, # 'NONE', 'HALDANE', or 'EB'
}
```

---

### BothResult

`scan(..., clustertype="BOTH")` returns an envelope rather than a `ScanResult`.
It has exactly these top-level fields; `primary`, `hot`, and `cold` are
directional `ScanResult` mappings, and the top-level `scan_settings` is required.

```python
{
    'type': 'BOTH',
    'primary_type': 'HOT' | 'COLD',
    'primary': ScanResult,
    'hot': ScanResult,
    'cold': ScanResult,
    'scan_settings': dict,
}
```

---

## Root Export Index

Every name in `flexscanx.__all__` is supported by the public package namespace.
The entries below provide the authoritative index for compact helpers whose
full behaviour is documented in the module docstrings and examples above.

| Export | Purpose |
| --- | --- |
| `__version__`, `core_version` | Package and compiled-core version strings. |
| `export_cluster_report`, `write_clusters_csv`, `write_series_clusters_csv`, `scan_settings_to_text` | Reporting and reproducible export helpers. `write_series_clusters_csv` accepts `max_dist` metadata when present. |
| `plot_consensus_frequency` | Plot consensus-frequency summaries. |
| `scan`, `scan_series`, `scan_from_geodataframe` | Spatial scan entry points. |
| `csr_from_pysal`, `csr_from_dense`, `csr_knn_from_coords`, `queen_adjacency`, `rook_adjacency`, `build_contiguity` | CSR graph and contiguity construction. |
| `summarize_result`, `flatten_clusters`, `clusters_to_dataframe` | Result summaries and tabular conversion. |
| `build_static`, `describe_static` | Reusable scan-static construction and inspection. |
| `cluster_pool`, `filter_significant_clusters`, `merge_overlapping_clusters` | Cluster-pool selection and set operations. |
| `ensure_projected`, `pool_membership_series`, `pool_to_geodataframe` | Geospatial projection and pool conversion. |
| `silhouette_graph`, `dunn_graph`, `morans_I_within`, `join_counts`, `cv_glm_gain`, `trend_level_slope` | Cluster quality, spatial association, and trend summaries. `trend_level_slope` estimates its slope over the trailing window while returning the complete segment times and level. |
| `consensus_sweep`, `consensus_to_series`, `conductance`, `boundary_contrast`, `stability_across_K` | Consensus and stability diagnostics. |
| `calibrate_llr_threshold`, `find_stable_params_all`, `find_stable_params_all_series`, `grid_search_stable`, `screen_stable_significant` | Calibration and stable-parameter searches. |

### `summarize_result()`

```python
flexscanx.summarize_result(result) -> pandas.DataFrame
```

Convert a single-direction result, a BOTH envelope, or a `scan_series()` result
into the standard table. The table includes `Rank`, `Type`, `NumArea`, `MaxDist`,
`Case`, `Expected`, `RR`, `Stats`, `P`, and `Sig`; rows are ordered by descending
statistic with p-value as the tie-breaker. `write_series_clusters_csv()` likewise
writes per-time `max_dist` metadata. For large sequence distance matrices, the
advisory is a `UserWarning`.

---

### SeriesResult

<!-- BEGIN GENERATED: series_result_contract -->
Result from `scan_series()`:

```python
{
    'per_time': List[dict],      # T scan results; each entry omits per-slice scan_settings
    'times': List[Any],          # time labels (provided or default range(T))
    'scan_settings': dict        # shared run metadata
}
```
<!-- END GENERATED: series_result_contract -->

---

## Performance Tips

### Choosing Parameters

**For Large N (>2000)**:
- Use `K <= 100` to avoid DFS explosion
- Use `mc_early_stop=True` for faster one-sided HOT/COLD p-values. BOTH scans
  deliberately exhaust `simcount` to preserve valid aligned two-sided inference.
- Consider `simcount=199` instead of 999

**For Time-Series**:
- Always use `crn=True` for stability
- Set `n_jobs_time` for parallelism across time
- Use `simcount=199` per time point (faster)

**For High Precision**:
- Use `simcount=9999` for publication-quality p-values
- Set `ralpha=0.2` for conservative detection
- Enable `secondary="ALL"` for complete enumeration

### Memory and Speed

**Distance Computation** (sequence analysis):
- O(N² × T) complexity; the full N×N output matrix is N² × 8 bytes
- For N > ~7 900 areas (> 500 MB), a `UserWarning` is emitted — use `condensed=True` for ~half the memory
- `condensed=True` returns a 1-D upper-triangle array compatible with `scipy.spatial.distance.squareform`
- For very large N, subsample `sequences` rows first

**Monte Carlo**:
- Set the `FLEXSCANX_MC_ALIAS=1` environment variable (or pass `mc_use_alias=True`) to use the alias sampler for the MULTINOMIAL null, a 10-20 % speedup in large runs (experimental). It changes which null sample a given `seed` produces, so p-values move within Monte Carlo error relative to the default sampler; `scan_settings["mc_alias_env_override"]` records when the variable, not the argument, turned it on

---

## Complete Workflow Example

```python
import flexscanx as flx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from flexscanx.graph import csr_from_geodataframe_contiguity
from flexscanx.sequence import (
    extract_sequences, seq_dist_hamming, seq_trans,
    seq_iplot, seq_dplot
)
from flexscanx import plotting

# 1. Load geographic data
gdf = gpd.read_file("regions.geojson").to_crs("EPSG:3857")
centroids = gdf.geometry.centroid
coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
row_ptr, col_idx = csr_from_geodataframe_contiguity(gdf, method='queen')

# 2. Time-series scan (12 months)
T, N = 12, len(gdf)
observed_TN = ...  # shape (12, N)
expected_TN = ...

result_series = flx.scan_series(
    observed=observed_TN,
    expected=expected_TN,
    coords=coords,
    adjacency=(row_ptr, col_idx),
    model="POISSON",
    clustertype="BOTH",
    K=30,
    simcount=199,
    crn=True,
    seed=42
)

# 3. Extract and analyze sequences
sequences, area_ids = extract_sequences(result_series, include_type='both')

# 4. Compute trajectory distances
# Use condensed=True for large N to halve memory usage
distances = seq_dist_hamming(sequences, norm=True)  # (N, N) full matrix
# For N > ~7900 areas use: cdist = seq_dist_hamming(sequences, condensed=True)

# 5. Analyze patterns
trans_matrix = seq_trans(sequences, lag=1)
print("Transition probabilities:")
print(trans_matrix)

# 6. Visualize
fig, ax = seq_iplot(sequences[:40], area_ids=area_ids[:40])
plt.savefig('trajectories.png')

fig, ax = seq_dplot(sequences, show_entropy=True)
plt.savefig('distribution.png')

# 7. Map spatial results
for t in [0, 6, 11]:  # Start, middle, end
    fig, ax = plotting.plot_cluster_membership(
        gdf,
        result_series['per_time'][t],
        id_col='id',
        title=f'Month {t+1}',
        save_path=f'map_t{t}.png'
    )
```

---

## See Also

- **Tutorial**: `vignettes/workflows/sequence_analysis_workflow.qmd`
- **Examples**: `vignettes/` directory
- **Sequence tests**: `tests/test_sequence_utils.py`

---

**flexscanx v0.3.3** | MIT License | 2026