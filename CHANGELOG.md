# Changelog

> **Versioning note.** The canonical package version is **0.3.0** — the single
> source of truth is `[project].version` in `pyproject.toml`, mirrored by
> `flexscanx.__version__` (`_meta.py`) and enforced by `tests/test_meta_version.py`.
> The section headers below labelled `v0.0.1b`, `v0.0.1c`, and
> `v0.0.1 post-remediation` are dated maintenance phases, not separately
> published package versions or tags. Each `vX.Y.Z` heading is a release: its
> section and the dated sections between it and the previous release.
> Test counts and gate results quoted under a heading describe the tree as it
> stood on that heading's date; they are historical records, not the current
> baseline, which is recorded in the most recent `audit/*/VERIFICATION.md`.

## v0.3.2 — 2026-11-08

A patch release, by the maintainer's decision of 2026-11-08, so that the repository can be made public.
Under `src/` only the version changed since v0.3.1; the parity tests now pin traminex 0.2.2
(`2ed89d77a75d67a4ac319c05da463a1758244503`), the same package code as traminex 0.2.1.

## Published files without local paths — 2026-11-08 (after v0.3.1)

So that the repository can be made public (maintainer's decision of 2026-11-08), the one published line that
named a user folder of the maintainer's former Windows machine, in `tests/test_scripts_smoke.py`, now checks
that `scripts/gen_compile_commands.ps1` names no Windows user folder at all, a stricter form of the same
check.

## v0.3.1 — 2026-11-08

A patch release, by the maintainer's decision of 2026-11-08. Under `src/` only the version changed since v0.3.0;
the parity tests now pin traminex 0.2.1 (`7bebb7a4026f7f8037e1e196d005f4cb5ffa03a4`).

## Tested traminex commit — 2026-11-07 (after v0.3.0)

- The parity tests pin traminex commit `928186e80ff263f4a5dca8270aca8407a20a12a9` (traminex 0.2.0 with
  its fixes of 2026-11-07; on 2026-11-08, `7bebb7a4026f7f8037e1e196d005f4cb5ffa03a4`, traminex 0.2.1) in place of `6e76440b8ed7fc4a1c885c78546425a877bac275`, in `pyproject.toml`,
  `tests/traminex_parity_utils.py`, the CI workflow and `README.md` (supervision repository, owner decision
  D6, carried out at the owner's request "fix all known issues"). `ECOSYSTEM.md` and `DUPLICATES.md` follow:
  the oracle at that commit is TraMineR 2.2-14 and TraMineRextras 0.6.9; `DUPLICATES.md` is again
  traminex's master copy, and `MIGRATION_PLAN.md` says how it differs from the master. flexscanx's own code
  is unchanged and still installs and runs without traminex.

## v0.3.0 — 2026-11-07

Numbered 0.3.0 by the maintainer's decision. Releases the two dated sections below it, the post-v0.0.2 audit of 2026-08-01 and the
whole-codebase audit and fix of 2026-09-07, which change public contracts. One test
also changed: `tests/test_fix20260907_contracts.py` reads the membership plot as
geopandas 1.2 draws it, one collection per category with no colour array; the plot
itself was right. Completed locally; no remote publication is claimed.

## Whole-codebase audit and fix — 2026-09-07

Remediation of the open findings recorded by the 2026-08-22, 2026-09-01 and
2026-09-02 audits (`FXA-160`–`FXA-188`) and of the blocked rows they carried.
The evidence ledger is `audit/whole_codebase_audit_and_fix_20260907/`.

### Changed (public contracts)

- `scan()`, `scan_series()` and `calibrate_llr_threshold()` now reject a
  caller-supplied `coords=` or `adjacency=` beside a reused `handle=` with a
  `ValueError` naming the argument, instead of silently discarding the caller's
  graph (`FXA-179`). Pass either a handle or a graph, never both.
- `csr_from_dense()` enforces its documented 0/1 contract: any entry outside
  `{0, 1}` (a weight, a negative value, `NaN`, a string) raises `ValueError`
  naming the offending `(row, column)` positions instead of being silently
  treated as an edge by Python truthiness, and symmetry is checked on the
  validated matrix. The diagonal is still ignored for edges. Every
  in-repository caller already passed a binary matrix (`U-5`).
- `queen_adjacency()`, `rook_adjacency()`, `build_contiguity()` and
  `csr_from_geodataframe_contiguity()` now evaluate their documented geometric
  predicates with shapely on every installation: queen = the polygons touch at
  any point, rook = they share a boundary segment of non-zero length. They used
  to delegate to libpysal when it was importable, and libpysal decides both by
  shared *vertices*, so on meshes that are not vertex-congruent (a vertex of one
  polygon inside an edge of another, as at a T-junction) the graph, the reported
  cluster and its p-value depended on whether an optional dependency was
  installed. The `[geo]` install is the one that departed from the documentation.
  `queen_include_intersects` now applies on every installation, and the
  O(N²) no-spatial-index fallback is gone (shapely ≥ 2.0 always provides an
  STRtree). `csr_from_pysal()` is unchanged for callers who want libpysal's
  vertex rule (`FXA-164`).
- Enabling `enable_frame_ub_prune` or `enable_center_ub_gating` on a FLEXIBLE
  scan now emits a `UserWarning` naming the flag: the greedy bound behind both
  opt-in heuristics is not a valid upper bound, and the 2026-09-02 audit measured
  the frame prune losing the true maximum in 144 of 7,056 scans. The options are
  unchanged and remain available (D5).
- On a regular lattice (or any coordinates rounded so that a centre's K-th and
  (K+1)-th nearest neighbours are exactly equidistant) the K-limited universe
  boundary falls inside a tied distance group, which flexscanx resolves by area
  index; the reported cluster, statistic and p-value can therefore change when
  the input rows are relabelled (measured on square grids: 2.9 % of
  relabellings). The compiled core now counts such centres at build time
  (`describe_static(handle)["knn_tie_boundary_centers"]`), `build_static()`,
  `scan()`, `scan_series()` and `calibrate_llr_threshold()` emit a `UserWarning`
  naming them, and `API_REFERENCE.md`'s `K` entry documents the index
  resolution and the remedies (a `K` that does not split a tied group, or a
  negligible coordinate jitter). Results are unchanged: this is decision D4's
  route one; admitting whole tie groups or a geometric tie-break would change
  reported results and remain maintainer decisions (`FXA-187`). The same note
  now states that without boundary ties the cluster and statistic are
  independent of the row order while the Monte Carlo p-value is not (the null
  stream is keyed by area index), measured in the 2026-09-07 determinism matrix.
- A running scan can be interrupted. The compiled core now polls for a pending
  Python signal about every 50 ms on the calling thread (between centres of the
  observed enumeration and between Monte Carlo replicates), so `Ctrl-C` raises
  `KeyboardInterrupt` promptly instead of after the native call completes; the
  worker threads are cancelled and joined through the schedulers' existing
  error path, and the calling thread now participates as one of the workers.
  The scan cores stay usable without Python (the hook is empty by default).
  Limits: a single centre's DFS is not interruptible mid-way, and with
  `both_parallel=True` the interrupt lands only when the two directional scans
  return (`FXA-188`, D6).
- `pool_to_geodataframe()` and `plotting.choropleth()` report a non-integer
  identifier column as `ValueError: id_col '<name>' must contain integer area
  IDs (or their decimal string form) ...; row i has <value>` instead of
  Python's bare `invalid literal for int()` message; numeric strings are still
  accepted (`FXA-190`).
- `docs/MIGRATION_FROM_RFLEXSCAN.md` no longer claims results reproduce
  "identically across ... platforms": the Monte Carlo null draws go through
  `std::poisson_distribution`, `std::binomial_distribution` and
  `std::discrete_distribution`, whose algorithms are implementation-defined, so
  reproducibility from `seed` alone holds across thread counts and scheduler
  backends on one build and across machines for identical builds (`FXA-189`).
- The calibration cache directory is now created `0700`. A pre-existing
  directory that is group- or world-writable is tightened to `0700`, its
  integrity key is rotated and the entries it held are discarded (with a
  warning); a directory owned by another user disables the cache for the
  process (one warning). The HMAC key sits beside the data it authenticates, so
  the integrity scheme only holds while nobody else can write the directory;
  with the workstation's `umask 002` the directory used to be created `0775`,
  which let any group member replace the key and choose the returned threshold
  (`FXA-182`). An attacker who already runs as the user is out of scope.
- The real-valued controls `clusterradius`, `ralpha`, `mc_alpha`,
  `secondary_p_stop`, `secondary_stats_stop` and the calibration `alpha` now
  report a conversion failure as `<name> must be a real number, got ...`
  instead of Python's bare `float()` message; the exception type is unchanged
  (`ValueError` for strings, `TypeError` for containers), and a `bool` is now
  rejected like the integer controls already did (`FXA-180`).
- `describe_static()`, and every entry point taking `handle=`, report an object
  that is not a static handle as `ValueError: handle must be a static precompute
  handle returned by build_static()` rather than leaking the compiled
  `DescribeStatic` symbol in a `TypeError` (`FXA-171`).
- Every warning the package emits (clustersize cap, exact-enumeration cost,
  warm-start override, numba fallback, BFS size, geographic CRS, ID mismatch,
  spatial-index fallback, calibration-cache save, sequence-matrix memory) is now
  attributed to the first stack frame outside `flexscanx`, so `-W` filters and
  the printed source line point at the caller's code (`FXA-173`).
- `calibrate_llr_threshold(handle=...)` works again. Its `knn_refill_force`
  default is now `None` ("not requested"), matching `scan()`; the wrapper used to
  forward a concrete `False`, which the handle-reuse path rejected as an explicit
  build-time request, so the documented parameter raised on every call
  (`FXA-165`). An explicit `knn_refill_force` beside a handle is still rejected,
  as in `scan()`.
- `scan_series()` stores `times` as strict-JSON scalars: numpy integers and
  floats become Python numbers and datetime-like labels (`datetime`,
  `pandas.Timestamp`, `numpy.datetime64`) become ISO 8601 strings, so
  `json.dumps(result, allow_nan=False)` succeeds for a pandas time index instead
  of raising `TypeError` (`FXA-172`).
- A non-finite `Rate_Ratio`/`Risk_Ratio` (reachable through a legal denormal
  `expected`) and its confidence interval are reported as `None`, the documented
  "degenerate" value, so the whole result envelope stays strict-JSON encodable;
  the provenance sub-tree already followed this rule (`FXA-181`).
- Each secondary cluster's `simcount_actual` is now a declared, documented key
  (`types.ScanClusterDict`, `API_REFERENCE.md`): the completed replicates behind
  that cluster's own `pval`. It differs from `scan_settings["simcount_actual"]`
  only under `secondary_null="conditional"` with `mc_early_stop=True`, where each
  secondary's conditional null stops independently and the primary's count would
  understate the denominator (`FXA-166`, `FXA-167`).
- `find_stable_params_all` (and through it `find_stable_params_all_series`)
  now actually searches the cluster-size axis at small N. The descent steps by
  at least one area while a smaller feasible value exists (a purely
  multiplicative `round(0.8 * 2) == 2` used to freeze it), and the default
  `S_min` scales as `max(1, min(5, round(6% of N)))` instead of coinciding with
  `S_start` for every N between 17 and 41. The search may therefore return a
  smaller `S` than before and evaluates a few more `(K, S)` states (`FXA-185`).
  `API_REFERENCE.md` now states the implemented, N-scaled `K_min`/`S_min`
  defaults instead of the constants 8 and 5 (`FXA-186`).
- `API_REFERENCE.md`'s `both_parallel` bullet now says what the `scan()`
  docstring already said: the option forces `n_jobs_centers = 1` per direction
  (recorded as `scan_settings["n_jobs_centers"]` beside the requested value), so
  a BOTH scan uses two threads however many were requested, and with
  `n_jobs_centers > 1` the default sequential path is faster (measured: 0.73 s
  at 8 threads sequentially against 3.3 s with `both_parallel=True` on a 49-area
  exact scan with 999 replicates) (`FXA-199`).
- `API_REFERENCE.md` corrections: `names` defaults to `None` and emits no
  `name` key (`FXA-175`); `hot` and `cold` in a BOTH result are never `None`
  (`FXA-176`); the clustersize cap warning is emitted only when `clustersize`
  was passed explicitly, as the `scan()` docstring now also says (`FXA-177`).
- The documented submodules `flexscanx.plotting`, `report`, `summary`, `graph`,
  `helpers` and `types` declare an explicit `__all__`, so `import *`,
  documentation generators and tab completion see only their public names
  (`FXA-178`). `tests/test_public_api_surface.py` pins each surface.
- An existing calibration cache file that cannot be parsed (truncated or
  corrupt JSON, undecodable bytes, an unreadable file, or a payload of an
  unexpected layout) is reported once per process as a `UserWarning` naming the
  file, then ignored and rebuilt; a cold cache (no file) stays silent as before
  (`FXA-191`). Silent rebuilding hid a damaged cache and cost a full
  re-calibration on every call.
- `API_REFERENCE.md` and `docs/ADVANCED_USAGE.md` state that `mc_use_alias=True`
  (or `FLEXSCANX_MC_ALIAS=1`) draws a *different* MULTINOMIAL null sample for the
  same `seed`: the alias sampler and the default `std::discrete_distribution`
  path consume the random stream differently, so p-values agree within Monte
  Carlo error but not bit-for-bit, and `calibrate_llr_threshold()`'s Python
  alias path draws from numpy's generator. The provenance keys `mc_use_alias`
  and `mc_alias_env_override` record which path ran, and the switch is inert for
  the BINOMIAL model (`FXA-192`). Behaviour unchanged.
- `plot_consensus_frequency()` converts the identifier column like the other
  geospatial helpers: integers and their decimal string form are accepted, any
  other value raises `ValueError` naming the column, row and value, and a column
  in which no value matches an area ID in `1..N` is rejected. Before, a
  numeric-string or non-integer column silently rendered every area in
  `missing_color` and a non-matching column produced an all-missing map with no
  diagnostic (`FXA-194`). Rows whose ID falls outside `1..N` still take the
  missing colour.
- `plot_cluster_membership()`, `plot_all_clusters()` and `plot_all_both_clusters()`
  convert the identifier column the same way: a non-integer value raises
  `ValueError` naming the column, row and value instead of being treated as
  "not in cluster", and a result whose clusters are entirely absent from the
  column raises `ValueError` ("No cluster IDs matched ...") instead of drawing
  an empty map. A cluster that is only partly present is still drawn partly
  (`FXA-195`). Before, a column of letters produced a map with no highlighted
  area and no signal in all three functions.
- Documentation metadata (`[tool.flexscanx.docs]` in `pyproject.toml`,
  propagated by `scripts/sync_docs_metadata.py`): `last_updated` is
  `2026-09-07` and the status string no longer claims "exhaustively audited and
  verified"; it names the unreleased state, this audit-and-fix run and the open
  licensing decision. `docs/MIGRATION_FROM_RFLEXSCAN.md` (which carried the
  managed markers but was never synchronised, so it showed 2026-06-17) and
  `docs/TRAMINEX_SEQUENCE_PROVENANCE.md` (hand-dated) are now in the managed set
  (`FXA-174`, D9).
- `docs/TRAMINEX_SEQUENCE_PROVENANCE.md` records the 2026-09-07 re-check of the
  licensing facts against traminex `6e76440b`: still `GPL-3.0-or-later`, no MIT
  grant on any ref, `NOTICE.txt` still asserting "traminex MIT". The three
  remediation routes are unchanged and the choice remains the maintainer's (D8).
- `dunn_graph()` and `conductance()` document their degenerate-case
  conventions (`dunn_graph`: `nan` when nothing is reachable, `inf` for a
  single-node cluster or an unreachable complement, `0.0` for mutually
  unreachable members; `conductance`: `inf` for a zero-volume side), which are
  returned rather than raised so callers can filter them.

### Build

- `CMakeLists.txt` now finds and links `Threads::Threads` for both the
  extension module and the `flexscanx_core_tests` executable. The core uses
  `std::thread`, `std::mutex` and `std::atomic` but never linked a threading
  library: the documented native test build failed to link under the
  conda-forge compiler that the family environment puts first on `PATH`
  (`undefined reference to pthread_create`), and an extension built with that
  toolchain carried unversioned `pthread_create`/`pthread_join` references that
  glibc older than 2.34 cannot resolve. With the system compiler the extension
  is byte-identical to before; with the conda-forge compiler both targets now
  link `-pthread` and the references are versioned. The `wheels.yml` path builds
  inside a manylinux container with its own GCC and was not affected (`FXA-160`).
- The compiled core embeds a build-time SHA-256 digest of the native sources
  (`flexscanx._core.source_digest()`, computed by CMake over
  `src/flexscanx_core/*.cpp|hpp|in` and `CMakeLists.txt`), and a test recomputes
  it from the working tree. The editable install does not rebuild the extension
  when only C++ changes, so a green Python suite could silently run against a
  stale binary; that now fails loudly with a rebuild instruction (`FXA-168`).

### Ecosystem

- The traminex revision the B1–B5 sequence-parity suite is tested against
  rolled forward from `c2e0bf3acec8bfec4dff239f902e97fb663208b7` (2026-07-18)
  to `6e76440b8ed7fc4a1c885c78546425a877bac275` (2026-09-05), the revision the
  two other family consumers already pin, in all four pin locations
  (`pyproject.toml`, `tests/traminex_parity_utils.py`,
  `.github/workflows/ci.yml`, `README.md`). Strict parity passed at the new
  revision against a detached checkout and the live checkout (25 tests), and the
  strict guard still refuses the old revision. traminex remains a test-time
  dependency only; the standalone-install guarantee is unchanged (D1).
- `ECOSYSTEM.md`, `AGENTS.md`, `CLAUDE.md`, `README.md` and
  `DOCUMENTATION_INDEX.md` now name the visualization sibling by its live
  identity, `py_seqplot` / `PROJECT_PY_SEQPLOT` (renamed 2026-08-14; the old
  directory no longer exists), without rewriting dated history. The charter's
  one-way `cp` synchronization commands, which would have reverted a sibling's
  verified rename, are replaced by the two-way reconciliation rule the
  pyweightedcluster and py_seqplot copies already carry. The traminex baseline
  copy still lags and is not edited from this repository (`FXA-147`).

### Packaging and environment

- CI's lint step pins `ruff==0.15.20` and `black==26.5.1`, the revisions the
  pre-commit hooks use (the hook's black moved from 24.4.2 to 26.5.1, the
  version the committed tree is formatted with). CI used to install the latest
  black while the hook ran 24.4.2, and the two disagree on the committed tree
  (24.4.2 re-wraps the `textwrap.dedent` calls in
  `tests/test_remediation_packaging_20260816.py`), so a commit that passed the
  local hook could fail CI (`FXA-197`).
- CI now checks native formatting with `clang-format 17.0.6` (the hook's
  revision, `--dry-run --Werror` over `src/flexscanx_core` and `tests/cpp`), and
  three headers that had drifted from it (`algorithms/universe.hpp`,
  `middle_p_binomial.hpp`, `middle_p_poisson.hpp`) are reformatted, whitespace
  only. Any edit to those headers used to drag unrelated re-wrapping into the
  diff (`FXA-198`).
- New optional extra `flexscanx[knn]` declares scikit-learn, which
  `csr_knn_from_coords(backend="balltree")` and `metric="haversine"` require;
  `[all]` includes it. The two `ImportError` diagnostics now name scikit-learn
  and the extra instead of advising a SciPy install that cannot satisfy the
  request (`FXA-162`, `FXA-163`).
- The source distribution no longer ships the repository-governance material
  (`AGENTS.md`, `ECOSYSTEM.md`, `DUPLICATES.md`, `MIGRATION_PLAN.md`,
  `CLAUDE.md`, `docs/DEVELOPER_GUIDE.md`, `docs/README.md`) or the test that
  reads them (`tests/test_repository_guidance.py`), and the shipped
  `tests/traminex_parity_utils.py` no longer embeds the maintainer's checkout
  path, so no sdist member carries a private home path.
  `scripts/verify_release_artifacts.py` now scans every UTF-8-decodable sdist
  member for one, not just `README.md` (`FXA-183`).
- The shipped test suite passes from the extracted sdist: the seven tests that
  read `scripts/` and `vignettes/` (which the sdist excludes by design) are
  declined by `tests/conftest.py` when those trees are absent, as the existing
  sdist logic already did for the other tooling tests (`FXA-170`).
- `FLEXSCANX_DISABLE_JIT=1` now forces the pure-Python graph-metric paths even
  when numba is installed, without the missing-numba warning. CI has set the
  variable on its non-`jit` cells since 2026-07 while nothing read it
  (`FXA-161`).

### Tests

- `scripts/render_all.ps1` (historical Windows helper) now classifies a
  vignette as a note by directory boundary rather than by name prefix, so a
  sibling directory such as `vignettes/notes_extra` would execute rather than be
  rendered without execution. The defect was latent (the helper is only called
  with two fixed directories); the fix is verified under Linux PowerShell in the
  Docker image `mcr.microsoft.com/powershell:latest`, and Windows-native path
  semantics remain untested here (`U-1`).
- The traminex parity harness (`tests/traminex_parity_utils.py`) now passes
  missing values to `seqdef` as genuine missing values kept as the not-recorded
  state, instead of stringifying `NaN` into an ordinary state label. The B3
  entropy and B5 distribution-plot parity tests gained a missing-value fixture
  that matches `seqstatd(include_missing=True)` and asserts that
  `include_missing=False` differs, so the relationship `NOTICE.txt` documents is
  finally executable (`FXA-184`).
- `tests/test_core_edgecases.py::test_radius_strict_no_out_of_universe_inclusions`
  and `tests/test_core_cold_and_circular.py::test_circular_restricted_hot_respects_radius_and_order`
  used null data, so their scans returned an empty cluster and every quantified
  assertion was vacuously true. Both now plant an excess, assert a non-empty
  cluster, and check the radius bound and the circular prefix order directly;
  each fails when the radius is ignored (`FXA-169`).
- `scripts/check_docs_signatures.py` now validates the trailing `# Default: ...`
  comments of the `find_stable_params_all` signature block against a registry
  rendered from the `tuning` module's constants, and rejects an unregistered
  `# Default:` comment on any registered signature. The checker used to discard
  comments, which is how the stale `K_min`/`S_min` defaults of `FXA-186` passed
  the documentation gate (`FXA-193`). Mutation controls: a stale percentage, a
  removed comment, an unregistered comment and the old "8"/"5" wording each fail.
- `tests/test_fix20260907_packaging.py::test_package_type_checks_apart_from_missing_third_party_stubs`
  runs `mypy src/flexscanx` and fails on any error other than a missing
  third-party stub, so the whole package stays type-clean, not only the two
  modules `scripts/release_sanity.py` checks. Four typing regressions
  introduced earlier in this run (a re-bound loop variable in
  `csr_from_dense`, an un-narrowed frame walk in `warn_at_caller`, and the
  numba module alias when JIT is disabled) were caught by the section 9
  matrix and fixed before merge (`FXA-196`).
- This run's tests that read the development-only `scripts/` tree (the
  signature-checker, metadata-management and `render_all.ps1` tests) live in
  `tests/test_fix20260907_scripts.py`, which `tests/conftest.py` declines when
  `scripts/` is absent, and the package type-check test tolerates an absent
  optional module (`import-not-found`) as well as a missing stub. The first cut
  of these tests failed eight times when the suite shipped in the sdist was run
  from the extracted archive, the FXA-170 property re-checked by the section 9
  matrix; a new test forbids any shipped `test_fix20260907_*` module from
  opening `scripts/` (`FXA-200`).
- The six static test-integrity guards of 2026-08-16
  (`tests/test_remediation_test_integrity_20260816.py`) were re-adjudicated: each
  still points at a live construct in the file it guards. The process-global
  numpy RNG rule now covers every test module instead of the two repaired in
  August (the suite has no offenders), and the calibration-cache redirection
  gained an in-process check that the public `calibrate_llr_threshold` reads its
  cache path from a pytest temporary directory rather than the user's cache.

## Post-v0.0.2 whole-codebase audit — 2026-08-01

### Correctness and reliability

- Graph metrics now define finite disconnected-graph silhouette conventions and
  use the same two-pass boundary-contrast variance calculation whether numba is
  available. Shapely contiguity fallback CSR entries are built-in integers,
  string geodata IDs map correctly in pool membership, and HOT/COLD sequence
  conflicts are neutral independent of input ordering. Tuning defaults explore
  the small-N cluster-size axis; stable grid searches have one schema and reject
  wholly infeasible `K <= S` grids.
- Reporting and plotting now reject unsupported result shapes instead of
  silently presenting arbitrary mappings as single scans; consensus maps require
  one frequency per geodata row. Missing p-values (`simcount=0`) follow one
  documented policy across filtering and plotting: exclude under a finite
  threshold, include only with `alpha=None`. BOTH overlays now draw only each
  directional union, palette fallback reaches `tab20` after one Okabe-Ito pass,
  and non-figure plotting returns release only figures they created.
- Rejected zero expected counts for POISSON scans before inference. BOTH result
  tables, CSVs, reports, and significance filters now use the envelope
  primary's combined-null two-sided p-value. BOTH reports retain complete HOT
  and COLD one-sided diagnostic tables, including non-winning-direction rows;
  stability summaries now score unmatched clusters as zero Jaccard similarity.
- Added a self-contained, executable vignette reproducing every worked example
  in the rflexscan 1.1.0 manual: the North Carolina SIDS scan, complete and
  cluster-specific output, summary table, three network plots, and three county
  maps. The exact historical neighbour list and matching county geometry are
  bundled with recorded checksums.
- Repaired 32 confirmed findings across Monte Carlo BOTH inference, static
  provenance and geometry, calibration caching, high-count tail probabilities,
  large-denominator LLR stability, tuning and consensus, sequence helpers,
  plotting, native input boundaries, maintained workflows, CI, benchmarks, and
  clean-room artifact verification.
- Added focused Python and native regressions for every behavioral repair. The
  complete finding evidence is in
  `audit/whole_codebase_audit_20260801/FINDINGS.md`.
- Per-cluster ratio provenance now distinguishes the requested
  `scan_settings["ratio_smoothing"]` mode from `ratio_smoothing_applied`
  (`"NONE"`, `"HALDANE"`, or `"EB"`). An EB request that cannot be applied is
  explicitly reported as applied `"NONE"` rather than silently relabelled.
- Preserved the standalone sequence package and strict B1–B5 test-only parity
  with traminex revision
  `c2e0bf3acec8bfec4dff239f902e97fb663208b7`.

### Verification

- Python suite: 721 passed, 2 expected platform skips, and 2 asserted
  sparse-radius warnings; measured source coverage was 82%.
- Native suite: 65/65 under the classified warning build, ASan/UBSan, and
  ThreadSanitizer.
- All 24 deterministic runtime-matrix combinations, six executable Quarto
  workflows, strict pinned parity, clean-room source tests, sdist/wheels,
  isolated installs, metadata checks, and quality gates passed.
- Full commands, artifact hashes, and classified diagnostics are recorded in
  `audit/whole_codebase_audit_20260801/VERIFICATION.md`.

## v0.0.2 — 2026-07-30

The locally completed `v0.0.2` release consolidates the corrected spatial-scan
implementation, sequence-parity work, Linux release workflow, and all repairs
from the exhaustive whole-codebase audit. This release is complete and tagged
locally; no remote publication is claimed.

### Correctness and reliability

- Repaired all 34 confirmed findings from the 2026-07-29 whole-codebase audit
  across Python and C++ validation boundaries, scan/calibration controls,
  deterministic parallel result ordering, cache concurrency, graph/geospatial
  helpers, sequence analysis, tuning, plotting, and reporting. The evidence
  ledger is `audit/whole_codebase_audit_20260729/FINDINGS.md`.
- Preserved the deliberate mathematical corrections to rflexscan/FleXScan and
  exact test-only sequence parity with traminex commit
  `c2e0bf3acec8bfec4dff239f902e97fb663208b7`.
- Added focused regressions for every confirmed behavioral defect and hardened
  public and direct-native contracts against invalid or silently narrowed input.

### Release verification

- Definitive clean-room release sanity: 130 passed, 2 optional skips.
- Definitive isolated full suite: 606 passed, 26 dependency-absence skips, and
  2 classified sparse-radius advisory warnings.
- Native C++ suite: 65/65 under the standard warning build, ASan/UBSan, and
  ThreadSanitizer, with no sanitizer diagnostics.
- Checkout wheel, source distribution, and source-distribution-rebuilt wheel
  passed metadata inspection, `twine check`, isolated installation, runtime
  smoke tests without traminex or weightedcluster, and `auditwheel show`.
- Repository-wide Ruff, Black, clang-format, documentation, exact parity,
  runtime-matrix, and pre-commit gates passed; mypy over the release-sanity contract set
  passed. **Historical correction (2026-08-16):** this entry previously
  overstated that limited mypy run as repository-wide mypy.

## Linux cleanup and artifact validation — 2026-07-19

### Changed (tooling and CI only)

- Active development, test, wheel, vignette, and release-validation jobs now
  use Ubuntu 24.04. Windows-oriented scripts remain historical compatibility
  helpers and are no longer part of active pre-commit execution.
- Added `scripts/render_all.py` as the deterministic Linux-native Quarto
  renderer, including dry-run, explicit heavyweight-vignette, and
  workflow-versus-note execution policies.
- Added `scripts/verify_release_artifacts.py` to build and inspect an sdist and
  Linux wheel, rebuild a second wheel from the sdist, run `twine check` and
  `auditwheel show`, and smoke-test both wheels in isolated environments without
  traminex or weightedcluster.
- Clean-room verification now creates its own virtual environment, installs the
  `all` extras plus explicit validation tools, excludes local build/audit/planning
  material from its committed snapshot, and exercises the artifact verifier
  after release sanity and the full Python suite.

This maintenance entry did not itself represent a version release, tag, upload,
or publication; it is included in `v0.0.2`.

## Maintenance remediation — 2026-07-18

### Fixed (build and tooling)

- `detail_crn::Xoshiro256ss` now declares the `result_type` required by the
  C++17 UniformRandomBitGenerator contract, allowing the Monte Carlo core to
  compile when instantiated through standard-library random distributions.
- `scripts/quickcheck_ot.py` now resolves GDAL and PROJ data from the active
  Python prefix using the platform's actual share layout and no longer
  overwrites Linux paths with a nonexistent Windows `Library/share` directory.

### Tests

- `tests/cpp/test_core.cpp` now checks the generator type contract at compile
  time and exercises it through `std::uniform_int_distribution`.
- The Manhattan quick-check smoke test now passes in the supported Linux
  development environment.

## rflexscan flaw audit — MC-3 / MC-6 / ENUM-2 — 2026-06-17

Second remediation batch from the phased rflexscan-vs-flexscanx flaw audit
(`archive/reports/2026/rflexscan_flaw_audit_2026-06-15/RFLEXSCAN_FLAWS_REPORT_2026-06-15.md`
§5; `archive/reports/2026/rflexscan_flaw_audit_2026-06-15/ISSUES_AND_FIXES.md`,
actions **A12 / A7 / A9**). User-facing summary: `docs/MIGRATION_FROM_RFLEXSCAN.md`. These are the two open flexscanx-side
DECISIONS, both resolved **Option A (paper-faithful)**, plus an internal comment
fix. Requires a C++ rebuild (`pip install -e .`).

### Changed (behaviour change — `clustertype="BOTH"` Monte-Carlo p-values)
- **BOTH is now a proper two-sided test (MC-3).** Previously `clustertype="BOTH"`
  reported the winning direction's **own one-sided** p-value. The BOTH primary
  (and its secondaries) are now ranked against **one combined two-sided null** =
  the per-replicate `max(hot_max, cold_max)`, per DEF-BOTH (Tango 2008,
  Jpn. J. Biometrics 29(2):75–95, p. 81: the two-sided statistic deletes the
  direction indicator). Mechanically, each one-sided sub-scan emits its
  full-region null-max array (`out["sim_max"]`, gated on `_emit_sim_max`) across
  the pybind boundary; `_build_both_result` forms the element-wise max and
  re-ranks the envelope with the Dwass rule. The two sub-scans share
  `seed`/`time_index` (counter-based CRN), so replicate *b* aligns across
  directions. `sim_max` is an internal transport array, always stripped before the
  result is returned (never on the public surface). Single-direction HOT/COLD
  results are byte-unchanged.
  `src/flexscanx_core/bindings_scan_result.cpp`, `src/flexscanx/_result_format.py`,
  `src/flexscanx/api.py`, `src/flexscanx/_scan_internal.py`.
- **Direction of effect:** because `max(hot, cold) ≥` either one-sided max, BOTH
  p-values are now **larger (more conservative)** — the multiplicity cost of a
  two-sided look. This differs from rflexscan's value, whose combined null is
  separately corrupted by MC-2 (zero-case cold windows dropped). The per-direction
  one-sided p-values remain available under the result's `hot`/`cold` keys.

### Changed (behaviour change — secondary-cluster Monte-Carlo p-values)
- **Secondary clusters now use the full-region null (MC-6).** Previously each
  secondary was ranked against a **conditional** null recomputed over only the
  non-overlapping windows. Secondaries are now ranked against the **unconditional
  full-region null** already computed for the primary (`primary_sim_max`), per
  DEF-STOP ("the same Monte-Carlo null distribution of the maximum … as if it were
  the primary cluster"; Otani 2021 §2.4). The secondary *window search* still
  excludes already-detected areas (non-overlap, correct per DEF-STOP) — only the
  null distribution it is ranked against changed.
  `src/flexscanx_core/bindings_scan_execution.cpp`, `src/flexscanx/api.py`,
  `src/flexscanx/_scan_internal.py`.
- **Direction of effect:** the conditional null's maxima are `≤` the full-region
  maxima, so the old secondary p-values were **anti-conservative**; the new values
  are **larger (more conservative)**. The primary/MLC is byte-identical.
- **Back-compat switch:** new parameter `secondary_null` (default `"full"`,
  Option A). Pass `secondary_null="conditional"` to reproduce the prior behaviour.
  Validated in both Python (`_normalize_secondary_null`) and C++; under
  `clustertype="BOTH"` the envelope primary is a full-region two-sided rank by
  construction (MC-3) regardless of the switch.

### Changed (internal — no behaviour change)
- **Corrected the flexible-DFS enumeration comment (ENUM-2).** The comment at
  `src/flexscanx_core/algorithms/scan_flexible.hpp` claimed each connected subset
  is "generated exactly once … verified against brute-force"; that is **false for
  cyclic induced subgraphs** (a subset is regenerated once per cyclic insertion
  order, up to ~107× on K₈). Only set *completeness* holds. The duplicates are
  inference-neutral — the scan keeps the max LLR under a strict total-order
  tie-break, and re-evaluating an identical window is idempotent — so the MLC and
  all p-values are unaffected. Comment only; no logic change. The true visit-once
  perf optimisation (a persistent per-frame forbid) is deferred to a separate
  ticket. (Commits `3192dd6` comment, `9760780` test.) *(Superseded by the
  2026-06-17 ENUM-2 correction: cyclic repeats remain, completeness only, and
  inference is neutral.)*

### Tests
- `tests/test_mc3_both_combined_null.py` — the BOTH primary is ranked against the
  element-wise `max` of the two sub-scans' null arrays; on data with comparable
  hot+cold and no zero-case windows the BOTH p ≥ each one-sided p; `simcount=0`
  leaves `pval` `None`; `sim_max` never appears on the public result.
- `tests/test_mc6_secondary_full_null.py` — secondary p-value equals ranking the
  secondary's `stats` against `primary_sim_max`; `secondary_null="conditional"`
  reproduces the pre-fix (smaller) value; the primary is identical under either
  null.
- `tests/test_enum2_cyclic_invariance.py` — the MLC and p-values are invariant to
  cyclic insertion order; engine `dfs_nodes_expanded` matches a faithful Python
  reconstruction; engine MLC == an independent brute-force max (Poisson & Binomial,
  HOT & COLD).

## rflexscan flaw audit — GATE-4 fix — 2026-06-15 (BOTH restricted screens at α1/2)

From the phased rflexscan-vs-flexscanx flaw audit
(`archive/reports/2026/rflexscan_flaw_audit_2026-06-15/findings/02_gating.md`,
finding **GATE-4**;
`archive/reports/2026/rflexscan_flaw_audit_2026-06-15/ISSUES_AND_FIXES.md`,
action **A5**). This is the one confirmed
flexscanx-side deviation from the method papers found in the LLR/gating layers;
the other audited flaws are rflexscan-only (flexscanx already paper-correct).

### Changed (behaviour change — `clustertype="BOTH"`, `stattype="RESTRICTED"`)
- **Two-sided restricted screening now uses `ralpha/2` per tail.** Tango (2008,
  Jpn. J. Biometrics 29(2):75–95, p. 81) specifies that the two-sided restricted
  statistic deletes the direction indicator and **"replace[s] α1 with α1/2"** — a
  Bonferroni split of the per-region significance level across the two tails.
  flexscanx builds BOTH as an envelope of two one-sided RESTRICTED sub-scans;
  previously each sub-scan screened at the full `ralpha` (matching rflexscan, but
  too permissive by 2× on the per-region level). Each sub-scan now screens at
  `ralpha/2`. The merged BOTH envelope still reports the user-configured `ralpha`;
  each sub-scan's `scan_settings` records the `ralpha/2` actually applied. No-op
  for `stattype="ORIGINAL"` (no per-region gate). Affects `scan()` and
  `scan_series()`. `src/flexscanx/api.py`, `src/flexscanx/_scan_internal.py`.
- **Direction of effect:** BOTH/RESTRICTED scans now admit fewer windows (the
  per-region screen is half as permissive), bringing flexscanx into agreement with
  Tango (2008). HOT-only and COLD-only scans are unchanged.

### Tests
- `tests/test_gate4_both_alpha_split.py` — metadata (sub-scans report `ralpha/2`,
  envelope reports `ralpha`; ORIGINAL untouched) plus an end-to-end behavioural
  check that a centre with HOT middle-p ≈ 0.134 is admitted one-sided at
  `ralpha=0.2` but screened out of BOTH (`ralpha/2 = 0.10`) and re-admitted at
  `ralpha=0.4`.

## Audit-Fix Phase D — 2026-06-13 (sequence parity re-confirmed; test-only)

Independent re-audit of `flexscanx.sequence` against the **Audit-Phase-B-fixed**
traminex oracle (`0.0.8`). B1–B5 parity is green with no flexscanx drift: the
Phase B TRATE fix changed only traminex's `distances/costs.py`, not the
`seqtrate` / `seqstatd` / `seqtab` targets these copies are checked against.
The B2 test also pins `seqtransn` as a distinct per-sequence statistic so it
cannot be mistaken for the transition-matrix API. **No behaviour change;
flexscanx stays `0.0.2`.**

### Changed (test-only, no behaviour change)
- **B2 parity now exercises the absent-source zero-row.** Added an
  `ABSENT_SOURCE` fixture (a state present only in the final column, so it is
  never a transition source, count `PA == 0`) plus a dedicated assertion that
  `seq_trans(normalize=True)` zeroes that rate row — matching R `seqtrate`'s
  `if (PA == 0) tmat[x, ] <- 0` (the exact convention pinned in the oracle by
  Audit Phase B) rather than emitting a 0/0 `NaN`. flexscanx already agreed
  bit-for-bit; the branch was simply untested.
  `tests/test_traminex_parity_b2_trans.py`.
- **B4 parity tolerance tightened** from `np.isclose`'s loose default
  (`rtol=1e-5`) to `rtol=0, atol=1e-12`, matching the other parity tests.
  `tests/test_traminex_parity_b4_freq.py`.

### Verified
- **Standalone guarantee re-proven:** built the `0.0.2` wheel and imported/ran it
  in a clean virtualenv with **traminex absent**, from a directory outside
  `PROJECT_TRAMINEX`; the sequence helpers run on numpy/pandas only and the
  parity tests skip cleanly (and fail under `FLEXSCANX_PARITY_STRICT=1`, proving
  the skip gate is real).
- Full suite **387 passed / 1 failed / 3 skipped** (plain and strict; the +1 over
  the Phase-A 386 baseline is the new B2 test; the single failure is the
  pre-existing numba-vs-NumPy issue).

## Sequence parity migration — 2026-06-13

Behaviour-identical alignment of the `flexscanx.sequence` helpers with their
traminex source-of-truth counterparts (ECOSYSTEM.md / DUPLICATES.md Section B,
MIGRATION_PLAN.md Phase 3). traminex remains a **test-only** dependency;
flexscanx still installs and runs without it.

### Changed (behaviour)
- **`seq_entropy(by_time=True)` is now normalized by default.** The
  cross-sectional per-position entropy is divided by `log(K)` (K = alphabet
  size = distinct states across the whole matrix), yielding a value in [0, 1]
  that matches `traminex.seqstatd`'s normalized `Entropy` row (the source of
  truth). Previously it returned the unnormalized entropy in `base` units,
  which drifted from traminex. A new `normalize` parameter (default `True`)
  recovers the old unnormalized bits with `normalize=False`. The longitudinal
  `by_time=False` overall-diversity form is unchanged (flexscanx-specific, no
  traminex counterpart). Verified by `tests/test_traminex_parity_b3_entropy.py`.

### Added (test-only, no behaviour change)
- Parity tests proving the remaining `sequence/` helpers already match
  traminex: `seq_dist_hamming` vs HAM / explicit-subm DHD (B1), `seq_trans`
  vs `seqtrate`, with a separate guard that `seqtransn` remains a per-sequence
  statistic (B2), `seq_freq` vs `seqtab` (B4), and the plot-driving data of
  `seq_dplot` / `seq_fplot` / `seq_iplot` vs `seqstatd` / `seqtab` (B5).
- `sequence/NOTICE.txt` rewritten per function with the correct traminex
  counterpart and parity-test reference (fixing the stale `seq_dist_gower`
  entry and the `seq_trans` → `seqtransn` mis-attribution; its rate-matrix
  counterpart is `seqtrate`).

## v0.0.1c — 2026-06-12 (independent audit + fixes, unreleased)

An independent end-to-end audit (see `AUDIT_REPORT.md`, produced in the audit
worktree) verified the engine core by brute-force enumeration and rflexscan
parity, and found and fixed four substantive defects plus several statistical
helper bugs. All fixes are test-first; the regression suite is
`tests/test_audit_v001c.py`.

### Critical / Correctness
- **Default scans are now exact.** The frame-level and center-level
  upper-bound pruning heuristics (`enable_frame_ub_prune`,
  `enable_center_ub_gating`) are **off by default**: the greedy bound was not
  a valid upper bound (early stop + frontier-only candidates) and default
  scans returned provably suboptimal clusters (76/80 sequential divergences
  on a 10×10 sweep; wrong BINOMIAL HOT primaries vs rflexscan). Both remain
  available as documented opt-in speed heuristics that may return suboptimal
  results.
- **Build repaired.** `pip install -e .` works again: `tests/cpp/test_core.cpp`
  updated to the `*_restricted_poisson` entry points and the C++ unit-test
  target gated behind `FLEXSCANX_BUILD_CPP_TESTS` (OFF for packaging builds;
  CI builds and runs it explicitly).
- **Deterministic parallel scans.** Exact LLR ties between distinct clusters
  were previously resolved by thread arrival order; a deterministic tie-break
  (smaller cluster, then lexicographically smaller area) makes parallel
  results equal the sequential scan bit-for-bit.
- **`binom_cdf` large-n underflow fixed** (`middle_p_binomial.hpp`): the
  recurrence base `(1-p)^n` underflowed for `n·|log(1-p)| > ~708`, corrupting
  RESTRICTED BINOMIAL gating at populations ≳1000 (inadmissible centres
  admitted via the `p>0.5` complement path). Hybrid linear/log-space CDF;
  scipy-referenced C++ regressions added.

### Performance
- **Canonical visit-once DFS enumeration**: each connected subset is now
  generated exactly once (previously once per insertion order — a factorial
  blowup that the invalid prune had masked). Exact flexible scans drop from
  intractable to seconds in the K≤15 regime; `scan()` warns when an exact
  FLEXIBLE scan is requested with K > 25 (inherently exponential).
  *(Superseded by the 2026-06-17 ENUM-2 correction: cyclic induced subgraphs
  can repeat visits; enumeration remains complete and the repeats are
  inference-neutral.)*

### Statistical helpers
- `silhouette_graph`: intra-cluster mean no longer includes the node's own
  zero distance (values were systematically inflated vs Rousseeuw 1987,
  including sign flips); singletons return 0.
- `boundary_contrast`: numba path now uses sample variance (ddof=1) matching
  the pure-Python fallback; results no longer depend on numba's presence.
- `dunn_graph`: documented as the cluster-only-diameter variant.
- `calibrate_llr_threshold`: cache key includes the RNG seed (schema v4) and
  cache writes are atomic.

### Tooling / docs
- Docs-hygiene checker resolves source-rooted references and exempts
  CHANGELOG history; pre-commit hooks made environment-robust;
  `environment.yml` can now actually run the test suite; benchmark scripts
  opt in to the pruning heuristics explicitly.
- Parity vs `RFLEXSCAN_ORIGINAL` (24-cell grid): identical primary clusters
  and ≤4e-13 relative LLR deltas in 20/24 cells; the 4 exceptions are the
  documented intentional RESTRICTED-COLD design difference. COLD/BOTH MC
  p-value comparability remains an open question (see AUDIT_REPORT.md).

## v0.0.1b — 2026-05-31 (architectural audit remediation, unreleased)

All 22 findings from the comprehensive architectural audit were addressed in this pass.
No C++ rebuild is required for the changes in this section.

### Critical Fix
- **`plot_consensus_frequency` blocking import fixed** (`plot_consensus.py`): `import matplotlib.pyplot as plt` was at module level, causing `import flexscanx` to raise `ImportError` in any environment without matplotlib. The import is now deferred inside the function body. A `plot = ["matplotlib>=3.6"]` optional dependency group was added to `pyproject.toml`.

### New Public API
- **`export_cluster_report()`** promoted from `flexscanx.report` (semi-private) to `flexscanx.__all__` and the top-level namespace. Previously the function was public-named but completely undiscoverable without reading source.

### Bug Fixes
- **`csr_from_dense` symmetry validation** (`graph.py`): Asymmetric adjacency matrices were silently forwarded to the C++ core (which requires symmetry), producing wrong results. `csr_from_dense` now validates both square-matrix shape and symmetry; raises `ValueError` with a precise diagnostic.
- **`consensus_sweep` length guard** (`_consensus.py`): Mismatched `observed`/`expected` lengths now raise a clear Python `ValueError` before reaching the C++ `ScanSlice` call.
- **`calibrate_llr_threshold` `n_sims` guard** (`_calibration.py`): `n_sims=0` or `n_sims<0` now raises `ValueError` instead of returning silently empty `nan` results.
- **`_Psort` sentinel changed to `float('inf')`** (`summary.py`): The previous sentinel of `1.1` could theoretically collide with a real p-value, causing `NaN` p-values (MC disabled) to sort incorrectly relative to clusters with `p > 1.0`. The sentinel is now `float('inf')`, which is unreachable by any valid p-value.
- **Redundant K provenance expression removed** (`api.py`): The three-way ternary for `"K"` in the provenance dict was equivalent to a simpler expression after the `K = eff_K` assignment at line 204.

### Dead Code Removed
- **`seq_dist_gower` deleted** (`sequence/distances.py`): The function was unexported from `flexscanx.sequence`, delegated 100 % to `seq_dist_hamming`, and its docstring promised generic Gower semantics it did not deliver. Removed with no backward-compat impact (v0.0.1, first release).
- **Numba cache binaries excluded** (`.gitignore`): `**/__pycache__/*.nbc` and `**/__pycache__/*.nbi` patterns added. The stale `_validation_graph.*_numba*.nbi/nbc` files (referencing the deleted `_validation_graph` module) were preventing clean builds on some machines.

### Improvements
- **`seq_dist_hamming` OOM protection** (`sequence/distances.py`): A `UserWarning` is now emitted when the N×N output matrix would exceed 500 MB (N ≳ 7 900 areas). New `condensed=True` parameter returns a 1-D upper-triangle array compatible with `scipy.spatial.distance.squareform`, using ~half the memory.
- **Pandas import made lazy** (`summary.py`): `import pandas as pd` moved from module level into each function that uses it. `import flexscanx` no longer forces an eager pandas import; users who only need `scan()` results can run without triggering the pandas initialisation overhead.
- **`_CRN_DEFAULT_SEED` named constant** (`_scan_internal.py`): The magic literal `0xC0FFEE` that anchors the CRN stream when `crn=True, seed=None` is now a named module constant with a docstring explaining its purpose. `scan_series()` docstring updated with a seed-note section.
- **`_bfs_distances_py` slow-path warning** (`_graph_metrics.py`): At this
  maintenance stage, a one-time `UserWarning` was emitted at module load when
  numba was absent. The later v0.0.2 audit remediation made base imports silent
  and deferred the warning until the first fallback-backed metric call. The
  numba-specific paths accelerate BFS for `silhouette_graph` and `dunn_graph`
  and the inner loop of `boundary_contrast`; `morans_I_within` does not use a
  numba-specific implementation.
- **`_CACHE_SCHEMA_VERSION` version history documented** (`_calibration.py`): A comment block explains the v1→v2→v3 progression so future maintainers understand why the initial release starts at version 3.
- **`ScanHandle` Protocol strengthened** (`_core.pyi`): `__hash__` method added so the Protocol provides actual structural type-checking; previously it was equivalent to `object`.

### Architecture
- **`validation.py` split into focused sub-modules** (`_calibration.py`, `_graph_metrics.py`, `_predictive.py`, `_consensus.py`): The 1 223-line God Module combining 7 distinct concerns was split into four focused files. `validation.py` is now a thin re-exporting facade; all existing imports and monkeypatches continue to work without change.

### Documentation
- `scan()` and `scan_series()` docstrings rewritten in NumPy `Parameters\n----------` format (was `-` bullet list; inconsistent with the rest of the codebase).
- `API_REFERENCE.md`: `export_cluster_report` import path corrected to root namespace; `seq_dist_hamming` condensed parameter documented; `seq_dist_gower` references removed; `calibrate_llr_threshold` `n_sims` constraint documented; `stability_across_K` signature corrected; `find_stable_params_all` / `find_stable_params_all_series` sections added.
- `docs/DEVELOPER_GUIDE.md`: `validation.py` architecture description updated to reflect the sub-module split.
- `vignettes/workflows/sequence_analysis_workflow.qmd`: `seq_dist_gower` import removed (function deleted); Gower distance section replaced with condensed-mode `seq_dist_hamming` demonstration.

### Tests
- `tests/test_audit_remediation_v001b.py` added: warm_start correctness, `csr_from_dense` symmetry detection, `consensus_sweep` length validation, `calibrate_llr_threshold` `n_sims` guard, `seq_dist_hamming` OOM warning and condensed mode, `seq_dist_gower` namespace absence, `export_cluster_report` root accessibility, `_Psort` sentinel correctness, validation.py facade coverage.
- `tests/test_public_api_surface.py` updated: `export_cluster_report` added to `EXPECTED_ROOT_EXPORTS` and `ROOT_EXPORT_SOURCES`.

---

## C++ Core Improvements — 2026-05-31

### Bug Fixes
- **`binom_cdf` underflow for `p > 0.5`** (`middle_p_binomial.hpp`): the recurrence
  `pmf *= (n-i+1)/i * p/(1-p)` underflows to 0 for large `n` and `p` near 1, making
  `binom_cdf` return 0 when it should return a value close to 1. Fixed with the symmetric
  complement identity `P(X≤k | n,p) = 1 − P(X≤n-k-1 | n,1-p)` for `p > 0.5`. Maximum
  recursion depth 1.
- **S2-2 double-precision radius** (`algorithms/universe.hpp`): `build_knn_order` (circular
  scanner) still used `float` stored distances for radius check; `build_universe` (flexible
  scanner) was fixed but the sibling function was missed. Both now recompute distance at
  full `double` precision when radius is finite, matching the refill path.

### Performance
- **LLR baseline precompute** (`llr_poisson.hpp`, `llr_binomial.hpp`, `policies/model_*.hpp`,
  `algorithms/bounds.hpp`, `algorithms/scan_flexible.hpp`, `algorithms/scan_circular.hpp`):
  the constant baseline term (`nG*log(nG/mG)` for Poisson; `nG*log(pTot)+(mG-nG)*log(1-pTot)`
  for Binomial) was recomputed at every DFS node expansion and every upper-bound evaluation.
  Now precomputed once per scan and threaded through via `Traits::llr_with_baseline<Tail>()`.
  For SIMCOUNT=999, N=1000, ~50 DFS expansions per center: eliminates ~50M `log()` calls
  for the baseline term alone.
- **Thread-local frontier pool** (`algorithms/scan_flexible.hpp`): per-DFS-node
  `std::vector<pair<int,double>> new_frontier` heap allocation replaced with a
  `thread_local` object pool. Pool grows to `clustersize` depth then stays constant;
  all DFS allocations amortise to O(1) `clear()` calls after warmup.
- **`std::move` initial frontier**: initial DFS frontier moved into stack frame
  instead of copied.

### Code Quality
- **DFS candidate LLR cache reset** (`algorithms/scan_flexible.hpp`): when building a
  child frontier, parent-frame cached LLR values were computed at `(nZ_parent, mZ_parent)`
  and incorrectly inherited at the child state. Child frontier entries now reset to
  `NEG_INF`, forcing recomputation at the correct `(nZ, mZ)`. Makes
  `cache_candidate_llr=True` consistently useful at every DFS depth.
- **Circular scanner dispatch cleanup**: `algorithms/circular_dispatch.hpp` deleted.
  `scanner_circular_poisson.cpp` and `scanner_circular_binomial.cpp` now call
  `scan_circular_core<Traits, Tail, StatTag>` directly, matching the flexible scanner
  architecture. Eliminates runtime `std::string` comparison to reach a compile-time decision.
- **Scheduler LIFO own queue** (`algorithms/scheduler.hpp`): work-steal scheduler now uses
  LIFO (`back()/pop_back()`) for own queue and FIFO (`front()/pop_front()`) for stealing —
  standard Cilk/TBB pattern for cache locality.
- **CSR validation sorted vectors** (`bindings_support.cpp`): `unordered_set<int>` × N
  replaced with sorted `vector<int>` × N. Reduces memory from O(N × hash_overhead) to
  O(E); duplicate detection is adjacent-element comparison after sort.

### Tests
- `test_core_bounds.py`: added 8 tests —
  `binom_cdf` extreme-`p` regression (`midp_lower + midp_upper = 1` identity for Poisson
  and Binomial including `p=0.99` overflow case), `poisson_middle_p_lower(0)` exact value,
  and three `cache_candidate_llr=True vs False` assertions (Poisson HOT, Binomial HOT,
  Poisson COLD) confirming baseline precompute and cache reset don't alter results.

### Rebuild Required
All changes in this section are C++ only. Run `pip install -e .` to rebuild the compiled
extension before the Python test suite will reflect them.

---

## v0.0.1 post-remediation — 2026-05-30 (second-pass audit fixes, unreleased)

### Bug Fixes
- **Fixed `scan_series` warm_start off-by-one** (`api.py`): priority centers built from
  1-based area IDs were incremented again (`+1`) before being passed to C++, causing
  warm_start to target adjacent nodes instead of the correct high-value areas.
  Fix: pass 1-based IDs directly; C++ binding subtracts 1 to 0-based index.
- **Removed dead denominator re-assignment block** in `scan_series` (`api.py:439-445`):
  two `if expected_arr is None` / `if population_arr is None` guards that could never
  beneficially fire were removed; `_validate_denominator` handles all cross-model rejection.

### New Public API
- `csr_knn_from_coords` promoted to `flexscanx.__all__` and the top-level namespace.
  Previously accessible only via `from flexscanx.graph import csr_knn_from_coords`.

### Safety
- `scan_from_geodataframe`: raises explicit `TypeError` when `coords` or `adjacency`
  are passed via `**kwargs` (previously raised a cryptic Python-internal "multiple values" error).
- `csr_from_geodataframe_contiguity`: migrated from deprecated `sindex.intersection(bounds)`
  to `sindex.query(geom, predicate='intersects')` (GeoPandas ≥ 0.14 / Shapely ≥ 2.0).
  Falls back to the deprecated form for older GeoPandas installations.

### Architecture
- `_validation_calibration.py`, `_validation_graph.py`, `_validation_predictive.py`,
  `_validation_consensus.py` inlined into `validation.py`. Eliminates the three-layer
  import chain `__init__.py → validation.py → _validation_*.py`.
  *(Superseded by the v0.0.1b pass above, which re-split the concerns into
  `_calibration.py`, `_graph_metrics.py`, `_predictive.py`, `_consensus.py`,
  with `validation.py` retained as a thin re-exporting facade — see the
  v0.0.1b Architecture entry for the shipped layout.)*
- `_run_one` closure in `scan_series` replaced by module-level `_run_one_slice(t, ctx)`
  function taking an explicit `_SeriesSliceContext` dataclass. All captured state is
  now explicit and inspectable; eliminates implicit closure over ~30 variables.
- `scan_flexible_hot_restricted` / `scan_flexible_cold_restricted` bare forwarding wrappers
  removed from `scanner_flexible_poisson.cpp`. All four Poisson scanner entry points now
  call `scan_flexible_core<PoissonTraits, …>` directly, matching the Binomial pattern.
- `ScanHandle` type alias added to `api.py` via `TYPE_CHECKING` guard. `handle:` params
  use `_HandleT` instead of `Any`.

### Performance
- `greedy_upper_bound` (C++): replaced per-call `std::vector<char> taken(size, 0)` heap
  allocation with a `thread_local` scratch buffer, eliminating DFS hot-loop heap pressure.
- `seq_entropy(by_time=True)`: O(N×T×K) inner state loop replaced with
  `np.unique(col, return_counts=True)` — vectorized per time column.
- `seq_trans`: Python list comprehension for state-index mapping replaced with a
  numpy lookup array built once from `all_states`.

### Code Quality
- `_validate_scan_common` clusterradius check simplified: `not (r >= 0.0 or (np.isinf(r) and r > 0.0))`
  → `r < 0.0` (equivalent; removes redundant `np.isinf` call).
- `calibrate_llr_threshold._calib_key_impl`: removed misleading `sort_keys=True` from
  `json.dumps(items, ...)` where `items` is a list (sort_keys has no effect on lists).
- `rank` counter in `bindings_scan_result.cpp` widened from `int` to `int64_t` to prevent
  theoretical overflow for very large `simcount` values.
- `seq_dist_gower` removed from `flexscanx.sequence.__all__`. The function body remains
  in `distances.py` (unexported) for potential future generalisation.
- `consensus_sweep` docstring documents the secondary-cluster exclusion behaviour.
- `both_parallel` parameter docstring updated in `scan()` and `scan_series()` to note
  the forced `n_jobs_centers=1` override.
- `sequence/__init__.py` docstring example notes matplotlib optional dependency.

### Tests
- `tests/test_warm_start.py`: warm_start=True must produce identical primary cluster
  areas to warm_start=False on a planted-cluster time series.
- `tests/test_regressions_audit_fixes_v010.py`:
  - `both_parallel=True` produces same HOT/COLD cluster areas as sequential path.
  - `ratio_smoothing='EB'` returns finite, positive Rate_Ratio / Risk_Ratio.
  - `calibrate_llr_threshold(model='BINOMIAL', population=...)` returns valid tau/mu/sigma.
  - `calibrate_llr_threshold(model='BINOMIAL', expected=...)` raises ValueError.
  - `scan_series(..., knn_method='geometry', ...)` completes without error.
  - `consensus_sweep` returns freq values in [0, 1] for N areas.
- `test_public_api_surface.py`: `csr_knn_from_coords` added to `EXPECTED_ROOT_EXPORTS`.
- `test_validation_calibration_cache.py`: updated import from deleted
  `_validation_calibration` to `flexscanx.validation`.
- `test_sequence_distances.py`: `seq_dist_gower` import updated to reference unexported
  symbol via `from flexscanx.sequence.distances import seq_dist_gower`.

---

## v0.0.1 - 2026-05-30

Initial public release of flexscanx.

### Bug Fixes
- Fixed calibration cache key inconsistency: `rantype=None` and
  `rantype="MULTINOMIAL"` now share the same cache entry (previously caused
  permanent cache misses and duplicate simulation runs).
- Fixed `calibrate_llr_threshold` API contract for BINOMIAL model: now accepts
  `population=` (trial counts) instead of `expected=`, matching the `scan()`
  interface. Passing `expected=` for BINOMIAL now raises a clear `ValueError`.
- Fixed `arr_obs` dtype in `scan_series` internal loop (`int` → `np.int32`),
  removing a silent int64→int32 coercion on 64-bit platforms.
- Removed dead code in `scan_series` warm-start block (redundant `prio`
  re-initialisation and unreachable `if prev_result is not None` guard).
- Replaced Python builtin `sum()` on numpy array with `np.sum()` in BINOMIAL
  calibration simulation (performance fix, not correctness).

### New Public API
- `plot_consensus_frequency`: choropleth map of consensus sweep frequencies,
  promoted from internal module to public `__all__`.

### Architecture
- `consensus_sweep` now validates `model`, `scanmethod`, and `stattype` via
  the shared normaliser functions before delegating to the C++ core.
- `_pool_stability_metrics` thin wrapper inlined into its single call site
  (`stability_across_K`).
- `compute_default_n_jobs` import in `_scan_internal.py` moved to module-level
  (removed bottom-of-file re-export anti-pattern).

### Performance
- `calibrate_llr_threshold` BINOMIAL simulation: `sum(population)` →
  `np.sum(population)`.
- `_bfs_distances_py`: `from collections import deque` moved to module level
  (was re-imported on every BFS call).

### Safety Warnings
- `silhouette_graph` / `dunn_graph`: emit `UserWarning` when BFS work exceeds
  250 000 steps (cluster size × graph size), alerting users to potential
  slowness on large graphs.
- `csr_from_geodataframe_contiguity`: emit `UserWarning` on O(N²) brute-force
  fallback when spatial index is unavailable.

### Code Quality
- Duplicate `numba` import in `_validation_graph.py` removed.
- `_rank_kind` named function replaces inline lambdas in `report.py`.
- `cv_glm_gain` fold index type corrected from `float` to `int`.
- `_core.pyi` module docstring added for `help()` discoverability.
- `haversine` `radius` docstring in `csr_knn_from_coords` clarified (radians,
  not degrees).
- `ruff format` pass: 12 files reformatted (blank-line violations in `_utils.py`
  and other modules).

### Documentation
- `pyproject.toml`: `status = "Stable"`, `release_date = "2026-05-30"`.
- 20 documentation files updated by `sync_docs_metadata.py`.

### Tests
- Added cache parity test: `rantype=None` and `rantype="MULTINOMIAL"` share
  the same cache entry.
- Added `calibrate_llr_threshold` BINOMIAL parameter validation tests.
- Added `scan_series(..., warm_start=True, n_jobs_time=1)` coverage.
- Added `export_cluster_report` tests for BOTH+HTML and consensus_areas paths.
- Fixed `test_calibrate_binomial_smoke` to use `population=` (correct kwarg).

### Core Features
- Flexible and Circular spatial scan statistics
- Models: Poisson (count data) and Binomial (population-based)
- Cluster detection: HOT (high-risk), COLD (low-risk), or BOTH
- Statistics: Restricted likelihood ratio (default, ralpha=0.2) and Original
- Time-series workflows with Common Random Numbers (CRN) for variance reduction
- KNN refill mode for robust neighbor selection (K_REFILL default for N ≤ 5000)

### Python API
- `scan()`: Single-slice spatial scan with optional Monte Carlo p-values
- `scan_series()`: Time-series workflow with warm-starts and CRN
- `scan(..., secondary="ALL")`: Enumerate all secondary clusters
- Validation helpers: conductance, boundary_contrast, stability_across_K
- Tuning helpers: consensus_sweep, calibrate_llr_threshold, find_stable_params_all
- Visualization: choropleth, plot_cluster_membership, plot_all_clusters

### C++ Core
- Center-level UB gating and frame-level UB pruning (20-50% speedup)
  *(2026-06 audit: the greedy bound is not a valid upper bound and could
  return suboptimal clusters; both heuristics are now opt-in, off by
  default — see v0.0.1c)*
- Configurable parallelism (smart defaults: 1 thread for N ≤ 5000)
- Middle p-values with ax<0 handling for discrete distributions
- Direction-specific restricted gating (HOT/COLD)
- Deterministic seeding for reproducibility *(2026-06 audit: parallel-path
  determinism under exact LLR ties required the v0.0.1c tie-break fix)*
- Version auto-generation from pyproject.toml

### Performance
- Handles N ≈ 2,000-10,000 areas
- Supports 199-9,999 Monte Carlo simulations
- Optional early stopping with Wilson confidence intervals
- Experimental Alias method for multinomial sampling (env flag: FLEXSCANX_MC_ALIAS=1)

### Build & Distribution
- Python ≥ 3.10
- C++17 core via pybind11
- Windows (MSVC), Linux (GCC), macOS support
- Optional dependencies: [geo] for GeoPandas/libpysal, [jit] for Numba acceleration

### Documentation
- README with usage examples
- `docs/DEVELOPER_GUIDE.md` for development notes
- Academic reference papers in DOCUMENTATION/
- Automated test coverage across models, helpers, and release-gated scripts
  (audit-measured line coverage ~60% overall; 359 tests as of v0.0.1c)

### Code Quality
- Policy-based C++ templates eliminate code duplication
- Comprehensive input validation with clear error messages
- Template-based LLR formulas (single source of truth)
- Explicit root-package re-exports with preserved signatures and docstrings
- Type hints throughout Python API

### Bug Fixes
- **Critical**: Fixed RR key mismatch between C++ bindings and Python export code
  - C++ bindings correctly set `Rate_Ratio` (POISSON) and `Risk_Ratio` (BINOMIAL)
  - Python summary/report/plotting code extracts these keys via `_extract_rr()` helper
  - v0.0.1 policy: no backward-compatibility alias for legacy `RR` input fixtures
  - Added regression tests to verify correct RR export in summaries, DataFrames, and CSV
  - Issue discovered during deep code audit prior to first release

### Post-Audit Hardening (v0.0.1)
- Enforced BINOMIAL population-only API (`expected` rejected for BINOMIAL in `scan()` and `scan_series()`)
- Removed non-public `scan_series_all()` wrapper and deprecated compatibility artifacts
- Added strict CSR validation in graph mode (shape, bounds, self-loop, duplicates, symmetry)
- Clarified `K` vs `clustersize` semantics in runtime and docs:
  - `K` is KNN horizon / candidate universe
  - `clustersize` is max cluster cardinality, capped to `K` when larger
- Added explicit seed range validation (`0..2147483647`)
- Standardized BOTH envelope to use `primary_type` as the canonical direction field
- Fixed sequence analysis defects:
  - `seq_freq()` unique-count path
  - `seq_entropy(by_time=False)` overall entropy computation
  - `seq_dist_hamming()` empty-sequence error contract
- Documentation refreshed to remove obsolete parameters (`normalize_expected`, `gating_threshold`) and stale BOTH/series result examples
- **`find_stable_params_all()` now validates output before returning**:
  - Clamps `K_start` and `S_start` to feasible bounds for small graphs
  - Uses `_validate_tuning_pair()` to fail-fast with clear errors if the search produces invalid `(K, S)`
  - No longer returns impossible values like `K > N`
- **BINOMIAL `rantype` validation is strict**:
  - Invalid `rantype` values for a model raise immediately
  - No silent rewrite of incompatible pairs (e.g., `MULTINOMIAL` with `BINOMIAL`)
- **Series time labels are preserved** through summary/export helpers:
  - `flatten_clusters()`, `clusters_to_dataframe()`, and `summarize_result()` use `result["times"]` when available
  - Integer fallback only when `times` is absent
- Performance script hardening:
  - `scripts/perf_checks.py` validates `grid >= 4` before running (prevents IndexError on small grids)
  - Uses model-aware `rantype` (BINOMIAL uses BINOMIAL, others use MULTINOMIAL) instead of None
- Version fallback now matches repository:
  - `_meta.py` fallback version is `"0.0.1"` (not `"0.0.0.dev"`) for local development

---

## Development History

This unreleased development line includes the following features and improvements:

- Initial Poisson Flexible scanner implementation
- Addition of COLD, CIRCULAR, and Binomial support
- Monte Carlo alias method and center ordering improvements
- KNN refill mode with smart default parallelism
- Code quality improvements:
  - Removed deprecated parameters (`strict_binomial_population`)
  - Standardized error formatting
  - Template-based LLR formulas
  - Version auto-generation system
  - Numba JIT for graph utilities
  - Documented advanced parameters (`mc_progress`, `enable_frame_ub_prune`)

For detailed development history, see `archive/development_audits/`.
