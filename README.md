# flexscanx

**Flexible and circular spatial scan statistics with a high-performance C++17 core**

<!-- BEGIN GENERATED: readme_release_metadata -->
Version 0.3.3 | Release 2026-11-08 | Release v0.3.3 — 2026-11-08; licensing decision D8 open
<!-- END GENERATED: readme_release_metadata -->

`flexscanx` is a Python package for flexible and circular spatial scan statistics with
a compiled C++17 core. It supports Poisson and Binomial models, HOT/COLD/BOTH scans,
time-series workflows, and downstream summary/plotting helpers.

> **Not a port — a corrected reimplementation.** flexscanx *starts from* the R
> package **rflexscan** (and the FleXScan engine it wraps) but is a **new
> implementation that deliberately fixes critical algorithmic and mathematical
> errors in rflexscan/FleXScan** and expands its capabilities. rflexscan is the
> **starting point, not a source of truth**: correctness is established against
> the scan statistics' mathematical definitions and flexscanx's own tests, not by
> reproducing rflexscan's output. Where the two differ and rflexscan is wrong,
> flexscanx's result is the intended correction. See
> [ECOSYSTEM.md](ECOSYSTEM.md) "Source of truth".

> **Project family.** flexscanx is the spatial-scan application layer of a
> four-project family: traminex is the sequence-analysis foundation,
> weightedcluster owns sequence clustering, py_seqplot owns the dedicated
> plotnine visualization layer, and flexscanx owns spatial scan statistics. Its
> `sequence/` tools mirror traminex, but flexscanx **installs and runs without**
> traminex (copies kept identical via test-only parity checks). See
> [ECOSYSTEM.md](ECOSYSTEM.md) for the charter, [DUPLICATES.md](DUPLICATES.md)
> for the duplicate inventory, and [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the
> convergence plan. Any AI assistant should read [AGENTS.md](AGENTS.md) first.

### Tested traminex parity revision

The `sequence/` parity suite is tested against traminex commit
`2cbb3472736d0ccce32a291241983a12b8921122`. This is test provenance only:
traminex remains absent from Flexscanx's runtime dependencies and Flexscanx
continues to install and run standalone. The exact commit matters because
traminex long changed contracts while keeping version `0.0.8`.

Strict parity runs verify the imported traminex checkout before executing B1-B5:

The `[test]` extra deliberately does not declare a traminex distribution. A
strict parity caller must supply a trusted checkout at the pinned commit; CI
clones, detaches, and verifies that checkout before installing it for B1-B5.

```bash
FLEXSCANX_PARITY_STRICT=1 pytest -q \
  tests/test_traminex_parity_b{1_hamming,2_trans,3_entropy,4_freq,5_plots}.py
```

**Divergences from rflexscan (what flexscanx corrects/changes and why):** see [docs/COMPARATIVE_AUDIT_RFLEXSCAN_VS_FLEXSCANX.md](docs/COMPARATIVE_AUDIT_RFLEXSCAN_VS_FLEXSCANX.md). Differences are analysed on the maths; where rflexscan is wrong, flexscanx's behaviour is the deliberate correction.

## Install a release

flexscanx is not on PyPI. Each release is published on the project's GitHub
[releases page](https://github.com/Guillaume-Ladmiral/flexscanx/releases), with
its packages and a `SHA256SUMS` file. The compiled wheel is built for Linux
x86-64 and CPython 3.12; pip takes NumPy, pandas and SciPy from PyPI:

```bash
python -m pip install https://github.com/Guillaume-Ladmiral/flexscanx/releases/download/v0.3.3/flexscanx-0.3.3-cp312-cp312-linux_x86_64.whl
python -c "import flexscanx as flx; print('core:', flx.core_version())"
```

On another platform or Python version (3.10 or newer), install the source
package `flexscanx-0.3.3.tar.gz` from the same release instead; pip then compiles
the C++17 core, which needs a C++17 compiler. Optional features come as extras
(`plot`, `geo`, `jit`, `knn`, `all`); to add one, put its name before the
address, for example `python -m pip install "flexscanx[plot] @ <wheel address>"`.

## Install From This Repo

Active development targets Ubuntu 24.04. The editable install builds the C++17
extension through scikit-build-core and pybind11. SciPy is a required runtime
dependency because the public KNN graph helper used below builds its default
graph with `scipy.spatial.cKDTree`; its `backend="balltree"` and
`metric="haversine"` options need scikit-learn, installed by the optional
`flexscanx[knn]` extra.

```bash
python -m pip install -e .
python -c \
  "import flexscanx as flx; print('core:', flx.core_version())"
```

## First Scan

```python
import numpy as np
import flexscanx as flx

coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
observed = np.random.poisson(10, size=len(coords))
expected = np.full(len(coords), 10.0)
row_ptr, col_idx = flx.csr_knn_from_coords(coords, k=8)

result = flx.scan(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed,
    expected=expected,
    model="POISSON",
    clustertype="HOT",
    K=20,
    simcount=199,
    seed=42,
)

print(result["area"], result["stats"], result["pval"])
```

## First Series Scan

```python
observed_tn = np.random.poisson(10, size=(12, len(coords)))
expected_tn = np.full_like(observed_tn, 10.0, dtype=float)

series = flx.scan_series(
    coords=coords,
    adjacency=(row_ptr, col_idx),
    observed=observed_tn,
    expected=expected_tn,
    model="POISSON",
    clustertype="BOTH",
    K=20,
    simcount=0,
    crn=True,
)

print(len(series["per_time"]), series["times"][:3])
```

## Documentation

- **All documentation in one place:** [docs/README.md](docs/README.md)
- Getting started and current file map: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- Public API contract: [API_REFERENCE.md](API_REFERENCE.md)
- Practical deeper usage: [docs/ADVANCED_USAGE.md](docs/ADVANCED_USAGE.md)
- Executable workflows: [vignettes/workflows/](vignettes/workflows/)
- Validation and method notes: [vignettes/notes/](vignettes/notes/)
- Vignette index: [vignettes/README.md](vignettes/README.md)
- Build and repository notes: [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)
- Supported tooling and release scripts: [scripts/README.md](scripts/README.md)
- Release history: [CHANGELOG.md](CHANGELOG.md)

## Worked Examples

- Manhattan end-to-end workflow: [vignettes/workflows/Manhattan_Breast_Cancer.qmd](vignettes/workflows/Manhattan_Breast_Cancer.qmd)
- Time-series trajectory analysis: [vignettes/workflows/sequence_analysis_workflow.qmd](vignettes/workflows/sequence_analysis_workflow.qmd)
- Stability-focused screening: [vignettes/workflows/stable_significant_workflow.qmd](vignettes/workflows/stable_significant_workflow.qmd)

For Monte Carlo controls, plotting/reporting helpers (including `plot_consensus_frequency`
for consensus sweep maps), KNN/GeoDataFrame patterns, and performance/diagnostic switches,
use [docs/ADVANCED_USAGE.md](docs/ADVANCED_USAGE.md).
