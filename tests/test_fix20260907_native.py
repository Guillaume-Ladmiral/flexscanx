"""Native-build and compiled-core regressions of the 2026-09-07 run.

FXA-160 (thread linkage), FXA-168 (source digest), D4 (tie-boundary warning),
D6 (interruptibility) and U-6 (effective-universe binding) are pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKELISTS = REPO_ROOT / "CMakeLists.txt"


# ---------------------------------------------------------------------------
# FXA-160: both native targets link the threading library
# ---------------------------------------------------------------------------


def test_cmake_finds_threads_and_links_them_into_both_targets() -> None:
    text = CMAKELISTS.read_text(encoding="utf-8")
    assert "find_package(Threads REQUIRED)" in text
    assert "set(THREADS_PREFER_PTHREAD_FLAG ON)" in text
    assert re.search(r"target_link_libraries\(_core\s+PRIVATE\s+Threads::Threads\)", text)
    assert re.search(
        r"target_link_libraries\(flexscanx_core_tests\s+PRIVATE\s+Threads::Threads\)", text
    )
    # The find_package sits outside the PYEXT block so the test-only build gets it too.
    assert text.index("find_package(Threads REQUIRED)") < text.index("if (FLEXSCANX_BUILD_PYEXT)")


# ---------------------------------------------------------------------------
# FXA-168: the loaded extension was built from the current native sources
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402

NATIVE_ROOT = REPO_ROOT / "src" / "flexscanx_core"
NATIVE_SUFFIXES = {".cpp", ".hpp", ".in"}


def expected_native_source_digest() -> str:
    """Recompute CMake's digest: sorted '<relative path>\\n<sha256>\\n' records."""
    paths = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in NATIVE_ROOT.rglob("*")
        if p.is_file() and p.suffix in NATIVE_SUFFIXES
    )
    paths = sorted(paths + ["CMakeLists.txt"])
    manifest = "".join(
        f"{rel}\n{hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()}\n" for rel in paths
    )
    return hashlib.sha256(manifest.encode("ascii")).hexdigest()


def test_loaded_extension_was_built_from_the_current_native_sources() -> None:
    import flexscanx._core as core

    assert hasattr(core, "source_digest"), (
        "the loaded flexscanx._core predates the source digest; rebuild it with "
        "`python -m pip install -e . --no-build-isolation`"
    )
    built = core.source_digest()
    assert re.fullmatch(r"[0-9a-f]{64}", built), built
    assert built == expected_native_source_digest(), (
        "flexscanx._core is STALE: it was built from different native sources than the "
        "working tree contains. The editable install does not rebuild C++ automatically; "
        "run `python -m pip install -e . --no-build-isolation` and re-run the suite."
    )


# ---------------------------------------------------------------------------
# D4 / FXA-187: a tied K-universe boundary is announced and documented
# ---------------------------------------------------------------------------

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import flexscanx as flx  # noqa: E402


def _grid(side: int) -> tuple[np.ndarray, tuple[list[int], list[int]]]:
    n = side * side
    coords = np.array([[i % side, i // side] for i in range(n)], dtype=float)
    row_ptr, col_idx = [0], []
    for i in range(n):
        r, c = divmod(i, side)
        neighbours = []
        if r > 0:
            neighbours.append(i - side)
        if r < side - 1:
            neighbours.append(i + side)
        if c > 0:
            neighbours.append(i - 1)
        if c < side - 1:
            neighbours.append(i + 1)
        col_idx.extend(sorted(neighbours))
        row_ptr.append(len(col_idx))
    return coords, (row_ptr, col_idx)


def _brute_force_tie_boundary_count(coords: np.ndarray, K: int, radius: float = np.inf) -> int:
    """Independent reference: centres whose K-th and (K+1)-th nearest are exactly equidistant."""
    n = len(coords)
    if K >= n:
        return 0
    count = 0
    for c in range(n):
        d = np.sqrt(((coords - coords[c]) ** 2).sum(axis=1))
        order = sorted(range(n), key=lambda i: (d[i], i))  # centre first at distance 0
        boundary, beyond = d[order[K - 1]], d[order[K]]
        if boundary == beyond and boundary <= radius:
            count += 1
    return count


@pytest.mark.parametrize("side,K", [(3, 5), (4, 8), (5, 6), (4, 2)])
def test_tie_boundary_count_matches_an_independent_reference(side: int, K: int) -> None:
    coords, adjacency = _grid(side)
    expected = _brute_force_tie_boundary_count(coords, K)
    assert expected > 0, "fixture must contain a tie at the boundary"
    with pytest.warns(UserWarning, match="equidistant neighbours") as record:
        handle = flx.build_static(coords, adjacency, K=K)
    assert flx.describe_static(handle)["knn_tie_boundary_centers"] == expected
    tie = [w for w in record if "equidistant neighbours" in str(w.message)]
    assert len(tie) == 1
    assert f"{expected} of {side * side} centres" in str(tie[0].message)
    assert f"K={K}" in str(tie[0].message)
    assert tie[0].filename == __file__


def test_no_tie_warning_on_tie_free_geometry_or_when_K_equals_N() -> None:
    coords, adjacency = _grid(4)
    jittered = coords + np.random.default_rng(1).uniform(-1e-3, 1e-3, coords.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            flx.describe_static(flx.build_static(jittered, adjacency, K=8))[
                "knn_tie_boundary_centers"
            ]
            == 0
        )
        assert (
            flx.describe_static(flx.build_static(coords, adjacency, K=16))[
                "knn_tie_boundary_centers"
            ]
            == 0
        )


def test_tie_beyond_the_radius_is_not_announced() -> None:
    coords, adjacency = _grid(3)
    # K=5 on the 3x3 lattice ties at distance 2 (corners) and sqrt(2) (edges).  A
    # radius of 1.5 excludes the corners' tied group but keeps the edges' one, so
    # exactly the four edge centres are announced; a radius of 1.2 excludes both.
    with pytest.warns(UserWarning, match="4 of 9 centres"):
        handle = flx.build_static(coords, adjacency, K=5, clusterradius=1.5)
    assert flx.describe_static(handle)["knn_tie_boundary_centers"] == 4
    assert _brute_force_tie_boundary_count(coords, 5, radius=1.5) == 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tight = flx.build_static(coords, adjacency, K=5, clusterradius=1.2)
    assert flx.describe_static(tight)["knn_tie_boundary_centers"] == 0
    assert _brute_force_tie_boundary_count(coords, 5, radius=1.2) == 0


def test_scan_and_scan_series_and_calibration_surface_the_warning() -> None:
    coords, adjacency = _grid(3)
    obs = np.array([1, 1, 1, 1, 9, 1, 1, 1, 1])
    with pytest.warns(UserWarning, match="equidistant neighbours"):
        flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=obs,
            expected=np.full(9, 2.0),
            K=5,
            simcount=0,
        )
    with pytest.warns(UserWarning, match="equidistant neighbours"):
        flx.scan_series(
            coords=coords,
            adjacency=adjacency,
            observed=[obs],
            expected=[np.full(9, 2.0)],
            K=5,
            simcount=0,
        )
    with pytest.warns(UserWarning, match="equidistant neighbours"):
        flx.calibrate_llr_threshold(
            coords=coords,
            adjacency=adjacency,
            observed=obs,
            expected=np.full(9, 2.0),
            K=5,
            clustersize=3,
            n_sims=2,
        )


def test_api_reference_documents_the_index_resolution_of_ties() -> None:
    text = (REPO_ROOT / "API_REFERENCE.md").read_text(encoding="utf-8")
    assert "Equidistant neighbours" in text
    assert "knn_tie_boundary_centers" in text
    assert "depends on the input row order" in text


# ---------------------------------------------------------------------------
# D6 / FXA-188: a running scan honours SIGINT within a bounded time
# ---------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402

_INTERRUPT_PROBE_SRC = """
import json, os, signal, sys, threading, time, warnings
import numpy as np
import flexscanx as flx
warnings.simplefilter("ignore")
mode, n_jobs, sim, delay = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
side = 10; n = side * side
coords = np.array([[i % side, i // side] for i in range(n)], dtype=float)
coords += np.random.default_rng(0).uniform(-1e-3, 1e-3, coords.shape)
row_ptr, col_idx = [0], []
for i in range(n):
    r, c = divmod(i, side); nb = []
    if r > 0: nb.append(i - side)
    if r < side - 1: nb.append(i + side)
    if c > 0: nb.append(i - 1)
    if c < side - 1: nb.append(i + 1)
    col_idx.extend(sorted(nb)); row_ptr.append(len(col_idx))
rng = np.random.default_rng(1)
observed = rng.poisson(4.0, n); expected = np.full(n, 4.0)
kw = dict(coords=coords, adjacency=(row_ptr, col_idx), observed=observed, expected=expected,
          K=10, clustersize=6, stattype="ORIGINAL", n_jobs_centers=n_jobs, seed=3)
if mode == "interrupt":
    threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
t0 = time.perf_counter()
outcome = "completed"
try:
    flx.scan(simcount=sim, **kw)
except KeyboardInterrupt:
    outcome = "KeyboardInterrupt"
print(json.dumps({"outcome": outcome, "elapsed": time.perf_counter() - t0}))
"""
_INTERRUPT_PROBE = textwrap.dedent(_INTERRUPT_PROBE_SRC)


def _probe(mode: str, n_jobs: int, simcount: int, delay: float = 0.0) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _INTERRUPT_PROBE, mode, str(n_jobs), str(simcount), str(delay)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_monte_carlo_scan_is_interrupted_within_a_bounded_time(n_jobs: int) -> None:
    """The matched control is essential: without it a short elapsed time could mean the
    scan simply finished.  The control is sized so that it takes several seconds."""
    pilot = _probe("control", n_jobs, 400)
    assert pilot["outcome"] == "completed"
    per_replicate = pilot["elapsed"] / 400
    simcount = int(max(400, min(200_000, 3.0 / max(per_replicate, 1e-6))))
    control = _probe("control", n_jobs, simcount)
    assert control["outcome"] == "completed"
    assert control["elapsed"] >= 1.5, control
    delay = 0.4
    interrupted = _probe("interrupt", n_jobs, simcount, delay)
    assert interrupted["outcome"] == "KeyboardInterrupt", interrupted
    # Honoured promptly: within the 50 ms poll cadence plus one replicate's worth of work,
    # generously bounded, and far below the uninterrupted run.
    assert interrupted["elapsed"] < delay + 0.75, (interrupted, control)
    assert interrupted["elapsed"] < 0.5 * control["elapsed"], (interrupted, control)


# ---------------------------------------------------------------------------
# U-6: the engine's maximum equals an independent enumeration over the engine's
# own effective universes on the ring geometries the 2026-09-02 audit could not
# account for
# ---------------------------------------------------------------------------

import itertools  # noqa: E402
import math  # noqa: E402


def _ring_case(n: int, seed: int):
    """The 2026-09-02 reference's ring fixture (evidence/31_enumeration_reference.py)."""
    r = np.random.default_rng(seed)
    pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    coords = np.array(pts, dtype=float)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if r.random() < 0.35:
                A[i, j] = A[j, i] = 1
    row_ptr, col_idx = flx.csr_from_dense(A)
    observed = r.integers(0, 12, size=n).astype(np.int64)
    expected = r.uniform(0.5, 6.0, size=n)
    return coords, (row_ptr, col_idx), observed, expected


def _llr_poisson(nZ: float, mZ: float, nG: float, mG: float, hot: bool) -> float:
    """Kulldorff Poisson LLR, one-sided, in the numerically stable form of llr_poisson.hpp."""
    nOut, mOut = nG - nZ, mG - mZ
    if mZ <= 0.0 or mOut <= 0.0 or nG <= 0.0 or mG <= 0.0:
        return 0.0
    rIn, rOut = nZ / mZ, nOut / mOut
    if hot and not rIn > rOut:
        return 0.0
    if not hot and not rIn < rOut:
        return 0.0

    def contribution(count: float, exposure: float) -> float:
        if not count > 0.0:
            return 0.0
        return count * math.log1p((count * mG - nG * exposure) / (nG * exposure))

    return max(0.0, contribution(nZ, mZ) + contribution(nOut, mOut))


def _connected_windows(universe: list[int], centre: int, adjacency, max_size: int):
    """Every subset of ``universe`` that contains ``centre``, has at most ``max_size``
    areas and is connected in the subgraph induced on ``universe``."""
    row_ptr, col_idx = adjacency
    uni = set(universe)
    nbrs = {i: {j for j in col_idx[row_ptr[i] : row_ptr[i + 1]] if j in uni} for i in universe}
    others = [i for i in universe if i != centre]
    for size in range(0, max_size):
        for combo in itertools.combinations(others, size):
            S = {centre, *combo}
            seen, stack = {centre}, [centre]
            while stack:
                node = stack.pop()
                for j in nbrs[node]:
                    if j in S and j not in seen:
                        seen.add(j)
                        stack.append(j)
            if seen == S:
                yield tuple(sorted(S))


def _reference_maximum(handle, adjacency, observed, expected, clustersize, hot):
    import flexscanx._core as core

    nG, mG = float(observed.sum()), float(expected.sum())
    best_stat, best_area = 0.0, ()
    n = len(observed)
    for centre in range(n):
        universe = core.DescribeUniverse(handle, centre)["flexible"]
        for S in _connected_windows(universe, centre, adjacency, clustersize):
            value = _llr_poisson(
                float(sum(observed[i] for i in S)), float(sum(expected[i] for i in S)), nG, mG, hot
            )
            if value > best_stat or (
                value == best_stat and value > 0.0 and (len(S), S) < (len(best_area), best_area)
            ):
                best_stat, best_area = value, S
    return best_stat, best_area


_RING_CASES = [  # (n, seed index, K, clustersize, tail) - the four unexplained 2026-09-02 cases
    (5, 3, 4, 2, "HOT"),
    (5, 5, 4, 4, "COLD"),
    (7, 1, 6, 3, "HOT"),
    (7, 3, 6, 4, "COLD"),
]


def test_describe_universe_reports_the_effective_universe() -> None:
    import flexscanx._core as core

    coords, adjacency, _obs, _exp = _ring_case(7, 1 * 97 + 7)
    handle = flx.build_static(coords, adjacency, K=4)
    for centre in range(7):
        info = core.DescribeUniverse(handle, centre)
        assert info["center"] == centre
        assert centre in info["flexible"] and len(info["flexible"]) == 4
        assert info["flexible"] == sorted(info["flexible"])
        assert set(info["circular_order"]) == set(info["flexible"])
        assert info["circular_order"][0] == centre
    masked = core.DescribeUniverse(handle, 0, excluded=[1, 2])
    assert 1 not in masked["flexible"] and 2 not in masked["flexible"]
    assert len(masked["flexible"]) == 4  # refill replaces the excluded neighbours
    with pytest.raises(ValueError):
        core.DescribeUniverse(handle, 99)


@pytest.mark.parametrize("n,seed_index,K,clustersize,tail", _RING_CASES)
def test_engine_maximum_equals_enumeration_over_its_own_universes(
    n: int, seed_index: int, K: int, clustersize: int, tail: str
) -> None:
    coords, adjacency, observed, expected = _ring_case(n, seed_index * 97 + n)
    handle = flx.build_static(coords, adjacency, K=K)
    result = flx.scan(
        handle=handle,
        observed=observed,
        expected=expected,
        K=K,
        clustersize=clustersize,
        stattype="ORIGINAL",
        clustertype=tail,
        scanmethod="FLEXIBLE",
        simcount=0,
    )
    ref_stat, ref_area = _reference_maximum(
        handle, adjacency, observed, expected, clustersize, hot=(tail == "HOT")
    )
    engine_area = tuple(sorted(a - 1 for a in result["area"]))
    assert result["stats"] == pytest.approx(ref_stat, rel=1e-9, abs=1e-12)
    if ref_stat > 0.0:
        assert engine_area == ref_area
