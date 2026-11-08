"""Regression tests for the 2026-06-11 independent audit (v0.0.1c).

Covers:
- AUD-1: default-configuration scans must find the true maximum-likelihood
  cluster (frame-UB prune invalidity regression; audit finding CC-1).
- AUD-2: parallel scans must be deterministic and equal to sequential results,
  including under exact LLR ties (audit finding CC-2).

Both tests FAIL at commit 20f90d7 and pass with the audit fixes
(frame prune default off + deterministic tie-breaking).
"""

import math

import numpy as np
import pytest

import flexscanx as flx
from flexscanx.graph import csr_from_dense


def _grid(n):
    coords = np.array([[i, j] for i in range(n) for j in range(n)], dtype=float)
    N = n * n
    adj = [[0] * N for _ in range(N)]
    for i in range(n):
        for j in range(n):
            c = i * n + j
            if i > 0:
                adj[c][(i - 1) * n + j] = 1
            if i < n - 1:
                adj[c][(i + 1) * n + j] = 1
            if j > 0:
                adj[c][i * n + j - 1] = 1
            if j < n - 1:
                adj[c][i * n + j + 1] = 1
    row_ptr, col_idx = csr_from_dense(adj)
    return coords, (row_ptr, col_idx), N


def _neighbors(row_ptr, col_idx, v):
    return col_idx[row_ptr[v] : row_ptr[v + 1]]


def _connected_subsets(row_ptr, col_idx, N, max_size):
    found = set()
    for start in range(N):
        nbrs0 = frozenset(_neighbors(row_ptr, col_idx, start))
        stack = [(frozenset([start]), nbrs0)]
        while stack:
            cur, frontier = stack.pop()
            if cur in found:
                continue
            found.add(cur)
            if len(cur) >= max_size:
                continue
            for nb in frontier:
                new = cur | {nb}
                if new not in found:
                    newf = (frontier | frozenset(_neighbors(row_ptr, col_idx, nb))) - new
                    stack.append((new, newf))
    return found


def _llr_poisson(nZ, mZ, nG, mG, hot):
    if mZ <= 0 or mG - mZ <= 0:
        return 0.0
    rin, rout = nZ / mZ, (nG - nZ) / (mG - mZ)
    if hot != (rin > rout) or rin == rout:
        return 0.0
    t1 = nZ * math.log(rin) if nZ > 0 else 0.0
    t2 = (nG - nZ) * math.log(rout) if (nG - nZ) > 0 else 0.0
    t0 = nG * math.log(nG / mG) if nG > 0 else 0.0
    return t1 + t2 - t0


def test_default_scan_finds_optimum_small_graph():
    """AUD-1: with library DEFAULTS, the flexible scan must equal brute force.

    Fails at 20f90d7: the default-ON frame UB prune discards the optimum.
    """
    n = 4  # 4x4 grid: brute force is exact and fast
    coords, (row_ptr, col_idx), N = _grid(n)
    S = 5
    subsets = _connected_subsets(list(row_ptr), list(col_idx), N, S)
    failures = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        obs = rng.poisson(8.0, size=N).astype(int)
        obs[0] += int(rng.integers(10, 40))
        exp = np.full(N, 8.0)
        nG, mG = float(obs.sum()), float(exp.sum())
        for hot in (True, False):
            brute = 0.0
            for sset in subsets:
                nZ = float(sum(obs[i] for i in sset))
                mZ = float(sum(exp[i] for i in sset))
                brute = max(brute, _llr_poisson(nZ, mZ, nG, mG, hot))
            res = flx.scan(
                coords=coords,
                adjacency=(row_ptr, col_idx),
                observed=obs.tolist(),
                expected=exp.tolist(),
                model="POISSON",
                scanmethod="FLEXIBLE",
                stattype="ORIGINAL",
                clustertype="HOT" if hot else "COLD",
                clustersize=S,
                K=N,
                simcount=0,
                n_jobs_centers=1,
                # NOTE: deliberately no pruning flags — this asserts DEFAULTS.
            )
            if abs(float(res["stats"]) - brute) > 1e-9 * max(1.0, brute):
                failures.append((seed, hot, float(res["stats"]), brute))
    assert not failures, (
        "default-configuration scan missed the maximum-likelihood cluster: "
        f"{failures[:5]} (scan_stats vs brute_force_stats)"
    )


def test_default_scan_matches_unpruned_10x10():
    """AUD-1b: defaults must equal the exact (prune-disabled) scan.

    Direct regression of the audit's 76/80 sequential divergence sweep.
    """
    coords, adjacency, N = _grid(10)
    diverged = []
    for seed in (32, 33, 34, 35, 36, 37):  # seeds with confirmed divergence
        rng = np.random.default_rng(seed)
        observed = rng.poisson(10, size=N)
        observed[25:35] += 50
        observed[65:75] = 0
        expected = np.full(N, 10.0)
        for ct in ("HOT", "COLD"):
            default = flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                scanmethod="FLEXIBLE",
                clustertype=ct,
                clustersize=15,
                n_jobs_centers=1,
                simcount=0,
            )
            exact = flx.scan(
                coords=coords,
                adjacency=adjacency,
                observed=observed,
                expected=expected,
                model="POISSON",
                scanmethod="FLEXIBLE",
                clustertype=ct,
                clustersize=15,
                n_jobs_centers=1,
                simcount=0,
                enable_frame_ub_prune=False,
                enable_center_ub_gating=False,
            )
            if default["stats"] != exact["stats"]:
                diverged.append((seed, ct, default["stats"], exact["stats"]))
    assert not diverged, f"default scan != exact scan: {diverged}"


def test_binomial_restricted_gating_large_population():
    """AUD-3: RESTRICTED BINOMIAL mid-p gating must be correct for large n.

    At 20f90d7, binom_cdf's recurrence base (1-p)^n underflows for
    n*|log(1-p)| > ~708 (here n=2000, p~0.5), making midp_upper collapse to
    0.5*pmf — far too small — so inadmissible centers pass the gate.

    Reference values (scipy.stats.binom, recorded in audit evidence
    audit_scratch/cpp_core/task1d_decisive.py):
      obs4=1010 -> true midp_upper = 0.3455 > ralpha=0.3 -> center DROPPED
      obs4=1040 -> true midp_upper = 0.0559 < ralpha=0.3 -> center SELECTED
    """
    N = 9
    coords = np.array([[i % 3, i // 3] for i in range(N)], dtype=float)
    adj = {i: [] for i in range(N)}
    for i in range(N):
        r, c = i // 3, i % 3
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                adj[i].append(nr * 3 + nc)
    row_ptr, col_idx = [0], []
    for i in range(N):
        col_idx.extend(sorted(adj[i]))
        row_ptr.append(len(col_idx))
    pop = np.full(N, 2000.0)

    def run(obs4):
        observed = np.full(N, 1000, dtype=int)
        observed[4] = obs4
        res = flx.scan(
            coords=coords,
            adjacency=(row_ptr, col_idx),
            observed=observed,
            population=pop,
            model="BINOMIAL",
            scanmethod="FLEXIBLE",
            clustertype="HOT",
            clustersize=1,
            n_jobs_centers=1,
            simcount=0,
            K=8,
            stattype="RESTRICTED",
            ralpha=0.3,
        )
        return sorted(res.get("area", []))

    # true midp 0.3455 > 0.3: every center inadmissible -> empty result
    assert run(1010) == [], (
        "center with true mid-p 0.3455 admitted at ralpha=0.3 " "(binom_cdf large-n underflow)"
    )
    # true midp 0.0559 < 0.3: node 4 (1-based id 5) must be selected
    assert run(1040) == [5], "admissible center not selected"


def test_silhouette_graph_matches_rousseeuw():
    """AUD-4: silhouette_graph must implement Rousseeuw (1987).

    At 20f90d7 the intra-cluster mean included each node's own zero distance
    (a(i) must average over the OTHER |C|-1 members) and singletons returned
    1.0 (Rousseeuw defines s(i)=0). Hand-computed reference on a path graph
    0-1-2-3 with membership {0,1}:
      node0: a=1,   b=(2+3)/2=2.5 -> s=0.6
      node1: a=1,   b=(1+2)/2=1.5 -> s=1/3
      mean = 7/15
    """
    from flexscanx.validation import silhouette_graph

    row_ptr = [0, 1, 3, 5, 6]
    col_idx = [1, 0, 2, 1, 3, 2]  # path 0-1-2-3
    res = silhouette_graph(row_ptr, col_idx, [1, 1, 0, 0])
    assert abs(res["values"][0] - 0.6) < 1e-12, res
    assert abs(res["values"][1] - (1.0 / 3.0)) < 1e-12, res
    assert abs(res["mean"] - (0.6 + 1.0 / 3.0) / 2.0) < 1e-12, res

    singleton = silhouette_graph(row_ptr, col_idx, [1, 0, 0, 0])
    assert singleton["values"] == [0.0], "Rousseeuw defines s(i)=0 for singletons"
    assert singleton["mean"] == 0.0


def test_boundary_contrast_numba_matches_fallback(monkeypatch):
    """AUD-5: boundary_contrast must not depend on whether numba is installed.

    At 20f90d7 the numba path used population variance while the pure-Python
    fallback used np.std(ddof=1) — std_delta/std_z differed by
    sqrt(n/(n-1)).
    """
    import flexscanx._graph_metrics as gm

    if gm._bc_core is None:
        pytest.skip("numba not installed; only one path available")

    rng = np.random.default_rng(7)
    n = 4
    coords, (row_ptr, col_idx), N = _grid(n)
    obs = rng.poisson(8.0, size=N).astype(int)
    exp = np.full(N, 8.0)
    mem = [1 if i < 6 else 0 for i in range(N)]

    jit = gm.boundary_contrast(obs, exp, row_ptr, col_idx, mem, model="POISSON")
    monkeypatch.setattr(gm, "_bc_core", None)  # force the pure-Python fallback
    py = gm.boundary_contrast(obs, exp, row_ptr, col_idx, mem, model="POISSON")
    for key in ("n_edges", "mean_delta", "std_delta", "mean_z", "std_z"):
        assert abs(jit[key] - py[key]) < 1e-12, (key, jit[key], py[key])


def test_calibration_cache_key_includes_seed():
    """AUD-6: the on-disk calibration cache key must distinguish seeds.

    At 20f90d7 the key omitted `seed`, so calibrate(seed=2) silently returned
    a cached calibrate(seed=1) estimate.
    """
    from flexscanx._calibration import _calib_key

    base = dict(model="POISSON", K=10, clustersize=10, N=50, n_sims=99)
    k1 = _calib_key(seed=1, **base)
    k2 = _calib_key(seed=2, **base)
    assert k1 != k2, "cache key must include the seed"
    assert _calib_key(seed=1, **base) == k1, "key must be deterministic"


def test_exact_flexible_large_k_warns():
    """AUD-7: requesting an exact FLEXIBLE scan with K>25 must warn.

    The exact connected-subset enumeration is exponential in K (audit F-7b);
    the bounded heuristics are opt-in because they may return suboptimal
    clusters. A path graph keeps the actual enumeration cheap (segments only).
    """
    N = 36
    coords = np.array([[float(i), 0.0] for i in range(N)])
    row_ptr, col_idx = [0], []
    for i in range(N):
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < N]
        col_idx.extend(nbrs)
        row_ptr.append(len(col_idx))
    obs = np.full(N, 5)
    obs[0] += 10
    with pytest.warns(UserWarning, match="exponential in K"):
        flx.scan(
            coords=coords,
            adjacency=(row_ptr, col_idx),
            observed=obs.tolist(),
            expected=[5.0] * N,
            model="POISSON",
            scanmethod="FLEXIBLE",
            stattype="ORIGINAL",
            clustertype="HOT",
            clustersize=10,
            K=30,
            simcount=0,
            n_jobs_centers=1,
        )


def test_parallel_ties_deterministic():
    """AUD-2: parallel scans must return identical clusters across repeats and
    match the sequential result, even when distinct clusters tie exactly.

    Fails (flakily but with high probability over 25 reps) at 20f90d7: two
    5-node zero-count COLD clusters have bit-identical LLR and the winner
    depends on thread arrival order.
    """
    coords, adjacency, N = _grid(10)
    rng = np.random.default_rng(42)
    observed = rng.poisson(10, size=N)
    observed[25:35] += 50
    observed[65:75] = 0  # zero block => exact COLD ties
    expected = np.full(N, 10.0)

    def run(jobs):
        return flx.scan(
            coords=coords,
            adjacency=adjacency,
            observed=observed,
            expected=expected,
            model="POISSON",
            scanmethod="FLEXIBLE",
            clustertype="BOTH",
            clustersize=15,
            n_jobs_centers=jobs,
            simcount=0,
        )

    seq = run(1)
    outcomes = {"hot": set(), "cold": set()}
    for _ in range(25):
        res = run(4)
        for k in ("hot", "cold"):
            outcomes[k].add((res[k]["stats"], tuple(res[k]["area"])))
    for k in ("hot", "cold"):
        assert (
            len(outcomes[k]) == 1
        ), f"parallel {k.upper()} results not unique across 25 runs: {outcomes[k]}"
        ((stats, area),) = outcomes[k]
        assert stats == seq[k]["stats"], f"parallel {k} stats != sequential"
        assert list(area) == list(seq[k]["area"]), f"parallel {k} area != sequential"
